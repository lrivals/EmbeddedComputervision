"""Integer failure head (T39.5): the `FailureHead` of T39.1 on the integer features.

Input: the integer pooled features of `yolo.infer.failure.features_int`, p_j (|p_j| ≤ 127):
per channel of each feature layer, the int32 sum shifted right by ⌈log2 HW⌉ with rounding,
and the max. One LSB of p_j is worth c_j (`dequant_scales`).

First layer, with the standardization folded in (z = (x − μ)/σ, x_j = c_j·p_j) and an
integer zero point zp_j = round(μ_j / c_j) per feature:

    pre_i = Σ_j W_ij (c_j p_j − μ_j)/σ_j + b_i = Σ_j W'_ij (p_j − zp_j) + b'_i
    W'_ij = W_ij c_j / σ_j            b'_i = b_i − Σ_j W_ij (μ_j − c_j zp_j) / σ_j

Without the zero point, Σ_j W'_ij p_j nearly cancels a large b' (the features sit far from 0)
and the int8 rounding of W' is multiplied by |p|; with it, p − zp is centred. W' is quantized
per row (int8, scale s_w,i), b' to int32 at s_w,i: acc_i = Σ_j qW'_ij (p_j − zp_j) + qb_i, so
pre_i ≈ s_w,i · acc_i. Hidden output: requantization to the scale s_h (M0_i, n as a
conv, §9.3), ReLU, saturation to [0, 127]. Second layer: int8 row at scale s_w2, int32 bias
at s_h·s_w2; the int32 accumulator is the logit at scale s_h·s_w2. The score is that
accumulator for the "miss" target and its opposite for the "tcp" target (the failure score
1 − σ(f) decreases with f), so a high score is an alarm at every stage.

Bounds: |p − zp| ≤ 255, |acc_i| ≤ 127·255·d + |qb| (d = 1,536 for Tiny-YOLOv3: 5·10⁷ < 2³¹);
|acc₂| ≤ 127·127·h + |qb₂|. `hls/kernels/failure_head.hpp` computes the same integers (C-sim).
"""

from dataclasses import dataclass

import numpy as np

from yolo.quant.int_layers import requantize
from yolo.quant.quantize import quantize_bias, requant_params, round_half_up

QMAX = 127


def _quant_rows(W):
    """Per-row int8 quantization: (qW, s_w)."""
    sw = np.maximum(np.abs(W).max(axis=1), 1e-12) / QMAX
    return np.clip(round_half_up(W / sw[:, None]), -QMAX, QMAX).astype(np.int8), sw


@dataclass
class IntFailureHead:
    zp: np.ndarray    # (d,) int, zero point of the features
    qW1: np.ndarray   # (h, d) int8
    qb1: np.ndarray   # (h,) int32
    M0: np.ndarray    # (h,) int32
    n: int
    qW2: np.ndarray   # (h,) int8
    qb2: int
    sign: int         # +1 (miss target) or −1 (tcp target)
    sw1: np.ndarray
    sh: float
    sw2: float

    @classmethod
    def from_float(cls, head, scales, calib_p):
        """`head`: `FailureHead`; `scales`: c_j (`dequant_scales`); `calib_p`: integer
        features of calibration images, which set the hidden scale s_h (max over them)."""
        (l1, l2) = head.params
        c = np.asarray(scales, np.float64)
        zp = round_half_up(head.mean / c).astype(np.int64)
        W = l1["W"] * (c / head.std)[None]
        b = l1["b"] - l1["W"] @ ((head.mean - c * zp) / head.std)
        qW1, sw1 = _quant_rows(W)
        qb1 = quantize_bias(b, 1.0, sw1)
        acc = (np.asarray(calib_p, np.int64) - zp) @ qW1.T.astype(np.int64) + qb1
        hmax = max(float((acc * sw1).max()), 1e-6)
        sh = hmax / QMAX
        M0, n = requant_params(1.0, sw1, sh)
        w2 = l2["W"][0]
        sw2 = max(float(np.abs(w2).max()), 1e-12) / QMAX
        qW2 = np.clip(round_half_up(w2 / sw2), -QMAX, QMAX).astype(np.int8)
        qb2 = int(quantize_bias(l2["b"], sh, np.array([sw2]))[0])
        return cls(zp, qW1, qb1, M0, n, qW2, qb2, 1 if head.target == "miss" else -1, sw1, sh,
                   sw2)

    @property
    def d(self):
        return self.qW1.shape[1]

    @property
    def hidden(self):
        return self.qW1.shape[0]

    def hidden_int(self, p):
        """(N, h) int64 in [0, 127]."""
        acc = ((np.asarray(p, np.int64) - self.zp) @ self.qW1.T.astype(np.int64)
               + self.qb1.astype(np.int64))
        y = requantize(acc[:, :, None, None], self.M0, self.n)[:, :, 0, 0]
        return np.clip(y, 0, QMAX)

    def score(self, p):
        """(N,) int64 failure score (high = alarm), the logit at scale s_h·s_w2."""
        h = self.hidden_int(p)
        return self.sign * (h @ self.qW2.astype(np.int64) + self.qb2)

    def logit(self, p):
        return self.sign * self.score(p) * self.sh * self.sw2

    @property
    def memory_bytes(self):
        """int8 weights and zero points, int32 biases and multipliers, one shift."""
        return (self.zp.size + self.qW1.size + 4 * self.qb1.size + 4 * self.M0.size
                + self.qW2.size + 4 + 1)

    @property
    def macs(self):
        return self.d * self.hidden + self.hidden  # + d subtractions of the zero point

    # ---------------------------------------------------------------- C-sim blob
    def blob(self, p):
        """Little-endian int32 stream read by `hls/tb/tb_failure.cpp`: d, h, n, qb2, count,
        zp (d), qW1 (h·d), qb1 (h), M0 (h), qW2 (h), then `count` input vectors (d each)."""
        p = np.asarray(p, np.int64).reshape(-1, self.d)
        parts = [np.array([self.d, self.hidden, self.n, self.qb2, len(p)]), self.zp,
                 self.qW1.ravel(), self.qb1, self.M0, self.qW2, p.ravel()]
        return np.concatenate([np.asarray(x, np.int64) for x in parts]).astype("<i4").tobytes()

    def arrays(self):
        return {"zp": self.zp, "qW1": self.qW1, "qb1": self.qb1, "M0": self.M0,
                "n": np.array(self.n), "qW2": self.qW2, "qb2": np.array(self.qb2), "sign": np.array(self.sign),
                "sw1": self.sw1, "sh": np.array(self.sh), "sw2": np.array(self.sw2)}

    @classmethod
    def from_arrays(cls, a):
        return cls(a["zp"], a["qW1"], a["qb1"], a["M0"], int(a["n"]), a["qW2"], int(a["qb2"]),
                   int(a["sign"]), a["sw1"], float(a["sh"]), float(a["sw2"]))
