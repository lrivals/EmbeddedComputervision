"""T25.2: algorithm-based fault tolerance (ABFT) of the integer convolutions.

A conv is linear before requantization, so the sum of its output channels equals the conv
of its input by the summed filters:

    Σ_f acc_f = conv(x, Σ_f qW_f) + Σ_f q_b,f

- `checksums(qW, qb)`: the checksum filter cs_w = Σ_f qW_f (int32, C × k × k) and bias
  cs_b = Σ_f q_b,f, computed once from the fault-free weights (exported in `checksum.bin`).
- `check`: the equality is tested on the int32 accumulator, before requantization, leaky and
  saturation (not linear). It is taken **modulo 2³²**, as a 32-bit checksum lane would compute
  it. It stays exact (no tolerance, no false alarm), and a single bit flip of an accumulator
  (bit b < 32) changes the sum by 2^b ≠ 0 mod 2³²: it is always detected.
- `AbftMonitor`: `IntNetwork` hook that runs the check on every conv and counts the errors.
- A fault of the conv *input* is not seen by the check of that conv (both sides read the same
  faulty input); it is the input of the next layers that carries it.
"""

import numpy as np

from yolo.layers.conv import conv_forward
from yolo.quant import int_layers as il

MOD = 1 << 32


def checksums(qW, qb):
    """(cs_w int32 (C, k, k) = Σ_f qW_f, cs_b = Σ_f q_b,f wrapped to int32)."""
    cs_w = np.asarray(qW, np.int64).sum(axis=0)
    if np.abs(cs_w).max(initial=0) >= il.ACC_LIMIT:
        raise OverflowError("checksum filter outside int32")
    return cs_w.astype(np.int32), wrap32(int(np.asarray(qb, np.int64).sum()))


wrap32 = il.wrap32


def lane(x, cs_w, cs_b, engine="f64", s=1):
    """Checksum lane: conv(x, cs_w) + cs_b (int64, not wrapped), shape (N, 1, H, W).

    |conv(x, cs_w)| ≤ 127 · C k² · 2¹⁸ < 2⁵³: exact on the f64 engine as on int64.
    """
    w = np.asarray(cs_w, np.int64)[None]
    dt = np.float64 if engine == "f64" else np.int64
    y, _ = conv_forward(np.asarray(x, dt), w.astype(dt), s=s)
    return y.astype(np.int64) + int(cs_b)


def mismatches(x, acc, cs_w, cs_b, engine="f64", s=1):
    """Boolean map (N, H, W): pixels where Σ_f acc_f ≠ checksum lane mod 2³²."""
    total = np.asarray(acc, np.int64).sum(axis=1)
    ref = lane(x, cs_w, cs_b, engine, s)[:, 0]
    return ((total - ref) & (MOD - 1)) != 0


def check(x, acc, cs_w, cs_b, engine="f64", s=1):
    """Number of output pixels whose checksum fails (0: the conv passes)."""
    return int(mismatches(x, acc, cs_w, cs_b, engine, s).sum())


class AbftMonitor:
    """`IntNetwork` hook: ABFT check of every conv; `errors[i]` = failing pixels of conv i
    since the last `reset()`, `checks[i]` = pixels checked."""

    def __init__(self, qm, engine="f64"):
        self.qm, self.engine = qm, engine
        self.reset()

    def reset(self):
        self.errors, self.checks = {}, {}

    @property
    def detected(self):
        return any(self.errors.values())

    def on_acc(self, i, acc, x):
        c = self.qm.convs[i]
        e = check(x, acc, c.cs_w, c.cs_b, self.engine, c.s)
        self.errors[i] = self.errors.get(i, 0) + e
        self.checks[i] = self.checks.get(i, 0) + acc.shape[0] * acc.shape[2] * acc.shape[3]
        return acc

    def on_out(self, i, y):
        return y


def widths(qm):
    """Per conv: bits of the checksum filter (signed), F, and the extra MACs of the lane
    (one more filter out of F: +1/F); also for the whole network (MAC-weighted)."""
    rows = []
    for i, c in sorted(qm.convs.items()):
        m = int(np.abs(c.cs_w.astype(np.int64)).max(initial=0))
        rows.append({"id": i, "cout": len(c.qW), "cs_max": m,
                     "cs_bits": int(m).bit_length() + 1, "extra": 1 / len(c.qW)})
    return rows
