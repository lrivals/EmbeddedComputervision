"""Learned failure prediction (T39.0, T39.1): a small head trained to flag the images where
the detector misses a person.

The monitors of T29.0 are post-hoc scores; this one is trained on the failures it must flag
(ConfidNet, Corbière 2019, for the TCP target; a binary "person missed" target next to it).

**Labels** (T39.0). A person is missed when no detection kept by the NMS has the person class
and IoU ≥ 0.5 with it (`yolo.explain.misses.matched`, the same test as the causes of T36.5);
an image fails when at least one of its persons is missed (the KGFP criterion). The **TCP**
of a person is the best score σ(t_o)·p_person over the candidates `classify_miss` looks at
(the pre-NMS boxes with IoU ≥ 0.5 and the responsible cell; 0 without candidate); the TCP
of an image is the
minimum over its persons (1 without person). Both depend on the operating point.

**Features**: the multi-scale maps the detector already computes, the input of the last conv
of each head (`feature_layers`), pooled per channel by mean and max (`pool_float`). On the
integer chain the mean is the int32 sum shifted right by ⌈log2 HW⌉ with rounding, the max is
exact (`pool_int_shift`); `dequant_scales` gives the float value of each integer feature.

**Head** (`FailureHead`): d → h (ReLU) → 1 logit, no softmax, so the integer form of T39.5
is a matrix product, a requantization and a second matrix product. The standardization of
the input is stored and folds into the first layer.

- target `"miss"`: BCE on the logit, score σ(f) (high = alarm);
- target `"tcp"`: ConfidNet, MSE between σ(f) and the image TCP, score 1 − σ(f).

Class imbalance (few failing images): a positive weight n_neg / n_pos in the loss
(`imbalance="weight"`) or batches drawn half from each class (`"balanced"`); the tools choose
between them on the validation split only.

**Metrics** (protocol of T29.1): AUROC of the event "this image has a missed person", share of
the missed persons whose image raises an alarm at a fixed alarm rate (1 %, 5 % of all
images), recall of the system "detector or monitor alarm" against the detector alone, and the
KGFP metric (`recall_at_false_alarm`: person recall among the images the monitor accepts, at
5 % false alarms).
"""

import math

import numpy as np

from yolo.data.targets import heads
from yolo.infer import ood
from yolo.quant.int_layers import rshift_round
from yolo.train.optim import AdamW

SPLITS = ("fit", "val", "test")
RATES = (0.01, 0.05)


# ------------------------------------------------------------------------ features
def feature_layers(net):
    """Per head of `net` (spec dict), the layer whose output feeds the head's last conv."""
    layers = net["layers"]
    out = []
    for hid, _ in heads(net):
        last = hid - 1
        if layers[last]["type"] != "conv":
            raise ValueError(f"head {hid}: layer {last} is not a conv")
        out.append(last - 1)
    return out


def pool_float(maps):
    """(N, C, H, W) → (N, 2C): per-channel mean, then max."""
    m = np.asarray(maps, np.float64)
    return np.concatenate([m.mean(axis=(2, 3)), m.max(axis=(2, 3))], axis=1)


def mean_shift(hw):
    """⌈log2 HW⌉: the shift that brings the int32 sum of an int8 map back into int8 range."""
    return int(math.ceil(math.log2(hw)))


def pool_int_shift(maps):
    """(N, C, H, W) int8 → (N, 2C) int64: rounded sum >> ⌈log2 HW⌉, then max."""
    m = np.asarray(maps, np.int64)
    k = mean_shift(m.shape[2] * m.shape[3])
    return np.concatenate([rshift_round(m.sum(axis=(2, 3)), k), m.max(axis=(2, 3))], axis=1)


def dequant_scales(scales, channels, hws):
    """Float value of one LSB of each integer feature of `pool_int_shift`, layer by layer:
    s·2^k / HW for the means, s for the maxes."""
    parts = []
    for s, c, hw in zip(scales, channels, hws):
        parts += [np.full(c, s * 2.0 ** mean_shift(hw) / hw), np.full(c, float(s))]
    return np.concatenate(parts)


def features_float(outs, layers):
    return np.concatenate([pool_float(outs[i]) for i in layers], axis=1)


def features_int(outs, layers):
    return np.concatenate([pool_int_shift(outs[i]) for i in layers], axis=1)


# -------------------------------------------------------------------------- labels
def person_tcp(net, box, cls, out, grids, match_iou=0.5):
    """Best σ(t_o)·p_cls over the candidates of `classify_miss` (pre-NMS boxes with IoU ≥
    `match_iou`, and the responsible cell and anchor)."""
    from yolo.explain.misses import responsible_index
    from yolo.infer.boxes import iou

    ov = iou(np.asarray(box, dtype=np.float64)[None], out.boxes)[0]
    resp = responsible_index(net, box, grids)
    cand = np.union1d(np.nonzero(ov >= match_iou)[0], [] if resp is None else [resp])
    return float(out.scores[cand.astype(int), cls].max()) if len(cand) else 0.0


def image_targets(causes, tcps):
    """(image fails, image TCP) from the per-person causes (`None` = detected) and TCPs."""
    return any(c is not None for c in causes), float(min(tcps, default=1.0))


def split_of(n, seed=0, fractions=(0.6, 0.2)):
    """Fixed split of n images: 0 fit, 1 val, 2 test (seeded permutation)."""
    perm = np.random.default_rng(seed).permutation(n)
    a = int(round(fractions[0] * n))
    b = a + int(round(fractions[1] * n))
    out = np.full(n, 2, np.int64)
    out[perm[:a]], out[perm[a:b]] = 0, 1
    return out


# ---------------------------------------------------------------------------- head
def _sigmoid(t):
    return 0.5 * (1.0 + np.tanh(0.5 * t))


def _softplus(t):
    return np.maximum(t, 0) + np.log1p(np.exp(-np.abs(t)))


class FailureHead:
    """MLP d → hidden (ReLU) → 1 in NumPy (float64), on standardized features."""

    TARGETS = ("miss", "tcp")

    def __init__(self, d, hidden=32, target="miss", seed=0):
        if target not in self.TARGETS:
            raise ValueError(f"unknown target {target!r}")
        rng = np.random.default_rng(seed)
        self.target = target
        self.params = [{"W": rng.normal(0, math.sqrt(2.0 / d), (hidden, d)),
                        "b": np.zeros(hidden)},
                       {"W": rng.normal(0, math.sqrt(1.0 / hidden), (1, hidden)),
                        "b": np.zeros(1)}]
        self.mean = np.zeros(d)
        self.std = np.ones(d)

    @property
    def d(self):
        return self.params[0]["W"].shape[1]

    @property
    def hidden(self):
        return self.params[0]["W"].shape[0]

    def standardize(self, x, std_floor=None):
        """Mean and standard deviation of the features; `std_floor` (per feature, e.g. the
        integer LSB of `dequant_scales`) keeps a near-constant feature (a channel saturated
        at 127) from getting a huge weight once σ is folded into the first layer."""
        x = np.asarray(x, np.float64)
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-6
        if std_floor is not None:
            self.std = np.maximum(self.std, std_floor)
        return self

    def forward(self, x):
        """Logits (N,) and the cache of `backward`."""
        z = (np.asarray(x, np.float64) - self.mean) / self.std
        (l1, l2) = self.params
        pre = z @ l1["W"].T + l1["b"]
        h = np.maximum(pre, 0.0)
        return h @ l2["W"][0] + l2["b"][0], (z, pre, h)

    def backward(self, cache, df):
        """Gradients of the parameters and of the input x for δ = ∂L/∂f (N,)."""
        z, pre, h = cache
        (l1, l2) = self.params
        g2 = {"W": (df @ h)[None], "b": np.array([df.sum()])}
        dh = np.outer(df, l2["W"][0]) * (pre > 0)
        g1 = {"W": dh.T @ z, "b": dh.sum(axis=0)}
        return [g1, g2], (dh @ l1["W"]) / self.std

    def loss_grad_f(self, f, y, w):
        """Weighted loss (mean over the batch) and ∂L/∂f for the head's target."""
        n = len(f)
        y = np.asarray(y, np.float64)
        if self.target == "miss":
            terms = _softplus(f) - y * f              # BCE(σ(f), y)
            return float((w * terms).sum() / n), w * (_sigmoid(f) - y) / n
        p = _sigmoid(f)
        return float((w * (p - y) ** 2).sum() / n), 2 * w * (p - y) * p * (1 - p) / n

    def loss_and_grad(self, x, y, w=None):
        f, cache = self.forward(x)
        w = np.ones(len(f)) if w is None else np.asarray(w, np.float64)
        loss, df = self.loss_grad_f(f, y, w)
        grads, dx = self.backward(cache, df)
        return loss, grads, dx

    def score(self, x):
        """Failure score, high = alarm."""
        p = _sigmoid(self.forward(x)[0])
        return p if self.target == "miss" else 1.0 - p

    def fit(self, x, y, pos=None, imbalance="weight", iters=2000, batch=64, lr=1e-3,
            weight_decay=1e-4, seed=0, std_floor=None):
        """Fits the head on (x, y); `pos` (bool, the failing images) drives the imbalance
        (default y > 0.5 for the miss target). Returns the loss of each iteration."""
        x = np.asarray(x, np.float64)
        y = np.asarray(y, np.float64)
        pos = (y > 0.5) if pos is None else np.asarray(pos, bool)
        self.standardize(x, std_floor)
        rng = np.random.default_rng(seed)
        opt = AdamW(self.params, weight_decay=weight_decay)
        n_pos = int(pos.sum())
        w_all = np.ones(len(x))
        if imbalance == "weight" and 0 < n_pos < len(x):
            w_all[pos] = (len(x) - n_pos) / n_pos
        elif imbalance not in ("weight", "balanced", "none"):
            raise ValueError(f"unknown imbalance {imbalance!r}")
        ip, ineg = np.nonzero(pos)[0], np.nonzero(~pos)[0]
        losses = []
        for _ in range(iters):
            if imbalance == "balanced" and len(ip) and len(ineg):
                idx = np.concatenate([rng.choice(ip, batch // 2),
                                      rng.choice(ineg, batch - batch // 2)])
                w = np.ones(len(idx))
            else:
                idx = rng.integers(len(x), size=batch)
                w = w_all[idx]
            loss, grads, _ = self.loss_and_grad(x[idx], y[idx], w)
            opt.step(grads, lr)
            losses.append(loss)
        return losses

    # ---------------------------------------------------------------- cost and I/O
    @property
    def memory_bytes(self):
        """float32 parameters; the standardization folds into the first layer."""
        return 4 * sum(v.size for p in self.params for v in p.values())

    @property
    def macs(self):
        return self.d * self.hidden + self.hidden

    def arrays(self, prefix=""):
        out = {f"{prefix}mean": self.mean, f"{prefix}std": self.std,
               f"{prefix}target": np.array(self.target)}
        for i, p in enumerate(self.params):
            out.update({f"{prefix}{i}/{k}": v for k, v in p.items()})
        return out

    @classmethod
    def from_arrays(cls, a, prefix=""):
        h = cls(a[f"{prefix}0/W"].shape[1], a[f"{prefix}0/W"].shape[0],
                str(a[f"{prefix}target"]))
        h.mean, h.std = np.asarray(a[f"{prefix}mean"]), np.asarray(a[f"{prefix}std"])
        for i, p in enumerate(h.params):
            for k in p:
                p[k] = np.array(a[f"{prefix}{i}/{k}"], np.float64)
        return h


# ------------------------------------------------------------------------- metrics
def alarm_threshold(scores, rate):
    """Threshold t such that `scores > t` flags ⌊rate·N⌋ images (no ties)."""
    s = np.sort(np.asarray(scores, np.float64))[::-1]
    k = int(math.floor(rate * len(s)))
    return float(s[k]) if k < len(s) else -np.inf


def flagged_miss_share(flags, person_image, person_missed):
    """Share of the missed persons whose image raises an alarm."""
    missed = np.asarray(person_missed, bool)
    if not missed.any():
        return float("nan")
    return float(np.asarray(flags, bool)[np.asarray(person_image)[missed]].mean())


def system_recall(flags, person_image, person_missed):
    """(recall of "detector or monitor alarm", recall of the detector alone) over persons."""
    found = ~np.asarray(person_missed, bool)
    flagged = np.asarray(flags, bool)[np.asarray(person_image)]
    return float((found | flagged).mean()), float(found.mean())


def monitor_metrics(scores, image_fails, person_image, person_missed, rates=RATES):
    """T29.1 protocol for one monitor on one set."""
    scores = np.asarray(scores, np.float64)
    fails = np.asarray(image_fails, bool)
    out = {"auroc": ood.auroc(scores[~fails], scores[fails]) if fails.any() and (~fails).any()
           else float("nan")}
    for r in rates:
        flags = scores > alarm_threshold(scores, r)
        key = f"{r * 100:g}"
        out[f"flagged_{key}"] = flagged_miss_share(flags, person_image, person_missed)
        out[f"system_{key}"], out["detector_recall"] = system_recall(flags, person_image,
                                                                    person_missed)
    return out


def recall_at_false_alarm(scores, image_fails, fa=0.05, person_image=None,
                          person_missed=None):
    """KGFP metric (Zimmermann 2026): threshold set so that `fa` of the images without a miss
    are rejected (false alarms); returns (share of the failing images flagged, share of the
    images accepted, person recall among the accepted images — the number KGFP reports,
    NaN without persons)."""
    scores = np.asarray(scores, np.float64)
    fails = np.asarray(image_fails, bool)
    thr = alarm_threshold(scores[~fails], fa)
    flags = scores > thr
    rec = float("nan")
    if person_image is not None:
        acc = ~flags[np.asarray(person_image)]
        if acc.any():
            rec = float((~np.asarray(person_missed, bool))[acc].mean())
    return (float(flags[fails].mean()) if fails.any() else float("nan"),
            float((~flags).mean()), rec)
