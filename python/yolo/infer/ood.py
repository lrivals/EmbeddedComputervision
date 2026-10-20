"""T25.5: out-of-distribution monitor on the backbone features of the detector.

The monitor reads the output of the last backbone conv (L13 of Tiny-YOLOv2, 1024 × 13 × 13,
dequantized from the int8 map that the accelerator writes) and gives a score. Three raw
scores:

- **norm**: L2 norm of the pooled features (global average and max, 2,048 values), the
  energy of the map;
- **Mahalanobis** (baseline): distance of the pooled features to their mean on the training
  images, with a Ledoit-Wolf shrunk covariance (more dimensions than images);
- **autoencoder**: per-cell 1×1 conv autoencoder (`Network`, layers of `yolo.layers`) on the
  standardized map; raw score = mean squared reconstruction error over the cells.

A domain shift to dark or thermal images *lowers* the activations: these images fall inside
the shell where the in-domain features lie, closer to the mean and easier to reconstruct, so
a raw score is lower for them. The monitor therefore uses the **typicality** of a raw score,
|s − s_ref| with s_ref its median on held-out in-domain images (not seen by the fit): far from
the typical value in either direction is out of distribution.

`auroc` (in-domain against out-of-domain scores) and `macs` (cost against the detector) give
the figures of `results/ood_monitor.md`.
"""

import numpy as np

from yolo.models.graph import Network
from yolo.models.specs import infer_shapes, layer_cost
from yolo.train.optim import AdamW


def typicality(scores, ref):
    """Two-sided score |s − ref|, `ref` = median of the raw score on held-out in-domain data."""
    return np.abs(np.asarray(scores, np.float64) - ref)


def norm(pooled):
    """Energy of the features: L2 norm of the pooled vector."""
    return np.linalg.norm(np.asarray(pooled, np.float64), axis=1)


def pool(feat):
    """(N, C, H, W) → (N, 2C): global average, then global max."""
    f = np.asarray(feat, np.float64)
    return np.concatenate([f.mean(axis=(2, 3)), f.max(axis=(2, 3))], axis=1)


def ledoit_wolf(x):
    """(covariance shrunk towards μ·I, shrinkage) of the rows of `x` (Ledoit & Wolf 2004)."""
    n, d = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / n
    mu = np.trace(s) / d
    delta = ((s - mu * np.eye(d)) ** 2).sum() / d
    beta = ((xc ** 2).sum(axis=1) ** 2).sum() / n ** 2 - (s ** 2).sum() / n
    beta = min(max(beta / d, 0.0), delta)
    a = beta / delta if delta > 0 else 1.0
    return (1 - a) * s + a * mu * np.eye(d), a


class Mahalanobis:
    """Mahalanobis distance of pooled features to the training distribution."""

    def fit(self, x):
        x = np.asarray(x, np.float64)
        self.mean = x.mean(axis=0)
        cov, self.shrinkage = ledoit_wolf(x)
        self.prec = np.linalg.inv(cov)
        return self

    def score(self, x):
        d = np.asarray(x, np.float64) - self.mean
        return np.sqrt(np.einsum("ij,jk,ik->i", d, self.prec, d))


def autoencoder_spec(c=1024, hw=(13, 13), hidden=(128, 32)):
    """Per-cell autoencoder c → hidden… → c, 1×1 convs, leaky, last one linear."""
    sizes = list(hidden) + list(hidden[-2::-1]) + [c]
    layers = [{"type": "conv", "k": 1, "s": 1, "cout": h, "bn": False,
               "act": "linear" if k == len(sizes) - 1 else "leaky"}
              for k, h in enumerate(sizes)]
    return {"name": "ood-ae", "input": (c, *hw), "classes": 0, "layers": layers}


def macs(net):
    """MACs of a network spec (per image)."""
    return sum(layer_cost(layer, i, o)[1] for layer, (i, o) in zip(net["layers"],
                                                                     infer_shapes(net)))


class Standardizer:
    """Per-channel mean and standard deviation of (N, C, H, W) maps."""

    def fit(self, maps):
        m = np.asarray(maps, np.float64)
        self.mean = m.mean(axis=(0, 2, 3))[None, :, None, None]
        self.std = m.std(axis=(0, 2, 3))[None, :, None, None] + 1e-6
        return self

    def __call__(self, maps):
        return ((np.asarray(maps, np.float64) - self.mean) / self.std).astype(np.float32)


def reconstruction_error(net, x):
    """Mean squared error per image of the autoencoder on standardized maps x."""
    y = net.forward(x, train=False)[len(net.layers) - 1]
    return ((y - x) ** 2).mean(axis=(1, 2, 3))


def train_ae(net, x, iters=2000, batch=16, lr=1e-3, seed=0, log_every=0):
    """Fits the autoencoder on standardized maps x (N, C, H, W): MSE, AdamW. Returns the
    loss of each iteration."""
    rng = np.random.default_rng(seed)
    opt = AdamW(net.params, weight_decay=1e-4)
    out = len(net.layers) - 1
    losses = []
    for it in range(iters):
        xb = x[rng.integers(len(x), size=batch)]
        y = net.forward(xb, train=True)[out]
        d = y - xb
        losses.append(float((d ** 2).mean()))
        _, grads = net.backward({out: (2 * d / d.size).astype(d.dtype)})
        opt.step(grads, lr)
        if log_every and (it + 1) % log_every == 0:
            print(f"  AE {it + 1}/{iters}  loss {np.mean(losses[-log_every:]):.4f}",
                  flush=True)
    return losses


def make_ae(c=1024, hw=(13, 13), hidden=(128, 32), seed=0):
    return Network(autoencoder_spec(c, hw, hidden), dtype=np.float32, rng=seed)


def auroc(inside, outside):
    """Area under the ROC curve, `outside` the positive class (score high = OOD); ties
    count one half (Mann-Whitney U)."""
    a, b = np.asarray(inside, np.float64), np.asarray(outside, np.float64)
    s = np.concatenate([a, b])
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    # Average rank of tied values.
    _, inv, counts = np.unique(s, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=ranks)
    ranks = (sums / counts)[inv]
    u = ranks[len(a):].sum() - len(b) * (len(b) + 1) / 2
    return float(u / (len(a) * len(b)))


def spearman(x, y):
    """Spearman rank correlation (no tie correction needed for continuous values)."""
    rx = np.asarray(x).argsort().argsort().astype(np.float64)
    ry = np.asarray(y).argsort().argsort().astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


# ------------------------------------------------------------- T29.0: more monitors
# Three monitors of the literature, computed on the **integer** features (int8 L13 map,
# pooled as integers: per-channel sum of the 169 cells, int32, and max, int8), next to the
# norm, Mahalanobis and autoencoder of T25.5. Each one has `fit` / `score` (high = alarm) and
# its cost: `memory_bytes` (stored state) and `ops` (operations per image after the pooling,
# compares counted as one).


def pool_int(maps):
    """(N, C, H, W) int8 → (N, 2C) int64: per-channel sum (int32 in hardware) and max."""
    m = np.asarray(maps, np.int64)
    return np.concatenate([m.sum(axis=(2, 3)), m.max(axis=(2, 3))], axis=1)


def fpr_at_tpr(inside, outside, tpr=0.95):
    """False-alarm rate on `inside` at the threshold that flags `tpr` of `outside`."""
    thr = np.quantile(np.asarray(outside, np.float64), 1 - tpr)
    return float((np.asarray(inside, np.float64) >= thr).mean())


class MaxScore:
    """Maximum class score of the detector on the image (MSP of Hendrycks 2017 for a
    detector: max over cells and classes of σ(t_o)·p_c, the top detection after NMS). Score
    = 1 − max: a low top confidence raises the alarm. No state, one compare per detection."""

    memory_bytes = 0

    def fit(self, _):
        return self

    def score(self, top):
        return 1.0 - np.asarray(top, np.float64)

    @staticmethod
    def ops(d):
        return 0


def kmeans(x, k, iters=50, seed=0):
    """Lloyd's k-means on the rows of x: (centres (k, d), labels)."""
    x = np.asarray(x, np.float64)
    rng = np.random.default_rng(seed)
    c = x[rng.choice(len(x), k, replace=False)]
    for _ in range(iters):
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1)
        lab = d.argmin(1)
        new = np.array([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
        if np.allclose(new, c):
            break
        c = new
    return c, lab


class AbstractionBoxes:
    """Per-cluster boxes [min, max] of each neuron on the fit features (Henzinger 2020).
    Score = smallest number of neurons outside the box over the clusters (0: inside one
    box). Compare-only: 2·d·k compares, 2·d·k stored integers (int32 sums, int8 maxes)."""

    def __init__(self, k=4, seed=0):
        self.k, self.seed = k, seed

    def fit(self, x):
        x = np.asarray(x, np.int64)
        _, lab = kmeans(x.astype(np.float64) / (np.abs(x).max(0) + 1), self.k, seed=self.seed)
        self.lo = np.stack([x[lab == j].min(0) for j in range(self.k) if (lab == j).any()])
        self.hi = np.stack([x[lab == j].max(0) for j in range(self.k) if (lab == j).any()])
        return self

    def score(self, x):
        x = np.asarray(x, np.int64)[:, None, :]
        out = ((x < self.lo[None]) | (x > self.hi[None])).sum(-1)
        return out.min(1).astype(np.float64)

    @property
    def memory_bytes(self):
        return self.lo.size * 4 * 2

    def ops(self, d):
        return 2 * d * len(self.lo)


class ActivationPatterns:
    """Binary activation patterns (Cheng 2018) of the `bits` pooled neurons of largest
    variance on the fit set: neuron on when above its fit median. Score = Hamming distance to
    the nearest pattern seen in the fit (0: a known pattern). XOR + popcount over the stored
    patterns."""

    def __init__(self, bits=64):
        self.bits = bits

    def fit(self, x):
        x = np.asarray(x, np.float64)
        self.idx = np.argsort(x.var(0))[::-1][:self.bits]
        self.thr = np.median(x[:, self.idx], 0)
        self.patterns = np.unique(self._binarize(x), axis=0)
        return self

    def _binarize(self, x):
        return (np.asarray(x, np.float64)[:, self.idx] > self.thr).astype(np.uint8)

    def score(self, x):
        b = self._binarize(x)
        d = np.empty(len(b))
        for s in range(0, len(b), 256):
            d[s:s + 256] = (b[s:s + 256, None, :] != self.patterns[None]).sum(-1).min(1)
        return d

    @property
    def memory_bytes(self):
        return len(self.patterns) * self.bits // 8 + self.bits * 4 * 2

    def ops(self, d):
        return self.bits + len(self.patterns) * (self.bits // 64 + 1) * 2
