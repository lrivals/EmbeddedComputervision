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
