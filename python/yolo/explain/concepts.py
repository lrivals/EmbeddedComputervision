"""Concepts and channels (T36.6): Network Dissection-lite and TCAV.

**Dissection** (Bau 2017, simplified): each channel's map is upsampled ×`up` and thresholded
at its `quantile` over the dataset (the threshold is taken on the native grid); its IoU with
the box masks of each class, summed over the images, says which class it detects. A channel
whose best IoU is under `MIN_IOU` (0.04, the threshold of Bau 2017) matches no class.

**TCAV** (Kim 2018): the activations of a layer, averaged over the pixels (one value per
channel, a simplification of the flattened map), of concept images against random images;
a logistic regression separates them, its unit normal is the CAV v. The TCAV score of a class
is the share of its images whose score s grows along v: Σ_c v_c Σ_{ij} ∂s/∂A_{cij} > 0. The
test: CAVs of the concept against `R` random sets, CAVs of random against random, and a
two-sided Welch t-test between the two lists of scores (the p-value from the regularized
incomplete beta function, no SciPy).

Degradations for the concepts "dark" and "fog" (until T27.3 gives its own): exposure and
gamma (x ↦ g·x^γ), and a uniform haze (x ↦ (1 − a)·x + a·L).
"""

import math

import numpy as np

from yolo.explain.maps import box_mask

MIN_IOU = 0.04


# ------------------------------------------------------------------- degradations
def dark(x, gain=0.35, gamma=1.6):
    return (gain * np.power(np.clip(x, 0, 1), gamma)).astype(x.dtype)


def fog(x, a=0.6, light=0.8):
    return ((1 - a) * x + a * light).astype(x.dtype)


# ---------------------------------------------------------------------- dissection
def upsample(a, up):
    return a.repeat(up, axis=-2).repeat(up, axis=-1)


def thresholds(acts, quantile=0.995):
    """Per-channel threshold (C,) over the native maps `acts` (N, C, h, w)."""
    c = acts.shape[1]
    return np.quantile(np.moveaxis(acts, 1, 0).reshape(c, -1), quantile, axis=1)


def class_masks(boxes, labels, classes, h, w):
    """(classes, h, w) union of the box masks of each class of one image."""
    m = np.zeros((classes, h, w), dtype=bool)
    for b, lab in zip(boxes, labels):
        m[int(lab)] |= box_mask(b, h, w)
    return m


def dissect(acts, boxes, labels, classes, quantile=0.995, up=4, thr=None):
    """IoU (C, classes) of each thresholded channel with each class, over the images.
    `acts` (N, C, h, w); `boxes[n]` (k, 4) normalised `(cx, cy, w, h)`, `labels[n]` (k,)."""
    n, c, h, w = acts.shape
    thr = thresholds(acts, quantile) if thr is None else thr
    inter = np.zeros((c, classes))
    union = np.zeros((c, classes))
    for k in range(n):
        on = upsample(acts[k], up) > thr[:, None, None]  # (C, H, W)
        cm = class_masks(boxes[k], labels[k], classes, h * up, w * up)  # (K, H, W)
        on_f = on.reshape(c, -1).astype(np.float64)
        cm_f = cm.reshape(classes, -1).astype(np.float64)
        i = on_f @ cm_f.T
        inter += i
        union += on_f.sum(1)[:, None] + cm_f.sum(1)[None, :] - i
    return np.where(union > 0, inter / np.maximum(union, 1), 0.0)


def top_channels(iou_cl, k=3):
    """{class: [(channel, IoU)] best `k`} of a `dissect` result."""
    return {cl: [(int(ch), float(iou_cl[ch, cl])) for ch in np.argsort(-iou_cl[:, cl])[:k]]
            for cl in range(iou_cl.shape[1])}


def unmatched_share(iou_cl, min_iou=MIN_IOU):
    return float((iou_cl.max(axis=1) < min_iou).mean())


# ---------------------------------------------------------------------------- TCAV
def logistic(xa, xb, l2=1e-2, iters=300, lr=0.5):
    """Unit normal of the logistic regression separating `xa` (label 1) from `xb` (0), on
    standardised features (the normal is mapped back to the raw features)."""
    x = np.concatenate([xa, xb]).astype(np.float64)
    y = np.r_[np.ones(len(xa)), np.zeros(len(xb))]
    mu, sd = x.mean(0), x.std(0) + 1e-12
    z = (x - mu) / sd
    w = np.zeros(z.shape[1])
    b = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(z @ w + b)))
        g = p - y
        w -= lr * (z.T @ g / len(y) + l2 * w)
        b -= lr * g.mean()
    v = w / sd
    return v / (np.linalg.norm(v) or 1.0)


def pooled(acts):
    """(N, C) mean over the pixels of (N, C, h, w)."""
    return np.asarray(acts, dtype=np.float64).mean(axis=(2, 3))


def tcav_score(cav, grads):
    """Share of the images whose score grows along `cav`; `grads` (N, C, h, w) ∂s/∂A."""
    d = np.asarray(grads, dtype=np.float64).sum(axis=(2, 3)) @ cav
    return float((d > 0).mean())


def _tiny(v, fpmin=1e-300):
    return v if abs(v) >= fpmin else fpmin


def _betacf(a, b, x, iters=300, eps=3e-16):
    """Continued fraction of the incomplete beta function (Lentz, Numerical Recipes 6.4)."""
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 / _tiny(1 - qab * x / qap)
    h = d
    for m in range(1, iters + 1):
        m2 = 2 * m
        for aa in (m * (b - m) * x / ((qam + m2) * (a + m2)),
                   -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))):
            d = 1 / _tiny(1 + aa * d)
            c = _tiny(1 + aa / c)
            delta = d * c
            h *= delta
        if abs(delta - 1) < eps:
            break
    return h


def betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b) (Numerical Recipes 6.4)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbt = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) \
        + b * math.log(1 - x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lbt) * _betacf(a, b, x) / a
    return 1 - math.exp(lbt) * _betacf(b, a, 1 - x) / b


def welch(a, b):
    """(t, two-sided p-value) of Welch's t-test between samples `a` and `b`."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    if va + vb == 0:
        return (0.0, 1.0) if a.mean() == b.mean() else (math.copysign(math.inf, a.mean()
                                                                       - b.mean()), 0.0)
    t = (a.mean() - b.mean()) / math.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return float(t), float(betainc(df / 2, 0.5, df / (df + t * t)))


def tcav(concept, randoms, grads, rng=0):
    """TCAV of one concept. `concept` (n, C) pooled activations; `randoms` [(m, C)] the R
    random sets; `grads` (N, C, h, w) ∂s/∂A of the class images. Returns {score (mean),
    scores, random_scores, t, p}."""
    rng = np.random.default_rng(rng)
    scores = [tcav_score(logistic(concept, r), grads) for r in randoms]
    base = []
    for k, r in enumerate(randoms):
        other = randoms[(k + 1 + int(rng.integers(len(randoms) - 1))) % len(randoms)]
        base.append(tcav_score(logistic(r, other), grads))
    t, p = welch(scores, base)
    return {"score": float(np.mean(scores)), "scores": scores, "random_scores": base,
            "t": t, "p": p}
