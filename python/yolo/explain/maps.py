"""Shared helpers of the attribution maps (M36): resizing, blur, comparison.

A map is a (H, W) float64 array over the network input (preprocessed image); larger means
more relevant to the explained score.
"""

import numpy as np

from yolo.infer.ood import spearman


def resize(m, h, w):
    """Bilinear resize of a (h0, w0) map to (h, w), pixel centres aligned."""
    m = np.asarray(m, dtype=np.float64)
    h0, w0 = m.shape

    def axis(n0, n):
        pos = np.clip((np.arange(n) + 0.5) * n0 / n - 0.5, 0, n0 - 1)
        lo = np.floor(pos).astype(int)
        hi = np.minimum(lo + 1, n0 - 1)
        return lo, hi, pos - lo

    yl, yh, fy = axis(h0, h)
    xl, xh, fx = axis(w0, w)
    top = m[yl][:, xl] * (1 - fx) + m[yl][:, xh] * fx
    bot = m[yh][:, xl] * (1 - fx) + m[yh][:, xh] * fx
    return top * (1 - fy[:, None]) + bot * fy[:, None]


def box_blur(x, r):
    """Box blur of radius `r` on the last two axes (edge-replicated), any leading axes."""
    if r <= 0:
        return np.array(x, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    pad = [(0, 0)] * (x.ndim - 2) + [(r, r), (r, r)]
    p = np.pad(x, pad, mode="edge")
    c = p.cumsum(-1)
    c = np.concatenate([np.zeros_like(c[..., :1]), c], -1)
    k = 2 * r + 1
    p = (c[..., k:] - c[..., :-k]) / k
    c = p.cumsum(-2)
    c = np.concatenate([np.zeros_like(c[..., :1, :]), c], -2)
    return (c[..., k:, :] - c[..., :-k, :]) / k


def normalize(m):
    """Map scaled to [0, 1] (a constant map becomes 0)."""
    m = np.asarray(m, dtype=np.float64)
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


def rank_corr(a, b):
    """Spearman correlation of two maps of the same shape (0 when one is constant)."""
    a, b = np.ravel(a), np.ravel(b)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return 0.0
    return spearman(a, b)


def box_mask(box, h, w):
    """Boolean (h, w) mask of a normalised `(cx, cy, w, h)` box."""
    cx, cy, bw, bh = box
    x0, x1 = int(np.floor((cx - bw / 2) * w)), int(np.ceil((cx + bw / 2) * w))
    y0, y1 = int(np.floor((cy - bh / 2) * h)), int(np.ceil((cy + bh / 2) * h))
    m = np.zeros((h, w), dtype=bool)
    m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m
