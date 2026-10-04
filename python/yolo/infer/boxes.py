"""Boîtes : conversions centre/coins et IoU vectorisées (§5.2, §5.3).

Format par défaut `(cx, cy, w, h)` (docs/conventions.md) ; les fonctions `*_xyxy` prennent
des coins `(x1, y1, x2, y2)`. Les IoU sont rendues en matrice `(N, M)`.
"""

import numpy as np


def cxcywh_to_xyxy(b):
    # §5.3 : x1 = x − w/2, x2 = x + w/2
    b = np.asarray(b, dtype=np.float64)
    half = b[..., 2:4] / 2
    return np.concatenate([b[..., 0:2] - half, b[..., 0:2] + half], axis=-1)


def xyxy_to_cxcywh(b):
    b = np.asarray(b, dtype=np.float64)
    return np.concatenate([(b[..., 0:2] + b[..., 2:4]) / 2, b[..., 2:4] - b[..., 0:2]], axis=-1)


def _ratio(inter, union):
    # Union nulle (deux boîtes d'aire nulle) → IoU 0.
    out = np.zeros_like(inter)
    np.divide(inter, union, out=out, where=union > 0)
    return out


def iou_xyxy(a, b):
    """IoU (N, M) de coins `a` (N, 4) et `b` (M, 4) — §5.3."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)[:, None, :]
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)[None, :, :]
    # §5.3 : |A∩B| = max(0, min(x2) − max(x1)) · max(0, min(y2) − max(y1))
    iw = np.maximum(0.0, np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]))
    ih = np.maximum(0.0, np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]))
    inter = iw * ih
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return _ratio(inter, area_a + area_b - inter)


def iou(a, b):
    """IoU (N, M) de boîtes `(cx, cy, w, h)` — §5.3."""
    return iou_xyxy(cxcywh_to_xyxy(np.reshape(a, (-1, 4))), cxcywh_to_xyxy(np.reshape(b, (-1, 4))))


def iou_wh(a, b):
    """IoU (N, M) de formes `(w, h)` à centres alignés — §5.2 (k-means, choix de l'ancre)."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 2)[:, None, :]
    b = np.asarray(b, dtype=np.float64).reshape(-1, 2)[None, :, :]
    # §5.2 : inter = min(w)·min(h)
    inter = np.minimum(a[..., 0], b[..., 0]) * np.minimum(a[..., 1], b[..., 1])
    return _ratio(inter, a[..., 0] * a[..., 1] + b[..., 0] * b[..., 1] - inter)
