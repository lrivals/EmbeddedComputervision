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


def ciou_alpha(p, g, eps=1e-7):
    """The α of `ciou` (the constant of its backward)."""
    p = np.asarray(p, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(g, dtype=np.float64).reshape(-1, 4)
    lt = np.maximum(p[:, :2] - p[:, 2:] / 2, g[:, :2] - g[:, 2:] / 2)
    rb = np.minimum(p[:, :2] + p[:, 2:] / 2, g[:, :2] + g[:, 2:] / 2)
    inter = np.prod(np.maximum(rb - lt, 0), axis=1)
    iou_ = inter / (np.prod(p[:, 2:], axis=1) + np.prod(g[:, 2:], axis=1) - inter + eps)
    v = (4 / np.pi ** 2) * (np.arctan(g[:, 2] / g[:, 3]) - np.arctan(p[:, 2] / p[:, 3])) ** 2
    return v / (v - iou_ + (1 + eps))


def ciou(p, g, eps=1e-7, alpha=None):
    """Complete IoU of box pairs (T33.6) and its gradient with respect to `p`.

    `p`, `g`: (n, 4) `(cx, cy, w, h)` (any unit, the same for both). As the YOLOv5
    `bbox_iou(..., CIoU=True)`:

        CIoU = IoU − ρ²/c² − α v,   v = 4/π² (atan(w_g/h_g) − atan(w_p/h_p))²,
        α = v / (v − IoU + 1 + ε)

    with ε added to the union and to c². α is a constant of the backward (no gradient
    through it, as the reference computes it under `no_grad`); `alpha` given replaces it
    (gradient checks). Returns (CIoU (n,), ∂CIoU/∂p (n, 4)).
    """
    p = np.asarray(p, dtype=np.float64).reshape(-1, 4)
    g = np.asarray(g, dtype=np.float64).reshape(-1, 4)
    x1, y1, w1, h1 = p.T
    x2, y2, w2, h2 = g.T
    l1, r1, t1, b1 = x1 - w1 / 2, x1 + w1 / 2, y1 - h1 / 2, y1 + h1 / 2
    l2, r2, t2, b2 = x2 - w2 / 2, x2 + w2 / 2, y2 - h2 / 2, y2 + h2 / 2
    iw_raw = np.minimum(r1, r2) - np.maximum(l1, l2)
    ih_raw = np.minimum(b1, b2) - np.maximum(t1, t2)
    iw, ih = np.maximum(iw_raw, 0), np.maximum(ih_raw, 0)
    inter = iw * ih
    union = w1 * h1 + w2 * h2 - inter + eps
    iou_ = inter / union
    cw = np.maximum(r1, r2) - np.minimum(l1, l2)
    ch = np.maximum(b1, b2) - np.minimum(t1, t2)
    c2 = cw ** 2 + ch ** 2 + eps
    rho2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
    da = np.arctan(w2 / h2) - np.arctan(w1 / h1)
    v = (4 / np.pi ** 2) * da ** 2
    if alpha is None:
        alpha = v / (v - iou_ + (1 + eps))
    out = iou_ - (rho2 / c2 + alpha * v)

    # Backward, through the box edges l, r, t, b (∂l/∂x = 1, ∂l/∂w = −1/2, ...).
    on_w, on_h = (iw_raw > 0) * 1.0, (ih_raw > 0) * 1.0
    d_iw_dr, d_iw_dl = (r1 < r2) * on_w, -on_w * (l1 > l2)
    d_ih_db, d_ih_dt = (b1 < b2) * on_h, -on_h * (t1 > t2)
    d_iou_dinter = (union + inter) / union ** 2      # union depends on −inter
    d_iou_darea = -inter / union ** 2                # area = w1·h1
    d_cw_dr, d_cw_dl = (r1 >= r2) * 1.0, -1.0 * (l1 <= l2)
    d_ch_db, d_ch_dt = (b1 >= b2) * 1.0, -1.0 * (t1 <= t2)
    k = -1 / c2                                      # ∂(−ρ²/c²)/∂ρ²
    kc = rho2 / c2 ** 2                              # ∂(−ρ²/c²)/∂c²

    def edges(d_r, d_l):
        """∂/∂centre and ∂/∂size of a function of the right and left edges."""
        return d_r + d_l, (d_r - d_l) / 2

    gx_e, gw_e = edges(d_iou_dinter * ih * d_iw_dr + kc * 2 * cw * d_cw_dr,
                       d_iou_dinter * ih * d_iw_dl + kc * 2 * cw * d_cw_dl)
    gy_e, gh_e = edges(d_iou_dinter * iw * d_ih_db + kc * 2 * ch * d_ch_db,
                       d_iou_dinter * iw * d_ih_dt + kc * 2 * ch * d_ch_dt)
    r_2 = w1 ** 2 + h1 ** 2
    dv = alpha * (8 / np.pi ** 2) * da               # α ∂v/∂atan(w1/h1) = −α·8/π²·da
    grad = np.stack([
        gx_e + k * -2 * (x2 - x1),
        gy_e + k * -2 * (y2 - y1),
        gw_e + d_iou_darea * h1 + dv * (h1 / r_2),
        gh_e + d_iou_darea * w1 - dv * (w1 / r_2),
    ], axis=1)
    return out, grad
