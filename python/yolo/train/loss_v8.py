"""YOLOv8 loss (`v8DetectionLoss`, T35.5), with gradients.

Each head map `(N, 4·16 + C, S_h, S_w)` is raw logits: 16 DFL bins per side (l, t, r, b),
then the class logits. Over the A anchor points of all levels (`decode.anchor_points`, grid
units), with the boxes decoded from the DFL expectation (`decode.dfl_expect`, `dist2bbox`):

    assignment     `assign_tal` per image, in pixels, on σ(class logits) and the decoded
                   boxes (detached): target boxes, soft class targets t̂, foreground fg
    S              max(Σ t̂, 1) over the batch
    L_cls          Σ over every anchor and class of BCE(z_c, t̂) / S
    L_box          Σ_{j ∈ fg} w_j (1 − CIoU(p_j, g_j)) / S,  w_j = Σ_c t̂_jc
    L_dfl          Σ_{j ∈ fg} w_j · mean_k [(g_r − g)·CE(z_k, g_l) + (g − g_l)·CE(z_k, g_r)] / S
                   g = target distance (grid units) clamped to [0, 15 − 0.01], g_l = ⌊g⌋,
                   g_r = g_l + 1

    L = N·(7.5 L_box + 0.5 L_cls + 1.5 L_dfl)

CIoU is the one of YOLOv5 (`infer.boxes.ciou`, α constant), here on boxes in grid units;
its gradient goes back through `dist2bbox` and the DFL expectation (∂d/∂z_i = p_i (i − d)).
The DFL term reaches the same box logits directly (soft cross-entropy: p − q).

The loss is a total for the batch (already multiplied by N, as the reference); `douts` are
its gradients with respect to the raw outputs.
"""

import numpy as np

from yolo.data.tal import assign_tal
from yolo.infer.boxes import ciou, ciou_alpha, xyxy_to_cxcywh
from yolo.infer.decode import (anchor_points, dfl_expect, dfl_expect_backward, dist2bbox,
                               split_v8)
from yolo.layers.activations import sigmoid
from yolo.train.loss import LossResult, bce_logits

# default.yaml of Ultralytics (YOLOv8): loss gains, TAL top-k.
HYP_V8 = {"box": 7.5, "cls": 0.5, "dfl": 1.5, "topk": 10}
REG_MAX = 16


def _xyxy_grad(dc):
    """∂/∂(x1, y1, x2, y2) from ∂/∂(cx, cy, w, h)."""
    return np.concatenate([dc[..., :2] / 2 - dc[..., 2:], dc[..., :2] / 2 + dc[..., 2:]],
                          axis=-1)


def _decode(outputs, levels, reg_max):
    zb, zc, grids = split_v8(outputs, levels, reg_max)
    pts, st = anchor_points(grids, [s for _, s in levels])
    d, p = dfl_expect(zb)
    return zb, zc, grids, pts, st, d, p, dist2bbox(d, pts)


def gt_pixels(boxes, size_hw):
    """Normalized `(cx, cy, w, h)` (k, 4) → xyxy pixels of the input (H, W)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    h, w = size_hw
    scale = np.array([w, h, w, h], dtype=np.float64)
    return np.concatenate([b[:, :2] - b[:, 2:] / 2, b[:, :2] + b[:, 2:] / 2], axis=1) * scale


def assign_batch(zc, boxes_xyxy, pts, st, gt_boxes, gt_labels, size_hw, topk=10):
    """`assign_tal` of each image: (target boxes in grid units (N, A, 4), target scores
    (N, A, C), fg (N, A))."""
    n, a, c = zc.shape
    tb, ts, fg = np.zeros((n, a, 4)), np.zeros((n, a, c)), np.zeros((n, a), dtype=bool)
    s = sigmoid(zc)
    for b in range(n):
        t_box, ts[b], fg[b], _ = assign_tal(s[b], boxes_xyxy[b] * st[:, None], pts * st[:, None],
                                            gt_labels[b], gt_pixels(gt_boxes[b], size_hw),
                                            topk=topk)
        tb[b] = t_box / st[:, None]
    return tb, ts, fg


def yolo_v8_loss(outputs, gt_boxes, gt_labels, levels, num_classes, hyp=HYP_V8,
                 reg_max=REG_MAX, targets_from=None):
    """Loss and gradients with respect to the raw head maps (see the module doc).

    `outputs`: {id: (N, 4·reg_max + C, S_h, S_w)}; `gt_boxes[n]`: (k, 4) normalized
    `(cx, cy, w, h)`; `gt_labels[n]`: (k,); `levels`: `v8_levels(net)`; `targets_from`: the
    outputs that set the assignment and the CIoU α (default: `outputs`), so that a gradient
    check can hold these constants fixed.
    """
    zb, zc, grids, pts, st, d, p, pbox = _decode(outputs, levels, reg_max)
    if zc.shape[-1] != num_classes:
        raise ValueError(f"{zc.shape[-1]} class channels, expected {num_classes}")
    n = zc.shape[0]
    size_hw = (grids[0][0] * levels[0][1], grids[0][1] * levels[0][1])
    if targets_from is None:
        ref_zc, ref_box = zc, pbox
    else:
        _, ref_zc, *_, ref_box = _decode(targets_from, levels, reg_max)
    tb, ts, fg = assign_batch(ref_zc, ref_box, pts, st, gt_boxes, gt_labels, size_hw,
                              hyp.get("topk", 10))
    tss = max(float(ts.sum()), 1.0)

    # Classes: every anchor and class.
    l_cls = float(bce_logits(zc, ts).sum()) / tss
    dzc = (sigmoid(zc) - ts) / tss * hyp["cls"]
    dzb = np.zeros_like(zb)
    l_box = l_dfl = 0.0
    bi, aj = np.nonzero(fg)
    if len(bi):
        w = ts[bi, aj].sum(axis=1)
        pc, gc = xyxy_to_cxcywh(pbox[bi, aj]), xyxy_to_cxcywh(tb[bi, aj])
        alpha = None if targets_from is None else ciou_alpha(xyxy_to_cxcywh(ref_box[bi, aj]), gc)
        ci, dci = ciou(pc, gc, alpha=alpha)
        l_box = float(((1 - ci) * w).sum()) / tss
        dxyxy = _xyxy_grad(-dci * (w / tss * hyp["box"])[:, None])
        dd = dxyxy * np.array([-1.0, -1.0, 1.0, 1.0])      # x1 = a_x − d_l, x2 = a_x + d_r
        dzb[bi, aj] += dfl_expect_backward(dd, p[bi, aj])
        # DFL: two-bin soft cross-entropy on each side.
        g = np.concatenate([pts[aj] - tb[bi, aj, :2], tb[bi, aj, 2:] - pts[aj]], axis=1)
        g = np.clip(g, 0, reg_max - 1 - 0.01)
        gl = np.floor(g).astype(np.int64)
        wl = gl + 1 - g
        z = zb[bi, aj]                                        # (K, 4, reg_max)
        zmax = z.max(axis=-1, keepdims=True)
        logp = z - zmax - np.log(np.exp(z - zmax).sum(axis=-1, keepdims=True))
        k_idx, s_idx = np.meshgrid(np.arange(len(bi)), np.arange(4), indexing="ij")
        ce = -(logp[k_idx, s_idx, gl] * wl + logp[k_idx, s_idx, gl + 1] * (1 - wl))
        l_dfl = float((ce.mean(axis=1) * w).sum()) / tss
        q = np.zeros_like(z)
        q[k_idx, s_idx, gl] = wl
        q[k_idx, s_idx, gl + 1] += 1 - wl
        dzb[bi, aj] += (np.exp(logp) - q) * (w / (4 * tss) * hyp["dfl"])[:, None, None]

    # Back to the head maps.
    grad = np.concatenate([dzb.reshape(n, len(pts), -1), dzc], axis=2).transpose(0, 2, 1) * n
    douts, a0 = {}, 0
    for (hid, _), (sh, sw) in zip(levels, grids):
        g_l = grad[:, :, a0:a0 + sh * sw]
        douts[hid] = g_l.reshape(outputs[hid].shape).astype(outputs[hid].dtype)
        a0 += sh * sw
    parts = {"box": hyp["box"] * l_box * n, "cls": hyp["cls"] * l_cls * n,
             "dfl": hyp["dfl"] * l_dfl * n}
    return LossResult(total=sum(parts.values()), douts=douts, parts=parts, terms={},
                      masks={"fg": fg, "target_scores": ts, "target_scores_sum": tss})
