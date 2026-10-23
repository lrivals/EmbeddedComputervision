"""Task-aligned assignment of YOLOv8 (`TaskAlignedAssigner`, T35.4), one image.

For ground truth i (class c_i, box g_i) and anchor point j (predicted scores s_j, box p_j):

    candidates     centre of j strictly inside g_i (margin 10⁻⁹)
    u_ij           CIoU(g_i, p_j) clamped at 0, on the candidates (0 elsewhere)
    t_ij           s_{j,c_i}^α · u_ij^β   (α = 0.5, β = 6)
    positives      the top-k (k = 10) anchors of t_i·, kept if they are candidates
    conflicts      an anchor positive for several ground truths goes to argmax_i u_ij
    soft targets   t̂_ij = t_ij · max_j u_ij / (max_j t_ij + ε) on the positives of i;
                   the class target of anchor j is max_i t̂_ij on the class of its ground truth

This is the rule of the YOLOv8 releases (8.0 to 8.2, the published weights); Ultralytics 8.3+
also enlarges ground-truth sides under 16 px before the candidate test (`stride_val`), which
is left out here. The top-k is taken over the whole row t_i· (zeros outside the candidates,
ties in index order), then masked by the candidates, as the reference.

The assignment works on detached predictions (plain arrays in, nothing for a backward):
it is a constant of the loss gradient (`yolo.train.loss_v8`).
"""

import numpy as np

from yolo.infer.boxes import ciou, xyxy_to_cxcywh

TOPK, ALPHA, BETA, EPS = 10, 0.5, 6.0, 1e-9


def candidates_in_gts(points, gt_boxes, eps=EPS):
    """(G, A) mask: anchor centre `points` (A, 2) strictly inside `gt_boxes` (G, 4) xyxy."""
    lt = points[None, :, :] - gt_boxes[:, None, :2]
    rb = gt_boxes[:, None, 2:] - points[None, :, :]
    return np.concatenate([lt, rb], axis=2).min(axis=2) > eps


def assign_tal(pred_scores, pred_boxes, anchor_points, gt_labels, gt_boxes, topk=TOPK,
               alpha=ALPHA, beta=BETA, eps=EPS):
    """One image, everything in pixels.

    `pred_scores` (A, C) probabilities (σ of the class logits); `pred_boxes` (A, 4) xyxy;
    `anchor_points` (A, 2); `gt_labels` (G,); `gt_boxes` (G, 4) xyxy.
    Returns (target_boxes (A, 4) xyxy, target_scores (A, C), fg_mask (A,) bool,
    gt_index (A,): the ground truth of each anchor, 0 for the background).
    """
    s = np.asarray(pred_scores, dtype=np.float64)
    pb = np.asarray(pred_boxes, dtype=np.float64).reshape(-1, 4)
    pts = np.asarray(anchor_points, dtype=np.float64).reshape(-1, 2)
    gl = np.asarray(gt_labels, dtype=np.int64).reshape(-1)
    gb = np.asarray(gt_boxes, dtype=np.float64).reshape(-1, 4)
    a, c = s.shape
    g = len(gb)
    if g == 0:
        return (np.zeros((a, 4)), np.zeros((a, c)), np.zeros(a, dtype=bool),
                np.zeros(a, dtype=np.int64))
    in_gts = candidates_in_gts(pts, gb, eps)
    gi, aj = np.nonzero(in_gts)
    overlaps = np.zeros((g, a))
    align = np.zeros((g, a))
    if len(gi):
        u = np.maximum(ciou(xyxy_to_cxcywh(pb[aj]), xyxy_to_cxcywh(gb[gi]))[0], 0)
        overlaps[gi, aj] = u
        align[gi, aj] = s[aj, gl[gi]] ** alpha * u ** beta
    top = np.argsort(-align, axis=1, kind="stable")[:, :topk]
    mask_pos = np.zeros((g, a), dtype=bool)
    np.put_along_axis(mask_pos, top, True, axis=1)
    mask_pos &= in_gts
    multi = mask_pos.sum(axis=0) > 1
    if multi.any():
        best = np.zeros((g, a), dtype=bool)
        best[overlaps.argmax(axis=0), np.arange(a)] = True
        mask_pos[:, multi] = best[:, multi]
    fg = mask_pos.any(axis=0)
    idx = mask_pos.argmax(axis=0)
    target_boxes = gb[idx]
    target_scores = np.zeros((a, c))
    target_scores[np.arange(a), gl[idx]] = fg
    align *= mask_pos
    pos_align = align.max(axis=1, keepdims=True)
    pos_ov = (overlaps * mask_pos).max(axis=1, keepdims=True)
    norm = (align * pos_ov / (pos_align + eps)).max(axis=0)
    return target_boxes, target_scores * norm[:, None], fg, idx
