"""YOLOv5 loss (T33.6) and the OTA assignment of YOLOv7 (T34.3), with gradients.

Each head output `(N, A·(5+C), S_h, S_w)` is raw logits (the head conv is linear, the
sigmoid is applied here), seen as `(N, A, 5+C, S_h, S_w)`. With the matches of
`build_targets_v5` (or `build_targets_ota`), per level l:

    p_xy = 2σ(t_xy) − 0.5,  p_wh = (2σ(t_wh))²·A           (grid units, cell offsets)
    L_box,l = mean over matches of 1 − CIoU(p, target)
    L_obj,l = mean over every cell and anchor of BCE(t_o, IoU*)   (IoU* detached, ≥ 0)
    L_cls,l = mean over matches × classes of BCE(t_c, one-hot)    (C > 1 only)

    L = N·(g_box Σ_l L_box,l + g_obj Σ_l balance_l L_obj,l + g_cls Σ_l L_cls,l)

with the gains of the hyperparameter file scaled as the reference `train.py`:
g_box = box·3/n_l, g_cls = cls·(C/80)·3/n_l, g_obj = obj·(S/640)²·3/n_l (S the training
image size). When two matches share a cell and an anchor, the last one sets IoU* (as the
indexed assignment of the reference); their box and class gradients add up.

OTA (`ComputeLossOTA` of YOLOv7, `loss_ota: 1` in `hyp.scratch.tiny.yaml`): the candidates
of `build_targets_v5` of each image (all levels together) are reassigned to its ground
truths by a SimOTA-style rule. Pairwise IoU in pixels between ground truth i and the decoded
candidate box j; cost C_ij = BCE(logit √(σ(t_c)·σ(t_o)), one-hot_i) summed over classes
+ 3·(−ln(IoU_ij + 10⁻⁸)); dynamic k_i = max(1, ⌊Σ top-10 IoU_i·⌋); ground truth i keeps its
k_i cheapest candidates; a candidate kept by several ground truths goes to the cheapest one.
The assignment uses the predictions but is a constant of the backward.

The loss is a total for the batch (already multiplied by N, as the reference); `douts` are
its gradients with respect to the raw outputs.
"""

import numpy as np

from yolo.data.targets import build_targets_v5, grid_hw
from yolo.infer.boxes import ciou, ciou_alpha, cxcywh_to_xyxy, iou_xyxy
from yolo.layers.activations import sigmoid
from yolo.train.loss import LossResult, bce_logits

# hyp.scratch-low.yaml (YOLOv5 v6.0+, T33.6) and hyp.scratch.tiny.yaml (YOLOv7, T34.3):
# loss gains and the anchor threshold. `balance`: objectness weight of the levels, small
# stride first (`ComputeLoss` with 3 levels). The training part is in `yolo.train.recipe_v5`.
HYP_V5 = {"box": 0.05, "cls": 0.5, "obj": 1.0, "anchor_t": 4.0, "balance": (4.0, 1.0, 0.4),
          "label_smoothing": 0.0, "ota": False}
HYP_V7_TINY = {**HYP_V5, "ota": True}


def gains(hyp, num_classes, n_levels, img_size):
    """(g_box, g_obj, g_cls): the gains of `hyp` scaled as the reference `train.py`."""
    s = 3 / n_levels
    return (hyp["box"] * s, hyp["obj"] * (img_size / 640) ** 2 * s,
            hyp["cls"] * num_classes / 80 * s)


def _views(outputs, levels, num_classes):
    c5 = 5 + num_classes
    views = {}
    for hid, _, anchors in levels:
        out = outputs[hid]
        n, ch, sh, sw = out.shape
        if ch != len(anchors) * c5:
            raise ValueError(f"head {hid}: {ch} channels, expected {len(anchors)}×{c5}")
        views[hid] = out.reshape(n, len(anchors), c5, sh, sw).astype(np.float64)
    return views


def _pred_boxes(ps, anch):
    """Matched logits (K, 5+C) → (box (K, 4) in grid units relative to the cell, σ (K, 4))."""
    s = sigmoid(ps[:, :4])
    return np.concatenate([2 * s[:, :2] - 0.5, (2 * s[:, 2:4]) ** 2 * anch], axis=1), s


def ota_match(cost, iou):
    """(G, M) bool: ground truth i keeps its k_i = max(1, ⌊Σ top-10 IoU_i·⌋) cheapest
    candidates; a candidate kept by several ground truths goes to the cheapest (the first
    on equal costs)."""
    top = -np.sort(-iou, axis=1)[:, :min(10, iou.shape[1])]
    ks = np.maximum(top.sum(axis=1).astype(np.int64), 1)
    match = np.zeros(cost.shape, dtype=bool)
    for g in range(len(cost)):
        match[g, np.argsort(cost[g], kind="stable")[:ks[g]]] = True
    multi = match.sum(axis=0) > 1
    if multi.any():
        match[:, multi] = False
        match[np.argmin(cost[:, multi], axis=0), np.nonzero(multi)[0]] = True
    return match


def build_targets_ota(outputs, boxes_list, labels_list, levels, num_classes, anchor_t=4.0):
    """OTA assignment (T34.3, see the module doc): same format as `build_targets_v5`."""
    views = _views(outputs, levels, num_classes)
    grids = {hid: views[hid].shape[-2:] for hid, _, _ in levels}
    cand = build_targets_v5(boxes_list, labels_list, levels, grids, anchor_t)
    keys = ("b", "a", "gj", "gi", "cls", "anch")
    picked = {hid: {k: [] for k in (*keys, "tbox")} for hid, _, _ in levels}
    for n, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        if len(boxes) == 0:
            continue
        rows = []  # (level index, candidate index) of the image's candidates
        pred, logit_cls, logit_obj = [], [], []
        for li, (hid, stride, _) in enumerate(levels):
            c = cand[hid]
            idx = np.nonzero(c["b"] == n)[0]
            if len(idx) == 0:
                continue
            ps = views[hid][n, c["a"][idx], :, c["gj"][idx], c["gi"][idx]]
            box, _ = _pred_boxes(ps, c["anch"][idx])
            box[:, 0] += c["gi"][idx]
            box[:, 1] += c["gj"][idx]
            pred.append(box * stride)
            logit_cls.append(ps[:, 5:])
            logit_obj.append(ps[:, 4:5])
            rows += [(li, j) for j in idx]
        if not rows:
            continue
        hid0, stride0, _ = levels[0]
        sh, sw = views[hid0].shape[-2:]
        img_wh = np.array([sw * stride0, sh * stride0], dtype=np.float64)
        gt_px = boxes * np.concatenate([img_wh, img_wh])
        pred = np.concatenate(pred)
        iou = iou_xyxy(cxcywh_to_xyxy(gt_px), cxcywh_to_xyxy(pred))       # (G, M)
        iou_cost = -np.log(iou + 1e-8)
        y = np.sqrt(sigmoid(np.concatenate(logit_cls)) * sigmoid(np.concatenate(logit_obj)))
        logit = np.log(y / (1 - y))
        onehot = labels[:, None, None] == np.arange(num_classes)[None, None, :]
        cls_cost = bce_logits(logit[None], onehot.astype(np.float64)).sum(axis=2)  # (G, M)
        match = ota_match(cls_cost + 3.0 * iou_cost, iou)
        fg = match.any(axis=0)
        owner = match[:, fg].argmax(axis=0)
        for (li, j), g in zip((r for r, f in zip(rows, fg) if f), owner):
            hid, _, _ = levels[li]
            c = cand[hid]
            sh, sw = grids[hid]
            d = picked[hid]
            for k in keys[:4]:
                d[k].append(c[k][j])
            d["cls"].append(labels[g])
            d["anch"].append(c["anch"][j])
            gxy = boxes[g, :2] * (sw, sh)
            d["tbox"].append([gxy[0] - c["gi"][j], gxy[1] - c["gj"][j],
                              boxes[g, 2] * sw, boxes[g, 3] * sh])
    out = {}
    for hid, _, _ in levels:
        d = picked[hid]
        out[hid] = {k: np.asarray(d[k], dtype=np.int64) for k in keys[:5]}
        out[hid]["anch"] = np.asarray(d["anch"], dtype=np.float64).reshape(-1, 2)
        out[hid]["tbox"] = np.asarray(d["tbox"], dtype=np.float64).reshape(-1, 4)
    return out


def yolo_v5_loss(outputs, gt_boxes, gt_labels, levels, num_classes, hyp=HYP_V5,
                 img_size=None, targets_from=None):
    """Loss and gradients with respect to the raw head outputs (see the module doc).

    `outputs`: {id: (N, A·(5+C), S_h, S_w)}; `gt_boxes[n]`: (k, 4) normalized
    `(cx, cy, w, h)`; `gt_labels[n]`: (k,); `levels`: `v5_levels(net)`; `img_size`: the S
    of the objectness gain (default: input size of the first level); `targets_from`: the
    outputs that set the assignment and the IoU targets (default: `outputs`), so that a
    gradient check can hold these constants fixed.
    """
    ref = outputs if targets_from is None else targets_from
    views = _views(outputs, levels, num_classes)
    ref_views = views if targets_from is None else _views(ref, levels, num_classes)
    grids = {hid: views[hid].shape[-2:] for hid, _, _ in levels}
    if hyp.get("ota"):
        targets = build_targets_ota(ref, gt_boxes, gt_labels, levels, num_classes,
                                    hyp["anchor_t"])
    else:
        targets = build_targets_v5(gt_boxes, gt_labels, levels, grids, hyp["anchor_t"])
    hid0, stride0, _ = levels[0]
    if img_size is None:
        img_size = max(grid_hw(grids[hid0])) * stride0
    g_box, g_obj, g_cls = gains(hyp, num_classes, len(levels), img_size)
    balance = hyp["balance"]
    eps_ls = hyp.get("label_smoothing", 0.0)
    cp, cn = 1.0 - 0.5 * eps_ls, 0.5 * eps_ls
    bs = next(iter(views.values())).shape[0]
    parts = {"box": 0.0, "obj": 0.0, "cls": 0.0}
    douts, masks = {}, {}
    for li, (hid, _, _) in enumerate(levels):
        p = views[hid]
        grad = np.zeros_like(p)
        t = targets[hid]
        b, a, gj, gi = t["b"], t["a"], t["gj"], t["gi"]
        k = len(b)
        tobj = np.zeros(p[:, :, 4].shape)
        if k:
            ps = p[b, a, :, gj, gi]                                   # (K, 5+C)
            pbox, s = _pred_boxes(ps, t["anch"])
            alpha = None
            if targets_from is not None:
                ref_box = _pred_boxes(ref_views[hid][b, a, :, gj, gi], t["anch"])[0]
                alpha = ciou_alpha(ref_box, t["tbox"])
            ci, dci = ciou(pbox, t["tbox"], alpha=alpha)
            parts["box"] += g_box * float((1.0 - ci).mean())
            # ∂/∂t of mean(1 − CIoU): −1/K ∂CIoU/∂p · ∂p/∂t
            dp = -dci / k * g_box
            dt = np.empty((k, 4))
            dt[:, :2] = dp[:, :2] * 2 * s[:, :2] * (1 - s[:, :2])
            dt[:, 2:] = dp[:, 2:] * 8 * s[:, 2:] * s[:, 2:] * (1 - s[:, 2:]) * t["anch"]
            gsel = np.zeros((k, p.shape[2]))
            gsel[:, :4] = dt
            iou_ref = ci if targets_from is None else ciou(ref_box, t["tbox"])[0]
            tobj[b, a, gj, gi] = np.maximum(iou_ref, 0)              # gr = 1, last wins
            if num_classes > 1:
                tc = np.full((k, num_classes), cn)
                tc[np.arange(k), t["cls"]] = cp
                parts["cls"] += g_cls * float(bce_logits(ps[:, 5:], tc).mean())
                gsel[:, 5:] = (sigmoid(ps[:, 5:]) - tc) / (k * num_classes) * g_cls
            np.add.at(grad, (b, a, slice(None), gj, gi), gsel)
        to = p[:, :, 4]
        w_obj = g_obj * balance[li]
        parts["obj"] += w_obj * float(bce_logits(to, tobj).mean())
        grad[:, :, 4] += (sigmoid(to) - tobj) / to.size * w_obj
        douts[hid] = (grad * bs).reshape(outputs[hid].shape).astype(outputs[hid].dtype)
        masks[hid] = {"tobj": tobj, "matches": k}
    parts = {key: v * bs for key, v in parts.items()}
    return LossResult(total=sum(parts.values()), douts=douts, parts=parts, terms={},
                      masks=masks)
