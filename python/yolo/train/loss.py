"""Perte YOLOv3 (§6.2) et variante YOLOv2 à softmax de classes (§2.2, §8.1), avec gradients.

Pour chaque tête, la sortie `(N, A·(5+C), S_h, S_w)` est vue en `(N, A, 5+C, S_h, S_w)`, canaux
`t_x, t_y, t_w, t_h, t_o`, puis les classes (docs/conventions.md).

    L = λ_coord Σ_obj ω [(σ(t_x)−x*)² + (σ(t_y)−y*)² + (t_w−t_w*)² + (t_h−t_h*)²]
      + Σ_obj BCE(σ(t_o), 1) + Σ_noobj BCE(σ(t_o), 0) + Σ_obj L_cls

- Coordonnées sur σ(t_x) (variante hors base du §6.2) ; ω = 2 − g_w g_h.
- *Ignore* (§2.3, §5.1) : ancre non responsable dont la boîte **prédite** recouvre une vérité
  de l'image avec une IoU > `ignore_thresh` ; exclue de tous les termes. Le masque est
  recalculé à chaque appel et traité comme une constante pour le gradient.
- `class_mode="sigmoid"` (v3) : L_cls = Σ_c BCE(σ(t_c), 1[c=c*]) ;
  `class_mode="softmax"` (v2) : L_cls = −ln softmax(t)_{c*}.
- La perte est une **somme** sur le lot ; la normalisation par lot est laissée au trainer.
"""

from dataclasses import dataclass, field

import numpy as np

from yolo.data.targets import ANCHOR_REF, anchors_frac, build_targets
from yolo.infer.boxes import iou
from yolo.layers.activations import sigmoid


@dataclass
class LossResult:
    total: float
    douts: dict                                  # {id de tête: δ sortie, forme de la sortie}
    parts: dict                                  # {coord, obj, noobj, cls}
    terms: dict                                  # {id: (N, A, 5+C, S, S)} ; Σ terms = total
    masks: dict = field(default_factory=dict)    # {id: {"obj", "ignore", "best_iou"}}


def softplus(t):
    """ln(1 + e^t) sans overflow."""
    return np.maximum(t, 0) + np.log1p(np.exp(-np.abs(t)))


def bce_logits(t, y):
    # BCE(σ(t), y) = −[y ln σ(t) + (1−y) ln(1−σ(t))] = softplus(t) − y t ; dérivée σ(t) − y (§6.2)
    return softplus(t) - y * t


def _grid(sh, sw):
    i, j = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    return i, j


def predicted_boxes(p, anchors):
    """Boîtes prédites (N, A, S_h, S_w, 4) en `(cx, cy, w, h)` normalisés (§8.1).

    `p` : (N, A, 5+C, S_h, S_w) ; `anchors` : (A, 2) en fraction de l'image.
    """
    sh, sw = p.shape[-2:]
    i, j = _grid(sh, sw)
    bx = (sigmoid(p[:, :, 0]) + j) / sw
    by = (sigmoid(p[:, :, 1]) + i) / sh
    # e^{t} borné : seule l'IoU du masque ignore en dépend.
    bw = anchors[None, :, 0, None, None] * np.exp(np.clip(p[:, :, 2], -50, 50))
    bh = anchors[None, :, 1, None, None] * np.exp(np.clip(p[:, :, 3], -50, 50))
    return np.stack([bx, by, bw, bh], axis=-1)


def ignore_mask(p, anchors, obj, gt_boxes, thresh):
    """(ignore, best_iou) de forme (N, A, S_h, S_w) — §5.1, 4."""
    pred = predicted_boxes(p, anchors)
    best = np.zeros(obj.shape)
    for n, gt in enumerate(gt_boxes):
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        if len(gt):
            best[n] = iou(pred[n].reshape(-1, 4), gt).max(axis=1).reshape(obj.shape[1:])
    return (best > thresh) & ~obj, best


def yolo_loss(outputs, gt_boxes, gt_labels, anchors_px, head_list, num_classes,
              class_mode="sigmoid", lambda_coord=1.0, ignore_thresh=0.5,
              anchor_ref=ANCHOR_REF):
    """Perte et gradients par rapport aux sorties brutes des têtes (§6.2).

    `outputs` : {id: (N, A·(5+C), S_h, S_w)} (sortie de `Network.forward`) ;
    `gt_boxes[n]` : (k, 4) `(cx, cy, w, h)` normalisés ; `gt_labels[n]` : (k,) ;
    `head_list` : `yolo.data.targets.heads(net)` ; `anchor_ref` :
    `yolo.data.targets.anchor_ref(net)`.
    """
    if class_mode not in ("sigmoid", "softmax"):
        raise ValueError(f"class_mode inconnu : {class_mode!r}")
    grids = {hid: tuple(outputs[hid].shape[-2:]) for hid, _ in head_list}
    targets = build_targets(gt_boxes, gt_labels, anchors_px, head_list, grids, anchor_ref)
    all_anchors = anchors_frac(anchors_px, anchor_ref)
    c5 = 5 + num_classes
    parts = dict.fromkeys(("coord", "obj", "noobj", "cls"), 0.0)
    douts, terms, masks = {}, {}, {}
    for hid, mask in head_list:
        out = outputs[hid]
        n, _, sh, sw = out.shape
        a = len(mask)
        if out.shape[1] != a * c5:
            raise ValueError(f"tête {hid} : {out.shape[1]} canaux, attendu {a}×{c5}")
        p = out.reshape(n, a, c5, sh, sw).astype(np.float64)
        t = targets[hid]
        obj = t["obj"]
        ignore, best = ignore_mask(p, all_anchors[mask], obj, gt_boxes, ignore_thresh)
        noobj = ~obj & ~ignore
        objf, noobjf = obj.astype(np.float64), noobj.astype(np.float64)

        term = np.zeros_like(p)
        grad = np.zeros_like(p)

        # §6.2 : coordonnées, λ ω (σ(t) − x*)² et λ ω (t_w − t_w*)², sur les ancres obj.
        w = lambda_coord * t["scale"] * objf
        for ch, key in ((0, "x"), (1, "y")):
            sg = sigmoid(p[:, :, ch])
            d = sg - t[key]
            term[:, :, ch] = w * d * d
            grad[:, :, ch] = 2 * w * d * sg * (1 - sg)          # 2λω(σ−x*)σ(1−σ)
        for ch, key in ((2, "tw"), (3, "th")):
            d = p[:, :, ch] - t[key]
            term[:, :, ch] = w * d * d
            grad[:, :, ch] = 2 * w * d                          # 2λω(t_w − t_w*)

        # §6.2 : objectness, BCE(σ(t_o), 1) si obj, BCE(σ(t_o), 0) si noobj, rien si ignore.
        to = p[:, :, 4]
        term_obj = objf * bce_logits(to, 1.0)
        term_noobj = noobjf * bce_logits(to, 0.0)
        term[:, :, 4] = term_obj + term_noobj
        grad[:, :, 4] = objf * (sigmoid(to) - 1) + noobjf * sigmoid(to)

        # Classes, sur les ancres obj.
        tc = p[:, :, 5:]
        onehot = (t["cls"][:, :, None] == np.arange(num_classes)[None, None, :, None, None])
        onehot = onehot.astype(np.float64)
        if class_mode == "sigmoid":
            # §2.3, §6.2 : BCE multi-label ; ∂/∂t_c = σ(t_c) − 1[c=c*]
            term[:, :, 5:] = objf[:, :, None] * bce_logits(tc, onehot)
            grad[:, :, 5:] = objf[:, :, None] * (sigmoid(tc) - onehot)
        else:
            # §2.2, §8.1 : softmax stabilisé (max soustrait) ; −ln softmax_{c*}, ∂ = softmax − 1
            z = tc - tc.max(axis=2, keepdims=True)
            lse = np.log(np.exp(z).sum(axis=2, keepdims=True))
            term[:, :, 5:] = objf[:, :, None] * onehot * (lse - z)
            grad[:, :, 5:] = objf[:, :, None] * (np.exp(z - lse) - onehot)

        parts["coord"] += float(term[:, :, :4].sum())
        parts["obj"] += float(term_obj.sum())
        parts["noobj"] += float(term_noobj.sum())
        parts["cls"] += float(term[:, :, 5:].sum())
        terms[hid] = term
        douts[hid] = grad.reshape(out.shape).astype(out.dtype)
        masks[hid] = {"obj": obj, "ignore": ignore, "best_iou": best}
    total = sum(parts.values())
    return LossResult(total=total, douts=douts, parts=parts, terms=terms, masks=masks)
