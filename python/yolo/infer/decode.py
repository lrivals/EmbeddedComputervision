"""Décodage des têtes : sorties brutes → boîtes, objectness, scores de classes (§8.1).

    b_x = (σ(t_x) + j)/S_w,  b_y = (σ(t_y) + i)/S_h,  b_w = p_w e^{t_w},  b_h = p_h e^{t_h}

YOLOv4-tiny (`scale_x_y` = α de la couche [yolo], T24.2) : σ(t) devient σ(t)·α − (α − 1)/2.

YOLOv7-tiny (`new_coords`, T26.0, as YOLOv5) : b_w = p_w·(2σ(t_w))², b_h = p_h·(2σ(t_h))², the
centre as above with α = 2 (σ·2 − 0,5) ; the sigmoid of the head conv is applied here.

YOLOv5 (`mode="v5"`, T33.4) : the same formulas, b_xy = (2σ(t_xy) − 0.5 + c_xy)·s,
b_wh = (2σ(t_wh))²·A, score σ(t_o)·σ(t_c). The anchors of these networks are pixels of the
actual input (`v5_levels`), not of a 416 reference: `decode` divides them by S·stride.

Score de la classe c : σ(t_o)·σ(t_c) (YOLOv3) ou σ(t_o)·softmax(t)_c (YOLOv2, softmax
stabilisé). Les boîtes sont rendues en `(cx, cy, w, h)` normalisés dans l'image letterbox ;
`to_original` les ramène dans l'image d'origine.
"""

import numpy as np

from yolo.data.letterbox import boxes_from_letterbox
from yolo.data.targets import anchor_ref, anchors_frac, heads, is_v5, v5_levels
from yolo.layers.activations import sigmoid


def scale_xy(sig, alpha):
    """σ(t)·α − (α − 1)/2 (`scale_x_y` de Darknet) ; l'identité pour α = 1."""
    return sig if alpha == 1 else sig * alpha - (alpha - 1) / 2


def wh_new_coords(sig):
    """(2σ)² : facteur d'ancre des têtes `new_coords` (Darknet : 4·σ²)."""
    return 4.0 * sig * sig


def decode_head(out, anchors, num_classes, mode="v3", alpha=1.0, new_coords=False):
    """`out` : (N, A·(5+C), S_h, S_w) ; `anchors` : (A, 2) en fraction de l'image ; `alpha` :
    `scale_x_y` de la tête ; `new_coords` : b_w = p_w·(2σ(t_w))² au lieu de p_w·e^{t_w} ;
    `mode="v5"` : sigmoid classes with α = 2 and `new_coords` forced (YOLOv5, T33.4).

    Rend `boxes` (N, A·S_h·S_w, 4), `obj` (N, A·S_h·S_w), `scores` (N, A·S_h·S_w, C), dans
    l'ordre (ancre, ligne, colonne).
    """
    if mode == "v5":
        mode, alpha, new_coords = "v3", 2.0, True
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    n, _, sh, sw = out.shape
    a = len(anchors)
    p = out.reshape(n, a, 5 + num_classes, sh, sw).astype(np.float64)
    i, j = np.meshgrid(np.arange(sh), np.arange(sw), indexing="ij")
    # §8.1
    bx = (scale_xy(sigmoid(p[:, :, 0]), alpha) + j) / sw
    by = (scale_xy(sigmoid(p[:, :, 1]), alpha) + i) / sh
    fw, fh = ((wh_new_coords(sigmoid(p[:, :, 2])), wh_new_coords(sigmoid(p[:, :, 3])))
              if new_coords else (np.exp(p[:, :, 2]), np.exp(p[:, :, 3])))
    bw = anchors[None, :, 0, None, None] * fw
    bh = anchors[None, :, 1, None, None] * fh
    obj = sigmoid(p[:, :, 4])
    tc = p[:, :, 5:]
    if mode == "v3":
        cls = sigmoid(tc)
    elif mode == "v2":
        e = np.exp(tc - tc.max(axis=2, keepdims=True))  # §8.1 : max soustrait
        cls = e / e.sum(axis=2, keepdims=True)
    else:
        raise ValueError(f"mode inconnu : {mode!r}")
    scores = obj[:, :, None] * cls
    boxes = np.stack([bx, by, bw, bh], axis=-1).reshape(n, -1, 4)
    return (boxes, obj.reshape(n, -1),
            scores.transpose(0, 1, 3, 4, 2).reshape(n, -1, num_classes))


def decode(outputs, net):
    """Toutes les têtes de `net` (format `specs`) concaténées ; mode v2 pour `region`,
    `decode_v5` pour les réseaux YOLOv5 / YOLOv7 (`is_v5`)."""
    if is_v5(net):
        return decode_v5(outputs, net)
    anchors = anchors_frac(net["anchors"], anchor_ref(net))
    parts = []
    for hid, mask in heads(net):
        layer = net["layers"][hid]
        mode = "v2" if layer["type"] == "region" else "v3"
        parts.append(decode_head(outputs[hid], anchors[mask], net["classes"], mode,
                                 layer.get("scale_x_y", 1.0), bool(layer.get("new_coords"))))
    return tuple(np.concatenate(x, axis=1) for x in zip(*parts))


def decode_v5(outputs, net):
    """Heads of a YOLOv5-style network (`v5_levels`), concatenated, small stride first;
    anchors in pixels of the input size S·stride of each map."""
    parts = []
    for hid, stride, anchors in v5_levels(net):
        sh, sw = outputs[hid].shape[-2:]
        frac = anchors / np.array([sw * stride, sh * stride], dtype=np.float64)
        parts.append(decode_head(outputs[hid], frac, net["classes"], "v5"))
    return tuple(np.concatenate(x, axis=1) for x in zip(*parts))


def to_original(boxes, width, height, size, mode="letterbox"):
    """Boîtes de l'image letterbox `size` (S ou (H, W)) → normalisées dans l'image d'origine
    (w × h) ; `mode` : `letterbox` ou `v5`."""
    return boxes_from_letterbox(boxes, width, height, size, mode)


# ------------------------------------------------------------------ décodage entier (§9.4)
def decode_head_int(q, anchors, num_classes, mode, luts, obj_thr, alpha=1.0,
                    new_coords=False):
    """Une image, une tête int8 `q` (A·(5+C), S_h, S_w) d'échelle `luts.scale`.

    §9.4 : on garde d'abord les cellules dont l'entier t_o dépasse le logit du seuil (le
    score σ(t_o)·p_c ≤ σ(t_o) : aucune boîte au-dessus du seuil n'est perdue), puis les
    tables (Q16) ne servent qu'aux survivantes ; `scale_x_y` (`alpha`) s'applique après la
    table ; `new_coords` : (2σ)² par la table sigmoïde au lieu de l'exponentielle. Rend `boxes` (K, 4), `obj` (K,),
    `scores` (K, C) dans l'ordre (ancre, ligne, colonne), comme `decode_head`.
    """
    from yolo.quant.lut import ONE, SOFTMAX_OFFSET, logit_threshold_q, lookup

    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    a = len(anchors)
    _, sh, sw = q.shape
    p = np.asarray(q, dtype=np.int64).reshape(a, 5 + num_classes, sh, sw)
    keep = p[:, 4] >= logit_threshold_q(obj_thr, luts.scale)
    k, i, j = np.nonzero(keep)
    t = p[k, :, i, j]  # (K, 5 + C) entiers
    bx = (scale_xy(lookup(luts.sigmoid, t[:, 0]) / ONE, alpha) + j) / sw
    by = (scale_xy(lookup(luts.sigmoid, t[:, 1]) / ONE, alpha) + i) / sh
    if new_coords:
        fw = wh_new_coords(lookup(luts.sigmoid, t[:, 2]) / ONE)
        fh = wh_new_coords(lookup(luts.sigmoid, t[:, 3]) / ONE)
    else:
        fw = lookup(luts.exp, t[:, 2]) / 2.0**luts.exp_frac
        fh = lookup(luts.exp, t[:, 3]) / 2.0**luts.exp_frac
    bw = anchors[k, 0] * fw
    bh = anchors[k, 1] * fh
    obj = lookup(luts.sigmoid, t[:, 4]) / ONE
    tc = t[:, 5:]
    if mode == "v3":
        cls = lookup(luts.sigmoid, tc) / ONE
    elif mode == "v2":
        # Softmax en trois étages : max entier, e^{(q − q_max)s} par table, division.
        e = lookup(luts.softmax_exp, tc - tc.max(axis=1, keepdims=True), SOFTMAX_OFFSET)
        cls = e / e.sum(axis=1, keepdims=True, dtype=np.int64)
    else:
        raise ValueError(f"mode inconnu : {mode!r}")
    boxes = np.stack([bx, by, bw, bh], axis=-1).reshape(-1, 4)
    return boxes, obj, obj[:, None] * cls


def decode_int(outputs, net, luts, obj_thr):
    """Têtes int8 de `net` → par image `(boxes, scores)` des cellules retenues.

    `luts` : {id de tête: HeadLuts}.
    """
    anchors = anchors_frac(net["anchors"], anchor_ref(net))
    n = len(next(iter(outputs.values())))
    out = []
    for b in range(n):
        parts = []
        for hid, mask in heads(net):
            layer = net["layers"][hid]
            mode = "v2" if layer["type"] == "region" else "v3"
            boxes, _, scores = decode_head_int(outputs[hid][b], anchors[mask], net["classes"],
                                               mode, luts[hid], obj_thr,
                                               layer.get("scale_x_y", 1.0),
                                               bool(layer.get("new_coords")))
            parts.append((boxes, scores))
        out.append(tuple(np.concatenate(x, axis=0) for x in zip(*parts)))
    return out
