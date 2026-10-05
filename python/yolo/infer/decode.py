"""Décodage des têtes : sorties brutes → boîtes, objectness, scores de classes (§8.1).

    b_x = (σ(t_x) + j)/S,  b_y = (σ(t_y) + i)/S,  b_w = p_w e^{t_w},  b_h = p_h e^{t_h}

Score de la classe c : σ(t_o)·σ(t_c) (YOLOv3) ou σ(t_o)·softmax(t)_c (YOLOv2, softmax
stabilisé). Les boîtes sont rendues en `(cx, cy, w, h)` normalisés dans l'image letterbox ;
`to_original` les ramène dans l'image d'origine.
"""

import numpy as np

from yolo.data.letterbox import boxes_from_letterbox
from yolo.data.targets import anchors_frac, heads
from yolo.layers.activations import sigmoid


def decode_head(out, anchors, num_classes, mode="v3"):
    """`out` : (N, A·(5+C), S, S) ; `anchors` : (A, 2) en fraction de l'image.

    Rend `boxes` (N, A·S·S, 4), `obj` (N, A·S·S), `scores` (N, A·S·S, C), dans l'ordre
    (ancre, ligne, colonne).
    """
    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    n, _, s, _ = out.shape
    a = len(anchors)
    p = out.reshape(n, a, 5 + num_classes, s, s).astype(np.float64)
    i, j = np.meshgrid(np.arange(s), np.arange(s), indexing="ij")
    # §8.1
    bx = (sigmoid(p[:, :, 0]) + j) / s
    by = (sigmoid(p[:, :, 1]) + i) / s
    bw = anchors[None, :, 0, None, None] * np.exp(p[:, :, 2])
    bh = anchors[None, :, 1, None, None] * np.exp(p[:, :, 3])
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
    """Toutes les têtes de `net` (format `specs`) concaténées ; mode v2 pour `region`."""
    anchors = anchors_frac(net["anchors"])
    parts = []
    for hid, mask in heads(net):
        mode = "v2" if net["layers"][hid]["type"] == "region" else "v3"
        parts.append(decode_head(outputs[hid], anchors[mask], net["classes"], mode))
    return tuple(np.concatenate(x, axis=1) for x in zip(*parts))


def to_original(boxes, width, height, size):
    """Boîtes de l'image letterbox `size` → normalisées dans l'image d'origine (w × h)."""
    return boxes_from_letterbox(boxes, width, height, size)


# ------------------------------------------------------------------ décodage entier (§9.4)
def decode_head_int(q, anchors, num_classes, mode, luts, obj_thr):
    """Une image, une tête int8 `q` (A·(5+C), S, S) d'échelle `luts.scale`.

    §9.4 : on garde d'abord les cellules dont l'entier t_o dépasse le logit du seuil (le
    score σ(t_o)·p_c ≤ σ(t_o) : aucune boîte au-dessus du seuil n'est perdue), puis les
    tables (Q16) ne servent qu'aux survivantes. Rend `boxes` (K, 4), `obj` (K,),
    `scores` (K, C) dans l'ordre (ancre, ligne, colonne), comme `decode_head`.
    """
    from yolo.quant.lut import ONE, SOFTMAX_OFFSET, logit_threshold_q, lookup

    anchors = np.asarray(anchors, dtype=np.float64).reshape(-1, 2)
    a = len(anchors)
    _, s, _ = q.shape
    p = np.asarray(q, dtype=np.int64).reshape(a, 5 + num_classes, s, s)
    keep = p[:, 4] >= logit_threshold_q(obj_thr, luts.scale)
    k, i, j = np.nonzero(keep)
    t = p[k, :, i, j]  # (K, 5 + C) entiers
    bx = (lookup(luts.sigmoid, t[:, 0]) / ONE + j) / s
    by = (lookup(luts.sigmoid, t[:, 1]) / ONE + i) / s
    bw = anchors[k, 0] * (lookup(luts.exp, t[:, 2]) / 2.0**luts.exp_frac)
    bh = anchors[k, 1] * (lookup(luts.exp, t[:, 3]) / 2.0**luts.exp_frac)
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
    anchors = anchors_frac(net["anchors"])
    n = len(next(iter(outputs.values())))
    out = []
    for b in range(n):
        parts = []
        for hid, mask in heads(net):
            mode = "v2" if net["layers"][hid]["type"] == "region" else "v3"
            boxes, _, scores = decode_head_int(outputs[hid][b], anchors[mask], net["classes"],
                                               mode, luts[hid], obj_thr)
            parts.append((boxes, scores))
        out.append(tuple(np.concatenate(x, axis=0) for x in zip(*parts)))
    return out
