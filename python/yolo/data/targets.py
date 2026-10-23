"""Encodage des cibles : cellule, ancre responsable, cibles de régression (§5.1).

Une **tête** est une couche de sortie `yolo` (YOLOv3) ou `region` (YOLOv2) ; `heads(net)` la
décrit par `(id de couche, mask)`, `mask` = indices de ses ancres dans `net["anchors"]`.

Le masque *ignore* dépend des boîtes **prédites** : il est calculé dans la perte à chaque
itération (`yolo.train.loss`), pas ici.
"""

import numpy as np

from yolo.infer.boxes import iou_wh

# docs/conventions.md : ancres en pixels pour une entrée de 416. Elles sont converties en
# fraction de l'image avec cette référence quelle que soit la taille d'entrée (multi-échelle) :
# une ancre couvre la même part de l'image à 320 qu'à 608. Réseau non carré (T11.4) : ancres
# en pixels de l'entrée width × height de la cfg (`anchor_ref`).
ANCHOR_REF = 416


def anchor_ref(net):
    """(W, H) de référence des ancres de `net` : 416 × 416 pour un réseau carré (quelle que
    soit sa taille), width × height de la cfg sinon."""
    _, h, w = net["input"]
    return (ANCHOR_REF, ANCHOR_REF) if h == w else (w, h)


def anchors_frac(anchors_px, ref=ANCHOR_REF):
    """Ancres (K, 2) en fraction de l'image ; `ref` : W = H, ou (W, H) (`anchor_ref`)."""
    ref = np.broadcast_to(np.asarray(ref, dtype=np.float64), (2,))
    return np.asarray(anchors_px, dtype=np.float64).reshape(-1, 2) / ref


def grid_hw(s):
    """Grille S (carrée) ou (S_h, S_w) → (S_h, S_w)."""
    return tuple(int(v) for v in s) if isinstance(s, (tuple, list)) else (int(s), int(s))


def heads(net):
    """[(id, mask)] des couches de sortie de `net` (format de `yolo.models.specs`)."""
    out = []
    for i, layer in enumerate(net["layers"]):
        if layer["type"] == "yolo":
            out.append((i, list(layer["mask"])))
        elif layer["type"] == "region":
            out.append((i, list(range(layer["num"]))))
    return out


def is_v5(net):
    """True for the networks decoded and trained the YOLOv5 way (T33.4): the Ultralytics
    descriptions (`head: "anchor"`) and the Darknet `new_coords` heads (YOLOv7-tiny)."""
    return net.get("head") == "anchor" or any(
        layer.get("new_coords") for layer in net["layers"] if layer["type"] == "yolo")


def v5_levels(net):
    """[(output id, stride, anchors (A, 2) in pixels)] of a YOLOv5-style network, small
    stride first. The anchors are pixels of the input at any input size (no reference size,
    unlike `anchors_frac`); the stride gives the input size of a head map: S·stride."""
    from yolo.models.specs import infer_shapes

    shapes = infer_shapes(net)
    if net.get("head") == "anchor":
        pairs = [(hid, np.asarray(a, dtype=np.float64).reshape(-1, 2))
                 for hid, a in zip(net["outputs"], net["anchors"])]
    else:
        anchors = np.asarray(net["anchors"], dtype=np.float64).reshape(-1, 2)
        pairs = [(hid, anchors[mask]) for hid, mask in heads(net)]
    levels = [(hid, net["input"][1] // shapes[hid][1][1], a) for hid, a in pairs]
    return sorted(levels, key=lambda lv: lv[1])


def responsible(boxes, anchors, owner, grid_sizes):
    """Cell and anchor of each box by the rule of §5.1 (used by `build_targets` and the miss
    target of T36.0). `boxes` (k, 4) normalised `(cx, cy, w, h)`; `anchors` (K, 2) fractions;
    `owner` {global anchor: (head id, position in its mask)}. Returns [(k, hid, a, i, j)],
    `None` for a box whose best anchor belongs to no head.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(boxes) == 0:
        return []
    # §5.1, 2 : ancre de meilleure IoU de forme parmi toutes les ancres de toutes les
    # têtes ; l'échelle retenue est celle de l'ancre gagnante.
    best = np.argmax(iou_wh(boxes[:, 2:4], anchors), axis=1)
    out = []
    for (gx, gy, _, _), k in zip(boxes, best):
        if k not in owner:
            out.append(None)
            continue
        hid, a = owner[k]
        sh, sw = grid_hw(grid_sizes[hid])
        # §5.1, 1 : cellule qui contient le centre (g = 1 bornée à la dernière cellule).
        j = min(int(np.floor(gx * sw)), sw - 1)
        i = min(int(np.floor(gy * sh)), sh - 1)
        out.append((int(k), hid, a, i, j))
    return out


def anchor_owner(head_list):
    """{global anchor: (head id, position in its mask)} of `heads(net)`."""
    return {k: (hid, a) for hid, mask in head_list for a, k in enumerate(mask)}


def build_targets(boxes_list, labels_list, anchors_px, head_list, grid_sizes, ref=ANCHOR_REF):
    """Cibles de chaque tête pour un lot.

    `boxes_list[n]` : (k, 4) `(cx, cy, w, h)` normalisés de l'image n ; `labels_list[n]` : (k,).
    `head_list` : sortie de `heads(net)` ; `grid_sizes` : {id: S ou (S_h, S_w)} ; `ref` :
    référence des ancres (`anchor_ref`).

    Rend {id: dict} avec, de forme (N, A, S_h, S_w) : `obj` (bool), `x`, `y`, `tw`, `th` (cibles
    §5.1), `scale` (ω = 2 − g_w g_h, §6.2), `cls` (int64, −1 hors *obj*).
    Si deux objets tombent sur la même cellule et la même ancre, le dernier l'emporte
    (comme Darknet, hors base).
    """
    anchors = anchors_frac(anchors_px, ref)
    n_img = len(boxes_list)
    owner = {}  # ancre globale -> (id de tête, position dans le mask)
    targets = {}
    for hid, mask in head_list:
        shape = (n_img, len(mask), *grid_hw(grid_sizes[hid]))
        targets[hid] = {
            "obj": np.zeros(shape, dtype=bool),
            "x": np.zeros(shape), "y": np.zeros(shape),
            "tw": np.zeros(shape), "th": np.zeros(shape),
            "scale": np.zeros(shape),
            "cls": np.full(shape, -1, dtype=np.int64),
        }
        for a, k in enumerate(mask):
            owner[k] = (hid, a)
    for n, (boxes, labels) in enumerate(zip(boxes_list, labels_list)):
        boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
        for (gx, gy, gw, gh), label, r in zip(boxes, labels,
                                              responsible(boxes, anchors, owner, grid_sizes)):
            if r is None:
                continue
            k, hid, a, i, j = r
            sh, sw = grid_hw(grid_sizes[hid])
            pw, ph = anchors[k]
            t = targets[hid]
            t["obj"][n, a, i, j] = True
            # §5.1, 3 : x* = g_x S − j, y* = g_y S − i, t_w* = ln(g_w/p_w), t_h* = ln(g_h/p_h)
            t["x"][n, a, i, j] = gx * sw - j
            t["y"][n, a, i, j] = gy * sh - i
            t["tw"][n, a, i, j] = np.log(gw / pw)
            t["th"][n, a, i, j] = np.log(gh / ph)
            t["scale"][n, a, i, j] = 2.0 - gw * gh  # §6.2 : ω = 2 − g_w g_h
            t["cls"][n, a, i, j] = int(label)
    return targets


# Neighbour cells of the YOLOv5 assignment (T33.5): the cell itself, left, up, right, down.
V5_OFFSETS = np.array([[0, 0], [1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64) * 0.5


def build_targets_v5(boxes_list, labels_list, levels, grid_sizes, anchor_t=4.0):
    """YOLOv5 targets (T33.5, `ComputeLoss.build_targets` of the reference) for a batch.

    `boxes_list[n]`: (k, 4) normalized `(cx, cy, w, h)`; `labels_list[n]`: (k,); `levels`:
    `v5_levels(net)`; `grid_sizes`: {id: S or (S_h, S_w)}.

    For each level, a box (in grid units) matches anchor a when max(r, 1/r) < `anchor_t` for
    r = w / A_w and r = h / A_h (several anchors and levels can match). Each match is
    assigned to its cell and to the two nearest neighbours: left when frac(x) < 0.5 and
    x > 1, right when frac(x) > 0.5 and x < S_w − 1, same in y, i.e. 3 cells inside the
    grid. Returns {id: dict} of flat arrays over the matches: `b` (image), `a` (anchor in the
    level), `gj`, `gi` (row, column), `tbox` (K, 4) = (x − gi, y − gj, w, h) in grid units
    (offsets in [−0.5, 1.5]), `cls`, `anch` (K, 2) anchors in grid units.
    """
    rows = [(n, int(c), *box) for n, (boxes, labels) in enumerate(zip(boxes_list, labels_list))
            for box, c in zip(np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
                              np.asarray(labels).reshape(-1))]
    t_all = np.array(rows, dtype=np.float64).reshape(-1, 6)  # (img, cls, x, y, w, h)
    out = {}
    for hid, stride, anchors_px in levels:
        sh, sw = grid_hw(grid_sizes[hid])
        anchors = np.asarray(anchors_px, dtype=np.float64) / stride
        na = len(anchors)
        t = np.repeat(t_all[None], na, axis=0)              # (A, k, 6)
        t[..., 2:6] *= np.array([sw, sh, sw, sh])
        a = np.repeat(np.arange(na)[:, None], len(t_all), axis=1)
        r = t[..., 4:6] / anchors[:, None]
        keep = np.maximum(r, 1 / r).max(axis=2) < anchor_t
        t, a = t[keep], a[keep]
        gxy = t[:, 2:4]
        gxi = np.array([sw, sh]) - gxy
        left, up = ((gxy % 1 < 0.5) & (gxy > 1)).T
        right, down = ((gxi % 1 < 0.5) & (gxi > 1)).T
        sel = np.stack([np.ones_like(left), left, up, right, down])   # (5, K)
        t = np.repeat(t[None], 5, axis=0)[sel]
        a = np.repeat(a[None], 5, axis=0)[sel]
        offsets = np.repeat(V5_OFFSETS[:, None], sel.shape[1], axis=1)[sel]
        gxy = t[:, 2:4]
        gij = np.floor(gxy - offsets).astype(np.int64)
        gi = np.clip(gij[:, 0], 0, sw - 1)
        gj = np.clip(gij[:, 1], 0, sh - 1)
        # The reference clamps gij in place before taking the offsets (x = S_w: offset 1).
        tbox = np.concatenate([gxy - np.stack([gi, gj], axis=1), t[:, 4:6]], axis=1)
        out[hid] = {"b": t[:, 0].astype(np.int64), "a": a.astype(np.int64), "gj": gj, "gi": gi,
                    "tbox": tbox,
                    "cls": t[:, 1].astype(np.int64), "anch": anchors[a]}
    return out
