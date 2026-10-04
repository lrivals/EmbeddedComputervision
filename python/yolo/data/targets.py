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
# une ancre couvre la même part de l'image à 320 qu'à 608.
ANCHOR_REF = 416


def anchors_frac(anchors_px):
    """Ancres (K, 2) en fraction de l'image."""
    return np.asarray(anchors_px, dtype=np.float64).reshape(-1, 2) / ANCHOR_REF


def heads(net):
    """[(id, mask)] des couches de sortie de `net` (format de `yolo.models.specs`)."""
    out = []
    for i, layer in enumerate(net["layers"]):
        if layer["type"] == "yolo":
            out.append((i, list(layer["mask"])))
        elif layer["type"] == "region":
            out.append((i, list(range(layer["num"]))))
    return out


def build_targets(boxes_list, labels_list, anchors_px, head_list, grid_sizes):
    """Cibles de chaque tête pour un lot.

    `boxes_list[n]` : (k, 4) `(cx, cy, w, h)` normalisés de l'image n ; `labels_list[n]` : (k,).
    `head_list` : sortie de `heads(net)` ; `grid_sizes` : {id: S}.

    Rend {id: dict} avec, de forme (N, A, S, S) : `obj` (bool), `x`, `y`, `tw`, `th` (cibles
    §5.1), `scale` (ω = 2 − g_w g_h, §6.2), `cls` (int64, −1 hors *obj*).
    Si deux objets tombent sur la même cellule et la même ancre, le dernier l'emporte
    (comme Darknet, hors base).
    """
    anchors = anchors_frac(anchors_px)
    n_img = len(boxes_list)
    owner = {}  # ancre globale -> (id de tête, position dans le mask)
    targets = {}
    for hid, mask in head_list:
        s = grid_sizes[hid]
        shape = (n_img, len(mask), s, s)
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
        if len(boxes) == 0:
            continue
        # §5.1, 2 : ancre de meilleure IoU de forme parmi toutes les ancres de toutes les
        # têtes ; l'échelle retenue est celle de l'ancre gagnante.
        best = np.argmax(iou_wh(boxes[:, 2:4], anchors), axis=1)
        for (gx, gy, gw, gh), label, k in zip(boxes, labels, best):
            if k not in owner:
                continue
            hid, a = owner[k]
            s = grid_sizes[hid]
            # §5.1, 1 : cellule qui contient le centre (g = 1 bornée à la dernière cellule).
            j = min(int(np.floor(gx * s)), s - 1)
            i = min(int(np.floor(gy * s)), s - 1)
            pw, ph = anchors[k]
            t = targets[hid]
            t["obj"][n, a, i, j] = True
            # §5.1, 3 : x* = g_x S − j, y* = g_y S − i, t_w* = ln(g_w/p_w), t_h* = ln(g_h/p_h)
            t["x"][n, a, i, j] = gx * s - j
            t["y"][n, a, i, j] = gy * s - i
            t["tw"][n, a, i, j] = np.log(gw / pw)
            t["th"][n, a, i, j] = np.log(gh / ph)
            t["scale"][n, a, i, j] = 2.0 - gw * gh  # §6.2 : ω = 2 − g_w g_h
            t["cls"][n, a, i, j] = int(label)
    return targets
