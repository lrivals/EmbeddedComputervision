"""Inférence par tuiles (VisDrone, M15) : objets vus à une résolution proche de l'origine.

L'image d'origine est découpée en tuiles `tile` × `tile` px qui se recouvrent de `overlap`
(fraction du côté) ; chaque tuile passe dans le réseau à son entrée (416), comme les
découpes de l'entraînement (`train.py --crop`). Les détections reviennent dans le repère de
l'image, avec en option celles de l'image entière (grands objets coupés par les tuiles),
puis une NMS par classe sur l'ensemble. Coût : une passe par tuile (+ 1).
"""

import numpy as np

from yolo.infer.nms import IOU_THR, nms


def _starts(n, tile, step):
    if n <= tile:
        return [0]
    s = list(range(0, n - tile, step)) + [n - tile]
    return sorted(set(s))


def tile_grid(width, height, tile, overlap=0.2):
    """[(x0, y0, w, h)] : tuiles couvrant l'image, la dernière de chaque rangée calée au bord."""
    step = max(1, int(round(tile * (1 - overlap))))
    tw, th = min(tile, width), min(tile, height)
    return [(x, y, tw, th) for y in _starts(height, tile, step)
            for x in _starts(width, tile, step)]


def tile_to_image(boxes, tile_box, width, height):
    """Boîtes `(cx, cy, w, h)` normalisées dans la tuile → normalisées dans l'image."""
    x0, y0, tw, th = tile_box
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, 0] = (b[:, 0] * tw + x0) / width
    b[:, 1] = (b[:, 1] * th + y0) / height
    b[:, 2] *= tw / width
    b[:, 3] *= th / height
    return b


def merge(parts, iou_thr=IOU_THR):
    """`parts` : [(boîtes normalisées image, scores, labels)] → une NMS par classe."""
    if not parts:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    boxes = np.concatenate([np.asarray(b, dtype=np.float64).reshape(-1, 4) for b, _, _ in parts])
    scores = np.concatenate([np.asarray(s, dtype=np.float64).reshape(-1) for _, s, _ in parts])
    labels = np.concatenate([np.asarray(c, dtype=np.int64).reshape(-1) for _, _, c in parts])
    keep = []
    for c in np.unique(labels):
        idx = np.flatnonzero(labels == c)
        keep.append(idx[nms(boxes[idx], scores[idx], iou_thr)])
    keep = np.concatenate(keep) if keep else np.zeros(0, dtype=np.int64)
    keep = keep[np.argsort(-scores[keep], kind="stable")]
    return boxes[keep], scores[keep], labels[keep]
