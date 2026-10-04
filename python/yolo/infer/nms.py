"""Seuil et suppression des non-maxima (§8.2).

On filtre les scores sous `conf_thr`, puis, classe par classe, on garde la boîte de meilleur
score et on élimine celles dont l'IoU avec elle dépasse `iou_thr` ; on recommence jusqu'à
épuisement. Seuils par défaut `conf = 0,25`, `iou = 0,45` (hors base).
"""

import numpy as np

from yolo.infer.boxes import iou
from yolo.infer.decode import decode

CONF_THR = 0.25
IOU_THR = 0.45


def nms(boxes, scores, iou_thr=IOU_THR):
    """Indices gardés, par score décroissant ; `boxes` (n, 4) `(cx, cy, w, h)`, `scores` (n,).

    À score égal, l'ordre d'origine départage (tri stable).
    """
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if scores.size == 0:
        return np.zeros(0, dtype=np.int64)
    order = np.argsort(-scores, kind="stable")
    ov = iou(np.asarray(boxes)[order], np.asarray(boxes)[order])
    alive = np.ones(len(order), dtype=bool)
    keep = []
    # §8.2 : garder la meilleure, éliminer les IoU > seuil, recommencer
    for k in range(len(order)):
        if not alive[k]:
            continue
        keep.append(order[k])
        alive[k + 1:] &= ov[k, k + 1:] <= iou_thr
    return np.array(keep, dtype=np.int64)


def filter_and_nms(boxes, scores, conf_thr=CONF_THR, iou_thr=IOU_THR):
    """Une image : `boxes` (K, 4), `scores` (K, C) → détections après seuil et NMS par classe.

    Rend `boxes` (m, 4), `scores` (m,), `labels` (m,) triés par score décroissant.
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float64)
    out_b, out_s, out_l = [], [], []
    for c in range(scores.shape[1]):
        cand = np.nonzero(scores[:, c] > conf_thr)[0]  # §8.2 : seuil puis NMS par classe
        keep = cand[nms(boxes[cand], scores[cand, c], iou_thr)]
        out_b.append(boxes[keep])
        out_s.append(scores[keep, c])
        out_l.append(np.full(len(keep), c, dtype=np.int64))
    b, s, lab = np.concatenate(out_b), np.concatenate(out_s), np.concatenate(out_l)
    order = np.argsort(-s, kind="stable")
    return b[order], s[order], lab[order]


def postprocess(outputs, net, conf_thr=CONF_THR, iou_thr=IOU_THR):
    """Sorties brutes → par image `(boxes, scores, labels)`, boîtes dans le repère letterbox."""
    boxes, _, scores = decode(outputs, net)
    return [filter_and_nms(b, s, conf_thr, iou_thr) for b, s in zip(boxes, scores)]

