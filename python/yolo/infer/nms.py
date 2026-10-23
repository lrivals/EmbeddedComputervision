"""Seuil et suppression des non-maxima (§8.2).

On filtre les scores sous `conf_thr`, puis, classe par classe, on garde la boîte de meilleur
score et on élimine celles dont l'IoU avec elle dépasse `iou_thr` ; on recommence jusqu'à
épuisement. Seuils par défaut `conf = 0,25`, `iou = 0,45` (hors base).

YOLOv5 evaluation settings (T33.4, those of the published mAP): conf 0.001, IoU 0.65, at
most 300 detections per image. Each (box, class) pair above the threshold is kept
(multi-label) and the NMS is class-aware, which the reference gets by offsetting the boxes
by `class·4096`: the per-class loop below is the same.
"""

import numpy as np

from yolo.infer.boxes import iou
from yolo.infer.decode import decode, decode_int

CONF_THR = 0.25
IOU_THR = 0.45
V5_CONF_THR = 0.001
V5_IOU_THR = 0.65
V5_MAX_DET = 300


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


def filter_and_nms(boxes, scores, conf_thr=CONF_THR, iou_thr=IOU_THR, max_det=None):
    """Une image : `boxes` (K, 4), `scores` (K, C) → détections après seuil et NMS par classe.

    Rend `boxes` (m, 4), `scores` (m,), `labels` (m,) triés par score décroissant ; au plus
    `max_det` détections (les meilleures) si donné.
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
    order = np.argsort(-s, kind="stable")[:max_det]
    return b[order], s[order], lab[order]


def postprocess(outputs, net, conf_thr=CONF_THR, iou_thr=IOU_THR, max_det=None):
    """Sorties brutes → par image `(boxes, scores, labels)`, boîtes dans le repère letterbox."""
    boxes, _, scores = decode(outputs, net)
    return [filter_and_nms(b, s, conf_thr, iou_thr, max_det) for b, s in zip(boxes, scores)]


def postprocess_int(outputs, net, luts, conf_thr=CONF_THR, iou_thr=IOU_THR):
    """Têtes int8 → par image `(boxes, scores, labels)` (§9.4 puis §8.2) ; `luts` :
    {id de tête: HeadLuts}. Le seuil sur t_o reprend `conf_thr` (score ≤ objectness).
    """
    return [filter_and_nms(b, s, conf_thr, iou_thr)
            for b, s in decode_int(outputs, net, luts, conf_thr)]
