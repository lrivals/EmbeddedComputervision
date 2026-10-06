"""mAP PASCAL VOC (§8.3), à l'identique du devkit officiel (`VOCevaldet.m`).

- Par classe : détections triées par score décroissant ; chacune est appariée à la vérité
  d'IoU maximale de son image. Vrai positif si IoU ≥ seuil (0,5) et vérité non encore
  appariée ; une vérité déjà appariée donne un faux positif ; une vérité `difficult` ne
  compte ni en vrai ni en faux positif, ni dans le nombre de vérités.
- Coordonnées en pixels VOC (1-indexés, bornes incluses) : largeur `x2 − x1 + 1`.
- AP 11 points (VOC2007) : moyenne de la précision interpolée en r ∈ {0, 0,1, …, 1}.
"""

from pathlib import Path

import numpy as np


def voc_ap(rec, prec, use_07=True):
    """AP d'une courbe rappel/précision cumulée."""
    rec, prec = np.asarray(rec, dtype=np.float64), np.asarray(prec, dtype=np.float64)
    if use_07:
        # §8.3 : AP_11 = 1/11 Σ_r max_{r̃ ≥ r} P(r̃)
        ap = 0.0
        for t in np.arange(0.0, 1.1, 0.1):
            p = prec[rec >= t]
            ap += (p.max() if p.size else 0.0) / 11.0
        return ap
    # VOC2010+ : aire sous l'enveloppe monotone de la courbe (hors base).
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    i = np.nonzero(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def _overlaps(bb, gt):
    """IoU (k,) de la détection `bb` (4,) avec les vérités `gt` (k, 4), convention pixel +1."""
    iw = np.minimum(gt[:, 2], bb[2]) - np.maximum(gt[:, 0], bb[0]) + 1.0
    ih = np.minimum(gt[:, 3], bb[3]) - np.maximum(gt[:, 1], bb[1]) + 1.0
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    union = ((bb[2] - bb[0] + 1.0) * (bb[3] - bb[1] + 1.0)
             + (gt[:, 2] - gt[:, 0] + 1.0) * (gt[:, 3] - gt[:, 1] + 1.0) - inter)
    return inter / union


def eval_class(image_ids, scores, boxes, gts, iou_thr=0.5, use_07=True):
    """Une classe. `image_ids` (n,), `scores` (n,), `boxes` (n, 4) coins pixels VOC ;
    `gts` : {image_id: (coins (k, 4), difficult (k,))}. Rend `rec`, `prec`, `ap`.
    """
    npos = sum(int((~np.asarray(d, bool)).sum()) for _, d in gts.values())
    used = {k: np.zeros(len(d), dtype=bool) for k, (_, d) in gts.items()}
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    for d, k in enumerate(order):
        gt, difficult = gts.get(image_ids[k], (np.zeros((0, 4)), np.zeros(0, bool)))
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        if len(gt) == 0:
            fp[d] = 1
            continue
        ov = _overlaps(boxes[k], gt)
        j = int(np.argmax(ov))
        if ov[j] >= iou_thr:  # §8.3 : IoU qui « atteint » le seuil (VOCevaldet : >=)
            if difficult[j]:
                continue  # ni TP ni FP
            if used[image_ids[k]][j]:
                fp[d] = 1  # vérité déjà appariée
            else:
                tp[d] = 1
                used[image_ids[k]][j] = True
        else:
            fp[d] = 1
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec, prec, voc_ap(rec, prec, use_07)


def class_gts(annotations, c):
    """{id: (coins, difficult)} de la classe `c` pour toutes les images du split."""
    return {a["id"]: (a["xyxy"][a["labels"] == c], a["difficult"][a["labels"] == c])
            for a in annotations}


def evaluate(detections, annotations, num_classes, iou_thr=0.5, use_07=True,
             with_recall=False):
    """`detections` : {c: (image_ids, scores, coins)} ; `annotations` : sortie de
    `yolo.data.voc.load_split` (ou `yolo.data.datasets.load`). Rend (AP par classe (C,), mAP),
    plus le rappel final par classe (C,) avec `with_recall` (T11.6).
    """
    empty = ([], np.zeros(0), np.zeros((0, 4)))
    res = [eval_class(*detections.get(c, empty), class_gts(annotations, c), iou_thr, use_07)
           for c in range(num_classes)]
    aps = np.array([r[2] for r in res])
    if with_recall:
        rec = np.array([r[0][-1] if len(r[0]) else 0.0 for r in res])
        return aps, float(aps.mean()), rec
    return aps, float(aps.mean())


def to_voc_pixels(boxes, width, height):
    """`(cx, cy, w, h)` normalisés dans l'image d'origine → coins pixels VOC, bornés à
    [1, W] × [1, H] (inverse de `yolo.data.voc.xyxy_to_cxcywh`, convention Darknet).
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1 = (b[:, 0] - b[:, 2] / 2) * width + 1
    y1 = (b[:, 1] - b[:, 3] / 2) * height + 1
    x2 = (b[:, 0] + b[:, 2] / 2) * width
    y2 = (b[:, 1] + b[:, 3] / 2) * height
    return np.stack([np.clip(x1, 1, width), np.clip(y1, 1, height),
                     np.clip(x2, 1, width), np.clip(y2, 1, height)], axis=1)


def write_detections(detections, classes, out_dir, prefix="comp4_det_test_"):
    """Fichiers du devkit : un par classe, lignes `id score x1 y1 x2 y2`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for c, name in enumerate(classes):
        ids, scores, boxes = detections.get(c, ([], [], np.zeros((0, 4))))
        with (out_dir / f"{prefix}{name}.txt").open("w") as f:
            for i, s, b in zip(ids, scores, boxes):
                f.write(f"{i} {s:.6f} {b[0]:.6f} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f}\n")


def read_detections(classes, in_dir, prefix="comp4_det_test_"):
    """Inverse de `write_detections` ; une classe sans fichier n'a aucune détection."""
    out = {}
    for c, name in enumerate(classes):
        path = Path(in_dir) / f"{prefix}{name}.txt"
        rows = [line.split() for line in path.read_text().splitlines()] if path.exists() else []
        out[c] = ([r[0] for r in rows], np.array([float(r[1]) for r in rows]),
                  np.array([[float(v) for v in r[2:6]] for r in rows]).reshape(-1, 4))
    return out
