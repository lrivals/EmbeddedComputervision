"""Métrique COCO des boîtes (T11.1, §8.3), à l'identique de `COCOeval` (pycocotools, bbox).

- Seuils d'IoU 0,50:0,05:0,95 ; 100 détections au plus par image et par classe (les mieux
  notées) ; précision interpolée sur 101 rappels (0:0,01:1) ; AP = moyenne sur les seuils,
  les rappels et les classes, les classes sans vérité étant exclues.
- Aires (petits < 32², moyens < 96², grands) : aire `area` de l'annotation (segmentation
  pour COCO), aire de la boîte pour les détections. Une vérité hors de la plage d'aire est
  ignorée ; une détection non appariée hors de la plage aussi.
- Foules (`iscrowd`) : ignorées, et l'IoU avec une foule vaut intersection / aire de la
  détection ; une foule peut absorber plusieurs détections.
- Coordonnées continues `(x, y, w, h)` sans le +1 de VOC : elles sont déduites des coins
  VOC (`xyxy`) des échantillons et des détections, qui gardent le format de
  `yolo.infer.metrics.evaluate`.

pycocotools ne sert qu'à la vérification (`python/tests/test_coco_eval.py`).
"""

import numpy as np

IOU_THRS = np.linspace(0.5, 0.95, 10)
REC_THRS = np.linspace(0.0, 1.0, 101)
MAX_DETS = (1, 10, 100)
AREAS = {"all": (0.0, 1e10), "small": (0.0, 32.0 ** 2), "medium": (32.0 ** 2, 96.0 ** 2),
         "large": (96.0 ** 2, 1e10)}
STATS = ("AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm",
         "ARl")


def voc_to_xywh(xyxy):
    """Coins VOC (1-indexés, bornes incluses) → `(x, y, w, h)` continus."""
    b = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0] - 1, b[:, 1] - 1, b[:, 2] - b[:, 0] + 1, b[:, 3] - b[:, 1] + 1],
                    axis=1)


def box_iou(dt, gt, crowd):
    """IoU (D, G) de boîtes `(x, y, w, h)` ; colonne de foule : inter / aire détection."""
    if len(dt) == 0 or len(gt) == 0:
        return np.zeros((len(dt), len(gt)))
    dx1, dy1 = dt[:, 0] + dt[:, 2], dt[:, 1] + dt[:, 3]
    gx1, gy1 = gt[:, 0] + gt[:, 2], gt[:, 1] + gt[:, 3]
    iw = np.minimum(dx1[:, None], gx1[None]) - np.maximum(dt[:, None, 0], gt[None, :, 0])
    ih = np.minimum(dy1[:, None], gy1[None]) - np.maximum(dt[:, None, 1], gt[None, :, 1])
    inter = np.where((iw > 0) & (ih > 0), iw * ih, 0.0)
    da = (dt[:, 2] * dt[:, 3])[:, None]
    union = np.where(np.asarray(crowd, bool)[None], da, da + (gt[:, 2] * gt[:, 3])[None] - inter)
    return inter / union


def _eval_image(dt, scores, gt, garea, crowd, area_rng, max_det):
    """Appariement d'une image et d'une classe pour une plage d'aire (`evaluateImg`).

    Rend (scores triés, dtm (T, D) appariées, dtIg (T, D) ignorées, gtIg (G,)), ou None si
    l'image n'a ni vérité ni détection.
    """
    if len(dt) == 0 and len(gt) == 0:
        return None
    ignore = crowd | (garea < area_rng[0]) | (garea > area_rng[1])
    gind = np.argsort(ignore, kind="mergesort")
    gt, crowd, ignore = gt[gind], crowd[gind], ignore[gind]
    dind = np.argsort(-scores, kind="mergesort")[:max_det]
    dt, scores = dt[dind], scores[dind]
    ious = box_iou(dt, gt, crowd)
    T, D, G = len(IOU_THRS), len(dt), len(gt)
    dtm = np.zeros((T, D), bool)
    dtig = np.zeros((T, D), bool)
    gtm = np.zeros((T, G), bool)
    if G:
        for t, thr in enumerate(IOU_THRS):
            for d in range(D):
                best = min(thr, 1 - 1e-10)
                m = -1
                for g in range(G):
                    if gtm[t, g] and not crowd[g]:
                        continue
                    # Vérités ignorées en fin de liste : on s'arrête dès qu'on y arrive avec
                    # déjà une vérité valide appariée.
                    if m > -1 and not ignore[m] and ignore[g]:
                        break
                    if ious[d, g] < best:
                        continue
                    best = ious[d, g]
                    m = g
                if m == -1:
                    continue
                dtig[t, d] = ignore[m]
                dtm[t, d] = True
                gtm[t, m] = True
    darea = dt[:, 2] * dt[:, 3]
    out = (darea < area_rng[0]) | (darea > area_rng[1])
    dtig |= ~dtm & out[None]
    return scores, dtm, dtig, ignore


def _accumulate(evals, max_det):
    """Précision (T, R) et rappel (T,) d'une classe, d'une aire et d'un maxDets ; None si
    aucune vérité non ignorée.
    """
    evals = [e for e in evals if e is not None]
    if not evals:
        return None
    scores = np.concatenate([e[0][:max_det] for e in evals])
    order = np.argsort(-scores, kind="mergesort")
    dtm = np.concatenate([e[1][:, :max_det] for e in evals], axis=1)[:, order]
    dtig = np.concatenate([e[2][:, :max_det] for e in evals], axis=1)[:, order]
    npig = int(sum((~e[3]).sum() for e in evals))
    if npig == 0:
        return None
    tps = np.cumsum(dtm & ~dtig, axis=1, dtype=np.float64)
    fps = np.cumsum(~dtm & ~dtig, axis=1, dtype=np.float64)
    T, R = len(IOU_THRS), len(REC_THRS)
    precision = np.zeros((T, R))
    recall = np.zeros(T)
    nd = tps.shape[1]
    for t in range(T):
        tp, fp = tps[t], fps[t]
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        recall[t] = rc[-1] if nd else 0.0
        pr = np.maximum.accumulate(pr[::-1])[::-1]  # enveloppe décroissante
        inds = np.searchsorted(rc, REC_THRS, side="left")
        valid = inds < nd
        precision[t, valid] = pr[inds[valid]]
    return precision, recall


def _per_image(detections, annotations, c):
    """{id: (boxes, scores)} des détections et {id: (boxes, aires, foules)} des vérités."""
    ids, scores, boxes = detections.get(c, ([], np.zeros(0), np.zeros((0, 4))))
    boxes = voc_to_xywh(boxes)
    scores = np.asarray(scores, dtype=np.float64)
    dets = {}
    for k, i in enumerate(ids):
        dets.setdefault(i, []).append(k)
    dets = {i: (boxes[k], scores[k]) for i, k in dets.items()}
    gts = {}
    for a in annotations:
        sel = a["labels"] == c
        if sel.any():
            xywh = voc_to_xywh(a["xyxy"][sel])
            area = a["area"][sel] if "area" in a else xywh[:, 2] * xywh[:, 3]
            crowd = a["crowd"][sel] if "crowd" in a else a["difficult"][sel]
            gts[a["id"]] = (xywh, np.asarray(area, np.float64), np.asarray(crowd, bool))
    return dets, gts


def evaluate(detections, annotations, num_classes):
    """`detections` : {c: (image_ids, scores, coins VOC)} ; `annotations` : échantillons
    (`yolo.data.datasets`). Rend {nom de `STATS`: valeur, "ap_class": AP par classe (C,)}
    (−1 : non défini, comme pycocotools).
    """
    image_ids = {a["id"] for a in annotations}
    empty = (np.zeros((0, 4)), np.zeros(0))
    nogt = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, bool))
    T, R, C = len(IOU_THRS), len(REC_THRS), num_classes
    prec = -np.ones((T, R, C, len(AREAS), len(MAX_DETS)))
    rec = -np.ones((T, C, len(AREAS), len(MAX_DETS)))
    for c in range(C):
        dets, gts = _per_image(detections, annotations, c)
        imgs = sorted(i for i in image_ids if i in dets or i in gts)
        for a, rng in enumerate(AREAS.values()):
            evals = [_eval_image(*dets.get(i, empty), *gts.get(i, nogt), rng, MAX_DETS[-1])
                     for i in imgs]
            for m, md in enumerate(MAX_DETS):
                res = _accumulate(evals, md)
                if res is not None:
                    prec[:, :, c, a, m], rec[:, c, a, m] = res

    def ap(iou=None, area=0, md=2):
        p = prec[:, :, :, area, md]
        if iou is not None:
            p = p[np.where(np.isclose(IOU_THRS, iou))[0]]
        p = p[p > -1]
        return float(p.mean()) if p.size else -1.0

    def ar(area=0, md=2):
        r = rec[:, :, area, md]
        r = r[r > -1]
        return float(r.mean()) if r.size else -1.0

    vals = [ap(), ap(0.5), ap(0.75), ap(area=1), ap(area=2), ap(area=3), ar(md=0), ar(md=1),
            ar(), ar(1), ar(2), ar(3)]
    out = dict(zip(STATS, vals))
    per = prec[:, :, :, 0, 2]
    out["ap_class"] = np.array([per[:, :, c][per[:, :, c] > -1].mean()
                                if (per[:, :, c] > -1).any() else -1.0 for c in range(C)])
    out["ap50_class"] = np.array([per[0, :, c][per[0, :, c] > -1].mean()
                                  if (per[0, :, c] > -1).any() else -1.0 for c in range(C)])
    return out


def to_xywh_pixels(boxes, width, height):
    """`(cx, cy, w, h)` normalisés → `(x, y, w, h)` continus en pixels (résultats COCO)."""
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    return np.stack([(b[:, 0] - b[:, 2] / 2) * width, (b[:, 1] - b[:, 3] / 2) * height,
                     b[:, 2] * width, b[:, 3] * height], axis=1)
