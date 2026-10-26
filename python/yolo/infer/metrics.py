"""mAP PASCAL VOC (§8.3), à l'identique du devkit officiel (`VOCevaldet.m`).

- Par classe : détections triées par score décroissant ; chacune est appariée à la vérité
  d'IoU maximale de son image. Vrai positif si IoU ≥ seuil (0,5) et vérité non encore
  appariée ; une vérité déjà appariée donne un faux positif ; une vérité `difficult` ne
  compte ni en vrai ni en faux positif, ni dans le nombre de vérités.
- Coordonnées en pixels VOC (1-indexés, bornes incluses) : largeur `x2 − x1 + 1`.
- AP 11 points (VOC2007) : moyenne de la précision interpolée en r ∈ {0, 0,1, …, 1}.

Person safety metrics (T27.0, T27.7), on the same matching (`match_class`, so `difficult`
and ignore regions count as in the mAP):

- miss rate against FPPI (false positives per image), from the detections sorted by score
  (`mr_fppi`); recall at a fixed FPPI: the operating point is the lowest threshold whose
  FPPI stays at or below the target (`recall_at_fppi`);
- log-average miss rate over FPPI ∈ [10⁻², 10⁰] (Caltech `dbEval`, Dollár 2012: nine points
  10^(−2:0.25:0), geometric mean of the miss rates, `log_average_miss_rate`);
- recall per box-height bin (distance) and per occlusion level (CrowdHuman visible box
  against full box, `visible` of the samples) at one threshold (`recall_by_bin`);
- unflagged-miss rate with a Clopper-Pearson interval (`unflagged_miss_rate`, T27.7).
"""

import math
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


def in_regions(bb, regions):
    """Le centre de la détection `bb` (4,) tombe-t-il dans une des `regions` (k, 4) ?"""
    if regions is None or len(regions) == 0:
        return False
    r = np.asarray(regions, dtype=np.float64).reshape(-1, 4)
    cx, cy = (bb[0] + bb[2]) / 2, (bb[1] + bb[3]) / 2
    return bool(((r[:, 0] <= cx) & (cx <= r[:, 2]) & (r[:, 1] <= cy) & (cy <= r[:, 3])).any())


def match_class(image_ids, scores, boxes, gts, iou_thr=0.5, ignore=None):
    """Greedy matching of `eval_class` (VOC rule), kept per detection and per ground truth.

    Returns `(order, tp, fp, gt_score)`: `order` sorts the detections by decreasing score,
    `tp` / `fp` (n,) are 0/1 in that order (both 0: detection on a `difficult` ground truth
    or in an ignore region), `gt_score` is {image_id: (k,)} the score of the detection that
    took each ground truth (−inf if none, NaN if `difficult`). Matching greedily by score
    means the matches above a threshold are those of the detections above it alone, so
    `gt_score >= t` is the recall test at any threshold `t`.
    """
    ignore = ignore or {}
    gt_score = {k: np.where(np.asarray(d, bool), np.nan, -np.inf) for k, (_, d) in gts.items()}
    used = {k: np.zeros(len(d), dtype=bool) for k, (_, d) in gts.items()}
    scores = np.asarray(scores, dtype=np.float64)
    order = np.argsort(-scores, kind="stable")
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    tp = np.zeros(len(order))
    fp = np.zeros(len(order))
    for d, k in enumerate(order):
        gt, difficult = gts.get(image_ids[k], (np.zeros((0, 4)), np.zeros(0, bool)))
        gt = np.asarray(gt, dtype=np.float64).reshape(-1, 4)
        if len(gt) == 0:
            fp[d] = not in_regions(boxes[k], ignore.get(image_ids[k]))
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
                gt_score[image_ids[k]][j] = scores[k]
        else:
            fp[d] = not in_regions(boxes[k], ignore.get(image_ids[k]))
    return order, tp, fp, gt_score


def eval_class(image_ids, scores, boxes, gts, iou_thr=0.5, use_07=True, ignore=None):
    """Une classe. `image_ids` (n,), `scores` (n,), `boxes` (n, 4) coins pixels VOC ;
    `gts` : {image_id: (coins (k, 4), difficult (k,))}. Rend `rec`, `prec`, `ap`.

    `ignore` : {image_id: coins (k, 4)} des régions ignorées (VisDrone, catégorie 0) ; une
    détection non appariée dont le centre y tombe n'est ni TP ni FP (T15.10).
    """
    npos = sum(int((~np.asarray(d, bool)).sum()) for _, d in gts.values())
    _, tp, fp, _ = match_class(image_ids, scores, boxes, gts, iou_thr, ignore)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    rec = tp / max(npos, 1)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    return rec, prec, voc_ap(rec, prec, use_07)


def class_gts(annotations, c):
    """{id: (coins, difficult)} de la classe `c` pour toutes les images du split."""
    return {a["id"]: (a["xyxy"][a["labels"] == c], a["difficult"][a["labels"] == c])
            for a in annotations}


def ignore_regions(annotations):
    """{id: coins (k, 4)} des régions ignorées (`ignore_xyxy` des échantillons VisDrone)."""
    return {a["id"]: a["ignore_xyxy"] for a in annotations
            if len(a.get("ignore_xyxy", ()))}


def evaluate(detections, annotations, num_classes, iou_thr=0.5, use_07=True,
             with_recall=False, use_ignore=True):
    """`detections` : {c: (image_ids, scores, coins)} ; `annotations` : sortie de
    `yolo.data.voc.load_split` (ou `yolo.data.datasets.load`). Rend (AP par classe (C,), mAP),
    plus le rappel final par classe (C,) avec `with_recall` (T11.6). `use_ignore` : les
    régions ignorées des échantillons (VisDrone) s'appliquent (`eval_class`).
    """
    empty = ([], np.zeros(0), np.zeros((0, 4)))
    ignore = ignore_regions(annotations) if use_ignore else None
    res = [eval_class(*detections.get(c, empty), class_gts(annotations, c), iou_thr, use_07,
                      ignore)
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


# ------------------------------------------------------------------ person safety (M27)
FPPI_TARGETS = (0.1, 0.01)
LAMR_RANGE = (1e-2, 1.0)
# Caltech bins (Dollár 2012): near ≥ 80 px, medium 30-80 px, far < 30 px (full box height
# in pixels of the original image).
HEIGHT_EDGES = (0.0, 30.0, 80.0, math.inf)
# Occluded fraction of the full box: none, partial (Caltech ≤ 35 %), heavy (≤ 80 %), beyond.
OCCLUSION_EDGES = (0.0, 0.1, 0.35, 0.8, 1.0 + 1e-9)


def mr_fppi(scores, tp, fp, npos, n_images):
    """Curve of the detections sorted by decreasing `scores` (order of `match_class`, `tp`
    and `fp` in that order): one point per distinct score. Returns {thr, fppi, recall, mr}
    (arrays, threshold decreasing, FPPI increasing). A point at threshold `inf` (nothing
    kept: FPPI 0, recall 0) comes first. `n_images`: the images FPPI is counted on; pass
    `fp` restricted to them (`fp * on_empty`) to count on the empty frames alone.
    """
    s = np.sort(np.asarray(scores, dtype=np.float64))[::-1]
    ctp, cfp = np.cumsum(tp), np.cumsum(fp)
    last = np.r_[s[1:] != s[:-1], True] if len(s) else np.zeros(0, bool)  # end of each tie
    thr = np.r_[np.inf, s[last]]
    rec = np.r_[0.0, ctp[last]] / max(npos, 1)
    fppi = np.r_[0.0, cfp[last]] / max(n_images, 1)
    return {"thr": thr, "fppi": fppi, "recall": rec, "mr": 1.0 - rec}


def recall_at_fppi(curve, target):
    """(recall, threshold) of the lowest threshold whose FPPI ≤ `target`."""
    j = int(np.nonzero(curve["fppi"] <= target)[0][-1])
    return float(curve["recall"][j]), float(curve["thr"][j])


def log_average_miss_rate(curve, lo=LAMR_RANGE[0], hi=LAMR_RANGE[1], n=9):
    """Caltech `dbEval`: miss rate at `n` FPPI values log-spaced over [lo, hi], each the last
    point with FPPI ≤ the reference (miss rate 1 if none), geometric mean with a 10⁻¹⁰
    floor."""
    refs = np.logspace(math.log10(lo), math.log10(hi), n)
    miss = []
    for r in refs:
        j = np.nonzero(curve["fppi"] <= r * (1 + 1e-12))[0]  # 1/100 hits 10⁻² exactly
        miss.append(curve["mr"][j[-1]] if len(j) else 1.0)
    return float(np.exp(np.mean(np.log(np.maximum(1e-10, miss)))))


def recall_by_bin(gt_scores, values, edges, thr):
    """Per bin [edges[i], edges[i+1]) of `values` (one per ground truth, NaN = no value):
    (found, total) at threshold `thr`; `gt_scores` from `match_class` (NaN = difficult,
    left out)."""
    s, v = np.asarray(gt_scores, np.float64), np.asarray(values, np.float64)
    keep = ~np.isnan(s) & ~np.isnan(v)
    s, v = s[keep], v[keep]
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (v >= a) & (v < b)
        out.append((int((s[m] >= thr).sum()), int(m.sum())))
    return out


def gt_heights(sample):
    """Full-box heights in pixels (VOC convention, y2 − y1 + 1)."""
    xyxy = np.asarray(sample["xyxy"], np.float64).reshape(-1, 4)
    return xyxy[:, 3] - xyxy[:, 1] + 1.0


def gt_occlusion(sample):
    """Occluded fraction 1 − visible of each ground truth; NaN without `visible`."""
    if "visible" not in sample:
        return np.full(len(sample["labels"]), np.nan)
    return 1.0 - np.asarray(sample["visible"], np.float64)


def person_metrics(dets, samples, iou_thr=0.5, fppi_targets=FPPI_TARGETS, use_ignore=True,
                   empty_only=False, height_edges=HEIGHT_EDGES,
                   occlusion_edges=OCCLUSION_EDGES):
    """Safety metrics of one class. `dets` = (image_ids, scores, VOC corners) of that class,
    `samples` with `labels` already reduced to it (class 0). `empty_only`: FPPI counted on
    the images without any ground truth (PREWID empty frames). Returns a dict: `curve`,
    `npos`, `n_images`, `lamr`, and per FPPI target the recall, threshold, and the
    per-height / per-occlusion (found, total) at that threshold."""
    gts = class_gts(samples, 0)
    ignore = ignore_regions(samples) if use_ignore else None
    ids, scores, boxes = dets
    order, tp, fp, gt_score = match_class(ids, scores, boxes, gts, iou_thr, ignore)
    npos = sum(int((~np.asarray(d, bool)).sum()) for _, d in gts.values())
    n_images = len(samples)
    if empty_only:
        empty = {s["id"] for s in samples if len(s["labels"]) == 0}
        ids_sorted = [ids[k] for k in order]
        fp = fp * np.array([i in empty for i in ids_sorted], dtype=np.float64)
        n_images = len(empty)
    curve = mr_fppi(np.asarray(scores, np.float64)[order] if len(order) else [], tp, fp,
                    npos, n_images)
    per_gt = np.concatenate([gt_score[s["id"]] for s in samples]) if samples else np.zeros(0)
    heights = np.concatenate([gt_heights(s)[s["labels"] == 0] for s in samples]) \
        if samples else np.zeros(0)
    occl = np.concatenate([gt_occlusion(s)[s["labels"] == 0] for s in samples]) \
        if samples else np.zeros(0)
    out = {"curve": curve, "npos": npos, "n_images": n_images,
           "lamr": log_average_miss_rate(curve), "at": {}, "gt_score": per_gt,
           "height": heights, "occlusion": occl}
    for f in fppi_targets:
        r, t = recall_at_fppi(curve, f)
        out["at"][f] = {"recall": r, "thr": t,
                        "height": recall_by_bin(per_gt, heights, height_edges, t),
                        "occlusion": recall_by_bin(per_gt, occl, occlusion_edges, t)}
    return out


def point_at_threshold(curve, thr):
    """(recall, FPPI) of the detections with score ≥ `thr`."""
    j = int(np.nonzero(curve["thr"] >= thr)[0][-1])
    return float(curve["recall"][j]), float(curve["fppi"][j])


def threshold_for_recall(curve, target):
    """Highest threshold whose recall reaches `target` (None if never reached)."""
    j = np.nonzero(curve["recall"] >= target)[0]
    return float(curve["thr"][j[0]]) if len(j) else None


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided (1 − alpha) interval of a binomial rate k / n."""
    from scipy.stats import beta

    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


def unflagged_miss_rate(missed, flagged, alpha=0.05):
    """T27.7: share of the dangerous items (persons, or events of T27.4) missed by the
    detector **and** not flagged by the monitor before the deadline, over all of them.
    `missed`, `flagged`: booleans per item (`flagged` all False: detector alone). Returns
    {k, n, rate, ci}."""
    missed, flagged = np.asarray(missed, bool), np.asarray(flagged, bool)
    k, n = int((missed & ~flagged).sum()), int(missed.size)
    return {"k": k, "n": n, "rate": k / n if n else float("nan"),
            "ci": clopper_pearson(k, n, alpha)}
