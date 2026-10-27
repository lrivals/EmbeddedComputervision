"""Conformal person threshold (T29.2): a score threshold, and a box margin, calibrated for a
target miss rate, with a finite-sample guarantee under exchangeability.

The risk controlled is the false negative (Andéol 2023 covers the boxes of the detected
objects, not the misses). A person's score is `gt_score` of `metrics.match_class`: the
score of the detection the greedy VOC matching gave it (−inf if none), so "missed at `t`"
is `score < t`, the same miss as the recall of M27.

- **Per person** (`person_threshold`): split conformal. With `n` calibration persons and
  `k = ⌊α (n + 1)⌋`, the threshold is the k-th smallest calibration score; a new person
  exchangeable with them is missed with probability ≤ k / (n + 1) ≤ α. Conditional on one
  calibration set, its miss rate follows Beta(k, n + 1 − k) (Vovk 2012; Angelopoulos and
  Bates 2021, §3), which gives the spread expected over random splits (`miss_beta`).
- **Per image** (`image_threshold`): conformal risk control (Angelopoulos 2022). The loss
  of an image is the share of its persons missed, monotone in `t` and bounded by 1; the
  largest `t` with (Σ L_i(t) + 1) / (n + 1) ≤ α bounds the expected loss of a new image by
  α. It only needs the images to be exchangeable: the persons of one frame are not.
- **Box margin** (`box_margins`, `margin_quantile`): the relative enlargement of the matched
  detection that covers the person's box; its conformal (1 − α) quantile covers a new
  detected person with probability ≥ 1 − α.

Exchangeability is the assumption: under the degradations of T27.3 the calibration no longer
holds, which `person_safety.py conformal` measures.
"""

import math

import numpy as np


def person_threshold(scores, alpha):
    """Split-conformal threshold of the per-person scores (−inf = never detected): the
    k-th smallest, k = ⌊α (n + 1)⌋. None when k = 0 (too few persons for this α) or when
    the k-th smallest is −inf (even keeping every detection misses more than α)."""
    s = np.sort(np.asarray(scores, np.float64))
    k = int(math.floor(alpha * (len(s) + 1)))
    if k < 1 or not np.isfinite(s[k - 1]):
        return None
    return float(s[k - 1])


def miss_beta(n, alpha, q=(0.05, 0.95)):
    """(k / (n + 1), quantiles q of Beta(k, n + 1 − k)): the bound and the spread of the
    miss rate of a split-conformal threshold over calibration sets of n persons."""
    from scipy.stats import beta

    k = int(math.floor(alpha * (n + 1)))
    if k < 1:
        return float("nan"), (float("nan"),) * len(q)
    return k / (n + 1), tuple(float(beta.ppf(p, k, n + 1 - k)) for p in q)


def image_losses(per_image, t):
    """Share of the persons of each image missed at `t` (images without persons: 0)."""
    return np.array([float((s < t).mean()) if len(s) else 0.0 for s in per_image])


def image_threshold(per_image, alpha):
    """Conformal risk control over images: the largest candidate `t` (a calibration score)
    with (Σ L_i(t) + 1) / (n + 1) ≤ α. `per_image`: list of score arrays, one per image.
    None when no candidate meets it."""
    n = len(per_image)
    flat = np.concatenate([np.asarray(s, np.float64) for s in per_image]) if n else np.zeros(0)
    cand = np.unique(flat[np.isfinite(flat)])[::-1]
    if not len(cand):
        return None
    allp = np.concatenate([np.asarray(s, np.float64) for s in per_image if len(s)])
    owner = np.concatenate([np.full(len(s), i) for i, s in enumerate(per_image) if len(s)])
    size = np.bincount(owner, minlength=n).astype(np.float64)
    # Loss sum at each candidate, from high to low: a person stops being missed when t
    # drops to its score, so the risk only falls along `cand`.
    order = np.argsort(-allp, kind="stable")
    w = 1.0 / size[owner[order]]
    found = np.cumsum(w)
    n_img = float((size > 0).sum())
    # candidates descending; persons with score ≥ cand[j] are found
    idx = np.searchsorted(-allp[order], -cand, side="right")
    risk = (n_img - np.where(idx > 0, found[np.maximum(idx - 1, 0)], 0.0) + 1) / (n + 1)
    ok = np.nonzero(risk <= alpha)[0]
    return float(cand[ok[0]]) if len(ok) else None


def box_margins(det, gt):
    """Relative margin each detection (n, 4) needs to contain its person box (n, 4): the
    largest of the four side gaps over the detection's width or height, ≥ 0 (VOC corners)."""
    det, gt = np.asarray(det, np.float64).reshape(-1, 4), np.asarray(gt, np.float64).reshape(-1, 4)
    w = det[:, 2] - det[:, 0] + 1.0
    h = det[:, 3] - det[:, 1] + 1.0
    gaps = np.stack([(det[:, 0] - gt[:, 0]) / w, (gt[:, 2] - det[:, 2]) / w,
                     (det[:, 1] - gt[:, 1]) / h, (gt[:, 3] - det[:, 3]) / h], axis=1)
    return np.maximum(gaps.max(axis=1), 0.0) if len(det) else np.zeros(0)


def expand(det, margin):
    """Detections (n, 4) enlarged by `margin` of their width / height on each side."""
    det = np.asarray(det, np.float64).reshape(-1, 4)
    w = det[:, 2] - det[:, 0] + 1.0
    h = det[:, 3] - det[:, 1] + 1.0
    return det + np.stack([-w, -h, w, h], axis=1) * margin


def margin_quantile(margins, alpha):
    """Conformal (1 − α) quantile: the ⌈(1 − α)(n + 1)⌉-th smallest margin (inf when that
    rank exceeds n)."""
    m = np.sort(np.asarray(margins, np.float64))
    r = int(math.ceil((1 - alpha) * (len(m) + 1)))
    return float(m[r - 1]) if 1 <= r <= len(m) else float("inf")


def matched_boxes(gt_score, gt_boxes, det_scores, det_boxes):
    """For the persons of one image found by the greedy matching (`gt_score` finite), the
    detection that took each: the one of the same score with the largest IoU. Returns
    (index of the persons, their detection boxes (m, 4))."""
    from .metrics import _overlaps

    gs = np.asarray(gt_score, np.float64)
    gb = np.asarray(gt_boxes, np.float64).reshape(-1, 4)
    ds = np.asarray(det_scores, np.float64)
    db = np.asarray(det_boxes, np.float64).reshape(-1, 4)
    idx, out = [], []
    for j in np.nonzero(np.isfinite(gs))[0]:
        cand = np.nonzero(ds == gs[j])[0]
        if not len(cand):
            continue
        ov = np.array([_overlaps(db[c], gb[j:j + 1])[0] for c in cand])
        idx.append(j)
        out.append(db[cand[int(np.argmax(ov))]])
    return np.array(idx, int), np.array(out).reshape(-1, 4)
