"""Reference for the log-average miss rate of T27.0, written independently of
`yolo.infer.metrics`: brute force over thresholds, Caltech averaging.

For every distinct score t, the detections with score ≥ t are matched again from scratch
(per image, by decreasing score, VOC rule: best-IoU ground truth, a second match on it is a
false positive, a `difficult` one is neither), giving (FPPI, miss rate) at t. The miss rate
at a reference FPPI r is the lowest miss rate among the thresholds with FPPI ≤ r (1 if none),
and the log-average is exp(mean(log(max(1e-10, miss)))) over r ∈ 10^(-2:0.25:0), as in
`dbEval.m` of the Caltech Pedestrian toolbox (Dollár 2012).
"""

import math


def _iou(a, b):
    iw = min(a[2], b[2]) - max(a[0], b[0]) + 1
    ih = min(a[3], b[3]) - max(a[1], b[1]) + 1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    return inter / ((a[2] - a[0] + 1) * (a[3] - a[1] + 1)
                    + (b[2] - b[0] + 1) * (b[3] - b[1] + 1) - inter)


def counts_at(dets, gts, thr, iou_thr=0.5):
    """(TP, FP) of the detections (image, score, box) with score ≥ thr; gts {image: [(box,
    difficult)]}."""
    tp = fp = 0
    for img in {d[0] for d in dets} | set(gts):
        mine = sorted((d for d in dets if d[0] == img and d[1] >= thr), key=lambda d: -d[1])
        g = gts.get(img, [])
        used = [False] * len(g)
        for _, _, box in mine:
            best, j = -1.0, -1
            for k, (gb, _) in enumerate(g):
                o = _iou(box, gb)
                if o > best:
                    best, j = o, k
            if j >= 0 and best >= iou_thr:
                if g[j][1]:
                    continue
                if used[j]:
                    fp += 1
                else:
                    tp += 1
                    used[j] = True
            else:
                fp += 1
    return tp, fp


def lamr(dets, gts, n_images, iou_thr=0.5):
    npos = sum(1 for g in gts.values() for _, d in g if not d)
    points = [(0.0, 1.0)]
    for t in sorted({d[1] for d in dets}):
        tp, fp = counts_at(dets, gts, t, iou_thr)
        points.append((fp / n_images, 1 - tp / npos))
    miss = []
    for i in range(9):
        r = 10 ** (-2 + 0.25 * i)
        ok = [m for f, m in points if f <= r + 1e-12]
        miss.append(min(ok) if ok else 1.0)
    return math.exp(sum(math.log(max(1e-10, m)) for m in miss) / len(miss))
