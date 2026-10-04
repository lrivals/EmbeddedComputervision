"""T3.2 : seuil et NMS par classe (§8.2), comparés à une implémentation naïve O(n²)."""

import numpy as np

from yolo.infer.nms import filter_and_nms, nms


def _iou1(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _nms_naive(boxes, scores, iou_thr):
    # Pseudo-code du §8.2, littéralement.
    order = sorted(range(len(scores)), key=lambda k: -scores[k])
    keep = []
    while order:
        best = order.pop(0)
        keep.append(best)
        order = [k for k in order if _iou1(boxes[best], boxes[k]) <= iou_thr]
    return keep


def _random_set(rng):
    n = int(rng.integers(0, 60))
    centers = rng.uniform(0.2, 0.8, (n, 2))
    sizes = rng.uniform(0.05, 0.4, (n, 2))
    boxes = np.concatenate([centers, sizes], axis=1)
    # Quelques quasi-doublons pour forcer des suppressions.
    dup = rng.random(n) < 0.3
    if n:
        src = rng.integers(0, n, n)
        boxes[dup] = boxes[src[dup]] + rng.normal(0, 0.01, (dup.sum(), 4))
    boxes[:, 2:] = np.abs(boxes[:, 2:]) + 1e-3
    return boxes, rng.random(n)


def test_nms_matches_naive():
    rng = np.random.default_rng(0)
    for _ in range(1000):
        boxes, scores = _random_set(rng)
        thr = float(rng.uniform(0.2, 0.7))
        assert nms(boxes, scores, thr).tolist() == _nms_naive(boxes, scores, thr)


def test_filter_and_nms_matches_naive():
    rng = np.random.default_rng(1)
    for _ in range(1000):
        boxes, _ = _random_set(rng)
        scores = rng.random((len(boxes), 3)) ** 2
        b, s, lab = filter_and_nms(boxes, scores, conf_thr=0.25, iou_thr=0.45)
        expected = []
        for c in range(3):
            cand = [k for k in range(len(boxes)) if scores[k, c] > 0.25]
            keep = _nms_naive(boxes[cand], scores[cand, c], 0.45)
            expected += [(scores[cand[k], c], c, cand[k]) for k in keep]
        expected.sort(key=lambda t: -t[0])
        assert s.tolist() == [t[0] for t in expected]
        assert lab.tolist() == [t[1] for t in expected]
        np.testing.assert_array_equal(b, boxes[[t[2] for t in expected]].reshape(-1, 4))


def test_per_class():
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2]])
    # Même boîte, classes différentes : aucune suppression.
    scores = np.array([[0.9, 0.0], [0.0, 0.8]])
    _, s, lab = filter_and_nms(boxes, scores)
    assert s.tolist() == [0.9, 0.8] and lab.tolist() == [0, 1]
    # Même classe : la seconde disparaît.
    scores = np.array([[0.9, 0.0], [0.8, 0.0]])
    _, s, lab = filter_and_nms(boxes, scores)
    assert s.tolist() == [0.9] and lab.tolist() == [0]


def test_conf_threshold_and_empty():
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]])
    scores = np.array([[0.25], [0.26]])  # strictement au-dessus du seuil
    _, s, _ = filter_and_nms(boxes, scores, conf_thr=0.25)
    assert s.tolist() == [0.26]
    b, s, lab = filter_and_nms(np.zeros((0, 4)), np.zeros((0, 2)))
    assert b.shape == (0, 4) and s.shape == (0,) and lab.shape == (0,)
    assert nms(np.zeros((0, 4)), np.zeros(0)).shape == (0,)
