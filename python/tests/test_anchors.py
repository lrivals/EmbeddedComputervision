"""T2.2 : ancres par k-means (§5.2)."""

import numpy as np

from yolo.data.anchors import kmeans_anchors, mean_best_iou


def _clusters(rng, centers, n=200, noise=0.05):
    pts = [c * np.exp(rng.normal(0, noise, (n, 2))) for c in centers]
    return np.concatenate(pts)


def test_recovers_synthetic_clusters():
    rng = np.random.default_rng(0)
    centers = np.array([[10, 14], [40, 30], [80, 160], [300, 250]], dtype=float)
    wh = _clusters(rng, centers)
    found = kmeans_anchors(wh, 4, rng=1)
    np.testing.assert_allclose(found, centers, rtol=0.05)
    assert mean_best_iou(wh, found) > 0.9


def test_sorted_and_deterministic():
    wh = np.random.default_rng(2).uniform(5, 400, (500, 2))
    a = kmeans_anchors(wh, 6, rng=3)
    np.testing.assert_array_equal(a, kmeans_anchors(wh, 6, rng=3))
    area = a[:, 0] * a[:, 1]
    assert np.all(np.diff(area) >= 0)


def test_better_than_random_anchors():
    rng = np.random.default_rng(4)
    wh = _clusters(rng, np.array([[20, 60], [60, 20], [150, 150]]), noise=0.2)
    a = kmeans_anchors(wh, 3, rng=5)
    rand = rng.uniform(5, 200, (3, 2))
    assert mean_best_iou(wh, a) > mean_best_iou(wh, rand)
