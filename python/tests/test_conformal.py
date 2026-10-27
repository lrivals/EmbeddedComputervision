"""T29.2: conformal person threshold, risk control over images, box margin."""

import numpy as np
import pytest

from yolo.infer import conformal as C


def test_person_threshold_rank():
    # n = 9, α = 0.2: k = ⌊0.2 · 10⌋ = 2, the second smallest score.
    s = np.array([0.9, 0.1, 0.5, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6])
    assert C.person_threshold(s, 0.2) == 0.2
    assert C.person_threshold(s, 0.05) is None  # k = 0: too few persons
    s[[1, 5]] = -np.inf  # two persons never detected: the 2nd smallest is −inf
    assert C.person_threshold(s, 0.2) is None


def test_person_threshold_coverage():
    # Miss rate of a fresh person, averaged over calibration draws: k / (n + 1) ≤ α.
    rng = np.random.default_rng(0)
    alpha, n = 0.1, 199
    miss = []
    for _ in range(2000):
        t = C.person_threshold(rng.random(n), alpha)
        miss.append(t)  # P(U < t) = t for a uniform test score
    bound, (lo, hi) = C.miss_beta(n, alpha)
    assert bound == pytest.approx(20 / 200)
    assert np.mean(miss) == pytest.approx(bound, abs=0.005)
    assert lo < np.quantile(miss, 0.5) < hi
    assert np.mean(np.array(miss) <= hi) == pytest.approx(0.95, abs=0.02)


def _risk_brute(per_image, t):
    return (C.image_losses(per_image, t).sum() + 1) / (len(per_image) + 1)


def test_image_threshold_brute_force():
    rng = np.random.default_rng(1)
    for _ in range(30):
        per = [rng.random(rng.integers(0, 4)) for _ in range(rng.integers(5, 40))]
        for p in per:
            p[rng.random(len(p)) < 0.1] = -np.inf
        alpha = rng.uniform(0.05, 0.4)
        flat = np.concatenate(per)
        cand = sorted(set(flat[np.isfinite(flat)]), reverse=True)
        want = next((c for c in cand if _risk_brute(per, c) <= alpha), None)
        assert C.image_threshold(per, alpha) == want


def test_image_threshold_risk_controlled():
    # Expected loss of a fresh image ≤ α (Angelopoulos 2022), images of 1-3 persons.
    rng = np.random.default_rng(2)
    alpha, losses = 0.1, []
    for _ in range(1000):
        per = [rng.random(rng.integers(1, 4)) for _ in range(100)]
        t = C.image_threshold(per, alpha)
        test = [rng.random(rng.integers(1, 4)) for _ in range(50)]
        losses.append(C.image_losses(test, t).mean())
    assert np.mean(losses) <= alpha + 0.003


def test_box_margin_and_expand():
    det = np.array([[11, 11, 20, 30]], float)  # width 10, height 20
    gt = np.array([[9, 11, 20, 34]], float)  # 2 px past the left side, 4 px below
    m = C.box_margins(det, gt)
    assert m == pytest.approx([max(2 / 10, 4 / 20)])
    big = C.expand(det, m[0])
    assert (big[:, :2] <= gt[:, :2]).all() and (big[:, 2:] >= gt[:, 2:]).all()
    assert C.box_margins(gt, gt) == pytest.approx([0.0])


def test_margin_quantile():
    m = np.arange(1, 20, dtype=float)  # n = 19
    assert C.margin_quantile(m, 0.1) == 18.0  # ⌈0.9 · 20⌉ = 18
    assert C.margin_quantile(m, 0.01) == np.inf


def test_matched_boxes():
    gs = np.array([0.8, -np.inf, 0.3])
    gb = np.array([[0, 0, 10, 10], [50, 50, 60, 60], [20, 20, 30, 30]], float)
    ds = np.array([0.3, 0.8, 0.3])
    db = np.array([[100, 100, 110, 110], [1, 1, 10, 10], [21, 20, 30, 31]], float)
    idx, boxes = C.matched_boxes(gs, gb, ds, db)
    assert idx.tolist() == [0, 2]
    assert boxes.tolist() == [[1, 1, 10, 10], [21, 20, 30, 31]]
