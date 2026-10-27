"""T29.3: k-of-n temporal vote on the person detections."""

import numpy as np
import pytest

from yolo.infer import temporal as T

A = [10.0, 10, 50, 90]
B = [200.0, 10, 240, 90]


def _frames(*present):
    """One frame per entry: the boxes present (A, B) with score 0.9."""
    return [(np.array(f, float).reshape(-1, 4), np.full(len(f), 0.9)) for f in present]


def _counts(out):
    return [len(s) for _, s in out]


def test_one_of_one_is_the_detector():
    rng = np.random.default_rng(0)
    frames = [(rng.uniform(0, 300, (m, 4)).cumsum(axis=1), rng.random(m))
              for m in rng.integers(0, 5, 30)]
    for (b, s), (rb, rs) in zip(frames, T.vote(frames, 1, 1)):
        assert sorted(rs.tolist()) == sorted(s.tolist())
        assert sorted(map(tuple, rb.tolist())) == sorted(map(tuple, b.tolist()))


def test_hold_bridges_a_gap():
    frames = _frames([A], [], [], [A], [], [], [], [])
    assert _counts(T.vote(frames, 1, 3)) == [1, 1, 1, 1, 1, 1, 0, 0]
    out = T.vote(frames, 1, 3)
    assert out[1][0].tolist() == [A]  # held with the last box
    assert _counts(T.vote(frames, 1, 1)) == [1, 0, 0, 1, 0, 0, 0, 0]


def test_confirmation_delays_and_filters():
    frames = _frames([A], [A, B], [A], [A])  # B is an isolated false alarm
    assert _counts(T.vote(frames, 2, 2)) == [0, 1, 1, 1]
    assert _counts(T.vote(frames, 3, 3)) == [0, 0, 1, 1]
    # 2 of 3: one missed frame keeps the track, the window [miss, hit, miss] does not
    assert _counts(T.vote(_frames([A], [A], [], [A], []), 2, 3)) == [0, 1, 1, 1, 0]


def test_association_follows_a_moving_box():
    moving = [[[10.0 + 5 * i, 10, 50 + 5 * i, 90]] for i in range(6)]
    out = T.vote(_frames(*moving), 3, 3)
    assert _counts(out) == [0, 0, 1, 1, 1, 1]
    assert out[-1][0].tolist() == moving[-1]
    # a jump beyond the IoU of the association starts a new track
    assert _counts(T.vote(_frames([A], [A], [B], [B]), 2, 2)) == [0, 1, 0, 1]


def test_bad_k_n():
    with pytest.raises(ValueError):
        T.vote([], 3, 2)


def test_iou_matrix_and_greedy():
    ov = T.iou_matrix([A, B], [A, [10.0, 10, 50, 50]])
    assert ov[0, 0] == pytest.approx(1.0) and ov[1, 0] == 0.0
    assert T.greedy_pairs(ov, 0.3) == [(0, 0)]  # column 1 overlaps A at 0.5 but A is taken
    assert T.greedy_pairs(np.zeros((0, 2)), 0.3) == []
