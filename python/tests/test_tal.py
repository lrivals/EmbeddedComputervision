"""T35.4: task-aligned assigner of YOLOv8, hand-built cases and the torch reference."""

from pathlib import Path

import numpy as np
import pytest

from yolo.data.tal import ALPHA, BETA, assign_tal
from yolo.infer.boxes import ciou, xyxy_to_cxcywh
from yolo.infer.decode import anchor_points

FIXTURE = Path(__file__).resolve().parents[2] / "weights" / "m33" / "yolov8n_loss.npz"


def _points(n=8, stride=8):
    pts, st = anchor_points([(n, n)], [stride])
    return pts * st[:, None]


def _reference(s, pb, pts, gl, gb, topk=10, eps=1e-9):
    """Element-by-element rule of the module doc (loops, no vectorization)."""
    g, a = len(gb), len(pts)
    u, t, cand = np.zeros((g, a)), np.zeros((g, a)), np.zeros((g, a), bool)
    for i in range(g):
        for j in range(a):
            x, y = pts[j]
            x1, y1, x2, y2 = gb[i]
            if min(x - x1, y - y1, x2 - x, y2 - y) > eps:
                cand[i, j] = True
                u[i, j] = max(ciou(xyxy_to_cxcywh(pb[j]), xyxy_to_cxcywh(gb[i]))[0][0], 0)
                t[i, j] = s[j, gl[i]] ** ALPHA * u[i, j] ** BETA
    pos = np.zeros((g, a), bool)
    for i in range(g):
        for j in sorted(range(a), key=lambda j: -t[i, j])[:topk]:
            pos[i, j] = cand[i, j]
    owner, conflicts = {}, []
    for j in range(a):
        claim = [i for i in range(g) if pos[i, j]]
        if len(claim) > 1:
            conflicts.append(j)
            claim = [int(np.argmax(u[:, j]))]
        if claim:
            owner[j] = claim[0]
    scores = np.zeros_like(s)
    for j, i in owner.items():
        mine = [k for k, o in owner.items() if o == i]
        t_max, u_max = max(t[i, k] for k in mine), max(u[i, k] for k in mine)
        scores[j, gl[i]] = t[i, j] * u_max / (t_max + eps)
    return owner, scores, conflicts


def _case(rng, gb, a, c=3, jitter=6.0):
    """Predicted boxes: each ground truth jittered, spread over the anchors."""
    pb = np.repeat(gb[:1], a, axis=0) + rng.standard_normal((a, 4)) * jitter
    for i in range(1, len(gb)):
        pb[i::len(gb)] = gb[i] + rng.standard_normal((len(pb[i::len(gb)]), 4)) * jitter
    pb[:, 2:] = np.maximum(pb[:, 2:], pb[:, :2] + 1)
    return rng.random((a, c)), pb


def test_one_box():
    rng = np.random.default_rng(0)
    pts = _points()
    gb = np.array([[6.0, 6.0, 46.0, 38.0]])
    gl = np.array([2])
    s, pb = _case(rng, gb, len(pts))
    tb, ts, fg, idx = assign_tal(s, pb, pts, gl, gb)
    owner, ref, _ = _reference(s, pb, pts, gl, gb)
    assert sorted(np.nonzero(fg)[0]) == sorted(owner)
    assert fg.sum() == 10                              # 5 × 4 centres inside, top 10 kept
    assert np.allclose(ts, ref, rtol=0, atol=1e-12)
    assert np.all(tb[fg] == gb[0]) and np.all(ts[:, [0, 1]] == 0)
    # Normalisation: the best positive gets the best CIoU of the positives.
    u = [max(ciou(xyxy_to_cxcywh(pb[j]), xyxy_to_cxcywh(gb[0]))[0][0], 0) for j in owner]
    assert ts[:, 2].max() == pytest.approx(max(u), rel=1e-6)  # ε of t_max + ε


def test_two_overlapping_boxes():
    """The anchors inside both boxes go to the one of highest CIoU with their prediction
    (`topk` = 64: every candidate is kept, so the 9 shared centres are all conflicts)."""
    rng = np.random.default_rng(1)
    pts = _points()
    gb = np.array([[4.0, 4.0, 44.0, 44.0], [14.0, 14.0, 54.0, 54.0]])
    gl = np.array([0, 1])
    s, pb = _case(rng, gb, len(pts), jitter=4.0)
    tb, ts, fg, idx = assign_tal(s, pb, pts, gl, gb, topk=64)
    owner, ref, conflicts = _reference(s, pb, pts, gl, gb, topk=64)
    assert {int(j): int(idx[j]) for j in np.nonzero(fg)[0]} == owner
    assert np.allclose(ts, ref, rtol=0, atol=1e-12)
    assert conflicts, "the case has anchors claimed by both boxes"
    for j in conflicts:
        u = [ciou(xyxy_to_cxcywh(pb[j]), xyxy_to_cxcywh(b))[0][0] for b in gb]
        assert idx[j] == int(np.argmax(u)) and ts[j, gl[1 - idx[j]]] == 0


def test_small_box_without_centre():
    """A box covering no anchor centre has no positive (classic v8 rule)."""
    pts = _points()
    gb = np.array([[9.0, 9.0, 11.0, 11.0]])
    rng = np.random.default_rng(2)
    s, pb = _case(rng, gb, len(pts))
    tb, ts, fg, _ = assign_tal(s, pb, pts, np.array([0]), gb)
    assert not fg.any() and not ts.any()


def test_no_ground_truth_and_inputs_untouched():
    pts = _points(4)
    s, pb = np.full((16, 2), 0.5), np.tile([0.0, 0.0, 8.0, 8.0], (16, 1))
    s0, pb0 = s.copy(), pb.copy()
    tb, ts, fg, idx = assign_tal(s, pb, pts, np.zeros(0), np.zeros((0, 4)))
    assert not fg.any() and ts.shape == (16, 2) and tb.shape == (16, 4)
    assign_tal(s, pb, pts, np.array([1]), np.array([[0.0, 0.0, 16.0, 16.0]]))
    assert np.array_equal(s, s0) and np.array_equal(pb, pb0)


@pytest.mark.skipif(not FIXTURE.exists(), reason="torch reference fixture (local only, M35)")
def test_equals_torch_assigner():
    """`TaskAlignedAssigner` of Ultralytics (`stride_val = 0`, the v8 rule) on the YOLOv8n
    predictions of 2 COCO images at 320: same positives, ground truths and boxes; soft
    scores to 1e-6 (the reference adds ε to the box heights of its xyxy CIoU)."""
    f = np.load(FIXTURE)
    for b in range(2):
        m = f["tal_mask_gt"][b, :, 0] > 0
        tb, ts, fg, idx = assign_tal(f["tal_pd_scores"][b], f["tal_pd_boxes"][b],
                                     f["tal_anchors"], f["tal_gt_labels"][b, m, 0],
                                     f["tal_gt_boxes"][b, m])
        assert np.array_equal(fg, f["tal_fg"][b])
        assert np.array_equal(idx[fg], f["tal_gt_idx"][b][fg])
        assert np.array_equal(tb[fg], f["tal_target_boxes"][b][fg])
        assert np.abs(ts - f["tal_target_scores"][b]).max() <= 1e-6
