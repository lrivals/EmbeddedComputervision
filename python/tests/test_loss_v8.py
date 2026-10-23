"""T35.5: YOLOv8 loss (BCE + CIoU + DFL), gradients and the torch reference."""

from pathlib import Path

import numpy as np
import pytest

from yolo.testing.gradcheck import check_grad
from yolo.train.loss_v8 import HYP_V8, yolo_v8_loss

FIXTURE = Path(__file__).resolve().parents[2] / "weights" / "m33" / "yolov8n_loss.npz"
LEVELS = [(0, 8), (1, 16)]
NC = 3


def _batch(rng):
    """2 images 32×32 (grids 4×4 and 2×2), box logits peaked on short distances so that the
    predicted boxes overlap the ground truths (non-zero alignment)."""
    bias = -0.8 * np.arange(16)
    outs = {}
    for hid, s in LEVELS:
        g = 32 // s
        o = rng.standard_normal((2, 64 + NC, g, g)) * 0.5
        o[:, :64] += np.tile(bias, 4)[None, :, None, None]
        outs[hid] = o
    boxes = [np.array([[0.4, 0.45, 0.5, 0.6], [0.7, 0.3, 0.35, 0.3]]), np.array([[0.5, 0.5, 0.8, 0.7]])]
    labels = [np.array([0, 2]), np.array([1])]
    return outs, boxes, labels


@pytest.mark.parametrize("term", ["box", "cls", "dfl"])
def test_gradcheck_each_term(term):
    """Assignment and CIoU α held by `targets_from` (a copy of the outputs); both levels
    have positives with real alignment weights."""
    outs, boxes, labels = _batch(np.random.default_rng(0))
    ref = {k: v.copy() for k, v in outs.items()}
    hyp = {**HYP_V8, **{k: (HYP_V8[k] if k == term else 0.0) for k in ("box", "cls", "dfl")}}
    res = yolo_v8_loss(outs, boxes, labels, LEVELS, NC, hyp, targets_from=ref)
    assert res.parts[term] > 0 and res.masks["fg"].sum() > 0

    def f():
        return yolo_v8_loss(outs, boxes, labels, LEVELS, NC, hyp, targets_from=ref).total

    for hid in outs:
        assert np.abs(res.douts[hid]).max() > 1e-2, (term, hid)
        assert check_grad(f, outs[hid], res.douts[hid]) <= 1e-7, (term, hid)


def test_assignment_from_targets_equals_own():
    outs, boxes, labels = _batch(np.random.default_rng(1))
    a = yolo_v8_loss(outs, boxes, labels, LEVELS, NC)
    b = yolo_v8_loss(outs, boxes, labels, LEVELS, NC, targets_from=outs)
    assert a.total == b.total
    assert all(np.array_equal(a.douts[h], b.douts[h]) for h in outs)


def test_empty_batch():
    """No ground truth: only the class term, BCE against 0 on every anchor, S = 1."""
    outs, _, _ = _batch(np.random.default_rng(2))
    res = yolo_v8_loss(outs, [np.zeros((0, 4))] * 2, [np.zeros(0, int)] * 2, LEVELS, NC)
    assert res.parts["box"] == 0 and res.parts["dfl"] == 0
    z = np.concatenate([o[:, 64:].reshape(2, NC, -1) for o in outs.values()], axis=2)
    want = 2 * 0.5 * float(np.sum(np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z)))))
    assert res.total == pytest.approx(want, rel=1e-12)
    assert all(not res.douts[h][:, :64].any() for h in outs)


def test_gains_and_batch_size():
    outs, boxes, labels = _batch(np.random.default_rng(3))
    res = yolo_v8_loss(outs, boxes, labels, LEVELS, NC)
    half = yolo_v8_loss(outs, boxes, labels, LEVELS, NC,
                        {**HYP_V8, "box": 3.75, "cls": 0.25, "dfl": 0.75})
    assert half.total == pytest.approx(res.total / 2, rel=1e-12)


@pytest.mark.skipif(not FIXTURE.exists(), reason="torch reference fixture (local only, M35)")
def test_loss_equals_torch_fixture():
    """`v8DetectionLoss` (Ultralytics, assigner with `stride_val = 0`, the v8 rule) on the
    YOLOv8n outputs of 2 COCO images at 320 and their 21 boxes: each term to 1e-5."""
    f = np.load(FIXTURE)
    outs = {i: f[f"out{i}"] for i in range(3)}
    tg = f["targets"]  # (k, 6): image, class, cx, cy, w, h normalized
    boxes = [tg[tg[:, 0] == b, 2:] for b in range(2)]
    labels = [tg[tg[:, 0] == b, 1].astype(int) for b in range(2)]
    res = yolo_v8_loss(outs, boxes, labels, [(0, 8), (1, 16), (2, 32)], 80)
    for k, want in zip(("box", "cls", "dfl"), f["loss"]):
        assert res.parts[k] == pytest.approx(float(want), abs=1e-5), k
