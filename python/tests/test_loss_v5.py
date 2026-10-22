"""YOLOv5 loss (T33.6) and the OTA assignment (T34.3)."""

from pathlib import Path

import numpy as np
import pytest

from yolo.data.targets import build_targets_v5
from yolo.infer.boxes import ciou, ciou_alpha, iou
from yolo.layers.activations import sigmoid
from yolo.testing.gradcheck import check_grad, numerical_grad, rel_error
from yolo.train.loss_v5 import (HYP_V5, HYP_V7_TINY, build_targets_ota, gains, ota_match,
                                yolo_v5_loss)

FIXTURE = Path(__file__).resolve().parents[2] / "weights" / "m33" / "yolov5n_loss.npz"


# --------------------------------------------------------------------------- CIoU
def _pairs(kind, rng, n=12):
    p = np.column_stack([rng.random((n, 2)) * 4, rng.random((n, 2)) * 2 + 0.2])
    if kind == "random":
        g = np.column_stack([rng.random((n, 2)) * 4, rng.random((n, 2)) * 2 + 0.2])
    elif kind == "nested":
        g = p.copy()
        g[:, 2:] *= rng.random((n, 2)) * 0.5 + 0.3
        g[:, :2] += (rng.random((n, 2)) - 0.5) * 0.05
    else:  # disjoint
        g = p.copy()
        g[:, 0] += p[:, 2] + 1 + rng.random(n)
        g[:, 1:] += (rng.random((n, 3)) - 0.5) * 0.2  # no tie in the min / max edges
    return p, g


def test_ciou_values():
    b = np.array([[1.0, 2.0, 3.0, 1.5]])
    assert np.isclose(ciou(b, b)[0][0], 1.0, atol=1e-6)
    p, g = _pairs("random", np.random.default_rng(0))
    c, _ = ciou(p, g)
    i = np.diag(iou(p, g))
    assert np.all(c <= i + 1e-12) and np.all(c >= -1.5)
    p, g = _pairs("disjoint", np.random.default_rng(1))
    assert np.all(ciou(p, g)[0] < 0)


@pytest.mark.parametrize("kind", ["random", "nested", "disjoint"])
def test_ciou_gradcheck(kind):
    rng = np.random.default_rng(2)
    p, g = _pairs(kind, rng)
    c, grad = ciou(p, g)
    # α is a constant of the backward: frozen at its value in the numerical loss.
    alpha = ciou_alpha(p, g)
    assert np.allclose(c, ciou(p, g, alpha=alpha)[0])
    assert check_grad(lambda: ciou(p, g, alpha=alpha)[0], p, grad) < 1e-7


# --------------------------------------------------------------------------- loss
NC = 3
LEVELS = [(5, 8, np.array([[10.0, 13.0], [16.0, 30.0], [33.0, 23.0]])),
          (9, 16, np.array([[30.0, 61.0], [62.0, 45.0], [59.0, 119.0]]))]


def _batch(rng, size=64):
    outs = {hid: rng.standard_normal((2, 3 * (5 + NC), size // s, size // s)) * 0.5
            for hid, s, _ in LEVELS}
    boxes = [np.array([[0.3, 0.4, 0.3, 0.4], [0.7, 0.6, 0.5, 0.3]]),
             np.array([[0.55, 0.45, 0.2, 0.25]])]
    labels = [np.array([0, 2]), np.array([1])]
    return outs, boxes, labels


def _reference_loss(outs, boxes, labels, hyp, size=64):
    """Element-by-element loop over the matches: the formulas of the module doc."""
    grids = {hid: o.shape[-2:] for hid, o in outs.items()}
    t = build_targets_v5(boxes, labels, LEVELS, grids, hyp["anchor_t"])
    g_box, g_obj, g_cls = gains(hyp, NC, len(LEVELS), size)
    total = 0.0
    for li, (hid, _, _) in enumerate(LEVELS):
        o = outs[hid]
        n, _, sh, sw = o.shape
        p = o.reshape(n, 3, 5 + NC, sh, sw)
        d = t[hid]
        tobj = np.zeros((n, 3, sh, sw))
        lbox, lcls = [], []
        for m in range(len(d["b"])):
            b, a, gj, gi = d["b"][m], d["a"][m], d["gj"][m], d["gi"][m]
            v = p[b, a, :, gj, gi]
            s = sigmoid(v)
            box = np.array([2 * s[0] - 0.5, 2 * s[1] - 0.5,
                            (2 * s[2]) ** 2 * d["anch"][m, 0], (2 * s[3]) ** 2 * d["anch"][m, 1]])
            c = ciou(box[None], d["tbox"][m][None])[0][0]
            lbox.append(1 - c)
            tobj[b, a, gj, gi] = max(c, 0)
            y = np.zeros(NC)
            y[d["cls"][m]] = 1
            lcls += list(-(y * np.log(s[5:]) + (1 - y) * np.log(1 - s[5:])))
        so = sigmoid(p[:, :, 4])
        lobj = -(tobj * np.log(so) + (1 - tobj) * np.log(1 - so)).mean()
        total += g_box * np.mean(lbox) + g_obj * hyp["balance"][li] * lobj + g_cls * np.mean(lcls)
    return total * n


def test_loss_matches_reference_loop():
    outs, boxes, labels = _batch(np.random.default_rng(3))
    res = yolo_v5_loss(outs, boxes, labels, LEVELS, NC)
    assert np.isclose(res.total, _reference_loss(outs, boxes, labels, HYP_V5), rtol=1e-10)
    assert np.isclose(res.total, sum(res.parts.values()))


@pytest.mark.parametrize("hyp", [HYP_V5, HYP_V7_TINY], ids=["v5", "ota"])
def test_loss_gradcheck(hyp):
    outs, boxes, labels = _batch(np.random.default_rng(4))
    ref = {k: v.copy() for k, v in outs.items()}
    res = yolo_v5_loss(outs, boxes, labels, LEVELS, NC, hyp, targets_from=ref)
    assert {k: v.shape for k, v in res.douts.items()} == {k: v.shape for k, v in outs.items()}
    for hid in outs:
        def f():
            return yolo_v5_loss(outs, boxes, labels, LEVELS, NC, hyp, targets_from=ref).total
        # Every matched cell plus random ones.
        assert check_grad(f, outs[hid], res.douts[hid], n_samples=150, rng=0) < 1e-7
        tobj = res.masks[hid]["tobj"]
        b, a, gj, gi = np.nonzero(tobj)
        c5 = 5 + NC
        idx = [(bb, aa * c5 + ch, jj, ii) for bb, aa, jj, ii in zip(b, a, gj, gi)
               for ch in range(c5)]
        num = numerical_grad(f, outs[hid], idx=idx)
        rows = tuple(np.array(c) for c in zip(*idx))
        assert rel_error(res.douts[hid][rows], num[rows]) < 1e-7


def test_obj_targets_detached():
    """The IoU targets of the objectness come from `targets_from`, not from `outputs`."""
    outs, boxes, labels = _batch(np.random.default_rng(5))
    a = yolo_v5_loss(outs, boxes, labels, LEVELS, NC)
    b = yolo_v5_loss(outs, boxes, labels, LEVELS, NC, targets_from=outs)
    assert np.isclose(a.total, b.total)
    for hid in outs:
        assert np.allclose(a.douts[hid], b.douts[hid])


def test_no_box_image():
    outs, _, _ = _batch(np.random.default_rng(6))
    res = yolo_v5_loss(outs, [np.zeros((0, 4))] * 2, [np.zeros(0)] * 2, LEVELS, NC)
    assert res.parts["box"] == 0 and res.parts["cls"] == 0 and res.parts["obj"] > 0


def test_gains_scaling():
    assert gains(HYP_V5, 80, 3, 640) == pytest.approx((0.05, 1.0, 0.5))
    assert gains(HYP_V5, 20, 3, 416) == pytest.approx((0.05, (416 / 640) ** 2, 0.125))


@pytest.mark.skipif(not FIXTURE.exists(), reason="torch reference fixture (local only, M33)")
def test_loss_equals_torch_fixture():
    """`yolov5n_loss.npz`: head outputs, targets and the `ComputeLoss` value of the reference
    (made once by a throwaway torch script, not in the repo)."""
    f = np.load(FIXTURE, allow_pickle=False)
    levels = [(i, s, f["anchors"][i]) for i, s in enumerate((8, 16, 32))]
    outs = {i: f[f"out{i}"] for i in range(3)}
    tg = f["targets"]  # (k, 6): image, class, cx, cy, w, h normalized
    n = outs[0].shape[0]
    boxes = [tg[tg[:, 0] == b, 2:] for b in range(n)]
    labels = [tg[tg[:, 0] == b, 1].astype(int) for b in range(n)]
    res = yolo_v5_loss(outs, boxes, labels, levels, 80)
    assert res.total == pytest.approx(float(f["loss"]), abs=1e-5)


# --------------------------------------------------------------------------- OTA
def test_ota_dynamic_k_and_cheapest():
    iou_m = np.array([[0.9, 0.8, 0.7, 0.1, 0.0],    # Σ = 2.5 → k = 2
                      [0.0, 0.1, 0.2, 0.3, 0.35]])  # Σ = 0.95 → k = 1
    cost = np.array([[1.0, 0.5, 2.0, 9.0, 9.0],
                     [9.0, 9.0, 9.0, 0.7, 0.4]])
    m = ota_match(cost, iou_m)
    assert m.tolist() == [[True, True, False, False, False],
                          [False, False, False, False, True]]


def test_ota_conflict_goes_to_cheapest():
    iou_m = np.array([[0.6, 0.6, 0.0], [0.6, 0.6, 0.0]])  # k = 1 each
    cost = np.array([[1.0, 3.0, 9.0],
                     [0.5, 4.0, 9.0]])
    m = ota_match(cost, iou_m)
    # Both pick candidate 0; it goes to ground truth 1 (0.5 < 1.0); ground truth 0 is left
    # without a match, as the reference (no second round).
    assert m.tolist() == [[False, False, False], [True, False, False]]


def test_ota_targets_subset_of_candidates():
    outs, boxes, labels = _batch(np.random.default_rng(7))
    grids = {hid: o.shape[-2:] for hid, o in outs.items()}
    cand = build_targets_v5(boxes, labels, LEVELS, grids)
    ota = build_targets_ota(outs, boxes, labels, LEVELS, NC)
    for hid in outs:
        c = set(zip(*(cand[hid][k].tolist() for k in ("b", "a", "gj", "gi"))))
        o = list(zip(*(ota[hid][k].tolist() for k in ("b", "a", "gj", "gi"))))
        assert set(o) <= c
        assert ota[hid]["tbox"].shape == (len(o), 4)
    # Every ground truth label of an image is one of its own labels.
    for hid in outs:
        for b, c in zip(ota[hid]["b"], ota[hid]["cls"]):
            assert c in labels[b]
