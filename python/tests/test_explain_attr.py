"""T36.2, T36.3: perturbation attributions and exact integer attribution."""

import copy
from pathlib import Path

import numpy as np
import pytest

from tests.test_explain_gradient import T, _net, _x
from tests.test_faults import _qm
from yolo.explain import int_attr as IA
from yolo.explain import perturb as P
from yolo.explain.targets import Target
from yolo.quant.int_model import IntNetwork, head_luts

ROOT = Path(__file__).resolve().parents[2]
HEADS = {"tiny-yolov2-voc": 15, "tiny-yolov3-voc": 16, "tiny-yolov4-coco": 30}


# ---------------------------------------------------------------------- perturbation
def test_rise_masks_reproducible():
    a = P.rise_masks(5, 32, 32, s=4, rng=3)
    b = P.rise_masks(5, 32, 32, s=4, rng=3)
    assert a.shape == (5, 32, 32) and np.array_equal(a, b)
    assert 0 <= a.min() and a.max() <= 1


def test_similarity():
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])
    scores = np.array([[0.1, 0.8], [0.9, 0.0]])
    assert P.similarity((boxes, scores), (0.5, 0.5, 0.2, 0.2), 1) == pytest.approx(0.8)
    assert P.similarity((boxes[:0], scores[:0]), (0.5, 0.5, 0.2, 0.2), 1) == 0.0


def test_float_maps_deterministic():
    net = _net("yolo")
    stage = P.float_stage(net, conf=0.0)
    x = _x()
    box = (0.5, 0.5, 0.4, 0.4)
    a = P.drise(stage, x, box, 1, n=12, s=4, rng=0)
    b = P.drise(stage, x, box, 1, n=12, s=4, rng=0)
    assert np.array_equal(a["map"], b["map"]) and a["passes"] == 12
    o = P.occlusion(stage, x, box, 1, patch=8, stride=8)
    assert o["map"].shape == (32, 32) and o["passes"] == 17


def test_int_stage_equal_maps_on_equal_outputs():
    qm, _ = _qm("tiny-yolov2-voc")
    luts = head_luts(qm)
    x = np.random.default_rng(0).uniform(0, 1, (1, 3, 64, 64))
    a = P.drise(P.int_stage(IntNetwork(qm), luts, 1e-6), x, (0.5, 0.5, 0.5, 0.5), 3, n=8, s=4)
    b = P.drise(P.int_stage(IntNetwork(qm, "int64"), luts, 1e-6), x, (0.5, 0.5, 0.5, 0.5), 3,
                n=8, s=4)
    assert np.array_equal(a["map"], b["map"])


def test_golden_stage_equals_int():
    net = "tiny-yolov2-voc"
    model = ROOT / "model" / net
    calib = ROOT / "build" / "quant" / net / "calib.json"
    if not (P.GOLDEN.exists() and model.exists() and calib.exists()):
        pytest.skip("golden_run or export absent (make golden-check export)")
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from calibrate import load_fused

    from yolo.quant.calibrate import load_scales
    from yolo.quant.int_model import QuantModel

    qm = QuantModel.from_fused(load_fused(net, dtype=np.float64), *load_scales(calib))
    luts = head_luts(qm)
    x = np.random.default_rng(1).uniform(0, 1, (1, 3, 416, 416))
    masks = P.rise_masks(3, 416, 416, rng=0)
    box = (0.5, 0.5, 0.3, 0.5)
    a = P.drise(P.int_stage(IntNetwork(qm), luts), x, box, 14, masks=masks)
    b = P.drise(P.golden_stage(model, qm, luts, jobs=3), x, box, 14, masks=masks)
    assert np.array_equal(a["weights"], b["weights"]) and np.array_equal(a["map"], b["map"])


# -------------------------------------------------------------------- integer LRP
@pytest.fixture(scope="module", params=list(HEADS))
def model(request):
    qm, qx = _qm(request.param)
    return qm, qx[:1], HEADS[request.param]


def test_contributions_exact(model):
    qm, qx, _ = model
    hook = IA.KeepInputs()
    inet = IntNetwork(qm, "int64")
    inet.hooks.append(hook)
    inet.forward(qx)
    bad = IA.check_exact(hook, qm.convs, n=8)
    assert set(bad) == set(qm.convs) and not any(bad.values())


def test_lrp_int_shapes_and_conservation(model):
    qm, qx, head = model
    inet = IntNetwork(qm)
    t = Target(head, 0, 1, 1, 0)
    r = IA.lrp_int(inet, qx, t, rule="zplus")
    assert r.map.shape == qx.shape[2:] and r.input.shape == qx.shape[1:]
    # z+: no conv creates relevance, and the input gets what the chain kept
    assert all(c <= 1 + 1e-9 for c in r.conservation.values())
    assert 0 < r.total <= 1 + 1e-9
    e = IA.lrp_int(inet, qx, t, rule="eps")
    assert set(e.conservation) == set(r.conservation)
    assert e.head_channels.shape == (qm.convs[IA.head_conv(qm.layers, t)].qW.shape[1],)


def test_lrp_eps_conserves_without_bias(model):
    qm, qx, head = model
    qm = copy.deepcopy(qm)
    for c in qm.convs.values():
        c.qb = np.zeros_like(c.qb)
    r = IA.lrp_int(IntNetwork(qm), qx, Target(head, 1, 0, 1, 2), rule="eps", eps=1e-9)
    assert r.total == pytest.approx(1.0, abs=1e-6)


def test_lrp_float_matches_head_split():
    net = _net("region")
    x = _x()
    r = IA.lrp_float(net, x, T, rule="eps", eps=1e-9)
    assert r.map.shape == (32, 32)
    assert r.head_channels.sum() == pytest.approx(r.conservation[5], rel=1e-9)


def test_cam_l13():
    qm, qx = _qm("tiny-yolov2-voc")
    outs = IntNetwork(qm).forward(qx[:1], all_outputs=True)
    t = Target(15, 2, 1, 0, 4)
    cam = IA.cam_l13(qm, outs, t)
    f = 2 * 25 + 5 + 4
    acc = (qm.convs[14].qW[f, :, 0, 0].astype(np.int64)
           * outs[13][0, :, 1, 0].astype(np.int64)).sum()
    assert cam.dtype == np.int64 and cam[1, 0] == acc
