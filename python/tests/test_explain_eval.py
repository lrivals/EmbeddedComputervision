"""T36.4, T36.5: evaluation of the maps and causes of the misses."""

import numpy as np
import pytest

from tests.test_explain_gradient import _net
from tests.test_faults import _qm
from yolo.explain import evaluate as E
from yolo.explain import misses as M
from yolo.explain.perturb import float_stage
from yolo.quant.int_model import IntNetwork, head_luts

BOX = (0.5, 0.5, 0.25, 0.25)


def test_size_bins():
    assert E.size_bin((0.5, 0.5, 0.05, 0.05), 500, 375) == "small"
    assert E.size_bin((0.5, 0.5, 0.15, 0.2), 500, 375) == "medium"
    assert E.size_bin((0.5, 0.5, 0.5, 0.5), 500, 375) == "large"


def test_pointing_and_energy():
    m = np.zeros((32, 32))
    m[16, 16] = 1
    assert E.pointing_game(m, BOX) and E.energy_in_box(m, BOX) == 1.0
    m = np.zeros((32, 32))
    m[0, 0] = 1
    assert not E.pointing_game(m, BOX, tol=2) and E.energy_in_box(m, BOX) == 0.0
    assert E.pointing_game(E.centre_map((32, 32)), BOX)


class _Stage:
    """A detector whose only candidate is BOX, scored by the contrast of the image inside
    BOX (2·mean |x − 0.5|): the box pixels are exactly the relevant ones."""

    def __call__(self, xs):
        from yolo.explain.maps import box_mask

        m = box_mask(BOX, *xs.shape[-2:])
        return [(np.array([BOX]), np.array([[2 * np.abs(x[:, m] - 0.5).mean()]])) for x in xs]


def test_deletion_insertion_rank_maps():
    from yolo.explain.maps import box_mask

    x = np.broadcast_to(np.indices((32, 32)).sum(0) % 2, (1, 3, 32, 32)).astype(float)
    good = box_mask(BOX, 32, 32).astype(float) + E.random_map((32, 32)) * 1e-3
    bad = 1 - good
    st = _Stage()
    d_good, curve = E.deletion(st, x, good, BOX, 0, steps=8)
    d_bad, _ = E.deletion(st, x, bad, BOX, 0, steps=8)
    assert curve[0] == pytest.approx(1.0) and d_good < d_bad
    i_good, _ = E.insertion(st, x, good, BOX, 0, steps=8, blur=1)
    i_bad, _ = E.insertion(st, x, bad, BOX, 0, steps=8, blur=1)
    assert i_good > i_bad
    s = E.scores(st, x, good, BOX, 0, steps=4)
    assert s["pointing"] and s["energy"] > 0.99


def test_scores_on_float_stage():
    net = _net("yolo")
    x = np.random.default_rng(0).uniform(0, 1, (1, 3, 32, 32))
    s = E.scores(float_stage(net, conf=0.0), x, E.centre_map((32, 32)), BOX, 1, steps=4)
    assert set(s) == {"pointing", "energy", "deletion", "insertion"}


# ---------------------------------------------------------------------- miss causes
NET = _net("yolo").net  # 3 anchors 60/150/300 px of 416, one 8 × 8 head (id 6), 3 classes
GRIDS = {6: (8, 8)}
GT = (0.5, 0.5, 60 / 416, 60 / 416)  # anchor 0, cell (4, 4)
K = M.responsible_index(NET, GT, GRIDS)


def _out(obj=0.01, probs=(0.8, 0.1, 0.1), box=GT, dets=None):
    n = 3 * 64
    boxes = np.tile([0.1, 0.1, 0.05, 0.05], (n, 1))
    o = np.full(n, 1e-3)
    sc = np.full((n, 3), 1e-4)
    boxes[K], o[K] = box, obj
    sc[K] = obj * np.asarray(probs)
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    return M.StageOutputs(boxes, o, sc, dets or empty)


def test_responsible_index():
    assert K == (0 * 8 + 4) * 8 + 4


@pytest.mark.parametrize("out,cause,kw", [
    (_out(dets=(np.array([GT]), np.array([0.9]), np.array([0]))), None, {}),
    (_out(), "quantization",
     {"float_out": _out(dets=(np.array([GT]), np.array([0.9]), np.array([0])))}),
    (_out(), "low_objectness", {}),
    (_out(obj=0.9, probs=(0.1, 0.8, 0.1)), "class_confusion", {}),
    (_out(obj=0.3, probs=(0.5, 0.3, 0.2)), "below_threshold", {}),
    (_out(obj=0.9), "nms_suppressed", {}),
    (_out(obj=0.9, box=(0.5, 0.5, 0.02, 0.02)), "mislocalized", {}),
])
def test_causes(out, cause, kw):
    assert M.classify_miss(NET, GT, 0, out, GRIDS, **kw) == cause


def test_no_anchor():
    thin = (0.5, 0.5, 0.9, 0.01)
    assert M.classify_miss(NET, thin, 0, _out(), GRIDS) == "no_anchor"


def test_partition_on_int_model():
    qm, qx = _qm("tiny-yolov2-voc")
    luts = head_luts(qm)
    heads = IntNetwork(qm).forward(qx[:1])
    out = M.int_outputs(heads, qm.net, luts, conf_thr=0.25)
    assert len(out.obj) == 5 * 2 * 2 and out.scores.shape == (20, 20)
    grids = {15: (2, 2)}
    rng = np.random.default_rng(0)
    causes = [M.classify_miss(qm.net, tuple(rng.uniform(0.2, 0.8, 4) * [1, 1, .6, .6]),
                              int(rng.integers(20)), out, grids) for _ in range(30)]
    assert all(c is None or c in M.CAUSES for c in causes)
    sh = M.shares(causes)
    assert set(sh) == set(M.CAUSES) and sum(sh.values()) == pytest.approx(1.0)
