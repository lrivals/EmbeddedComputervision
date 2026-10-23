"""T36.0, T36.1: explanation targets (gradcheck) and gradient attributions."""

import copy

import numpy as np
import pytest

from yolo.explain import gradient as G
from yolo.explain.targets import (Target, detections, explain, miss_target, target_douts,
                                  target_score)
from yolo.models.graph import Network
from yolo.testing.gradcheck import check_grad


def _conv(cout, k=3, act="leaky"):
    return {"type": "conv", "k": k, "s": 1, "cout": cout, "act": act, "bn": False}


def _net(head):
    """Tiny folded network (no BN) with a `yolo` (sigmoid) or `region` (softmax) head."""
    layers = [_conv(4), {"type": "maxpool", "k": 2, "s": 2}, _conv(8),
              {"type": "maxpool", "k": 2, "s": 2}, _conv(6),
              _conv(3 * 8, k=1, act="linear")]
    layers.append({"type": "yolo", "mask": [0, 1, 2]} if head == "yolo"
                  else {"type": "region", "num": 3})
    desc = {"name": f"tiny-{head}", "input": (3, 32, 32), "classes": 3,
            "anchors": [[60, 60], [150, 120], [300, 300]], "layers": layers}
    net = Network(desc, dtype=np.float64, rng=0)
    rng = np.random.default_rng(1)
    for p in net.params:
        if "b" in p:
            p["b"][:] = rng.normal(0, 0.3, p["b"].shape)
    return net


@pytest.fixture(params=["yolo", "region"])
def net(request):
    return _net(request.param)


def _x(seed=2):
    return np.random.default_rng(seed).uniform(0, 1, (1, 3, 32, 32))


T = Target(head=6, anchor=1, i=3, j=5, cls=2)


def test_target_grad_input(net):
    x = _x()
    e = explain(net, x, T)

    def f():
        return target_score(net.forward(x, train=False), net.net, T)

    assert check_grad(f, x, e.dx, n_samples=60, rng=0) <= 1e-6


def test_target_grad_head_and_layer(net):
    x = _x()
    outs = net.forward(x, train=False)
    _, douts = target_douts(outs, net.net, T)
    head = outs[6].copy()

    def f():
        return target_score({6: head}, net.net, T)

    assert check_grad(f, head, douts[6], h=1e-6) <= 1e-6
    # δ of an intermediate layer (L4 output), through `keep`
    e = explain(net, x, T, keep=[4])
    acts = net.forward(x, train=False, all_outputs=True)
    a4 = acts[4].copy()

    def g():  # L5 (head conv) on a4, L6 is the identity
        return target_score({6: net._conv_forward(5, a4, False)[0]}, net.net, T)

    assert check_grad(g, a4, e.dlayers[4], n_samples=40, rng=1) <= 1e-6


def test_backward_without_keep_unchanged(net):
    x = _x()
    outs = net.forward(x, train=False)
    _, douts = target_douts(outs, net.net, T)
    a = net.backward(douts)
    net.forward(x, train=False)
    b = net.backward(douts, keep={2})
    assert len(a) == 2 and len(b) == 3
    assert np.array_equal(a[0], b[0])
    assert set(b[2]) == {2}


def test_detections_and_miss(net):
    outs = net.forward(_x(), train=False)
    dets = detections(outs, net.net, conf_thr=0.0, iou_thr=0.45)
    assert dets and all(d.score >= dets[-1].score for d in dets)
    d = dets[0]
    assert np.isclose(target_score(outs, net.net, d)[0], d.score)
    t = miss_target(net.net, (0.5, 0.4, 0.3, 0.3), 1)
    assert t.kind == "miss" and (t.i, t.j) == (3, 4) and t.cls == 1


@pytest.mark.parametrize("baseline", ["black", "blur"])
def test_ig_completeness(net, baseline):
    # The cell whose score moves most between the baseline and x: on a near-zero
    # s(x) − s(x') the relative error says nothing about the quadrature.
    x = _x()
    base = G.baseline_of(x, baseline)
    sx, sb = net.forward(x, train=False), net.forward(base, train=False)
    cands = [Target(6, a, i, j, c) for a in range(3) for i in range(8) for j in range(8)
             for c in range(3)]
    t = max(cands, key=lambda t: abs(target_score(sx, net.net, t)[0]
                                     - target_score(sb, net.net, t)[0]))
    # 128 steps: the tiny net has few kinks, each one costs the midpoint rule more than on
    # the real networks, where the 64-step acceptance is measured (tools/explain.py sanity).
    r = G.integrated_gradients(net, x, t, steps=128, baseline=baseline)
    assert r["completeness"] <= 0.01


def test_maps_shapes_and_costs(net):
    x = _x()
    for name, fn in G.METHODS.items():
        r = fn(net, x, T)
        assert r["map"].shape == (32, 32), name
        assert r["passes"] >= 2
    assert G.gradcam(net, x, T)["layer"] == 4


def test_randomize_from_top(net):
    x = _x()
    before = copy.deepcopy(net.params)
    r = G.randomize_from_top(net, 2)
    assert all(np.array_equal(a["W"], b["W"]) for a, b in zip(before, net.params) if "W" in a)
    ids = G.conv_ids(net)
    assert not np.array_equal(r.params[ids[-1]]["W"], net.params[ids[-1]]["W"])
    assert np.array_equal(r.params[ids[0]]["W"], net.params[ids[0]]["W"])
    m0 = G.gradcam(net, x, T)["map"]
    m1 = G.gradcam(r, x, T)["map"]
    assert not np.allclose(m0, m1)
