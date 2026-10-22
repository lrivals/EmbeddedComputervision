"""YOLOv7-tiny without torch (M34): layer table (T34.0), routes and stride-1 pools in the
backward (T34.1), the whole network in train and eval modes (T34.2)."""

from pathlib import Path

import numpy as np
import pytest

from yolo.io.darknet_weights import load_darknet_weights
from yolo.layers.pool import maxpool_backward, maxpool_forward
from yolo.models.graph import Network
from yolo.models.specs import (NETWORKS, _conv, _elan, _route, infer_shapes, layer_cost)
from yolo.models.tiny_yolo import build, load_cfg
from yolo.data.targets import v5_levels
from yolo.testing.gradcheck import check_grad, numerical_grad, rel_error
from yolo.train.loss_v5 import HYP_V7_TINY, yolo_v5_loss

WEIGHTS = Path(__file__).resolve().parents[2] / "weights" / "yolov7-tiny.weights"


# ------------------------------------------------------------------------ T34.0
def _counts(net):
    shapes = infer_shapes(net)
    costs = [layer_cost(layer, *sh) for layer, sh in zip(net["layers"], shapes)]
    return sum(p for p, _ in costs), sum(m for _, m in costs)


def test_layer_table_matches_published():
    """YOLOv7 paper (arXiv 2207.02696, Table 1): 6.2 M parameters, 5.8 GFLOPs at 416."""
    net = load_cfg("tiny-yolov7-coco")
    assert net["layers"] == NETWORKS["tiny-yolov7-coco"]["layers"]
    params, macs = _counts(net)
    assert round(params / 1e6, 1) == 6.2
    assert round(2 * macs / 1e9, 1) == 5.8


@pytest.mark.skipif(not WEIGHTS.exists(), reason="weights/yolov7-tiny.weights absent")
def test_weights_read_to_last_byte():
    net = build("tiny-yolov7-coco")
    load_darknet_weights(net, WEIGHTS)  # raises if a byte is missing or left over
    n = sum(v.size for p in net.params for v in p.values())
    n += sum(s["mean"].size + s["var"].size for s in net.state if s)
    header = 20  # 3 int32 + int64 `seen` (version 0.2)
    assert WEIGHTS.stat().st_size == header + 4 * n


# ------------------------------------------------------------------------ T34.1
def _distinct(shape, rng):
    return np.random.default_rng(rng).permutation(np.prod(shape)).reshape(shape) / 7.0


def _pool_ref(x, k):
    """Window max of the valid cells only (padding −∞, Darknet and PyTorch)."""
    lo = (k - 1) // 2
    h, w = x.shape[2:]
    y = np.empty_like(x)
    for i in range(h):
        for j in range(w):
            r = slice(max(i - lo, 0), min(i - lo + k, h))
            c = slice(max(j - lo, 0), min(j - lo + k, w))
            y[:, :, i, j] = x[:, :, r, c].max(axis=(2, 3))
    return y


def _first_max_grad(x, k, dy):
    """Brute-force backward: each window sends its δ to its first maximum (raster order of
    the valid cells)."""
    lo = (k - 1) // 2
    n, c, h, w = x.shape
    dx = np.zeros_like(x)
    for b in range(n):
        for ch in range(c):
            for i in range(h):
                for j in range(w):
                    r0, c0 = max(i - lo, 0), max(j - lo, 0)
                    win = x[b, ch, r0:min(i - lo + k, h), c0:min(j - lo + k, w)]
                    a, bb = np.unravel_index(np.argmax(win), win.shape)
                    dx[b, ch, r0 + a, c0 + bb] += dy[b, ch, i, j]
    return dx


@pytest.mark.parametrize("k", [5, 9, 13])
@pytest.mark.parametrize("const", [False, True])
def test_stride1_pool_backward_is_first_valid_max(k, const):
    x = np.full((1, 2, 8, 7), 0.3) if const else _distinct((1, 2, 8, 7), k)
    y, cache = maxpool_forward(x, k=k, s=1)
    np.testing.assert_array_equal(y, _pool_ref(x, k))
    dy = np.random.default_rng(k).standard_normal(y.shape)
    dx, _ = maxpool_backward(dy, cache)
    assert dx.shape == x.shape  # the padded border is cropped: it never keeps a gradient
    np.testing.assert_allclose(dx, _first_max_grad(x, k, dy), atol=1e-12)
    assert np.isclose(dx.sum(), dy.sum())  # gradient mass conserved


def _small(layers, outputs):
    return {"name": "small", "input": (3, 8, 8), "layers": layers, "outputs": outputs}


def _randomize(net, rng):
    for p in net.params:
        for k in ("gamma", "beta", "b"):
            if k in p:
                p[k][:] = rng.standard_normal(p[k].shape)
    for s in net.state:
        if s:
            s["mean"][:] = rng.standard_normal(s["mean"].shape) * 0.1
            s["var"][:] = rng.random(s["var"].shape) + 0.5
    return net


# ELAN-tiny of width 3 after a stem conv (layer 0): layers 1-7 as `specs._elan`.
ELAN = _small([_conv(4)] + _elan(3, 1), [7])
# SPP-CSP of YOLOv7-tiny (layers 33-44 of the cfg) after a stem conv, width 4.
SPP_CSP = _small([
    _conv(4),                                                        # 0
    _conv(4, k=1),                                                   # 1: u
    _route(0),                                                       # 2
    _conv(4, k=1),                                                   # 3: v
    {"type": "maxpool", "k": 5, "s": 1},                             # 4
    _route(3),                                                       # 5
    {"type": "maxpool", "k": 9, "s": 1},                             # 6
    _route(3),                                                       # 7
    {"type": "maxpool", "k": 13, "s": 1},                            # 8
    _route(8, 6, 4, 3),                                              # 9: v read 4 times
    _conv(4, k=1),                                                   # 10
    _route(1, 10),                                                   # 11
    _conv(4, k=1),                                                   # 12
], [12])


@pytest.mark.parametrize("desc", [ELAN, SPP_CSP], ids=["elan", "spp-csp"])
@pytest.mark.parametrize("train", [True, False])
def test_block_gradcheck(desc, train):
    rng = np.random.default_rng(3)
    net = _randomize(Network(desc, dtype=np.float64, rng=rng), rng)
    x = _distinct((2, 3, 8, 8), 4) / 55 - 1.0  # distinct values in [−1, 1]: no pool tie
    out = desc["outputs"][0]
    r = rng.standard_normal(net.forward(x, train=train)[out].shape)
    state = [{k: v.copy() for k, v in s.items()} for s in net.state]

    def loss():
        for s, s0 in zip(net.state, state):  # running averages unchanged by the probes
            for k in s:
                s[k][...] = s0[k]
        return net.forward(x, train=train)[out] * r

    loss()
    dx, grads = net.backward({out: r})
    assert check_grad(loss, x, dx) <= 1e-7
    for i, p in enumerate(net.params):
        for k in p:
            assert check_grad(loss, p[k], grads[i][k]) <= 1e-7, (i, k)


def test_map_read_four_times_sums_contributions():
    """δv of a map read by 3 pools and the route = route slice + the 3 pool backwards."""
    layers = [_conv(4),
              {"type": "maxpool", "k": 5, "s": 1, "from": 0},
              {"type": "maxpool", "k": 9, "s": 1, "from": 0},
              {"type": "maxpool", "k": 13, "s": 1, "from": 0},
              _route(3, 2, 1, 0)]
    net = Network(_small(layers, [4]), dtype=np.float64, rng=0)
    x = _distinct((2, 3, 8, 8), 5)
    outs = net.forward(x, train=False, all_outputs=True)
    r = np.random.default_rng(6).standard_normal(outs[4].shape)
    dx, grads = net.backward({4: r})
    r13, r9, r5, r1 = np.split(r, 4, axis=1)
    v = outs[0]
    dv = r1 + sum(maxpool_backward(rk, maxpool_forward(v, k=k, s=1)[1])[0]
                  for rk, k in ((r5, 5), (r9, 9), (r13, 13)))
    net.forward(x, train=False)
    dx_ref, grads_ref = net.backward({0: dv})
    np.testing.assert_allclose(dx, dx_ref, atol=1e-12)
    np.testing.assert_allclose(grads[0]["W"], grads_ref[0]["W"], atol=1e-12)


# ------------------------------------------------------------------------ T34.2
@pytest.mark.slow  # ≈ 1 min
def test_whole_network_gradcheck():
    """Whole YOLOv7-tiny, 2 images 64 × 64, train mode (batch BN): sampled entries of x and
    of parameters from the stem, an ELAN, the SPP-CSP, the PAN and a head."""
    rng = np.random.default_rng(7)
    net = _randomize(build("tiny-yolov7-coco", dtype=np.float64, rng=rng), rng)
    x = rng.uniform(0, 1, (2, 3, 64, 64))
    outs = net.forward(x, train=True)
    rs = {h: rng.standard_normal(o.shape) for h, o in outs.items()}
    state = [{k: v.copy() for k, v in s.items()} for s in net.state]

    def loss():
        for s, s0 in zip(net.state, state):
            for k in s:
                s[k][...] = s0[k]
        o = net.forward(x, train=True)
        return np.concatenate([(o[h] * rs[h]).ravel() for h in rs])

    loss()
    dx, grads = net.backward(rs)
    assert check_grad(loss, x, dx, n_samples=30, rng=0) <= 1e-7
    for i, k in ((0, "W"), (4, "gamma"), (35, "W"), (42, "beta"), (69, "W"), (89, "b")):
        assert check_grad(loss, net.params[i][k], grads[i][k], n_samples=12, rng=i) <= 1e-7


def test_eval_mode_uses_running_statistics():
    rng = np.random.default_rng(8)
    net = _randomize(build("tiny-yolov7-coco", dtype=np.float64, rng=rng), rng)
    x = rng.uniform(0, 1, (2, 3, 64, 64))
    a = net.forward(x, train=False)
    b = net.forward(x[:1], train=False)
    for h in a:  # no batch statistics: an image's output does not depend on the batch
        np.testing.assert_allclose(a[h][:1], b[h], atol=1e-12)


# ------------------------------------------------------------------------ T34.3
def test_v7_loss_on_darknet_heads():
    """`yolo_v5_loss` with the OTA assignment (`hyp.scratch.tiny.yaml`, `loss_ota: 1`) on the
    three Darknet head outputs of YOLOv7-tiny: gradient check through the heads."""
    rng = np.random.default_rng(9)
    net = build("tiny-yolov7-coco", dtype=np.float64, rng=rng)
    levels = v5_levels(net.net)
    assert [s for _, s, _ in levels] == [8, 16, 32]
    outs = {h: rng.standard_normal(o.shape) * 0.5
            for h, o in net.forward(rng.uniform(0, 1, (2, 3, 128, 128)), train=False).items()}
    boxes = [np.array([[0.3, 0.4, 0.2, 0.3], [0.6, 0.6, 0.5, 0.4]]), np.array([[0.5, 0.5, 0.1, 0.1]])]
    labels = [np.array([3, 17]), np.array([0])]
    ref = {h: o.copy() for h, o in outs.items()}
    res = yolo_v5_loss(outs, boxes, labels, levels, 80, HYP_V7_TINY, img_size=640,
                       targets_from=ref)
    assert all(res.masks[h]["matches"] > 0 for h, _, _ in levels[:2])
    for h, _, _ in levels:
        def f():
            return yolo_v5_loss(outs, boxes, labels, levels, 80, HYP_V7_TINY, img_size=640,
                                targets_from=ref).total
        tobj = res.masks[h]["tobj"]
        idx = [(b, a * 85 + c, j, i) for b, a, j, i in zip(*np.nonzero(tobj)) for c in (0, 3, 4, 7)]
        idx += [(0, 4, 0, 0), (1, 89, 1, 2)]
        num = numerical_grad(f, outs[h], idx=idx)
        rows = tuple(np.array(c) for c in zip(*idx))
        assert rel_error(res.douts[h][rows], num[rows]) <= 1e-7
