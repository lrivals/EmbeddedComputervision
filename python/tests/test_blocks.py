"""T33.2, T35.1, T35.2: YOLOv5 / YOLOv8 blocks, gradcheck of every block and parameter
(float64, small sizes)."""

import numpy as np
import pytest

from yolo.layers import blocks
from yolo.layers.conv import conv_forward, conv_forward_naive
from yolo.layers.pool import maxpool_backward, maxpool_forward
from yolo.testing.gradcheck import check_layer


def _conv_params(p, prefix, cin, cout, k, rng):
    p[prefix + "W"] = rng.standard_normal((cout, cin, k, k)) / np.sqrt(k * k * cin)
    p[prefix + "gamma"] = rng.random(cout) + 0.5
    p[prefix + "beta"] = rng.standard_normal(cout) * 0.1


def c3_params(c1, c2, n, rng):
    p, ch = {}, c2 // 2
    _conv_params(p, "cv1.", c1, ch, 1, rng)
    _conv_params(p, "cv2.", c1, ch, 1, rng)
    _conv_params(p, "cv3.", 2 * ch, c2, 1, rng)
    for j in range(n):
        _conv_params(p, f"m.{j}.cv1.", ch, ch, 1, rng)
        _conv_params(p, f"m.{j}.cv2.", ch, ch, 3, rng)
    return p


def c2f_params(c1, c2, n, rng):
    p, c = {}, c2 // 2
    _conv_params(p, "cv1.", c1, 2 * c, 1, rng)
    _conv_params(p, "cv2.", (2 + n) * c, c2, 1, rng)
    for j in range(n):
        _conv_params(p, f"m.{j}.cv1.", c, c, 3, rng)
        _conv_params(p, f"m.{j}.cv2.", c, c, 3, rng)
    return p


def detect_v8_params(chs, nc, rng, reg_max=4):
    p = {}
    cb, cc = max(16, chs[0] // 4, reg_max * 4), max(chs[0], min(nc, 100))
    for lv, c in enumerate(chs):
        for br, cm, cout in (("cv2", cb, 4 * reg_max), ("cv3", cc, nc)):
            _conv_params(p, f"{br}.{lv}.0.", c, cm, 3, rng)
            _conv_params(p, f"{br}.{lv}.1.", cm, cm, 3, rng)
            p[f"{br}.{lv}.2.W"] = rng.standard_normal((cout, cm, 1, 1)) / np.sqrt(cm)
            p[f"{br}.{lv}.2.b"] = rng.standard_normal(cout) * 0.1
    return p


def _check(fwd, bwd, x, params):
    errors = check_layer(fwd, bwd, x, params)
    assert set(errors) == {"x", *params}
    assert max(errors.values()) <= 1e-7, errors


@pytest.mark.parametrize("act", ["silu", "leaky"])
@pytest.mark.parametrize("train", [True, False])
def test_conv_bn_act(act, train):
    rng = np.random.default_rng(0)
    p, state = {}, {}
    _conv_params(p, "", 4, 6, 3, rng)
    x = rng.standard_normal((2, 4, 8, 8))
    if not train:
        state[""] = {"mean": rng.standard_normal(6) * 0.1, "var": rng.random(6) + 0.5}
    _check(lambda x, p: blocks.conv_bn_act_forward(x, p, state, act=act, train=train),
           blocks.conv_bn_act_backward, x, p)


@pytest.mark.parametrize("shortcut", [True, False])
def test_bottleneck(shortcut):
    rng = np.random.default_rng(1)
    p, state = {}, {}
    _conv_params(p, "cv1.", 4, 4, 1, rng)
    _conv_params(p, "cv2.", 4, 4, 3, rng)
    x = rng.standard_normal((2, 4, 8, 8))
    _check(lambda x, p: blocks.bottleneck_forward(x, p, state, shortcut=shortcut),
           blocks.bottleneck_backward, x, p)


@pytest.mark.parametrize("n,shortcut", [(1, True), (2, True), (1, False)])
def test_c3(n, shortcut):
    rng = np.random.default_rng(2)
    p, state = c3_params(6, 8, n, rng), {}
    x = rng.standard_normal((2, 6, 8, 8))
    _check(lambda x, p: blocks.c3_forward(x, p, state, n=n, shortcut=shortcut),
           blocks.c3_backward, x, p)


@pytest.mark.parametrize("n", [1, 2])
@pytest.mark.parametrize("shortcut", [True, False])
@pytest.mark.parametrize("train", [True, False])
def test_c2f(n, shortcut, train):
    """T35.1: x and every W, γ, β; y_1 is read by the concat and the first bottleneck."""
    rng = np.random.default_rng(10 + n)
    p, state = c2f_params(6, 8, n, rng), {}
    x = rng.standard_normal((2, 6, 6, 6))
    if not train:
        blocks.c2f_forward(x, p, state, n=n, shortcut=shortcut)  # running statistics
    _check(lambda x, p: blocks.c2f_forward(x, p, state, n=n, shortcut=shortcut, train=train),
           blocks.c2f_backward, x, p)


@pytest.mark.parametrize("train", [True, False])
def test_detect_v8(train):
    """T35.2: two levels (the second at half resolution), box and class branches."""
    rng = np.random.default_rng(13)
    chs, nc = (4, 8), 3
    p, state = detect_v8_params(chs, nc, rng), {}
    x = rng.standard_normal((2, 4, 6, 6))
    x2 = rng.standard_normal((2, 8, 3, 3))
    if not train:
        blocks.detect_v8_forward([x, x2], p, state)

    def fwd(x, p):
        outs, cache = blocks.detect_v8_forward([x, x2], p, state, train=train)
        return np.concatenate([o.ravel() for o in outs]), ([o.shape for o in outs], cache)

    def bwd(dy, cache):
        shapes, c = cache
        sizes = np.cumsum([np.prod(s) for s in shapes])[:-1]
        douts = [d.reshape(s) for d, s in zip(np.split(dy, sizes), shapes)]
        dxs, grads = blocks.detect_v8_backward(douts, c)
        return dxs[0], grads

    _check(fwd, bwd, x, p)
    outs, _ = blocks.detect_v8_forward([x, x2], p, state, train=train)
    assert [o.shape for o in outs] == [(2, 16 + nc, 6, 6), (2, 16 + nc, 3, 3)]


def test_sppf():
    rng = np.random.default_rng(3)
    p, state = {}, {}
    _conv_params(p, "cv1.", 8, 4, 1, rng)
    _conv_params(p, "cv2.", 16, 8, 1, rng)
    x = rng.standard_normal((2, 8, 8, 8))
    _check(lambda x, p: blocks.sppf_forward(x, p, state), blocks.sppf_backward, x, p)


def test_sppf_is_three_chained_pools():
    """x1, x2, x3 = 5×5, 9×9, 13×13 windows of x0 (SPPF identity)."""
    rng = np.random.default_rng(4)
    x0 = rng.standard_normal((1, 2, 16, 16))
    y = x0
    for k in (5, 9, 13):
        y = maxpool_forward(y, k=5, s=1)[0]
        assert np.array_equal(y, maxpool_forward(x0, k=k, s=1)[0])


def test_concat_and_add():
    rng = np.random.default_rng(5)
    xs = [rng.standard_normal((2, c, 4, 4)) for c in (3, 5, 2)]
    y, cache = blocks.concat_forward(xs)
    dy = rng.standard_normal(y.shape)
    dxs, _ = blocks.concat_backward(dy, cache)
    assert [d.shape for d in dxs] == [x.shape for x in xs]
    assert np.array_equal(np.concatenate(dxs, axis=1), dy)
    (da, db), _ = blocks.add_backward(dy, blocks.add_forward(y, y)[1])
    assert da is dy and db is dy


def test_shortcut_gradient_is_sum_of_branches():
    """dx of a bottleneck with shortcut = dx without shortcut (residual branch) + dy."""
    rng = np.random.default_rng(6)
    p = {}
    _conv_params(p, "cv1.", 4, 4, 1, rng)
    _conv_params(p, "cv2.", 4, 4, 3, rng)
    x = rng.standard_normal((2, 4, 8, 8))
    dy = rng.standard_normal(x.shape)
    y1, c1 = blocks.bottleneck_forward(x, p, {}, shortcut=True)
    y0, c0 = blocks.bottleneck_forward(x, p, {}, shortcut=False)
    assert np.allclose(y1, y0 + x, rtol=0, atol=1e-15)
    dx1, g1 = blocks.bottleneck_backward(dy, c1)
    dx0, g0 = blocks.bottleneck_backward(dy, c0)
    assert np.allclose(dx1, dx0 + dy, rtol=0, atol=1e-15)
    assert all(np.array_equal(g1[k], g0[k]) for k in g0)


def test_first_conv_6x6_stride2_pad2():
    """YOLOv5 layer 0: 6×6 / 2, padding 2, == the naive conv; 640 → 320."""
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, 3, 12, 12))
    W = rng.standard_normal((4, 3, 6, 6))
    y, _ = conv_forward(x, W, s=2, pad=2)
    ref, _ = conv_forward_naive(x, W, s=2, pad=2)
    assert y.shape == (2, 4, 6, 6)
    assert np.abs(y - ref).max() <= 1e-12
    assert conv_forward(np.zeros((1, 3, 64, 64)), W, s=2, pad=2)[0].shape[2:] == (32, 32)


def test_sppf_pool_ties():
    """Equal values: the first maximum of the window takes the gradient, as in the forward;
    the gradient mass is conserved."""
    x = np.zeros((1, 1, 6, 6))
    x[0, 0, 2:4, 2:4] = 1.0  # a plateau of four equal maxima
    y, cache = maxpool_forward(x, k=5, s=1)
    dy = np.random.default_rng(8).random(y.shape)
    dx, _ = maxpool_backward(dy, cache)
    assert dx.sum() == pytest.approx(dy.sum(), abs=1e-12)
    # Reference: each window (centred, edge replicated) routes to its first maximum in
    # row-major order.
    ref = np.zeros_like(x)
    for i in range(6):
        for j in range(6):
            win = [(min(max(u, 0), 5), min(max(v, 0), 5))
                   for u in range(i - 2, i + 3) for v in range(j - 2, j + 3)]
            vals = [x[0, 0, u, v] for u, v in win]
            ref[0, 0][win[int(np.argmax(vals))]] += dy[0, 0, i, j]
    assert np.allclose(dx, ref, rtol=0, atol=1e-12)
