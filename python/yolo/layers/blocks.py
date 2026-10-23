"""YOLOv5 / YOLOv8 blocks, forward and backward (T33.2).

Same interface as the layers of M1: `*_forward(...) -> (y, cache)`, `*_backward(dy, cache)
-> (dx, grads)`. Parameters are one flat dict named as the torch state dict below a prefix
(`cv1.W`, `cv1.gamma`, `m.0.cv2.beta`, ...); `state` holds the BN running statistics per
conv prefix (`{"cv1.": {"mean", "var"}}`), created on first use and updated in place in
training mode.

- `conv_bn_act`: conv (no bias, padding k // 2 unless given) + BN + SiLU or leaky;
- `bottleneck`: y = x + cv2(cv1(x)) (no add without shortcut), hidden width = c;
- `c3`: cv3([m(cv1(x)), cv2(x)]), cv1 / cv2 1×1 to c2 / 2, m = n bottlenecks (k 1, 3);
- `sppf`: x0 = cv1(x), x_{i+1} = P(x_i) (maxpool k / 1), y = cv2([x0, x1, x2, x3]);
- `concat`, `add`.

A map read twice (the C3 input by cv1 and cv2, x1 of the SPPF by x2 and the concat, a
shortcut input) receives the sum of its gradients. BN: ε = 10⁻³, momentum 0.03 (PyTorch
convention, keep factor 0.97 here). The whole-network version is `yolo.models.graph`
running the flat list of `yolo.models.ul_yaml`; these blocks are its checked units.
"""

from yolo.layers.activations import (leaky_backward, leaky_forward, silu_backward,
                                      silu_forward)
from yolo.layers.batchnorm import bn_backward, bn_forward, bn_init_state
from yolo.layers.conv import conv_backward, conv_forward
from yolo.layers.pool import maxpool_backward, maxpool_forward
from yolo.layers.route import route_backward, route_forward

BN_EPS = 1e-3
BN_MOMENTUM = 0.97  # keep factor: PyTorch momentum 0.03
ACTS = {"silu": (silu_forward, silu_backward), "leaky": (leaky_forward, leaky_backward)}


def conv_bn_act_forward(x, params, state, prefix="", s=1, pad=None, act="silu", train=True):
    W = params[prefix + "W"]
    y, c_conv = conv_forward(x, W, None, s=s, pad=pad)
    st = state.setdefault(prefix, bn_init_state(W.shape[0], W.dtype))
    y, c_bn = bn_forward(y, params[prefix + "gamma"], params[prefix + "beta"], st, train=train,
                         momentum=BN_MOMENTUM, eps=BN_EPS, unbiased=True)
    y, c_act = ACTS[act][0](y)
    return y, (prefix, act, c_conv, c_bn, c_act)


def conv_bn_act_backward(dy, cache):
    prefix, act, c_conv, c_bn, c_act = cache
    dy, _ = ACTS[act][1](dy, c_act)
    dy, g_bn = bn_backward(dy, c_bn)
    dx, g_conv = conv_backward(dy, c_conv)
    return dx, {prefix + "W": g_conv["W"], prefix + "gamma": g_bn["gamma"],
                prefix + "beta": g_bn["beta"]}


def add_forward(a, b):
    return a + b, None


def add_backward(dy, cache):
    """The gradient of a sum goes unchanged to both terms. Returns ([da, db], {})."""
    return [dy, dy], {}


def concat_forward(xs):
    return route_forward(xs)


def concat_backward(dy, cache):
    """Splits dy along C by the source widths. Returns (list of dx, {})."""
    return route_backward(dy, cache)


def bottleneck_forward(x, params, state, prefix="", shortcut=True, k=(1, 3), train=True):
    a, c1 = conv_bn_act_forward(x, params, state, prefix + "cv1.", pad=k[0] // 2, train=train)
    y, c2 = conv_bn_act_forward(a, params, state, prefix + "cv2.", pad=k[1] // 2, train=train)
    if shortcut:
        y = y + x
    return y, (shortcut, c1, c2)


def bottleneck_backward(dy, cache):
    shortcut, c1, c2 = cache
    da, g2 = conv_bn_act_backward(dy, c2)
    dx, g1 = conv_bn_act_backward(da, c1)
    if shortcut:
        dx = dx + dy
    return dx, {**g1, **g2}


def c3_forward(x, params, state, prefix="", n=1, shortcut=True, train=True):
    a, c_cv1 = conv_bn_act_forward(x, params, state, prefix + "cv1.", train=train)
    c_m = []
    for j in range(n):
        a, c = bottleneck_forward(a, params, state, f"{prefix}m.{j}.", shortcut, train=train)
        c_m.append(c)
    b, c_cv2 = conv_bn_act_forward(x, params, state, prefix + "cv2.", train=train)
    cat, c_cat = concat_forward([a, b])
    y, c_cv3 = conv_bn_act_forward(cat, params, state, prefix + "cv3.", train=train)
    return y, (c_cv1, c_m, c_cv2, c_cat, c_cv3)


def c3_backward(dy, cache):
    c_cv1, c_m, c_cv2, c_cat, c_cv3 = cache
    dcat, grads = conv_bn_act_backward(dy, c_cv3)
    (da, db), _ = concat_backward(dcat, c_cat)
    dx, g = conv_bn_act_backward(db, c_cv2)
    grads.update(g)
    for c in reversed(c_m):
        da, g = bottleneck_backward(da, c)
        grads.update(g)
    dx1, g = conv_bn_act_backward(da, c_cv1)
    grads.update(g)
    return dx + dx1, grads


def sppf_forward(x, params, state, prefix="", k=5, train=True):
    xs, c_pool = [None], []
    xs[0], c_cv1 = conv_bn_act_forward(x, params, state, prefix + "cv1.", train=train)
    for _ in range(3):
        y, c = maxpool_forward(xs[-1], k=k, s=1)
        xs.append(y)
        c_pool.append(c)
    cat, c_cat = concat_forward(xs)
    y, c_cv2 = conv_bn_act_forward(cat, params, state, prefix + "cv2.", train=train)
    return y, (c_cv1, c_pool, c_cat, c_cv2)


def sppf_backward(dy, cache):
    c_cv1, c_pool, c_cat, c_cv2 = cache
    dcat, grads = conv_bn_act_backward(dy, c_cv2)
    ds = list(concat_backward(dcat, c_cat)[0])
    # x_i feeds the concat and the next pool: walk the chain backwards, summing.
    for i in reversed(range(3)):
        ds[i] = ds[i] + maxpool_backward(ds[i + 1], c_pool[i])[0]
    dx, g = conv_bn_act_backward(ds[0], c_cv1)
    grads.update(g)
    return dx, grads
