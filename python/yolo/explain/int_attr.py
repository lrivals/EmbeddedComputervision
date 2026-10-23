"""Exact integer attribution (T36.3): contributions qW·qx and LRP over the integer network.

**Exact part.** The int32 accumulator of an output neuron (f, i, j) of a conv is

    acc[f, i, j] = Σ_{c, u, v} qW[f, c, u, v] · qx[c, i·s + u − p, j·s + v − p] + qb[f]

`contributions` returns these int8 × int8 products in int64, summed per input channel and
per input pixel; Σ == acc − qb holds **exactly**. Integer == golden == C-sim (M28), so this
split is the one of the circuit.

**Propagation** (LRP, Bach 2015; rules of Montavon 2019). The relevance starts at 1 on the
explained cell, split equally between the t_o and t_c neurons of its anchor in the head conv,
and goes down to the input:

- conv, ε rule: R_x = x ⊙ Wᵀ(R / (z + ε·sign z)), z = acc (bias included: what the bias
  takes is lost); ε = `eps` × the standard deviation of z over the layer;
- conv, z⁺ rule: only the positive products, R_x = x⁺ ⊙ W⁺ᵀ(R/Z⁺) + x⁻ ⊙ W⁻ᵀ(R/Z⁺),
  Z⁺ = W⁺x⁺ + W⁻x⁻ (bias ignored, conservative where Z⁺ > 0);
- requantization (scale M0 / 2ⁿ), leaky and clip: the relevance of the output passes to the
  accumulator unchanged (an approximation, measured as the per-layer leak);
- maxpool: winner takes all (argmax of the int input, first position on ties); route: split
  by channels; upsample: summed back over the 2 × 2 copies; a map read twice sums the two.

The same propagation runs on the folded float network (`lrp_float`) for the comparison.
"""

from dataclasses import dataclass, field

import numpy as np

from yolo.layers.conv import conv_backward, conv_forward
from yolo.layers.pool import maxpool_backward, maxpool_forward
from yolo.layers.route import route_backward, route_forward
from yolo.layers.upsample import upsample_backward

OUTPUT_TYPES = ("yolo", "region")


class KeepInputs:
    """`IntNetwork` hook: keeps the int64 input `x[i]`, accumulator `acc[i]` and output
    `y[i]` of every conv of the last forward pass."""

    def __init__(self):
        self.x, self.acc, self.y = {}, {}, {}

    def on_acc(self, i, acc, x):
        self.x[i], self.acc[i] = x, acc
        return acc

    def on_out(self, i, y):
        self.y[i] = y
        return y


def contributions(qx, qW, f, i, j, s=1):
    """Products qW·qx (int64) of output neuron (f, i, j) of a conv with input `qx` (C, H, W):
    returns (per channel (C,), per input pixel (H, W), total)."""
    qx = np.asarray(qx, dtype=np.int64)
    qW = np.asarray(qW, dtype=np.int64)
    c, h, w = qx.shape
    k = qW.shape[2]
    p = k // 2
    xp = np.pad(qx, ((0, 0), (p, p), (p, p)))
    patch = xp[:, i * s:i * s + k, j * s:j * s + k]
    prod = qW[f] * patch  # (C, k, k)
    pix = np.zeros((h + 2 * p, w + 2 * p), dtype=np.int64)
    pix[i * s:i * s + k, j * s:j * s + k] = prod.sum(axis=0)
    return prod.sum(axis=(1, 2)), pix[p:p + h, p:p + w], int(prod.sum())


def check_exact(hook, convs, n=16, rng=0):
    """For `n` random neurons of every conv of the last forward: Σ contributions == acc − qb.
    Returns {conv id: number of mismatches} (all 0 expected)."""
    rng = np.random.default_rng(rng)
    out = {}
    for i, acc in hook.acc.items():
        c = convs[i]
        _, f_, h, w = acc.shape
        bad = 0
        for _ in range(n):
            f, y, x = (int(rng.integers(v)) for v in (f_, h, w))
            _, pix, total = contributions(hook.x[i][0], c.qW, f, y, x, c.s)
            bad += int(total != int(acc[0, f, y, x]) - int(c.qb[f])
                       or int(pix.sum()) != total)
        out[i] = bad
    return out


def head_conv(layers, t):
    """Id of the conv read by the target's head (the `yolo` / `region` layer reads `id − 1`
    in Darknet nets)."""
    return t.head - 1 if layers[t.head]["type"] in OUTPUT_TYPES else t.head


def start_relevance(shape, classes, t):
    """Relevance on the head conv output: 1/2 on t_o and 1/2 on t_c of the target's cell."""
    r = np.zeros(shape)
    a0 = t.anchor * (5 + classes)
    r[0, a0 + 4, t.i, t.j] = 0.5
    r[0, a0 + 5 + t.cls, t.i, t.j] = 0.5
    return r


def _sources(layers, i):
    from yolo.models.specs import layer_sources

    return layer_sources(i, layers[i])


def _conv_relevance(x, W, z, r, s, rule, eps):
    """Relevance of the input `x` (1, C, H, W) of a conv from that of its accumulator `z`."""
    if rule == "eps":
        e = eps * (z.std() or 1.0)
        q = r / (z + e * np.where(z >= 0, 1.0, -1.0))
        _, cache = conv_forward(x, W, s=s)
        return x * conv_backward(q, cache)[0]
    if rule == "zplus":
        xp, xn = np.maximum(x, 0), np.minimum(x, 0)
        wp, wn = np.maximum(W, 0), np.minimum(W, 0)
        zp = conv_forward(xp, wp, s=s)[0] + conv_forward(xn, wn, s=s)[0]
        q = np.where(zp > 0, r / np.where(zp > 0, zp, 1.0), 0.0)
        return (xp * conv_backward(q, conv_forward(xp, wp, s=s)[1])[0]
                + xn * conv_backward(q, conv_forward(xn, wn, s=s)[1])[0])
    raise ValueError(f"rule: {rule!r}")


@dataclass
class Relevance:
    """`map` (H, W): input relevance summed over the channels; `input` (C, H, W);
    `conservation` {conv id: Σ R_in / Σ R_out}; `total` Σ map (1 for a lossless chain);
    `head_channels`: per-channel relevance of the head conv's input (the L13 split)."""

    map: np.ndarray
    input: np.ndarray
    conservation: dict = field(default_factory=dict)
    total: float = 0.0
    head_channels: np.ndarray = None


def _propagate(layers, classes, t, inputs, out_of, conv_of, rule, eps):
    """Generic LRP. `inputs(j)`: output of layer j as float64 (j = −1: the network input);
    `out_of(i)`: shape of layer i's output; `conv_of(i)`: (W, b, z) of conv i."""
    hc = head_conv(layers, t)
    rel = {hc: start_relevance(out_of(hc), classes, t)}
    cons, head_ch = {}, None
    for i in range(hc, -1, -1):
        r = rel.pop(i, None)
        if r is None:
            continue
        layer = layers[i]
        typ = layer["type"]
        src = _sources(layers, i)
        if typ == "conv":
            W, _, z = conv_of(i)
            x = inputs(src[0])
            rx = _conv_relevance(x, W, z, r, layer["s"], rule, eps)
            tot = r.sum()
            cons[i] = float(rx.sum() / tot) if tot else 1.0
            if i == hc:
                head_ch = rx[0].sum(axis=(1, 2))
            rxs = [rx]
        elif typ == "maxpool":
            _, cache = maxpool_forward(inputs(src[0]), k=layer["k"], s=layer["s"])
            rxs = [maxpool_backward(r, cache)[0]]
        elif typ == "upsample":
            rxs = [upsample_backward(r, layer["s"])[0]]
        elif typ == "route":
            _, cache = route_forward([inputs(j) for j in src], layer.get("groups", 1),
                                     layer.get("group_id", 0))
            rxs = route_backward(r, cache)[0]
        else:
            rxs = [r]
        for j, rx in zip(src, rxs):
            rel[j] = rel[j] + rx if j in rel else rx
    rin = rel[-1][0]
    return Relevance(rin.sum(axis=0), rin, cons, float(rin.sum()), head_ch)


def lrp_int(inet, qx, t, rule="eps", eps=0.25, outs=None):
    """LRP of target `t` over the integer network `inet` for the int8 input `qx` (1, 3, H, W);
    `outs`: the int outputs of every layer if already computed."""
    from yolo.quant.int_layers import conv_acc

    qm = inet.qm
    layers = qm.layers
    outs = inet.forward(qx, all_outputs=True) if outs is None else outs

    def inputs(j):
        return np.asarray(qx if j < 0 else outs[j], dtype=np.float64)

    def conv_of(i):
        c = qm.convs[i]
        acc = conv_acc(inputs(_sources(layers, i)[0]).astype(np.int64), c.qW, c.qb, "int64",
                       c.s)
        return c.qW.astype(np.float64), c.qb, acc.astype(np.float64)

    return _propagate(layers, qm.net["classes"], t, inputs, lambda i: outs[i].shape, conv_of,
                      rule, eps)


def lrp_float(net, x, t, rule="eps", eps=0.25, acts=None):
    """The same propagation over the folded float `Network` `net`, input `x` (1, 3, H, W)."""
    acts = net.forward(x, train=False, all_outputs=True) if acts is None else acts

    def inputs(j):
        return np.asarray(x if j < 0 else acts[j], dtype=np.float64)

    def conv_of(i):
        p = net.params[i]
        W = np.asarray(p["W"], dtype=np.float64)
        b = np.asarray(p["b"], dtype=np.float64)
        z = conv_forward(inputs(net.inputs[i][0]), W, b, s=net.layers[i]["s"])[0]
        return W, b, z

    return _propagate(net.layers, net.net["classes"], t, inputs, lambda i: acts[i].shape,
                      conv_of, rule, eps)


def cam_l13(qm, outs, t):
    """Integer CAM of the 1 × 1 linear head (T36.10 preview): Σ_c qW[f, c]·L[c, :, :] for the
    neuron f = t_c of the target's anchor, over the map L read by the head conv; int64."""
    hc = head_conv(qm.layers, t)
    c = qm.convs[hc]
    if c.qW.shape[2] != 1:
        raise ValueError("head conv is not 1 × 1")
    f = t.anchor * (5 + qm.net["classes"]) + 5 + t.cls
    src = np.asarray(outs[_sources(qm.layers, hc)[0]][0], dtype=np.int64)
    return np.tensordot(c.qW[f, :, 0, 0].astype(np.int64), src, axes=1)

