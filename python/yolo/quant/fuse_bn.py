"""Fusion de la BN dans la convolution qui la précède — §9.1.

    s_f = γ_f / √(σ²_f + ε),   W'_f = s_f W_f,   b'_f = β_f + s_f (b_f − μ_f)

Le réseau fusionné est un `Network` de même graphe dont les convolutions n'ont plus de BN
(`bn = False`, donc un biais) : sa passe avant d'inférence est celle du réseau d'origine.

It is also the graph the engine computes (T26.0): the stride-1 maxpools wider than
`POOL_K_MAX` of a SPP (YOLOv7-tiny: 5 / 9 / 13 of one map, each read through a single-source
route) are rewritten with the SPPF identity pool_a ∘ pool_b = pool_{a+b−1}: the route before
the pool becomes an exact identity 1×1 conv (W = I, b = 0) reading the previous pool, the pool
becomes a 5×5 one. Same layer count and numbering, same outputs (`lower_spp`).
"""

import copy

import numpy as np

from yolo.models.graph import Network
from yolo.models.specs import infer_shapes


def fuse_bn(W, gamma, beta, mean, var, eps=1e-5, b=None):
    """§9.1 : (W', b') de la convolution suivie de la BN d'inférence, en float64."""
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(gamma, dtype=np.float64) / np.sqrt(np.asarray(var, dtype=np.float64) + eps)
    b = np.zeros(W.shape[0]) if b is None else np.asarray(b, dtype=np.float64)
    return W * s[:, None, None, None], beta + s * (b - np.asarray(mean, dtype=np.float64))


POOL_K_MAX = 5  # widest stride-1 maxpool of the engine output stage (T26.0)


def lower_spp(layers, shapes):
    """Rewrites in place the SPP pools wider than `POOL_K_MAX` (see the module docstring).

    A stride-1 pool i of size k reading a single-source route i − 1 of X, right after a
    stride-1 pool i − 2 that covers k' of X: route i − 1 → identity conv reading i − 2, pool
    i → k − k' + 1. Returns {identity conv id: channels}.
    """
    cover = {}  # stride-1 pool -> (base map, window it covers on it)
    identity = {}
    for i, layer in enumerate(layers):
        if layer["type"] != "maxpool" or layer["s"] != 1:
            continue
        prev = layers[i - 1] if i > 0 else {"type": None}
        view = prev["type"] == "route" and len(prev["from"]) == 1 and prev.get("groups", 1) == 1
        base = prev["from"][0] if view else i - 1
        k = layer["k"]
        if k > POOL_K_MAX:
            k2 = k - cover.get(i - 2, (None, 0))[1] + 1
            if not view or cover.get(i - 2, (None,))[0] != base or not 2 <= k2 <= POOL_K_MAX:
                raise ValueError(f"maxpool {i} : {k}×{k} > {POOL_K_MAX} hors d'un SPP")
            layers[i - 1] = {"type": "conv", "k": 1, "s": 1, "cout": shapes[i][1][0],
                             "act": "linear", "bn": False, "identity": True}
            identity[i - 1] = shapes[i][1][0]
            layer["k"] = k2
        cover[i] = (base, k)
    return identity


def engine_net(net):
    """Copy of the description `net` (format `specs`) as the engine computes it (`lower_spp`)."""
    desc = copy.deepcopy(net)
    lower_spp(desc["layers"], infer_shapes(desc))
    return desc


def fuse_network(net, dtype=np.float64):
    """`Network` sans BN, à paramètres fusionnés, en `dtype` (float64 par défaut) ; SPP
    réécrit pour le moteur (`lower_spp`)."""
    desc = copy.deepcopy(net.net)
    for layer in desc["layers"]:
        if layer["type"] == "conv":
            layer["bn"] = False
    identity = lower_spp(desc["layers"], net.shapes)
    fused = Network(desc, dtype=dtype, rng=0)
    for i, c in identity.items():
        fused.params[i] = {"W": np.eye(c, dtype=dtype)[:, :, None, None],
                           "b": np.zeros(c, dtype=dtype)}
    for i, layer in enumerate(net.layers):
        if layer["type"] != "conv":
            continue
        p = net.params[i]
        if layer["bn"]:
            st = net.state[i]
            W, b = fuse_bn(p["W"], p["gamma"], p["beta"], st["mean"], st["var"], net.bn_eps)
        else:
            W, b = np.asarray(p["W"], dtype=np.float64), np.asarray(p["b"], dtype=np.float64)
        fused.params[i] = {"W": W.astype(dtype), "b": b.astype(dtype)}
    return fused
