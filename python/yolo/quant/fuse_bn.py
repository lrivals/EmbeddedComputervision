"""Fusion de la BN dans la convolution qui la précède — §9.1.

    s_f = γ_f / √(σ²_f + ε),   W'_f = s_f W_f,   b'_f = β_f + s_f (b_f − μ_f)

Le réseau fusionné est un `Network` de même graphe dont les convolutions n'ont plus de BN
(`bn = False`, donc un biais) : sa passe avant d'inférence est celle du réseau d'origine.
"""

import copy

import numpy as np

from yolo.models.graph import Network


def fuse_bn(W, gamma, beta, mean, var, eps=1e-5, b=None):
    """§9.1 : (W', b') de la convolution suivie de la BN d'inférence, en float64."""
    W = np.asarray(W, dtype=np.float64)
    s = np.asarray(gamma, dtype=np.float64) / np.sqrt(np.asarray(var, dtype=np.float64) + eps)
    b = np.zeros(W.shape[0]) if b is None else np.asarray(b, dtype=np.float64)
    return W * s[:, None, None, None], beta + s * (b - np.asarray(mean, dtype=np.float64))


def fuse_network(net, dtype=np.float64):
    """`Network` sans BN, à paramètres fusionnés, en `dtype` (float64 par défaut)."""
    desc = copy.deepcopy(net.net)
    for layer in desc["layers"]:
        if layer["type"] == "conv":
            layer["bn"] = False
    fused = Network(desc, dtype=dtype, rng=0)
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
