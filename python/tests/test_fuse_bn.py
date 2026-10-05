"""T4.1 : réseau à BN fusionnée == réseau avec BN en inférence, à 1e-12 en float64 (§9.1)."""

from pathlib import Path

import numpy as np
import pytest

from yolo.layers.batchnorm import bn_forward
from yolo.layers.conv import conv_forward
from yolo.models.tiny_yolo import build
from yolo.quant.fuse_bn import fuse_bn, fuse_network

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights" / "yolov2-tiny-voc.weights"


def _randomize_bn(net, rng):
    """γ, β, μ_run, σ²_run aléatoires : s_f très différents d'un canal à l'autre."""
    for i, layer in enumerate(net.layers):
        if layer["type"] == "conv" and layer["bn"]:
            c = layer["cout"]
            net.params[i]["gamma"][:] = rng.uniform(0.2, 3.0, c)
            net.params[i]["beta"][:] = rng.normal(0, 0.5, c)
            net.state[i]["mean"][:] = rng.normal(0, 0.3, c)
            net.state[i]["var"][:] = rng.uniform(0.05, 4.0, c)


def test_fuse_single_layer():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 5, 9, 9))
    W = rng.standard_normal((7, 5, 3, 3))
    gamma, beta = rng.uniform(0.1, 2, 7), rng.normal(size=7)
    state = {"mean": rng.normal(size=7), "var": rng.uniform(0.1, 3, 7)}
    y_conv, _ = conv_forward(x, W)
    ref, _ = bn_forward(y_conv, gamma, beta, state, train=False)
    Wf, bf = fuse_bn(W, gamma, beta, state["mean"], state["var"])
    y, _ = conv_forward(x, Wf, bf)
    assert np.abs(y - ref).max() <= 1e-12


def test_fuse_with_conv_bias():
    # §9.1 : b' = β + s (b − μ) quand la conv a déjà un biais.
    rng = np.random.default_rng(1)
    x = rng.standard_normal((1, 3, 6, 6))
    W, b = rng.standard_normal((4, 3, 1, 1)), rng.normal(size=4)
    gamma, beta = rng.uniform(0.1, 2, 4), rng.normal(size=4)
    state = {"mean": rng.normal(size=4), "var": rng.uniform(0.1, 3, 4)}
    ref, _ = bn_forward(conv_forward(x, W, b)[0], gamma, beta, state, train=False)
    Wf, bf = fuse_bn(W, gamma, beta, state["mean"], state["var"], b=b)
    assert np.abs(conv_forward(x, Wf, bf)[0] - ref).max() <= 1e-12


@pytest.mark.parametrize("name", ["tiny-yolov2-voc", "tiny-yolov3-voc"])
def test_fused_network_matches_bn(name):
    rng = np.random.default_rng(2)
    net = build(name, dtype=np.float64, rng=3)
    _randomize_bn(net, rng)
    x = rng.uniform(0, 1, (2, 3, 64, 64))
    ref = net.forward(x, train=False)
    fused = fuse_network(net)
    assert not any(layer.get("bn") for layer in fused.layers)
    out = fused.forward(x, train=False)
    assert ref.keys() == out.keys()
    for k in ref:
        scale = max(1.0, np.abs(ref[k]).max())
        assert np.abs(out[k] - ref[k]).max() <= 1e-12 * scale, k


@pytest.mark.skipif(not WEIGHTS.exists(), reason="poids Darknet absents (make get-weights)")
def test_fused_darknet_weights_same_detections():
    """« mAP inchangée » : mêmes détections avec les poids Darknet (float64)."""
    from yolo.infer.nms import postprocess
    from yolo.io.darknet_weights import load_darknet_weights

    net = build("tiny-yolov2-voc", dtype=np.float64)
    load_darknet_weights(net, WEIGHTS)
    x = np.random.default_rng(4).uniform(0, 1, (1, 3, 416, 416))
    ref = net.forward(x, train=False)
    out = fuse_network(net).forward(x, train=False)
    for k in ref:
        assert np.abs(out[k] - ref[k]).max() <= 1e-9
    for (b0, s0, l0), (b1, s1, l1) in zip(postprocess(ref, net.net, 0.005),
                                          postprocess(out, net.net, 0.005)):
        assert np.array_equal(l0, l1)
        assert np.allclose(b0, b1, atol=1e-9) and np.allclose(s0, s1, atol=1e-12)
