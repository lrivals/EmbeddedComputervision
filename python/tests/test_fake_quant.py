"""T9.3.2 : fake-quant, STE et pas d'activation puissance de 2 appris ; réseau QAT."""

import math

import numpy as np
import pytest

from yolo.models.tiny_yolo import build
from yolo.quant.fake_quant import (QATNetwork, fq_act_backward, fq_act_forward, fq_weight,
                                   pow2_act_steps, pow2_step, qmax_of_bits)
from yolo.quant.fuse_bn import fuse_network
from yolo.quant.int_model import IntNetwork, QuantModel
from yolo.quant.quantize import INPUT_SCALE, quantize_input, quantize_weights_per_channel
from yolo.testing.gradcheck import numerical_grad, rel_error


def test_qmax_and_pow2_step():
    assert qmax_of_bits(4) == 7 and qmax_of_bits(8) == 127
    # arrondi ⌊k + ½⌋ : −2,5 → −2
    assert pow2_step(np.array([-2.5])) == 0.25 and pow2_step(np.array([-2.6])) == 0.125


@pytest.mark.parametrize("qmax", [7, 127])
def test_fq_weight_matches_integer_quantization(qmax):
    W = np.random.default_rng(0).standard_normal((6, 3, 3, 3))
    qW, sw = quantize_weights_per_channel(W, qmax)
    assert np.abs(qW).max() <= qmax
    assert np.array_equal(fq_weight(W, qmax), qW * sw[:, None, None, None])


def test_fq_act_forward_grid_and_clip():
    x = np.array([-10.0, -0.3, -0.125, 0.0, 0.12, 0.13, 0.874, 0.876, 5.0])
    y, _ = fq_act_forward(x, np.array([-3.0]), 7)  # s = 1/8, plage ±7/8
    assert np.allclose(y, [-0.875, -0.25, -0.125, 0, 0.125, 0.125, 0.875, 0.875, 0.875])


@pytest.mark.parametrize("leaky", [False, True])
def test_ste_gradients_match_surrogate(leaky):
    """Gradcheck de la fonction de substitution de l'estimateur : arrondis remplacés par
    leur développement d'ordre 1 au point courant, s = 2^k continu. Points loin des bords
    d'arrondi et de la bande |v| ∈ [qmax − ½, qmax + ½] (où r = ±qmax tombe sur le bord de
    l'écrêtage de la fonction de substitution)."""
    rng = np.random.default_rng(1)
    qmax, k0 = 7, -2.0
    u = rng.uniform(-12, 12, 400)
    u = u[(np.abs(np.abs(u) - qmax) > 0.6) & (np.abs(u - np.floor(u) - 0.5) > 0.05)]
    x = u * 2.0**k0
    y, cache = fq_act_forward(x, np.array([k0]), qmax, leaky)
    v0, r0, a = cache[0].copy(), cache[1].copy(), cache[2]

    def surrogate(xv, k):
        s = 2.0 ** k
        return s * np.clip(a * xv / s + (r0 - v0), -qmax, qmax)

    assert np.allclose(surrogate(x, k0), y)
    dy = rng.standard_normal(x.shape)
    dx, dk = fq_act_backward(dy, cache)
    xs = x.copy()
    num_dx = numerical_grad(lambda: np.sum(dy * surrogate(xs, k0)), xs)
    assert rel_error(dx, num_dx) < 1e-7
    kk = np.array([k0])
    num_dk = numerical_grad(lambda: np.sum(dy * surrogate(x, kk[0])), kk)
    assert rel_error(np.array([dk]), num_dk) < 1e-7


def test_leaky_rounding_matches_integer_layer():
    from yolo.quant.int_layers import leaky_int

    r = np.arange(-200, 50)
    y, _ = fq_act_forward(r.astype(np.float64), np.array([0.0]), 127, leaky=True)
    assert np.array_equal(y, np.clip(leaky_int(r), -127, 127))


def test_pow2_act_steps_picks_mse_minimum():
    v = np.random.default_rng(2).standard_normal(20000)
    j = pow2_act_steps(lambda i: v, [0], lambda i: 7)[0]
    errs = {jj: np.mean((v - np.clip(np.floor(v / 2.0**jj + 0.5), -7, 7) * 2.0**jj) ** 2)
            for jj in range(j - 2, j + 3)}
    assert min(errs, key=errs.get) == j


def _qat_tiny(bits=4):
    net = build("tiny-yolov2-voc", dtype=np.float64, rng=0)
    fused = fuse_network(net)
    convs = [i for i, layer in enumerate(fused.layers) if layer["type"] == "conv"]
    head = convs[-1]
    q = qmax_of_bits(bits)
    w_qmax = {i: q for i in convs}
    a_qmax = {i: q for i in convs if i != head}
    steps = {i: -2.0 for i in a_qmax}
    return fused, QATNetwork(fused, w_qmax, a_qmax, steps, {head: 1 / 8}), head


def test_qat_network_close_to_integer_model():
    """La passe QAT (flottante) et le modèle entier exporté de la même QAT donnent presque
    les mêmes sorties : seuls diffèrent les arrondis flottants de la convolution et de M0
    (rares écarts d'un pas, qui se propagent)."""
    fused, qat, head = _qat_tiny()
    x = np.random.default_rng(3).uniform(0, 1, (1, 3, 64, 64))
    y_qat = qat.forward(x, train=False)
    qm = QuantModel.from_fused(qat, INPUT_SCALE, qat.act_scales(), w_qmax=qat.w_qmax,
                               a_qmax=qat.a_qmax)
    for i, c in qm.convs.items():
        assert c.qmax == (127 if i == head else 7) and np.abs(c.qW).max() <= 7
    y_int = IntNetwork(qm, "int64").forward(quantize_input(x))
    for h in y_int:
        steps = np.round(y_qat[h] * 8).astype(np.int64) - y_int[h].astype(np.int64)
        assert np.mean(steps == 0) > 0.95


def test_qat_backward_has_step_gradients():
    fused, qat, head = _qat_tiny()
    x = np.random.default_rng(4).uniform(0, 1, (1, 3, 64, 64))
    out = qat.forward(x)
    _, grads = qat.backward({h: np.ones_like(v) for h, v in out.items()})
    for i in qat.a_qmax:
        assert grads[i]["log2_s"].shape == (1,) and math.isfinite(grads[i]["log2_s"][0])
    assert "log2_s" not in grads[head]


def test_layer_qmax_keeps_int8_layers():
    """T12.4 : w4a4 avec la 1re et la dernière conv en INT8 (poids et sortie)."""
    from yolo.models.tiny_yolo import load_cfg
    from yolo.quant.lowbit import head_convs, layer_qmax

    net = load_cfg("tiny-yolov2-voc")
    convs = [i for i, layer in enumerate(net["layers"]) if layer["type"] == "conv"]
    first, last = convs[0], convs[-1]
    w, a = layer_qmax(net, 4, 4, int8_layers=(first, last))
    assert w[first] == w[last] == a[first] == 127
    assert set(head_convs(net)) == {last} and a[last] == 127
    for i in convs[1:-1]:
        assert w[i] == a[i] == 7
    assert layer_qmax(net, 4, 4) == ({i: 7 for i in convs},
                                     {i: 127 if i == last else 7 for i in convs})
    with pytest.raises(ValueError, match="pas des convs"):
        layer_qmax(net, 4, 4, int8_layers=(1,))  # couche 1 : maxpool
