"""T4.4 : sigmoïde et exponentielle par LUT de 256 entrées, seuil sur le logit (§9.4)."""

import numpy as np
import pytest

from yolo.infer.decode import decode_head, decode_head_int
from yolo.layers.activations import sigmoid
from yolo.quant.lut import (ONE, HeadLuts, exp_lut, logit_threshold_q, lookup,
                            sigmoid_lut, softmax_exp_lut)
from yolo.quant.quantize import quantize

S = 1 / 16
Q = np.arange(-128, 128)
THETAS = [0.001, 0.005, 0.01, 0.1, 0.25, 0.45, 0.5, 0.7, 0.9, 0.99]


def test_lut_shapes():
    for lut in (sigmoid_lut(S), exp_lut(S), softmax_exp_lut(S)):
        assert lut.shape == (256,) and lut.dtype == np.uint32
    assert sigmoid_lut(S).max() < ONE  # σ < 1 : tient sur 16 bits


def test_lut_error_on_grid():
    # Aux entiers q, la seule erreur est l'arrondi Q16.
    assert np.abs(sigmoid_lut(S) / ONE - sigmoid(Q * S)).max() <= 2**-17
    assert np.all(np.abs(exp_lut(S) / ONE - np.exp(Q * S)) <= 2**-17)
    d = np.arange(-255, 1)
    assert np.abs(softmax_exp_lut(S) / ONE - np.exp(d * S)).max() <= 2**-17


def test_sigmoid_error_end_to_end():
    """t réel → q = round(t/s) → table : erreur max σ'(0)·s/2 = 1/128 ≈ 0,0078 au pas 1/16
    (le §9.4 annonce 0,0075 ; l'écart vient de la pente maximale σ'(0) = 1/4)."""
    t = np.linspace(-7.9, 7.9, 400_001)
    err = np.abs(lookup(sigmoid_lut(S), quantize(t, S, 127)) / ONE - sigmoid(t))
    assert err.max() <= 1 / 128 + 2**-17
    assert err.max() == pytest.approx(sigmoid(1 / 32) - 0.5, abs=2**-16)
    # Au pas 1/32 (table 2× plus fine en entrée), l'erreur passe sous 0,0075.
    t2 = np.linspace(-3.9, 3.9, 200_001)
    err2 = np.abs(lookup(sigmoid_lut(S / 2), quantize(t2, S / 2, 127)) / ONE - sigmoid(t2))
    assert err2.max() < 0.0075


@pytest.mark.parametrize("scale", [1 / 16, 1 / 8, 0.0473])
def test_logit_threshold_identical_to_sigmoid_threshold(scale):
    lut = sigmoid_lut(scale)
    for theta in THETAS:
        qmin = logit_threshold_q(theta, scale)
        by_logit = Q >= qmin
        assert np.array_equal(by_logit, sigmoid(Q * scale) > theta), theta
        assert np.array_equal(by_logit, lookup(lut, Q) > theta * ONE), theta


@pytest.mark.parametrize("mode", ["v2", "v3"])
def test_decode_head_int_matches_float(mode):
    rng = np.random.default_rng(0)
    a, c, s = 3, 4, 5
    q = rng.integers(-127, 128, (a * (5 + c), s, s)).astype(np.int8)
    anchors = rng.uniform(0.05, 0.5, (a, 2))
    theta = 0.3
    boxes, obj, scores = decode_head_int(q, anchors, c, mode, HeadLuts(S), theta)
    fb, fo, fs = decode_head(q[None].astype(np.float64) * S, anchors, c, mode)
    keep = fo[0] > theta
    assert keep.sum() == len(obj) > 0
    assert np.abs(obj - fo[0][keep]).max() <= 2**-16
    assert np.abs(scores - fs[0][keep]).max() <= 2e-4
    assert np.abs(boxes[:, :2] - fb[0][keep, :2]).max() <= 2**-16
    rel = np.abs(boxes[:, 2:] / fb[0][keep, 2:] - 1).max()
    assert rel <= 2**-16 * np.exp(127 * S)


@pytest.mark.parametrize("scale", [1 / 16, 1 / 8, 0.149])
def test_exp_lut_fits_uint32(scale):
    from yolo.quant.lut import exp_frac_bits

    f = exp_frac_bits(scale)
    lut = exp_lut(scale)
    assert lut.max() < 2**32 and (f == 16 or lut.max() >= 2**31)
    assert np.abs(lut / 2.0**f - np.exp(Q * scale)).max() <= 2.0**-(f + 1) + 1e-12
    assert exp_frac_bits(1 / 16) == 16
