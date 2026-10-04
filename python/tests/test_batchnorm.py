"""T1.3 : batch normalization."""

import numpy as np

from yolo.layers.batchnorm import bn_backward, bn_forward, bn_init_state
from yolo.layers.conv import conv_backward, conv_forward
from yolo.testing.gradcheck import check_layer


def _data(rng=10):
    rng = np.random.default_rng(rng)
    x = rng.standard_normal((2, 3, 4, 5)) * 2 + 1
    params = {"gamma": rng.standard_normal(3), "beta": rng.standard_normal(3)}
    return x, params


def test_gradcheck_train():
    x, params = _data()

    def fwd(x, p):
        return bn_forward(x, p["gamma"], p["beta"], bn_init_state(3), train=True)

    errors = check_layer(fwd, bn_backward, x, params)
    assert max(errors.values()) <= 1e-7, errors


def test_gradcheck_inference():
    x, params = _data(1)
    state = {"mean": np.array([0.5, -1.0, 2.0]), "var": np.array([1.5, 0.3, 4.0])}

    def fwd(x, p):
        return bn_forward(x, p["gamma"], p["beta"], state, train=False)

    errors = check_layer(fwd, bn_backward, x, params)
    assert max(errors.values()) <= 1e-7, errors


def test_train_normalizes():
    x, p = _data()
    y, _ = bn_forward(x, np.ones(3), np.zeros(3), bn_init_state(3))
    np.testing.assert_allclose(y.mean(axis=(0, 2, 3)), 0, atol=1e-12)
    np.testing.assert_allclose(y.var(axis=(0, 2, 3)), 1, atol=1e-4)  # ε = 1e-5


def test_running_stats_and_inference():
    x, p = _data(2)
    state = bn_init_state(3)
    m, eps = 0.8, 1e-3
    bn_forward(x, p["gamma"], p["beta"], state, train=True, momentum=m, eps=eps)
    mu, var = x.mean(axis=(0, 2, 3)), x.var(axis=(0, 2, 3))
    np.testing.assert_allclose(state["mean"], (1 - m) * mu, atol=1e-14)
    np.testing.assert_allclose(state["var"], m + (1 - m) * var, atol=1e-14)
    # §4.2, inférence : y = γ (x − μ_run)/√(σ²_run + ε) + β
    y, _ = bn_forward(x, p["gamma"], p["beta"], state, train=False, eps=eps)
    sh = (1, 3, 1, 1)
    ref = (p["gamma"].reshape(sh) * (x - state["mean"].reshape(sh))
           / np.sqrt(state["var"].reshape(sh) + eps) + p["beta"].reshape(sh))
    np.testing.assert_allclose(y, ref, atol=1e-12)
    # L'inférence ne modifie pas les moyennes glissantes.
    before = {k: v.copy() for k, v in state.items()}
    bn_forward(x, p["gamma"], p["beta"], state, train=False)
    np.testing.assert_array_equal(state["mean"], before["mean"])


def test_conv_bias_before_bn_has_zero_gradient():
    # §4.2 : la soustraction de μ_c annule le biais de la convolution.
    rng = np.random.default_rng(3)
    x = rng.standard_normal((2, 3, 6, 6))
    W, b = rng.standard_normal((4, 3, 3, 3)), rng.standard_normal(4)
    y, c_conv = conv_forward(x, W, b)
    z, c_bn = bn_forward(y, rng.standard_normal(4), rng.standard_normal(4), bn_init_state(4))
    dz = rng.standard_normal(z.shape)
    dy, _ = bn_backward(dz, c_bn)
    _, g = conv_backward(dy, c_conv)
    assert np.max(np.abs(g["b"])) < 1e-12
    assert np.max(np.abs(g["W"])) > 1e-3
