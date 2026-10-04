"""T1.4 et T1.6 : leaky ReLU, sigmoïde."""

import numpy as np

from yolo.layers.activations import (
    leaky_backward,
    leaky_forward,
    sigmoid,
    sigmoid_backward,
    sigmoid_forward,
)
from yolo.testing.gradcheck import check_layer


def test_leaky_gradcheck():
    rng = np.random.default_rng(10)
    x = rng.standard_normal((2, 3, 4, 4))
    x[np.abs(x) < 1e-3] = 0.5  # §4.3 : φ non dérivable en 0
    errors = check_layer(lambda x, p: leaky_forward(x), leaky_backward, x, {})
    assert errors["x"] <= 1e-7


def test_leaky_values():
    y, _ = leaky_forward(np.array([-2.0, 0.0, 3.0]))
    np.testing.assert_array_equal(y, [-0.2, 0.0, 3.0])


def test_sigmoid_gradcheck():
    x = np.random.default_rng(1).standard_normal((2, 5)) * 4
    errors = check_layer(lambda x, p: sigmoid_forward(x), sigmoid_backward, x, {})
    assert errors["x"] <= 1e-7


def test_sigmoid_stable():
    with np.errstate(over="raise", under="ignore"):
        y = sigmoid(np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0]))
    assert np.all(np.isfinite(y))
    np.testing.assert_allclose(y, [0.0, 1 / (1 + np.exp(50)), 0.5, 1 / (1 + np.exp(-50)), 1.0],
                               rtol=1e-15, atol=0)
