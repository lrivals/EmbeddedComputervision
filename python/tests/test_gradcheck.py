"""T1.1 : l'outil de gradcheck retrouve le gradient de f(x) = Σ x³."""

import numpy as np

from yolo.testing.gradcheck import check_grad, numerical_grad, rel_error


def test_cube_full():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((3, 4, 5))
    err = check_grad(lambda: x**3, x, 3 * x**2)
    assert err <= 1e-7


def test_cube_sampled():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 16, 13, 13))
    err = check_grad(lambda: x**3, x, 3 * x**2, n_samples=50, rng=2)
    assert err <= 1e-7


def test_input_restored():
    x = np.arange(6, dtype=np.float64).reshape(2, 3)
    ref = x.copy()
    numerical_grad(lambda: x**3, x)
    np.testing.assert_array_equal(x, ref)


def test_detects_wrong_gradient():
    x = np.random.default_rng(3).standard_normal(10)
    assert check_grad(lambda: x**3, x, 2 * x**2) > 0.1


def test_scalar_loss():
    x = np.array([0.5, -1.5, 2.0])
    assert check_grad(lambda: float(np.sum(x**3)), x, 3 * x**2) <= 1e-7


def test_detects_single_wrong_element():
    x = np.random.default_rng(4).standard_normal(1000)
    g = 3 * x**2
    g[123] *= 1.01
    assert check_grad(lambda: x**3, x, g) > 1e-5


def test_rel_error():
    assert rel_error(np.array([1.0, 0.0]), np.array([1.0, 0.0])) == 0.0
    assert np.isclose(rel_error(np.array([2.0]), np.array([1.0])), 0.5)
    assert np.isclose(rel_error(np.array([4.0, 1.0]), np.array([4.0, 2.0])), 0.25)
