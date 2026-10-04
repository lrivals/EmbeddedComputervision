"""T1.2 : convolution, gradcheck et im2col == naïf."""

import numpy as np
import pytest

from yolo.layers.conv import conv_backward, conv_backward_naive, conv_forward, conv_forward_naive
from yolo.testing.gradcheck import check_layer

CASES = [(1, 1), (1, 2), (3, 1), (3, 2)]


def _data(k, rng=10, bias=True):
    rng = np.random.default_rng(rng)
    x = rng.standard_normal((2, 3, 7, 6))
    params = {"W": rng.standard_normal((4, 3, k, k))}
    if bias:
        params["b"] = rng.standard_normal(4)
    return x, params


@pytest.mark.parametrize("k,s", CASES)
@pytest.mark.parametrize("impl", ["im2col", "naive"])
def test_gradcheck(k, s, impl):
    fwd, bwd = (conv_forward, conv_backward) if impl == "im2col" else (
        conv_forward_naive, conv_backward_naive)
    x, params = _data(k)
    errors = check_layer(lambda x, p: fwd(x, p["W"], p["b"], s=s), bwd, x, params)
    assert max(errors.values()) <= 1e-7, errors


def test_gradcheck_no_bias():
    x, params = _data(3, bias=False)
    errors = check_layer(lambda x, p: conv_forward(x, p["W"], s=1), conv_backward, x, params)
    assert max(errors.values()) <= 1e-7, errors


@pytest.mark.parametrize("k,s", CASES)
def test_im2col_matches_naive(k, s):
    x, p = _data(k, rng=1)
    y, cache = conv_forward(x, p["W"], p["b"], s=s)
    y_ref, cache_ref = conv_forward_naive(x, p["W"], p["b"], s=s)
    assert y.shape == y_ref.shape
    np.testing.assert_allclose(y, y_ref, rtol=0, atol=1e-12)
    dy = np.random.default_rng(2).standard_normal(y.shape)
    dx, g = conv_backward(dy, cache)
    dx_ref, g_ref = conv_backward_naive(dy, cache_ref)
    np.testing.assert_allclose(dx, dx_ref, rtol=0, atol=1e-12)
    np.testing.assert_allclose(g["W"], g_ref["W"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(g["b"], g_ref["b"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("k,s,hw", [(3, 1, (13, 13)), (3, 2, (7, 7)), (1, 1, (13, 13))])
def test_output_size(k, s, hw):
    # §4.1 : H_o = ⌊(H + 2P − k)/s⌋ + 1 avec P = k//2
    x = np.zeros((1, 2, 13, 13))
    y, _ = conv_forward(x, np.zeros((5, 2, k, k)), s=s)
    assert y.shape == (1, 5) + hw
