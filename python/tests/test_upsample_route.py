"""T1.6 : upsample ×2 et route."""

import numpy as np

from yolo.layers.route import route_backward, route_forward
from yolo.layers.upsample import upsample_backward, upsample_forward
from yolo.testing.gradcheck import check_layer


def test_upsample_values():
    x = np.arange(4.0).reshape(1, 1, 2, 2)
    y, _ = upsample_forward(x)
    np.testing.assert_array_equal(y[0, 0], [[0, 0, 1, 1], [0, 0, 1, 1],
                                            [2, 2, 3, 3], [2, 2, 3, 3]])


def test_upsample_gradcheck():
    x = np.random.default_rng(10).standard_normal((2, 3, 4, 5))
    errors = check_layer(lambda x, p: upsample_forward(x), upsample_backward, x, {})
    assert errors["x"] <= 1e-7


def test_route_gradcheck():
    rng = np.random.default_rng(1)
    a, b = rng.standard_normal((2, 3, 4, 4)), rng.standard_normal((2, 5, 4, 4))

    def fwd(x, p):
        return route_forward([x, p["b"]])

    def bwd(dy, cache):
        (da, db), _ = route_backward(dy, cache)
        return da, {"b": db}

    errors = check_layer(fwd, bwd, a, {"b": b})
    assert max(errors.values()) <= 1e-7, errors


def test_route_shapes():
    y, cache = route_forward([np.zeros((1, 128, 26, 26)), np.zeros((1, 256, 26, 26))])
    assert y.shape == (1, 384, 26, 26)  # couche 20 de Tiny-YOLOv3
    dxs, _ = route_backward(y, cache)
    assert [d.shape[1] for d in dxs] == [128, 256]
