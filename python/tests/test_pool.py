"""T1.5 : max pooling stride 2 et stride 1."""

import numpy as np
import pytest

from yolo.layers.pool import maxpool_backward, maxpool_forward
from yolo.testing.gradcheck import check_layer


def _distinct(shape, rng=0):
    # Valeurs distinctes et espacées : pas d'égalité d'argmax, ni de bascule sous ±h.
    rng = np.random.default_rng(rng)
    return rng.permutation(np.prod(shape)).reshape(shape) * 0.1


@pytest.mark.parametrize("s,hw", [(2, (6, 8)), (2, (7, 7)), (1, (5, 5))])
def test_gradcheck(s, hw):
    x = _distinct((2, 3) + hw)
    errors = check_layer(lambda x, p: maxpool_forward(x, s=s), maxpool_backward, x, {})
    assert errors["x"] <= 1e-7


def test_shapes():
    x = _distinct((1, 4, 13, 13))
    assert maxpool_forward(x, s=1)[0].shape == (1, 4, 13, 13)  # §4.4 : 13×13 conservé
    assert maxpool_forward(x, s=2)[0].shape == (1, 4, 6, 6)
    assert maxpool_forward(_distinct((1, 4, 26, 26)), s=2)[0].shape == (1, 4, 13, 13)


def test_stride1_values():
    x = np.array([[1.0, 5.0, 2.0], [3.0, 0.0, 4.0], [7.0, 6.0, 8.0]])[None, None]
    y, _ = maxpool_forward(x, s=1)
    # Bord droit/bas répliqué : la dernière ligne/colonne ne voit que ses voisins réels.
    np.testing.assert_array_equal(y[0, 0], [[5, 5, 4], [7, 8, 8], [7, 8, 8]])


@pytest.mark.parametrize("s", [1, 2])
def test_gradient_routed_to_argmax_only(s):
    x = _distinct((1, 2, 6, 6), rng=3)
    y, cache = maxpool_forward(x, s=s)
    dx, _ = maxpool_backward(np.ones_like(y), cache)
    # Seules les positions qui sont le max d'au moins une fenêtre reçoivent un gradient.
    winners = np.isin(x, y)
    assert np.all(dx[~winners] == 0)
    assert np.all(dx[winners] > 0)
    assert dx.sum() == y.size
