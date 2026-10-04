"""T1.7 : graphe de réseau, gradcheck du mini-réseau du §11."""

import numpy as np
import pytest

from yolo.models.graph import Network
from yolo.testing.gradcheck import check_grad


def _conv(cout, k=3, act="leaky"):
    return {"type": "conv", "k": k, "s": 1, "cout": cout, "act": act, "bn": act == "leaky"}


# §11 : conv-BN-leaky-pool-conv-BN-leaky-pool(s1)-route-conv1×1.
# La route lit la couche 2, déjà lue par la couche 3 : son gradient est une somme (§4.5).
MINI = {
    "name": "mini",
    "input": (3, 8, 8),
    "layers": [
        _conv(4),                                  # 0
        {"type": "maxpool", "k": 2, "s": 2},       # 1 : 4×4
        _conv(6),                                  # 2
        {"type": "maxpool", "k": 2, "s": 1},       # 3 : 4×4 conservé
        {"type": "route", "from": [3, 2]},         # 4 : 12 canaux
        _conv(5, k=1, act="linear"),               # 5 : tête linéaire avec biais
    ],
}


@pytest.fixture(scope="module")
def mini():
    rng = np.random.default_rng(10)
    net = Network(MINI, dtype=np.float64, rng=rng)
    for p in net.params:  # γ, β et biais non triviaux
        for k in ("gamma", "beta", "b"):
            if k in p:
                p[k][:] = rng.standard_normal(p[k].shape)
    x = rng.standard_normal((2, 3, 8, 8))
    r = rng.standard_normal((2, 5, 4, 4))

    def loss():
        return net.forward(x)[5] * r

    loss()
    dx, grads = net.backward({5: r})
    return net, x, loss, dx, grads


def test_output_and_shapes(mini):
    net, x, *_ = mini
    out = net.forward(x)
    assert list(out) == [5]
    assert out[5].shape == (2, 5, 4, 4)


def test_gradcheck_input(mini):
    _, x, loss, dx, _ = mini
    assert check_grad(loss, x, dx) <= 1e-7


@pytest.mark.parametrize("layer,name", [(0, "W"), (0, "gamma"), (0, "beta"), (2, "W"),
                                        (2, "gamma"), (2, "beta"), (5, "W"), (5, "b")])
def test_gradcheck_params(mini, layer, name):
    net, _, loss, _, grads = mini
    assert check_grad(loss, net.params[layer][name], grads[layer][name]) <= 1e-7


def test_reused_map_accumulates():
    # Sans cumul, la contribution de la route à la couche 2 serait perdue.
    net = Network(MINI, dtype=np.float64, rng=1)
    x = np.random.default_rng(2).standard_normal((2, 3, 8, 8))
    net.forward(x)
    r = np.zeros((2, 5, 4, 4))
    r[0, 0, 0, 0] = 1.0
    _, g_full = net.backward({5: r})
    # Couper la branche maxpool : route vers [2] seule.
    w = net.params[5]["W"].copy()
    net.params[5]["W"][:, :6] = 0
    net.forward(x)
    _, g_route = net.backward({5: r})
    net.params[5]["W"][:] = w
    assert np.abs(g_route[2]["W"]).max() > 0
    assert not np.allclose(g_full[2]["W"], g_route[2]["W"])


def test_inference_uses_running_stats():
    net = Network(MINI, dtype=np.float64, rng=3)
    x = np.random.default_rng(4).standard_normal((2, 3, 8, 8))
    y_eval = net.forward(x, train=False)[5]
    mean0 = net.state[0]["mean"].copy()
    net.forward(x, train=True)
    assert not np.array_equal(mean0, net.state[0]["mean"])
    assert not np.allclose(y_eval, net.forward(x, train=True)[5])


def test_he_init_and_counts():
    net = Network(MINI, dtype=np.float32, rng=0)
    assert net.params[0]["W"].dtype == np.float32
    np.testing.assert_array_equal(net.params[0]["gamma"], 1)
    np.testing.assert_array_equal(net.params[0]["beta"], 0)
    np.testing.assert_array_equal(net.params[5]["b"], 0)
    assert net.num_params(0) == 27 * 4 + 2 * 4
    assert net.num_params(5) == 12 * 5 + 5
    assert net.num_params(1) == 0
