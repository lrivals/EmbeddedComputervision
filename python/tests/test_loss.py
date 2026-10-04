"""T2.4 / T2.5 : perte YOLOv3 (§6.2), variante softmax v2, gradcheck sur le mini-réseau §11."""

import numpy as np
import pytest

from yolo.data.targets import build_targets, heads
from yolo.layers.activations import sigmoid
from yolo.models.graph import Network
from yolo.testing.gradcheck import check_grad
from yolo.train.loss import yolo_loss

C = 3                       # classes
A = 2                       # ancres par cellule
ANCHORS = [[60, 70], [170, 150]]  # px pour 416 → 0,14 et 0,38 de l'image
IGNORE = 0.3


def _conv(cout, k=3, act="leaky"):
    return {"type": "conv", "k": k, "s": 1, "cout": cout, "act": act, "bn": act == "leaky"}


def mini_net(out_type):
    # §11 : conv-BN-leaky-pool-conv-BN-leaky-pool(s1)-route-conv1×1, puis la sortie.
    out = ({"type": "yolo", "mask": [0, 1]} if out_type == "yolo"
           else {"type": "region", "num": A})
    return {
        "name": "mini-" + out_type,
        "input": (3, 16, 16),
        "classes": C,
        "anchors": ANCHORS,
        "layers": [
            _conv(4),                                  # 0
            {"type": "maxpool", "k": 2, "s": 2},       # 1 : 8×8
            _conv(6),                                  # 2
            {"type": "maxpool", "k": 2, "s": 1},       # 3 : 8×8 conservé
            {"type": "route", "from": [3, 2]},         # 4
            _conv(A * (5 + C), k=1, act="linear"),     # 5 : tête
            out,                                       # 6
        ],
    }


# 2 images, 3 objets (§11).
GT_BOXES = [
    np.array([[0.30, 0.40, 0.15, 0.12], [0.70, 0.62, 0.40, 0.36]]),
    np.array([[0.55, 0.25, 0.35, 0.42]]),
]
GT_LABELS = [np.array([0, 2]), np.array([1])]


def _setup(out_type, class_mode, seed):
    rng = np.random.default_rng(seed)
    desc = mini_net(out_type)
    net = Network(desc, dtype=np.float64, rng=rng)
    for p in net.params:
        for k in ("gamma", "beta"):
            if k in p:
                p[k][:] = 1 + 0.3 * rng.standard_normal(p[k].shape)
        if "b" in p:
            p["b"][:] = 0.3 * rng.standard_normal(p["b"].shape)
    net.params[5]["W"] *= 0.2  # boîtes prédites proches des ancres : des ancres sont ignorées
    x = rng.standard_normal((2, 3, 16, 16))
    hl = heads(desc)

    def run():
        return yolo_loss(net.forward(x), GT_BOXES, GT_LABELS, ANCHORS, hl, C,
                         class_mode=class_mode, ignore_thresh=IGNORE)

    def loss():
        return np.concatenate([t.ravel() for t in run().terms.values()])

    res = run()
    dx, grads = net.backward(res.douts)
    return net, x, res, loss, dx, grads


CASES = [("yolo", "sigmoid"), ("region", "softmax")]


@pytest.fixture(scope="module", params=CASES, ids=["v3", "v2-softmax"])
def setup(request):
    return _setup(*request.param, seed=3)


def test_masks_all_present(setup):
    _, _, res, *_ = setup
    m = res.masks[6]
    obj, ign = m["obj"], m["ignore"]
    assert obj.sum() == 3
    assert ign.sum() > 0
    assert (~obj & ~ign).sum() > 0
    # Pas d'IoU au voisinage du seuil : le masque ignore ne bascule pas pendant le gradcheck.
    assert np.abs(m["best_iou"] - IGNORE).min() > 1e-3


def test_total_equals_terms(setup):
    _, _, res, *_ = setup
    assert res.total == pytest.approx(sum(t.sum() for t in res.terms.values()), rel=1e-12)
    assert all(v > 0 for v in res.parts.values())


def test_gradcheck_input(setup):
    _, x, _, loss, dx, _ = setup
    assert check_grad(loss, x, dx) <= 1e-7


@pytest.mark.parametrize("layer,name", [(0, "W"), (0, "gamma"), (0, "beta"), (2, "W"),
                                        (2, "gamma"), (2, "beta"), (5, "W"), (5, "b")])
def test_gradcheck_params(setup, layer, name):
    net, _, _, loss, _, grads = setup
    assert check_grad(loss, net.params[layer][name], grads[layer][name]) <= 1e-7


@pytest.mark.parametrize("class_mode", ["sigmoid", "softmax"])
def test_gradients_match_table(class_mode):
    # §6.2 : tableau des ∂L/∂t, élément par élément, sur des sorties brutes.
    rng = np.random.default_rng(5)
    s = 8
    out = rng.standard_normal((2, A * (5 + C), s, s))
    hl = [(0, [0, 1])]
    lam = 1.7
    res = yolo_loss({0: out}, GT_BOXES, GT_LABELS, ANCHORS, hl, C, class_mode=class_mode,
                    lambda_coord=lam, ignore_thresh=IGNORE)
    g = res.douts[0].reshape(2, A, 5 + C, s, s)
    p = out.reshape(2, A, 5 + C, s, s)
    t = build_targets(GT_BOXES, GT_LABELS, ANCHORS, hl, {0: s})[0]
    obj, ign = res.masks[0]["obj"], res.masks[0]["ignore"]
    assert obj.sum() == 3
    for n, a, i, j in np.ndindex(obj.shape):
        tt = p[n, a, :, i, j]
        gg = g[n, a, :, i, j]
        if obj[n, a, i, j]:
            w = lam * t["scale"][n, a, i, j]
            sx, sy = sigmoid(tt[0]), sigmoid(tt[1])
            exp = [2 * w * (sx - t["x"][n, a, i, j]) * sx * (1 - sx),
                   2 * w * (sy - t["y"][n, a, i, j]) * sy * (1 - sy),
                   2 * w * (tt[2] - t["tw"][n, a, i, j]),
                   2 * w * (tt[3] - t["th"][n, a, i, j]),
                   sigmoid(tt[4]) - 1]
            onehot = np.eye(C)[t["cls"][n, a, i, j]]
            if class_mode == "sigmoid":
                exp += list(sigmoid(tt[5:]) - onehot)
            else:
                e = np.exp(tt[5:] - tt[5:].max())
                exp += list(e / e.sum() - onehot)
            np.testing.assert_allclose(gg, exp, rtol=1e-12, atol=1e-15)
        elif ign[n, a, i, j]:
            assert np.all(gg == 0)         # ignore : exclue de tous les termes
        else:
            np.testing.assert_allclose(gg[4], sigmoid(tt[4]), rtol=1e-12)
            assert np.all(gg[[0, 1, 2, 3, 5, 6, 7]] == 0)


def test_perfect_prediction_has_low_loss():
    # Sorties égales aux cibles : seule l'objectness/classes reste, et tend vers 0.
    s = 8
    hl = [(0, [0, 1])]
    t = build_targets(GT_BOXES, GT_LABELS, ANCHORS, hl, {0: s})[0]
    p = np.zeros((2, A, 5 + C, s, s))
    x = np.clip(t["x"], 1e-12, 1 - 1e-12)
    y = np.clip(t["y"], 1e-12, 1 - 1e-12)
    p[:, :, 0], p[:, :, 1] = np.log(x / (1 - x)), np.log(y / (1 - y))
    p[:, :, 2], p[:, :, 3] = t["tw"], t["th"]
    p[:, :, 4] = np.where(t["obj"], 30.0, -30.0)
    p[:, :, 5:] = -30.0
    for n, a, i, j in zip(*np.nonzero(t["obj"])):
        p[n, a, 5 + t["cls"][n, a, i, j], i, j] = 30.0
    res = yolo_loss({0: p.reshape(2, -1, s, s)}, GT_BOXES, GT_LABELS, ANCHORS, hl, C,
                    ignore_thresh=0.99)
    assert res.total < 1e-9
