"""T9.2.3 : ADMM sur un problème quadratique — les poids convergent vers des niveaux."""

import numpy as np

from yolo.quant.pow2 import levels, project
from yolo.train.admm import ADMM


class _Net:
    def __init__(self, W):
        self.params = [{"W": W}]


class _Tr:
    it = 0


def test_admm_converges_to_levels():
    rng = np.random.default_rng(0)
    target = rng.standard_normal((4, 3, 3, 3))
    net = _Net(target.copy())
    admm = ADMM(net, {0: "mixed6"}, rho=0.05, every=20, growth=1.2, rho_max=5.0)
    tr, lr = _Tr(), 0.05
    res0 = admm.residuals()[0]
    for it in range(3000):
        tr.it = it
        grads = [{"W": net.params[0]["W"] - target}]  # L = ½‖W − cible‖²
        admm.hook(tr, grads)
        net.params[0]["W"] -= lr * grads[0]["W"]
    res = admm.residuals()[0]
    assert res < 1e-3 < res0
    q, s = project(net.params[0]["W"], "mixed6")
    assert set(np.abs(q).ravel()) <= set(levels("mixed6"))
