"""T2.7 : SGD et AdamW (problème quadratique), taux d'apprentissage, trainer et checkpoints."""

import numpy as np
import pytest

from yolo.models.graph import Network
from yolo.train.optim import SGD, AdamW
from yolo.train.schedule import StepSchedule, lr_at
from yolo.train.trainer import MULTISCALE_SIZES, Trainer, copy_matching, multiscale_size


def _quadratic(seed=0, n=5):
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n, n))
    a = q @ q.T + n * np.eye(n)      # définie positive
    c = rng.standard_normal(n)
    return a, c                      # f(x) = ½ (x−c)ᵀ A (x−c), ∇f = A (x−c)


def test_sgd_matches_recurrence_on_quadratic():
    a, c = _quadratic()
    mu, wd, lr = 0.9, 5e-4, 0.01
    x0 = np.ones(5)
    params = [{"W": x0.copy()}]
    opt = SGD(params, momentum=mu, weight_decay=wd)
    x, v = x0.copy(), np.zeros(5)
    for _ in range(200):
        g = a @ (params[0]["W"] - c)
        opt.step([{"W": g}], lr)
        # §7.1, récurrence écrite à la main
        v = mu * v - lr * (a @ (x - c) + wd * x)
        x = x + v
        np.testing.assert_allclose(params[0]["W"], x, rtol=1e-13, atol=1e-15)
    # Convergence vers le minimum de f + (wd/2)|x|² : (A + wd I) x* = A c.
    x_star = np.linalg.solve(a + wd * np.eye(5), a @ c)
    for _ in range(2000):
        opt.step([{"W": a @ (params[0]["W"] - c)}], lr)
    np.testing.assert_allclose(params[0]["W"], x_star, atol=1e-10)
    assert not np.allclose(x_star, c)  # le weight decay a bien un effet


def test_sgd_no_decay_on_gamma_beta_bias():
    params = [{"W": np.ones(3), "gamma": np.ones(3), "beta": np.ones(3), "b": np.ones(3)}]
    opt = SGD(params, momentum=0.9, weight_decay=0.1)
    zero = {k: np.zeros(3) for k in params[0]}
    opt.step([zero], lr=1.0)
    np.testing.assert_allclose(params[0]["W"], 0.9)
    for k in ("gamma", "beta", "b"):
        np.testing.assert_array_equal(params[0][k], 1.0)


def test_adamw_matches_recurrence_on_quadratic():
    a, c = _quadratic()
    b1, b2, eps, wd, lr = 0.9, 0.999, 1e-8, 1e-2, 0.01
    x0 = np.ones(5)
    params = [{"W": x0.copy()}]
    opt = AdamW(params, betas=(b1, b2), eps=eps, weight_decay=wd)
    x, m, v = x0.copy(), np.zeros(5), np.zeros(5)
    for t in range(1, 201):
        opt.step([{"W": a @ (params[0]["W"] - c)}], lr)
        # récurrence écrite à la main (decay découplé, corrections de biais)
        g = a @ (x - c)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * g * g
        m_hat, v_hat = m / (1 - b1 ** t), v / (1 - b2 ** t)
        x = x - lr * wd * x - lr * m_hat / (np.sqrt(v_hat) + eps)
        np.testing.assert_allclose(params[0]["W"], x, rtol=1e-12, atol=1e-14)
    # Point fixe (g → 0 aux erreurs près) : la décroissance découplée tire vers 0, sans
    # normalisation, d'où un minimum décalé de c mais proche (wd petit devant la courbure).
    for _ in range(3000):
        opt.step([{"W": a @ (params[0]["W"] - c)}], 1e-3)
    assert np.linalg.norm(params[0]["W"] - c) < 0.05 * np.linalg.norm(c)


def test_adamw_no_decay_on_gamma_beta_bias():
    params = [{"W": np.ones(3), "gamma": np.ones(3), "beta": np.ones(3), "b": np.ones(3)}]
    opt = AdamW(params, weight_decay=0.1)
    zero = {k: np.zeros(3) for k in params[0]}
    opt.step([zero], lr=1.0)
    np.testing.assert_allclose(params[0]["W"], 0.9)
    for k in ("gamma", "beta", "b"):
        np.testing.assert_array_equal(params[0][k], 1.0)


def test_adamw_state_roundtrip():
    rng = np.random.default_rng(0)
    p0 = {"W": rng.standard_normal((2, 3)), "b": rng.standard_normal(2)}
    grads = [[{k: rng.standard_normal(a.shape) for k, a in p0.items()}] for _ in range(6)]
    ref = AdamW([{k: a.copy() for k, a in p0.items()}])
    for g in grads:
        ref.step(g, 1e-2)
    params = [{k: a.copy() for k, a in p0.items()}]
    opt = AdamW(params)
    for g in grads[:3]:
        opt.step(g, 1e-2)
    state = {k: np.array(a) for k, a in opt.state_dict().items()}
    resumed = AdamW(params)
    resumed.load_state_dict(state)
    for g in grads[3:]:
        resumed.step(g, 1e-2)
    for k in p0:
        np.testing.assert_array_equal(params[0][k], ref.params[0][k])
    with pytest.raises(ValueError, match="SGD"):
        SGD(params).load_state_dict(state)
    with pytest.raises(ValueError, match="AdamW"):
        AdamW(params).load_state_dict(SGD(params).state_dict())


def test_schedule():
    assert lr_at(0, 1e-3, burn_in=100) == pytest.approx(1e-3 * (1 / 100) ** 4)
    assert lr_at(49, 1e-3, burn_in=100) == pytest.approx(1e-3 * 0.5 ** 4)
    assert lr_at(99, 1e-3, burn_in=100) == pytest.approx(1e-3)
    lrs = [lr_at(i, 1e-3, burn_in=100) for i in range(100)]
    assert np.all(np.diff(lrs) > 0)
    s = StepSchedule(1e-3, burn_in=10, steps=(400, 450), scales=(0.1, 0.1))
    assert s(10) == s(399) == pytest.approx(1e-3)
    assert s(400) == pytest.approx(1e-4)
    assert s(450) == pytest.approx(1e-5)


def test_multiscale_size():
    sizes = [multiscale_size(it, seed=3) for it in range(500)]
    assert set(sizes) <= set(MULTISCALE_SIZES)
    assert len(set(sizes)) > 5
    for k in range(0, 500, 10):  # §2.2 : constante par bloc de 10
        assert len(set(sizes[k:k + 10])) == 1
    assert MULTISCALE_SIZES[0] == 320 and MULTISCALE_SIZES[-1] == 608


def _conv(cout, k=3, act="leaky"):
    return {"type": "conv", "k": k, "s": 1, "cout": cout, "act": act, "bn": act == "leaky"}


TINY = {
    "name": "tiny-test",
    "input": (3, 32, 32),
    "classes": 2,
    "anchors": [[60, 60], [150, 120], [300, 300]],
    "layers": [
        _conv(4), {"type": "maxpool", "k": 2, "s": 2},
        _conv(8), {"type": "maxpool", "k": 2, "s": 2},
        _conv(3 * 7, k=1, act="linear"), {"type": "yolo", "mask": [0, 1, 2]},
    ],
}


class _Loader:
    """Chargeur déterministe en mémoire, même interface que `DataLoader.epoch`."""

    def __init__(self, n=6, batch=2):
        rng = np.random.default_rng(0)
        self.images = rng.uniform(0, 1, (n, 3, 64, 64)).astype(np.float32)
        self.boxes = [rng.uniform(0.2, 0.8, (2, 4)) * [1, 1, 0.5, 0.5] for _ in range(n)]
        self.labels = [rng.integers(0, 2, 2) for _ in range(n)]
        self.batch = batch

    def __len__(self):
        return len(self.images) // self.batch

    def epoch(self, epoch, size_fn, first_iter=0, skip=0):
        order = np.random.default_rng([1, epoch]).permutation(len(self.images))
        for b in range(skip, len(self)):
            idx = order[b * self.batch:(b + 1) * self.batch]
            s = size_fn(first_iter + b - skip)
            imgs = self.images[idx][:, :, :s, :s]
            yield imgs, [self.boxes[i] for i in idx], [self.labels[i] for i in idx]


def _trainer(log_path=None, size=32, optim=SGD):
    net = Network(TINY, dtype=np.float64, rng=0)
    return Trainer(net, optim(net.params), StepSchedule(1e-2, burn_in=3), size=size,
                   log_path=log_path)


def test_trainer_reduces_loss():
    tr = _trainer()
    loader = _Loader()
    losses = []
    tr.fit(loader, 60, callback=lambda t, res, lr: losses.append(res.total))
    assert tr.it == 60
    assert np.mean(losses[-6:]) < 0.5 * np.mean(losses[:6])


@pytest.mark.parametrize("optim", [SGD, AdamW])
def test_checkpoint_resume_is_exact(tmp_path, optim):
    loader = _Loader()
    ref = _trainer(optim=optim)
    ref.fit(loader, 5)

    first = _trainer(optim=optim)
    first.fit(loader, 4, checkpoint=tmp_path / "ck.npz")   # s'arrête au milieu de l'époque 1
    resumed = _trainer(optim=optim)
    resumed.net.init_he(rng=123)                           # état de départ différent
    resumed.load_checkpoint(tmp_path / "ck.npz")
    assert resumed.it == 4
    resumed.fit(loader, 5)
    for pa, pb in zip(ref.net.params, resumed.net.params):
        for k in pa:
            np.testing.assert_array_equal(pa[k], pb[k])
    for sa, sb in zip(ref.net.state, resumed.net.state):
        for k in sa:
            np.testing.assert_array_equal(sa[k], sb[k])


def test_log_and_multiscale(tmp_path):
    tr = _trainer(log_path=tmp_path / "log.csv", size=None)
    tr.size_for = lambda it: 32 if (it // 2) % 2 == 0 else 64  # tailles variables
    tr.fit(_Loader(), 4)
    rows = (tmp_path / "log.csv").read_text().splitlines()
    assert rows[0].startswith("it,lr,size,loss")
    assert [r.split(",")[2] for r in rows[1:]] == ["32", "32", "64", "64"]


def test_copy_matching():
    src = Network(TINY, dtype=np.float64, rng=1)
    other = dict(TINY, layers=TINY["layers"][:4] + [_conv(3 * 9, k=1, act="linear"),
                                                     TINY["layers"][5]])
    dst = Network(other, dtype=np.float64, rng=2)
    assert copy_matching(src, dst) == [4]
    np.testing.assert_array_equal(dst.params[2]["W"], src.params[2]["W"])
    assert not np.array_equal(dst.params[4]["W"][:21], src.params[4]["W"])


def test_copy_matching_rgb_to_gray():
    # T11.7 : L00 d'un réseau RGB vers un réseau à un canal, par somme sur les canaux ; même
    # sortie de L00 sur une image grise recopiée sur R, G et B.
    src = Network(TINY, dtype=np.float64, rng=1)
    dst = Network(dict(TINY, input=(1, 32, 32)), dtype=np.float64, rng=2)
    assert copy_matching(src, dst) == []
    gray = np.random.default_rng(0).uniform(0, 1, (1, 1, 32, 32))
    a = src.forward(np.repeat(gray, 3, axis=1), all_outputs=True)[0]
    b = dst.forward(gray, all_outputs=True)[0]
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12)
