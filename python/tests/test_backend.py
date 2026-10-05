"""T12.11 : backend de l'entraînement, NumPy (défaut) ou CuPy (`--device gpu`).

Sur le CPU : choix du module, refus clair sans CuPy, et non-régression (point f) — un
entraînement court donne exactement les mêmes tableaux avec ou sans `use("cpu")` et
`Network.to_device()`. Les tests GPU (points a à d) sont sautés quand CuPy ou CUDA manque.
"""

import sys

import numpy as np
import pytest

from tests.test_train import TINY, _Loader
from yolo import backend
from yolo.layers.activations import leaky_backward, leaky_forward
from yolo.layers.batchnorm import bn_backward, bn_forward, bn_init_state
from yolo.layers.conv import conv_backward, conv_forward
from yolo.layers.pool import maxpool_backward, maxpool_forward
from yolo.models.graph import Network
from yolo.quant.fake_quant import fq_act_backward, fq_act_forward, fq_weight
from yolo.testing.gradcheck import check_layer
from yolo.train.admm import ADMM
from yolo.train.optim import SGD
from yolo.train.schedule import StepSchedule
from yolo.train.trainer import Trainer


@pytest.fixture(autouse=True)
def _cpu_after():
    yield
    backend.use("cpu")


def test_default_is_numpy():
    x = np.zeros(3)
    assert backend.device() == "cpu"
    assert backend.get_xp(x) is np
    assert backend.to_numpy(x) is x
    assert backend.to_device(x) is x  # backend CPU : pas de copie


def test_gpu_without_cupy_fails_clearly(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)  # import cupy → ImportError
    with pytest.raises(RuntimeError, match="CuPy absent"):
        backend.use("gpu")
    assert backend.device() == "cpu"  # pas de repli silencieux, ni de bascule partielle
    with pytest.raises(ValueError):
        backend.use("tpu")


def _arrays(tr):
    """Paramètres, état BN et vitesses SGD, par clé."""
    out = {}
    for i, (p, s, v) in enumerate(zip(tr.net.params, tr.net.state, tr.optimizer.velocity)):
        out.update({f"p{i}{k}": a for k, a in p.items()})
        out.update({f"s{i}{k}": a for k, a in s.items()})
        out.update({f"v{i}{k}": a for k, a in v.items()})
    return out


def _train(tmp_path, tag, cpu_backend, admm=False, dtype=np.float32):
    if cpu_backend:
        backend.use("cpu")
    net = Network(TINY, dtype=dtype, rng=0)
    if cpu_backend:
        net.to_device()  # backend CPU : no-op
    hook = None
    if admm:
        hook = ADMM(net, {0: "mixed6", 2: "mixed6"}, rho=0.05, every=2).hook
    tr = Trainer(net, SGD(net.params), StepSchedule(1e-2, burn_in=3), size=32, grad_hook=hook)
    tr.fit(_Loader(), 6, checkpoint=tmp_path / f"{tag}.npz", save_every=3)
    return tr


@pytest.mark.parametrize("admm", [False, True])
def test_cpu_backend_is_bit_identical(tmp_path, admm):
    """Point f : checkpoint et état identiques à l'octet, avec ou sans le backend explicite."""
    ref = _train(tmp_path, "ref", cpu_backend=False, admm=admm)
    got = _train(tmp_path, "cpu", cpu_backend=True, admm=admm)
    assert got.array_module() is np
    a, b = _arrays(ref), _arrays(got)
    assert a.keys() == b.keys()
    for k in a:
        assert type(b[k]) is np.ndarray and a[k].dtype == b[k].dtype
        assert a[k].tobytes() == b[k].tobytes(), k
    with np.load(tmp_path / "ref.npz") as da, np.load(tmp_path / "cpu.npz") as db:
        assert da.files == db.files
        for k in da.files:
            assert da[k].dtype == db[k].dtype and da[k].tobytes() == db[k].tobytes(), k


def test_network_to_numpy_roundtrip():
    net = Network(TINY, dtype=np.float32, rng=0)
    before = {i: p["W"].copy() for i, p in enumerate(net.params) if p}
    net.to_device().to_numpy()
    for i, w in before.items():
        assert type(net.params[i]["W"]) is np.ndarray
        np.testing.assert_array_equal(net.params[i]["W"], w)


# ------------------------------------------------------------------------------ GPU
@pytest.fixture
def cp():
    cupy = pytest.importorskip("cupy")
    try:
        backend.use("gpu")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    return cupy


def _rel(a, b):
    a, b = backend.to_numpy(a).astype(np.float64), backend.to_numpy(b).astype(np.float64)
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-30))


def test_gpu_gradcheck_conv_float64(cp):
    """Point a : gradcheck en float64 sur le GPU, même seuil que sur le CPU."""
    rng = np.random.default_rng(0)
    x = cp.asarray(rng.standard_normal((2, 3, 7, 7)))
    params = {"W": cp.asarray(rng.standard_normal((4, 3, 3, 3))),
              "b": cp.asarray(rng.standard_normal(4))}
    errs = check_layer(lambda x, p: conv_forward(x, p["W"], p["b"], s=1),
                       conv_backward, x, params, n_samples=40)
    assert max(errs.values()) <= 1e-7


def _layer_cases(rng, dtype):
    x = rng.standard_normal((2, 4, 9, 9)).astype(dtype)
    W = rng.standard_normal((5, 4, 3, 3)).astype(dtype)
    dy_c = rng.standard_normal((2, 5, 9, 9)).astype(dtype)
    gamma, beta = rng.standard_normal(4).astype(dtype), rng.standard_normal(4).astype(dtype)

    def conv(m, x, W, dy):
        y, c = conv_forward(x, W)
        dx, g = conv_backward(dy, c)
        return y, dx, g["W"]

    def pool(m, x, s):
        y, c = maxpool_forward(x, k=2, s=s)
        return y, maxpool_backward(m.ones_like(y), c)[0]

    def bn(m, x):
        state = {k: m.asarray(v) for k, v in bn_init_state(4, dtype).items()}
        y, c = bn_forward(x, m.asarray(gamma), m.asarray(beta), state)
        dx, g = bn_backward(y * x, c)
        return y, dx, g["gamma"], g["beta"], state["mean"], state["var"]

    def leaky(m, x):
        y, c = leaky_forward(x)
        return y, leaky_backward(x, c)[0]

    def fq(m, x, W):
        y, c = fq_act_forward(x, -3.0, 7, leaky=True)
        dx, dk = fq_act_backward(x, c)
        return y, dx, m.asarray(dk), fq_weight(W, 7)

    return [(conv, (x, W, dy_c)), (lambda m, x: pool(m, x, 2), (x,)),
            (lambda m, x: pool(m, x, 1), (x,)), (bn, (x,)), (leaky, (x,)), (fq, (x, W))]


@pytest.mark.parametrize("dtype,tol", [(np.float64, 1e-12), (np.float32, 1e-5)])
def test_gpu_layers_match_cpu(cp, dtype, tol):
    """Point b : passes avant et arrière GPU == CPU, mêmes entrées."""
    for fn, args in _layer_cases(np.random.default_rng(1), dtype):
        ref = fn(np, *args)
        got = fn(cp, *(cp.asarray(a) for a in args))
        for r, g in zip(ref, got):
            assert _rel(g, r) <= tol


def test_gpu_short_training_matches_cpu(cp, tmp_path):
    """Point d (réduit) : quelques itérations en float64, ADMM compris, GPU ≈ CPU."""
    ref = _train(tmp_path, "cpu", cpu_backend=False, admm=True, dtype=np.float64)
    backend.use("gpu")
    net = Network(TINY, dtype=np.float64, rng=0).to_device()
    admm = ADMM(net, {0: "mixed6", 2: "mixed6"}, rho=0.05, every=2)
    tr = Trainer(net, SGD(net.params), StepSchedule(1e-2, burn_in=3), size=32,
                 grad_hook=admm.hook)
    tr.fit(_Loader(), 6, checkpoint=tmp_path / "gpu.npz", save_every=3)
    assert tr.array_module() is cp
    for pa, pb in zip(ref.net.params, tr.net.params):
        for k in pa:
            assert _rel(pb[k], pa[k]) <= 1e-8, k
    with np.load(tmp_path / "gpu.npz") as d:  # checkpoint lisible en NumPy
        assert all(isinstance(d[k], np.ndarray) for k in d.files)
