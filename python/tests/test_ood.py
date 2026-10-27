"""T25.5: out-of-distribution monitor (Mahalanobis baseline, per-cell autoencoder, AUROC)."""

import numpy as np
import pytest

from yolo.infer import ood
from yolo.models.tiny_yolo import load_cfg
from yolo.testing.gradcheck import check_grad


def test_auroc_known_cases():
    assert ood.auroc([0, 1, 2], [3, 4, 5]) == 1.0
    assert ood.auroc([3, 4, 5], [0, 1, 2]) == 0.0
    assert ood.auroc([1, 1], [1, 1]) == 0.5
    assert ood.auroc([0, 2], [1, 3]) == 0.75
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 4000), rng.normal(1, 1, 4000)
    # Two unit Gaussians one σ apart: AUROC = Φ(1/√2) ≈ 0.760.
    assert ood.auroc(a, b) == pytest.approx(0.760, abs=0.015)


def test_spearman():
    x = np.arange(10.0)
    assert ood.spearman(x, x ** 3) == pytest.approx(1.0)
    assert ood.spearman(x, -x) == pytest.approx(-1.0)


def test_ledoit_wolf_and_mahalanobis():
    rng = np.random.default_rng(1)
    d = 50
    x = rng.normal(0, 1, (30, d))  # fewer samples than dimensions: S is singular
    cov, a = ood.ledoit_wolf(x)
    assert 0 < a <= 1 and np.linalg.eigvalsh(cov).min() > 0
    m = ood.Mahalanobis().fit(rng.normal(0, 1, (400, 8)))
    inside, outside = m.score(rng.normal(0, 1, (300, 8))), m.score(rng.normal(1.5, 1, (300, 8)))
    assert ood.auroc(inside, outside) > 0.95
    assert m.score(m.mean[None])[0] == pytest.approx(0.0, abs=1e-9)


def test_pool_and_standardizer():
    f = np.arange(2 * 3 * 2 * 2, dtype=np.float64).reshape(2, 3, 2, 2)
    p = ood.pool(f)
    assert p.shape == (2, 6) and p[0, 0] == 1.5 and p[0, 3] == 3.0
    st = ood.Standardizer().fit(f)
    z = st(f)
    assert np.allclose(z.mean(axis=(0, 2, 3)), 0, atol=1e-6)
    assert np.allclose(z.std(axis=(0, 2, 3)), 1, atol=1e-4)


def test_autoencoder_spec_and_macs():
    spec = ood.autoencoder_spec(1024, (13, 13), (128, 32))
    assert [layer["cout"] for layer in spec["layers"]] == [128, 32, 128, 1024]
    assert spec["layers"][-1]["act"] == "linear" and spec["layers"][0]["act"] == "leaky"
    per_cell = 1024 * 128 + 128 * 32 + 32 * 128 + 128 * 1024
    assert ood.macs(spec) == 169 * per_cell
    assert ood.macs(spec) / ood.macs(load_cfg("tiny-yolov2-voc")) < 0.02


def test_autoencoder_gradient_and_training():
    rng = np.random.default_rng(2)
    net = ood.make_ae(6, (3, 3), (4, 2), seed=0)
    net.params = [{k: v.astype(np.float64) for k, v in p.items()} for p in net.params]
    net.dtype = np.dtype(np.float64)
    x = rng.normal(0, 1, (2, 6, 3, 3))
    out = len(net.layers) - 1

    def loss():
        return float(((net.forward(x, train=True)[out] - x) ** 2).mean())

    y = net.forward(x, train=True)[out]
    _, grads = net.backward({out: 2 * (y - x) / y.size})
    for i in (0, out):
        w = net.params[i]["W"]
        assert check_grad(loss, w, grads[i]["W"], n_samples=20, rng=0) < 1e-6
    # Low-rank data (rank 2 in 6 channels): the 2-unit bottleneck learns it.
    net = ood.make_ae(6, (3, 3), (4, 2), seed=0)
    basis = rng.normal(0, 1, (2, 6))
    data = np.einsum("nkhw,kc->nchw", rng.normal(0, 1, (64, 2, 3, 3)), basis)
    losses = ood.train_ae(net, data.astype(np.float32), iters=400, batch=8, lr=1e-2)
    assert np.mean(losses[-20:]) < 0.3 * np.mean(losses[:20])
    err_in = ood.reconstruction_error(net, data[:16].astype(np.float32)).mean()
    err_out = ood.reconstruction_error(net, rng.normal(0, 1, (16, 6, 3, 3))
                                       .astype(np.float32)).mean()
    assert err_out > 2 * err_in


def test_typicality_catches_low_energy_inputs():
    # In-domain points on a shell of radius ≈ √d; OOD points near the centre (low energy):
    # their raw distance is smaller (AUROC < 0.5), their typicality larger.
    rng = np.random.default_rng(3)
    d = 64
    m = ood.Mahalanobis().fit(rng.normal(0, 1, (2000, d)))
    ref = np.median(m.score(rng.normal(0, 1, (500, d))))
    inside, low = rng.normal(0, 1, (300, d)), rng.normal(0, 0.3, (300, d))
    assert ood.auroc(m.score(inside), m.score(low)) < 0.1
    assert ood.auroc(ood.typicality(m.score(inside), ref),
                     ood.typicality(m.score(low), ref)) > 0.9
    assert ood.norm(np.array([[3.0, 4.0]]))[0] == 5.0


# --------------------------------------------------------------------------- T29.0
def test_new_monitors_separate_shifted_gaussians():
    from yolo.infer import ood

    rng = np.random.default_rng(0)
    fit = rng.normal(0, 1, (600, 32)).round().astype(np.int64) * 10
    inside = rng.normal(0, 1, (200, 32)).round().astype(np.int64) * 10
    outside = rng.normal(2.5, 1, (200, 32)).round().astype(np.int64) * 10
    for mon in (ood.AbstractionBoxes(k=3), ood.ActivationPatterns(bits=16)):
        mon.fit(fit)
        a = ood.auroc(mon.score(inside), mon.score(outside))
        assert a > 0.9, (type(mon).__name__, a)
        assert mon.memory_bytes > 0 and mon.ops(fit.shape[1]) > 0
    assert ood.AbstractionBoxes(k=2).fit(fit).score(fit).max() == 0  # every fit point inside


def test_fpr_at_tpr_and_max_score():
    from yolo.infer import ood

    inside = np.linspace(0, 1, 101)
    outside = np.linspace(2, 3, 101)
    assert ood.fpr_at_tpr(inside, outside) == 0.0
    assert ood.fpr_at_tpr(outside, inside) == 1.0
    assert ood.MaxScore().score([0.9, 0.2]).tolist() == pytest.approx([0.1, 0.8])


def test_pool_int():
    from yolo.infer import ood

    m = np.arange(2 * 3 * 2 * 2, dtype=np.int8).reshape(2, 3, 2, 2)
    p = ood.pool_int(m)
    assert p.shape == (2, 6) and p[0, 0] == 0 + 1 + 2 + 3 and p[0, 3] == 3


# ------------------------------------------------------------- alarms for T27.7 (T29.1)
def test_write_flags(tmp_path):
    import json
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
    import ood_monitor as OM

    assert OM.flag_name("failure (miss, int)") == "failure-miss-int"
    L = SimpleNamespace(name="fit", ids=np.array([f"a{i}" for i in range(100)]))
    A = SimpleNamespace(name="lab", ids=np.array([f"b{i}" for i in range(200)]))
    val = np.arange(100) < 100
    scorers = {"s": lambda X: np.arange(len(X.ids), dtype=float) + (1000 if X is A else 0)}
    OM.write_flags(L, A, scorers, tmp_path, val, rates=(0.05,))
    rows = [json.loads(x) for x in (tmp_path / "lab__s__0.05.jsonl").read_text().splitlines()]
    assert len(rows) == 200 and all(r["alarm"] for r in rows)  # threshold of L: all above
    OM.write_flags(L, A, scorers, tmp_path, val, rates=(0.05,), rate_on="apply")
    rows = [json.loads(x) for x in (tmp_path / "lab__s__0.05__apply.jsonl").read_text().splitlines()]
    assert sum(r["alarm"] for r in rows) == 10 and rows[-1] == {"image": "b199", "alarm": True}
