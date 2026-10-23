"""T36.6 to T36.8: concepts, training recorder, CKA."""

import math

import numpy as np
import pytest

from tests.test_train import TINY, _Loader
from yolo.explain import cka as K
from yolo.explain import concepts as C
from yolo.explain import recorder as R
from yolo.models.graph import Network
from yolo.train.optim import SGD
from yolo.train.schedule import StepSchedule
from yolo.train.trainer import Trainer


# --------------------------------------------------------------------------- CKA
def test_cka_invariances():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    assert K.linear_cka(x, x) == pytest.approx(1.0)
    assert K.linear_cka(x, x @ q) == pytest.approx(1.0)
    assert K.linear_cka(x, 3.7 * x) == pytest.approx(1.0)
    assert K.linear_cka(x, rng.normal(size=(40, 12))) < 0.6
    assert K.pool(rng.normal(size=(4, 3, 6, 6)), grid=2).shape == (4, 12)
    m = K.cka_matrix([x, x @ q], [x])
    assert m.shape == (2, 1) and np.allclose(m, 1)


# ---------------------------------------------------------------------- concepts
def test_betainc_and_welch():
    # I_x(a, b) against known values; t-test p-value against Student's t (df large ≈ normal)
    assert C.betainc(2, 3, 0.4) == pytest.approx(0.5248, abs=1e-4)
    assert C.betainc(0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-12)
    rng = np.random.default_rng(0)
    a, b = rng.normal(0, 1, 2000), rng.normal(0, 1, 2000)
    t, p = C.welch(a, b)
    assert p == pytest.approx(math.erfc(abs(t) / math.sqrt(2)), abs=2e-3)
    assert C.welch(a + 1, b)[1] < 1e-10


def test_dissect_finds_planted_channel():
    rng = np.random.default_rng(0)
    n, ch, h = 12, 5, 8
    acts = rng.normal(0, 0.1, (n, ch, h, h))
    boxes, labels = [], []
    for k in range(n):
        cx, cy = rng.uniform(0.3, 0.7, 2)
        boxes.append(np.array([[cx, cy, 0.25, 0.25]]))
        labels.append(np.array([1]))
        i, j = int(cy * h), int(cx * h)
        acts[k, 3, i - 1:i + 1, j - 1:j + 1] = 5.0
    iou = C.dissect(acts, boxes, labels, 2, quantile=0.9, up=4)
    assert iou.shape == (5, 2) and int(np.argmax(iou[:, 1])) == 3
    assert C.top_channels(iou, 1)[1][0][0] == 3
    assert 0 < C.unmatched_share(iou) < 1


def test_tcav_planted_direction():
    rng = np.random.default_rng(0)
    c = 6
    v = np.zeros(c)
    v[2] = 1
    concept = rng.normal(0, 1, (30, c)) + 3 * v
    randoms = [rng.normal(0, 1, (30, c)) for _ in range(20)]  # 20 sets, as T36.6
    # the class score grows along channel 2 for 90 % of the images
    grads = rng.normal(0, 0.1, (50, c, 2, 2))
    grads[:45, 2] += 1.0
    grads[45:, 2] -= 1.0
    r = C.tcav(concept, randoms, grads)
    assert r["score"] > 0.85 and r["p"] < 0.01


def test_degradations():
    x = np.random.default_rng(0).uniform(0, 1, (3, 4, 4)).astype(np.float32)
    assert C.dark(x).max() <= 0.35 + 1e-6 and C.dark(x).dtype == np.float32
    assert C.fog(x).min() >= 0.48 - 1e-6


# ---------------------------------------------------------------------- recorder
class _OrderedLoader(_Loader):
    """`_Loader` with the `_order` / `batch_size` of `DataLoader`."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.batch_size = self.batch

    def _order(self, epoch):
        return np.random.default_rng([1, epoch]).permutation(len(self.images))


def test_recorder(tmp_path):
    net = Network(TINY, dtype=np.float64, rng=0)
    tr = Trainer(net, SGD(net.params), StepSchedule(1e-2, burn_in=3), size=32)
    loader = _OrderedLoader(n=6, batch=2)
    rec = R.Recorder(net, loader.images[:2, :, :32, :32], 6, every=2, out=tmp_path,
                     snap_every=3)
    wrapped, cb = rec.attach(tr, loader)
    tr.fit(wrapped, 9, callback=cb)
    a = rec.arrays()
    assert a["loss_img"].shape == (3, 6) and not np.isnan(a["loss_img"]).any()
    n_conv = len(rec.convs)
    assert a["w_norm"].shape == (4, n_conv) and a["pct"].shape == (4, n_conv, 3)
    assert (a["update"] > 0).all() and (a["g_norm"] > 0).all()
    assert list(a["it"]) == [2, 4, 6, 8]
    assert sorted(p.name for p in tmp_path.glob("snap_*.npz")) == [
        "snap_000003.npz", "snap_000006.npz", "snap_000009.npz"]
    assert rec.save().exists()


def test_summary_and_corruption():
    loss = np.array([[5, 1, 4, 3], [1, 1, 5, 4], [3, 1, 6, 1.0]])
    s = R.summary(loss)
    assert s["tau"] == 2.0
    assert list(s["never_learned"]) == [False, False, True, False]
    assert list(s["forgetting"]) == [1, 0, 0, 0]
    samples = [{"labels": np.array([0, 1])} for _ in range(20)]
    out, bad = R.corrupt_labels(samples, 0.1, 5, rng=0)
    assert len(bad) == 2
    for k in bad:
        assert (out[k]["labels"] != samples[k]["labels"]).all()
    assert all(np.array_equal(out[k]["labels"], samples[k]["labels"])
               for k in range(20) if k not in bad)
