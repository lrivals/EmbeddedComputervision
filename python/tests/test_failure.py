"""T39: failure labels, learned failure head, joint training term, integer head, calibration."""

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tests.test_explain_eval import GRIDS, GT, K, NET, _out
from tests.test_train import TINY, _Loader
from yolo.infer import confidence as CF
from yolo.infer import failure as F
from yolo.models.graph import Network
from yolo.models.tiny_yolo import load_cfg
from yolo.quant.int_failure import IntFailureHead
from yolo.testing.gradcheck import check_grad
from yolo.train.failure import FailureTerm, pool_backward, pool_forward
from yolo.train.optim import SGD
from yolo.train.schedule import StepSchedule
from yolo.train.trainer import Trainer

ROOT = Path(__file__).resolve().parents[2]


# -------------------------------------------------------------------------- labels
def test_feature_layers_tiny_yolov3():
    # Heads 16 (13×13) and 23 (26×26): their last conv reads the 3×3 convs 13 and 20.
    net = load_cfg("tiny-yolov3-coco")
    assert F.feature_layers(net) == [14, 21]


def test_person_tcp_and_image_targets():
    out = _out(obj=0.3, probs=(0.5, 0.3, 0.2))  # below the 0.25 threshold: 0.15
    assert F.person_tcp(NET, GT, 0, out, GRIDS) == pytest.approx(0.15)
    # A better pre-NMS box with IoU ≥ 0.5 counts as a candidate.
    out.boxes[K + 1], out.scores[K + 1, 0] = GT, 0.4
    assert F.person_tcp(NET, GT, 0, out, GRIDS) == pytest.approx(0.4)
    assert F.image_targets([None, "below_threshold"], [0.9, 0.15]) == (True, 0.15)
    assert F.image_targets([None], [0.9]) == (False, 0.9)
    assert F.image_targets([], []) == (False, 1.0)


def test_unowned_anchor_is_no_anchor():
    # Tiny-YOLOv3: anchor 0 (10×14 px) belongs to no head; a box of that shape has no
    # responsible cell.
    from yolo.explain import misses as M

    net = load_cfg("tiny-yolov3-coco")
    box = (0.5, 0.5, 10 / 416, 14 / 416)
    grids = {16: (13, 13), 23: (26, 26)}
    assert M.responsible_index(net, box, grids) is None
    e = np.zeros((0, 4))
    out = M.StageOutputs(e, np.zeros(0), np.zeros((0, 80)), (e, np.zeros(0), np.zeros(0, int)))
    assert M.classify_miss(net, box, 0, out, grids) == "no_anchor"
    assert F.person_tcp(net, box, 0, out, grids) == 0.0


def test_split_is_fixed_and_proportioned():
    s = F.split_of(100)
    np.testing.assert_array_equal(s, F.split_of(100))
    assert np.bincount(s).tolist() == [60, 20, 20]


def test_integer_pooling_matches_float_pooling():
    rng = np.random.default_rng(0)
    q = rng.integers(-128, 128, (3, 5, 13, 13)).astype(np.int8)
    s = 0.05
    p = F.pool_int_shift(q)
    c = F.dequant_scales([s], [5], [169])
    ref = F.pool_float(q.astype(np.float64) * s)
    # The mean loses at most half an LSB of the shifted sum.
    assert np.abs(p * c - ref).max() <= 0.5 * c.max() + 1e-12
    np.testing.assert_allclose((p * c)[:, 5:], ref[:, 5:])
    assert np.abs(p).max() <= 128


# ---------------------------------------------------------------------------- head
@pytest.mark.parametrize("target", ["miss", "tcp"])
def test_head_gradcheck(target):
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, (12, 7))
    y = (rng.uniform(size=12) > 0.6).astype(float) if target == "miss" else rng.uniform(size=12)
    w = rng.uniform(0.5, 3, 12)
    head = F.FailureHead(7, 5, target, seed=2).standardize(x)
    for p in head.params:
        p["b"] += rng.normal(0, 0.3, p["b"].shape)
    _, grads, dx = head.loss_and_grad(x, y, w)

    def loss():
        return head.loss_and_grad(x, y, w)[0]
    assert check_grad(loss, x, dx) <= 1e-6
    for p, g in zip(head.params, grads):
        for k in p:
            assert check_grad(loss, p[k], g[k]) <= 1e-6


@pytest.mark.parametrize("imbalance", ["weight", "balanced"])
def test_head_learns_rare_failures(imbalance):
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (2000, 10))
    y = (x[:, 0] + 0.5 * x[:, 1] > 1.8).astype(float)  # ≈ 5 % positives
    head = F.FailureHead(10, 16, "miss", seed=0)
    head.fit(x[:1500], y[:1500], imbalance=imbalance, iters=800)
    s = head.score(x[1500:])
    yt = y[1500:] > 0.5
    assert F.ood.auroc(s[~yt], s[yt]) > 0.95


def test_tcp_head_scores_low_tcp_high():
    rng = np.random.default_rng(4)
    x = rng.normal(0, 1, (1000, 4))
    tcp = 1 / (1 + np.exp(-2 * x[:, 0]))
    head = F.FailureHead(4, 8, "tcp", seed=0)
    head.fit(x, tcp, pos=tcp < 0.25, iters=600)
    s = head.score(x)
    assert F.ood.spearman(s, -tcp) > 0.9


def test_head_arrays_round_trip():
    head = F.FailureHead(6, 4, "tcp", seed=1).standardize(np.random.default_rng(0).normal(
        size=(5, 6)))
    back = F.FailureHead.from_arrays(head.arrays())
    x = np.random.default_rng(2).normal(size=(3, 6))
    np.testing.assert_array_equal(head.score(x), back.score(x))
    assert head.macs == 6 * 4 + 4 and head.memory_bytes == 4 * (24 + 4 + 4 + 1)


# ------------------------------------------------------------------------- metrics
def test_alarm_threshold_flags_the_rate():
    s = np.arange(100.0)
    assert (s > F.alarm_threshold(s, 0.05)).sum() == 5
    assert (s > F.alarm_threshold(s, 0.01)).sum() == 1
    assert (s > F.alarm_threshold(s, 0.0)).sum() == 0


def test_miss_metrics_by_hand():
    # 4 images; persons: img0 (missed), img0 (found), img1 (missed), img2 (found).
    flags = np.array([True, False, False, False])
    img, missed = np.array([0, 0, 1, 2]), np.array([True, False, True, False])
    assert F.flagged_miss_share(flags, img, missed) == 0.5
    assert F.system_recall(flags, img, missed) == (0.75, 0.5)
    m = F.monitor_metrics([0.9, 0.5, 0.1, 0.2], [True, True, False, False], img, missed,
                          rates=(0.25,))
    assert m["auroc"] == 1.0 and m["flagged_25"] == 0.5 and m["system_25"] == 0.75
    assert m["detector_recall"] == 0.5


def test_recall_at_false_alarm():
    neg = np.linspace(0, 1, 100)
    pos = np.array([0.999, 2.0, 0.5])
    scores = np.concatenate([neg, pos])
    fails = np.r_[np.zeros(100, bool), np.ones(3, bool)]
    # One person per image; the persons of the failing images are missed.
    rec, accepted, prec = F.recall_at_false_alarm(scores, fails, 0.05, np.arange(103), fails)
    assert rec == pytest.approx(2 / 3)
    assert accepted == pytest.approx((95 + 1) / 103)
    assert prec == pytest.approx(95 / 96)  # the accepted failing image keeps its miss


# ------------------------------------------------------------------- calibration
def test_ece_and_aurc_by_hand():
    assert CF.ece([0.9, 0.9, 0.1, 0.1], [1, 0, 0, 0], bins=10) == pytest.approx(
        (2 * 0.4 + 2 * 0.1) / 4)
    assert CF.aurc([0.9, 0.8, 0.1], [False, False, True]) == pytest.approx((0 + 0 + 1 / 3) / 3)
    assert CF.aurc([0.1, 0.8, 0.9], [False, False, True]) == pytest.approx((1 + 0.5 + 1 / 3) / 3)


def test_temperature_recovers_the_true_one():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    labels = rng.uniform(size=z.size) < 1 / (1 + np.exp(-z))
    over = 1 / (1 + np.exp(-2.5 * z))          # over-confident: logit × 2.5
    t = CF.fit_temperature(over, labels)
    assert t == pytest.approx(2.5, rel=0.05)
    cal = CF.apply_temperature(over, t)
    assert CF.ece(cal, labels) < 0.5 * CF.ece(over, labels)
    # Monotone: the ranking, hence AUROC and AURC, do not move.
    assert CF.aurc(cal, ~labels) == pytest.approx(CF.aurc(over, ~labels))


# ------------------------------------------------------------- T39.2 joint training
def test_pool_backward_gradcheck():
    rng = np.random.default_rng(0)
    m = rng.normal(size=(2, 3, 4, 5))
    w = rng.normal(size=(2, 6))
    p, cache = pool_forward(m)
    d = pool_backward(w, cache)

    def loss():
        return (pool_forward(m)[0] * w).sum()
    assert check_grad(loss, m, d) <= 1e-6


@pytest.mark.parametrize("coverage", [0.0, 0.99])
def test_failure_term_gradcheck(coverage):
    rng = np.random.default_rng(5)
    outs = {3: rng.normal(size=(4, 8, 4, 4)), 7: rng.normal(size=(4, 5, 2, 2))}
    term = FailureTerm(TINY, hidden=6, pos_weight=3.0, coverage=coverage, lam=2.0, weight=1.0)
    term.layers = [3, 7]
    y = np.array([1.0, 0.0, 0.0, 1.0])
    _, douts, grads = term(outs, y)
    if coverage:
        assert term.last["penalty"] > 0  # the constraint is active

    def loss():
        return term(outs, y)[0]
    for i in (3, 7):
        assert check_grad(loss, outs[i], douts[i]) <= 1e-6
    for p, g in zip(term.head.params, grads):
        for k in p:
            assert check_grad(loss, p[k], g[k]) <= 1e-6


def _joint_trainer(log_path=None, **kw):
    net = Network(TINY, dtype=np.float64, rng=0)
    term = FailureTerm(TINY, hidden=4, cls=0, conf=0.25, **kw)
    return Trainer(net, SGD(net.params), StepSchedule(1e-2, burn_in=3), size=32,
                   log_path=log_path, failure=term)


def test_joint_training_runs_logs_and_resumes(tmp_path):
    loader = _Loader()
    ref = _joint_trainer()
    ref.fit(loader, 4)
    first = _joint_trainer(log_path=tmp_path / "loss.csv")
    first.fit(loader, 3, checkpoint=tmp_path / "ck.npz")
    header = (tmp_path / "loss.csv").read_text().splitlines()[0].split(",")
    assert header[-3:] == ["fail", "coverage", "positives"]
    resumed = _joint_trainer()
    resumed.load_checkpoint(tmp_path / "ck.npz")
    resumed.fit(loader, 4)
    for pa, pb in zip(ref.failure.head.params, resumed.failure.head.params):
        for k in pa:
            np.testing.assert_allclose(pa[k], pb[k], rtol=1e-12)
    for pa, pb in zip(ref.net.params, resumed.net.params):
        for k in pa:
            np.testing.assert_allclose(pa[k], pb[k], rtol=1e-12)


def test_freeze_after_stops_the_detector():
    tr = _joint_trainer(freeze_after=0)
    before = [{k: v.copy() for k, v in p.items()} for p in tr.net.params]
    head0 = None
    tr.fit(_Loader(), 2)
    for pa, pb in zip(before, tr.net.params):
        for k in pa:
            np.testing.assert_array_equal(pa[k], pb[k])
    head0 = F.FailureHead(tr.failure.head.d, 4, "miss", seed=0)
    assert not np.array_equal(head0.params[0]["W"], tr.failure.head.params[0]["W"])


# --------------------------------------------------------------- T39.5 integer head
def _int_head(target="miss", seed=0):
    rng = np.random.default_rng(seed)
    c, hw, s = 24, 169, 0.04
    q = rng.integers(-40, 128, (300, c, 13, 13)).astype(np.int8)
    p = F.pool_int_shift(q)
    scales = F.dequant_scales([s], [c], [hw])
    x = p * scales
    head = F.FailureHead(2 * c, 8, target, seed=seed)
    y = (x[:, 0] + x[:, c + 3] > np.quantile(x[:, 0] + x[:, c + 3], 0.9)).astype(float)
    head.fit(x[:200], y[:200], iters=300, std_floor=scales)
    return head, IntFailureHead.from_float(head, scales, p[:200]), p, x


@pytest.mark.parametrize("target", ["miss", "tcp"])
def test_int_head_follows_the_float_head(target):
    head, ih, p, x = _int_head(target)
    f = head.forward(x)[0]
    sign = 1 if target == "miss" else -1
    assert F.ood.spearman(ih.score(p), sign * f) > 0.99
    assert np.corrcoef(ih.logit(p), f)[0, 1] > 0.995
    assert np.abs(ih.logit(p) - f).max() < 0.2 * np.abs(f).max()
    assert ih.memory_bytes < head.memory_bytes / 2


def test_int_network_failure_score():
    from tests.test_faults import _qm
    from yolo.quant.int_model import IntNetwork

    qm, qx = _qm("tiny-yolov3-coco")
    inet = IntNetwork(qm)
    inet.failure_layers = F.feature_layers(qm.net)
    outs = inet.forward(qx, all_outputs=True)
    p = F.features_int(outs, inet.failure_layers)
    head = F.FailureHead(p.shape[1], 4, "miss", seed=0).standardize(p.astype(float),
                                                                     std_floor=1.0)
    inet.failure_head = IntFailureHead.from_float(head, np.ones(p.shape[1]), p)
    np.testing.assert_array_equal(inet.failure_score(outs), inet.failure_head.score(p))
    # The detector path is unchanged.
    heads = inet.forward(qx)
    for i in qm.outputs:
        np.testing.assert_array_equal(heads[i], outs[i])


def build_tb(out_dir):
    exe = Path(out_dir) / "tb_failure"
    subprocess.run(["g++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-I",
                    str(ROOT / "hls" / "kernels"), str(ROOT / "hls" / "tb" / "tb_failure.cpp"),
                    "-o", str(exe)], check=True)
    return exe


@pytest.mark.skipif(shutil.which("g++") is None, reason="g++ absent")
@pytest.mark.parametrize("target", ["miss", "tcp"])
def test_int_head_equals_csim(tmp_path, target):
    _, ih, p, _ = _int_head(target, seed=1)
    exe = build_tb(tmp_path)
    blob = tmp_path / "head.bin"
    blob.write_bytes(ih.blob(p))
    out = subprocess.run([str(exe), str(blob), str(ih.sign)], check=True, capture_output=True,
                         text=True).stdout.split()
    np.testing.assert_array_equal(np.array(out, np.int64), ih.score(p))
