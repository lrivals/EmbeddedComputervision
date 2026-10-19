"""T25.0: fault model and injector (bit flips, BER, stuck-at) on the integer model."""

from pathlib import Path

import numpy as np
import pytest

from yolo.io.export import load_model
from yolo.models.tiny_yolo import build
from yolo.quant import faults as F
from yolo.quant.calibrate import ActStats, choose_scales
from yolo.quant.fuse_bn import fuse_network
from yolo.quant.int_model import IntNetwork, QuantModel
from yolo.quant.quantize import INPUT_SCALE, quantize_input

ROOT = Path(__file__).resolve().parents[2]
NETS = ["tiny-yolov2-voc", "tiny-yolov3-voc", "tiny-yolov4-coco"]
EXPORTS = [ROOT / "model" / n for n in ("tiny-yolov2-voc", "tiny-yolov3-coco",
                                        "tiny-yolov4-coco")]


def _qm(name):
    rng = np.random.default_rng(0)
    fused = fuse_network(build(name, dtype=np.float64, rng=1))
    stats = ActStats(fused, per_image=1024, rng=0)
    x = rng.uniform(0, 1, (2, 3, 64, 64))
    stats.update(fused.forward(x, train=False, all_outputs=True))
    scales, _, _ = choose_scales(fused, stats)
    return QuantModel.from_fused(fused, INPUT_SCALE, scales), quantize_input(x)


@pytest.fixture(scope="module", params=NETS)
def model(request):
    return _qm(request.param)


def _same(a, b):
    return all(np.array_equal(u, v) for u, v in zip(a, b))


# --- bit helpers ---------------------------------------------------------------------------

def test_flip_changes_one_element_by_power_of_two():
    v = np.arange(-127, 128, dtype=np.int64)
    for bit in range(7):
        out = F.flip_bits(v, [130], [bit], 8)
        d = out - v
        assert np.count_nonzero(d) == 1 and abs(d[130]) == 1 << bit
    # Sign bit of an int8: 3 → 3 − 128, 0 → −128 (outside the fault-free contract, kept).
    assert F.flip_bits(np.array([3, 0]), [0, 1], [7, 7], 8).tolist() == [-125, -128]
    acc = np.array([5, -(1 << 31), (1 << 31) - 1], np.int64)
    out = F.flip_bits(acc, [0, 1, 2], [31, 31, 0], 32)
    assert out.tolist() == [5 - (1 << 31), 0, (1 << 31) - 2]
    assert (np.abs(out) <= 1 << 31).all()


def test_double_flip_cancels_and_stuck_is_idempotent():
    v = np.array([7, -9], np.int64)
    assert np.array_equal(F.flip_bits(v, [0, 0], [2, 2], 8), v)
    once = F.force_bits(v, [0, 1], [0, 7], True, 8)
    assert once.tolist() == [7, -9]  # bit 0 of 7 and sign of −9 already at 1
    assert np.array_equal(F.force_bits(once, [0, 1], [0, 7], True, 8), once)
    assert F.force_bits(v, [1], [7], False, 8).tolist() == [7, 119]


def test_fault_validation():
    with pytest.raises(ValueError):
        F.Fault("weights")
    with pytest.raises(ValueError):
        F.Fault("act", "flip", bit=8)
    with pytest.raises(ValueError):
        F.Fault("acc", "ber", rate=2.0)
    assert F.Fault("acc", "flip").single and not F.Fault("acc", "stuck1", rate=0.1).single


def test_ber_count_follows_binomial():
    rng = np.random.default_rng(0)
    n, p = 1_000_000, 1e-4
    counts = [len(F.draw(rng, F.Fault("weight", "ber", rate=p), n)[0]) for _ in range(50)]
    mean, sigma = n * 8 * p, np.sqrt(n * 8 * p)
    assert abs(np.mean(counts) - mean) < 4 * sigma / np.sqrt(50)


# --- fault-free reference ------------------------------------------------------------------

def test_no_fault_equals_int_network(model):
    qm, qx = model
    ref = IntNetwork(qm).forward(qx, all_outputs=True)
    inet, inj = F.apply(IntNetwork(qm), [], seed=0)
    assert _same(inet.forward(qx, all_outputs=True), ref)
    zero = [F.Fault(s, "ber", rate=0.0) for s in F.WIDTH]
    inet, inj = F.apply(IntNetwork(qm), zero, seed=0)
    assert _same(inet.forward(qx, all_outputs=True), ref)
    assert sum(inj.total.values()) == 0


@pytest.mark.parametrize("d", EXPORTS, ids=lambda d: d.name)
def test_no_fault_real_dumps(d):
    if not (d / "manifest.json").exists():
        pytest.skip(f"export absent ({d.name}): make export")
    qm, _ = load_model(d)
    x = np.load(d / "dumps" / "000001" / "input.npy")
    inet, _ = F.apply(IntNetwork(qm), [F.Fault("acc", "ber", rate=0.0)], seed=1)
    outs = inet.forward(x, all_outputs=True)
    for i, o in enumerate(outs):
        assert np.array_equal(o, np.load(d / "dumps" / "000001" / f"L{i:02d}.npy")), i


# --- persistent faults ---------------------------------------------------------------------

def test_inject_params_copies_and_hits_one_weight(model):
    qm, _ = model
    before = {i: (c.qW.copy(), c.qb.copy(), c.cs_w.copy()) for i, c in qm.convs.items()}
    i0 = sorted(qm.convs)[1]
    fq, n = F.inject_params(qm, [F.Fault("weight", "flip", layer=i0, bit=6, index=5)], 0)
    assert n == 1
    for i, c in qm.convs.items():  # original untouched
        assert all(np.array_equal(a, b) for a, b in zip((c.qW, c.qb, c.cs_w), before[i]))
    d = fq.convs[i0].qW.astype(np.int64) - qm.convs[i0].qW
    assert np.count_nonzero(d) == 1 and abs(d.ravel()[5]) == 64
    assert fq.convs[i0].qW.dtype == np.int8
    assert np.array_equal(fq.convs[i0].cs_w, qm.convs[i0].cs_w)  # fault-free checksum kept
    assert all(fq.convs[i] is qm.convs[i] for i in qm.convs if i != i0)


def test_bias_ber_and_stuck(model):
    qm, _ = model
    fq, n = F.inject_params(qm, [F.Fault("bias", "ber", rate=0.05)], seed=2)
    total = sum(c.qb.size for c in qm.convs.values()) * 32
    assert 0 < n < 3 * 0.05 * total
    fq, n = F.inject_params(qm, [F.Fault("bias", "stuck1", bit=31, rate=1.0)], seed=2)
    assert all((c.qb < 0).all() for c in fq.convs.values())


# --- transient faults ----------------------------------------------------------------------

def test_transient_reproducible_per_image(model):
    qm, qx = model
    faults = [F.Fault("act", "ber", rate=1e-3), F.Fault("acc", "ber", rate=1e-4)]
    a, ia = F.apply(IntNetwork(qm), faults, seed=4)
    b, ib = F.apply(IntNetwork(qm), faults, seed=4)
    c, _ = F.apply(IntNetwork(qm), faults, seed=5)
    outs = {}
    for key in (3, 7):
        ia.start(key)
        ib.start(key)
        outs[key] = a.forward(qx, all_outputs=True)
        assert _same(outs[key], b.forward(qx, all_outputs=True))
        assert ia.flips == ib.flips and ia.flips["act"] > 0
    assert not _same(outs[3], outs[7])
    ia.start(3)
    assert _same(a.forward(qx, all_outputs=True), outs[3])  # order does not matter
    assert not _same(c.forward(qx, all_outputs=True), outs[3])


def test_single_acc_flip_without_layer(model):
    qm, qx = model
    sizes = F.conv_sizes(qm, qx.shape[1:])
    assert set(sizes) == set(qm.convs)
    ref = IntNetwork(qm).forward(qx, all_outputs=True)
    inet, inj = F.apply(IntNetwork(qm), [F.Fault("acc", "flip", bit=30)], seed=0, sizes=sizes)
    inet.forward(qx, all_outputs=True)
    assert inj.flips["acc"] == 1
    hit = 0
    for key in range(20):
        inj.start(key)
        hit += not _same(inet.forward(qx, all_outputs=True), ref)
    assert hit > 0
    with pytest.raises(ValueError):
        F.FaultInjector([F.Fault("acc", "flip")], seed=0)


def test_act_fault_on_given_layer(model):
    qm, qx = model
    i0 = sorted(qm.convs)[2]
    ref = IntNetwork(qm).forward(qx, all_outputs=True)
    inet, _ = F.apply(IntNetwork(qm), [F.Fault("act", "flip", layer=i0, bit=7, index=0)])
    outs = inet.forward(qx, all_outputs=True)
    assert all(np.array_equal(outs[i], ref[i]) for i in range(i0))
    assert np.count_nonzero(outs[i0].astype(np.int64) - ref[i0]) == 1


def test_bias_overflow_wraps_and_is_detected(model):
    from yolo.quant.abft import AbftMonitor

    qm, qx = model
    i0 = sorted(qm.convs)[0]
    inet = IntNetwork(qm)
    mon = AbftMonitor(qm)
    inet.hooks.append(mon)
    # Bits 0-30 of a non-negative bias stuck at 1: q_b = 2³¹ − 1, the accumulator leaves int32
    # wherever the products are positive. The faulty network wraps like a 32-bit register
    # (the fault-free model raises), and the check (mod 2³²) sees it.
    f0 = int(np.flatnonzero(qm.convs[i0].qb >= 0)[0])
    faults = [F.Fault("bias", "stuck1", layer=i0, bit=b, index=f0) for b in range(31)]
    fq, _ = F.inject_params(qm, faults)
    assert fq.convs[i0].qb[f0] == 2**31 - 1
    with pytest.raises(OverflowError):
        IntNetwork(fq).forward(qx)
    finet, _ = F.apply(inet, faults)
    outs = finet.forward(qx, all_outputs=True)
    assert outs[i0].dtype == np.int8 and mon.errors[i0] > 0
