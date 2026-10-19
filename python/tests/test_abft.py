"""T25.2: ABFT checksum of the integer convolutions (checksum filter Σ_f qW_f, bias Σ_f q_b,f,
exact check on the int32 accumulator modulo 2³²)."""

import json
from pathlib import Path

import numpy as np
import pytest

from tests.test_faults import EXPORTS, _qm
from yolo.io.export import export_model, load_model
from yolo.quant import abft
from yolo.quant import faults as F
from yolo.quant import int_layers as il
from yolo.quant.int_model import IntNetwork

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ("000001", "000002", "000003")


@pytest.fixture(scope="module")
def model():
    return _qm("tiny-yolov3-voc")


def test_checksums_and_wrap():
    rng = np.random.default_rng(0)
    qW = rng.integers(-127, 128, (8, 3, 3, 3)).astype(np.int8)
    qb = np.array([2**31 - 1] * 4 + [5] * 4, np.int32)
    cs_w, cs_b = abft.checksums(qW, qb)
    assert cs_w.dtype == np.int32 and cs_w.shape == (3, 3, 3)
    assert np.array_equal(cs_w, qW.astype(np.int64).sum(axis=0))
    assert cs_b == abft.wrap32(4 * (2**31 - 1) + 20) and -(2**31) <= cs_b < 2**31
    assert abft.wrap32(2**31) == -(2**31) and abft.wrap32(-1) == -1
    assert np.array_equal(abft.wrap32(np.array([2**32 + 3, -(2**31) - 1])), [3, 2**31 - 1])


@pytest.mark.parametrize("engine", il.ENGINES)
def test_check_passes_on_a_random_conv(engine):
    rng = np.random.default_rng(1)
    x = rng.integers(-127, 128, (2, 5, 9, 7))
    for s in (1, 2):
        qW = rng.integers(-127, 128, (6, 5, 3, 3)).astype(np.int8)
        qb = rng.integers(-2**20, 2**20, 6).astype(np.int32)
        acc = il.conv_acc(x, qW, qb, engine, s)
        cs_w, cs_b = abft.checksums(qW, qb)
        assert abft.check(x, acc, cs_w, cs_b, engine, s) == 0
        acc[1, 3, 2, 3] += 1
        assert abft.check(x, acc, cs_w, cs_b, engine, s) == 1


def test_every_single_acc_flip_is_detected():
    # Any flip of bit b < 32 of one accumulator changes the sum by ±2^b ≠ 0 mod 2³².
    rng = np.random.default_rng(2)
    x = rng.integers(-127, 128, (1, 4, 6, 6))
    qW = rng.integers(-127, 128, (5, 4, 3, 3)).astype(np.int8)
    qb = rng.integers(-1000, 1000, 5).astype(np.int32)
    acc = il.conv_acc(x, qW, qb)
    cs_w, cs_b = abft.checksums(qW, qb)
    for bit in range(32):
        for idx in rng.integers(acc.size, size=8):
            bad = F.flip_bits(acc, [idx], [bit], 32)
            m = abft.mismatches(x, bad, cs_w, cs_b)
            assert m.sum() == 1


def test_monitor_fault_free_and_acc_faults(model):
    qm, qx = model
    inet = IntNetwork(qm)
    mon = abft.AbftMonitor(qm)
    inet.hooks.append(mon)
    ref = inet.forward(qx, all_outputs=True)
    assert not mon.detected and set(mon.checks) == set(qm.convs)
    assert all(mon.checks[i] > 0 for i in qm.convs)
    # The check does not change the outputs.
    assert all(np.array_equal(a, b) for a, b in
               zip(ref, IntNetwork(qm).forward(qx, all_outputs=True)))
    sizes = F.conv_sizes(qm, qx.shape[1:])
    finet, inj = F.apply(inet, [F.Fault("acc", "flip")], seed=0, sizes=sizes)
    for key in range(30):  # 100 % of the accumulator faults
        mon.reset()
        inj.start(key)
        finet.forward(qx)
        assert inj.flips["acc"] == 1 and sum(mon.errors.values()) == 1, key


def test_weight_fault_detected_input_fault_not(model):
    qm, qx = model
    i0 = sorted(qm.convs)[1]
    mon = abft.AbftMonitor(qm)
    inet = IntNetwork(qm)
    inet.hooks.append(mon)
    # Weight fault: faulty qW against the fault-free checksum (kept by inject_params).
    c = qm.convs[i0]
    centre = np.ravel_multi_index((0, 0, c.qW.shape[2] // 2, c.qW.shape[3] // 2), c.qW.shape)
    finet, _ = F.apply(inet, [F.Fault("weight", "flip", layer=i0, bit=6, index=int(centre))])
    finet.forward(qx)
    assert mon.errors[i0] > 0
    assert all(mon.errors[i] == 0 for i in qm.convs if i < i0)
    # Activation fault on the output of conv i0: the check of i0 (before it) and the check of
    # the next conv (both sides read the same faulty input) do not see it.
    mon.reset()
    finet, _ = F.apply(inet, [F.Fault("act", "stuck1", layer=i0, bit=6, rate=0.01)], seed=3)
    finet.forward(qx)
    assert not mon.detected


def test_export_roundtrip_and_schema(model, tmp_path):
    import jsonschema

    qm, qx = model
    m = export_model(qm, tmp_path)
    schema = json.loads((ROOT / "docs" / "manifest.schema.json").read_text())
    jsonschema.Draft202012Validator(schema).validate(m)
    assert "checksum.bin" in m["blobs"]
    rq, _ = load_model(tmp_path)
    for i, c in qm.convs.items():
        assert np.array_equal(rq.convs[i].cs_w, c.cs_w) and rq.convs[i].cs_b == c.cs_b
    # A checksum blob that disagrees with the weights is reported (corrupted weights.bin).
    raw = np.fromfile(tmp_path / "checksum.bin", "<i4")
    raw[0] += 1
    raw.tofile(tmp_path / "checksum.bin")
    rq, _ = load_model(tmp_path)
    mon = abft.AbftMonitor(rq)
    inet = IntNetwork(rq)
    inet.hooks.append(mon)
    inet.forward(qx)
    assert mon.errors[min(rq.convs)] > 0


@pytest.mark.parametrize("d", EXPORTS, ids=lambda d: d.name)
def test_fault_free_real_dumps(d):
    if not (d / "manifest.json").exists():
        pytest.skip(f"export absent ({d.name}): make export")
    qm, m = load_model(d)
    assert "checksum.bin" in m["blobs"], "export without checksums: make export"
    inet = IntNetwork(qm)
    mon = abft.AbftMonitor(qm)
    inet.hooks.append(mon)
    for image in IMAGES:
        mon.reset()
        inet.forward(np.load(d / "dumps" / image / "input.npy"))
        assert set(mon.errors) == set(qm.convs) and not mon.detected, image


def test_widths(model):
    qm, _ = model
    rows = abft.widths(qm)
    assert [r["id"] for r in rows] == sorted(qm.convs)
    for r in rows:
        c = qm.convs[r["id"]]
        assert r["cs_max"] == np.abs(c.qW.astype(np.int64).sum(axis=0)).max()
        assert -(2 ** (r["cs_bits"] - 1)) <= -r["cs_max"] and r["cs_max"] < 2 ** (r["cs_bits"] - 1)
        assert r["cs_bits"] <= 18  # Σ of ≤ 1,024 int8 filters fits 18 bits signed
