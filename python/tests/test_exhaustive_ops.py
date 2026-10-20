"""T28.1: the scalar operators are equal in Python, golden C++ and HLS C-sim on every value
that matters, not on a sample.

- Output stage (requantization + leaky + saturation): for every conv of every export, every
  channel, the accumulators of its exact reachable range (`bounds.acc_range`): all of them
  when the range holds ≤ 2^16 values, else every value next to a step of the requantized
  integer that can still change the int8 output, both ends, zero and a dense grid
  (`bounds.sweep`). Python (`int_layers.conv_out`) == `golden_ops requant` == `tb_ops`
  (`store_tile` of the HLS kernel).
- Tables (σ, e^t, softmax e^{d·s}) of every head, over their 256 inputs: recomputed in Python
  from the head scale == what the golden loads from `luts.bin` and indexes.

The golden tool is built by `make test-cpp`, tb_ops by `make csim-gcc`; each comparison is
skipped when its binary is missing (YOLO_REQUIRE_MODEL=1 makes the missing export fail).
"""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest

from yolo.io.export import load_model
from yolo.quant import bounds
from yolo.quant import int_layers as il
from yolo.quant import lut

ROOT = Path(__file__).resolve().parents[2]
GOLDEN_OPS = ROOT / "build" / "golden" / "golden_ops"
TB_OPS = ROOT / "build" / "hls" / "tb_ops"
NETS = ["tiny-yolov2-voc", "tiny-yolov3-coco", "tiny-yolov4-coco", "tiny-yolov7-coco"]


def _model(net):
    d = ROOT / "model" / net
    if not (d / "manifest.json").exists():
        if os.environ.get("YOLO_REQUIRE_MODEL"):
            pytest.fail(f"export absent: {d}")
        pytest.skip(f"export absent: {d} (make export)")
    return d


def _python_out(M0, n, qmax, act, acc):
    y = il.requantize(acc[None, None, :, None], np.array([M0]), n)
    if act == "leaky":
        y = il.leaky_int(y)
    return il.clip_q(y, qmax).ravel().astype(np.int8)


@pytest.mark.parametrize("net", NETS)
def test_accumulators_fit_int32(net):
    """T4.3 per layer: the exact reachable accumulator range fits int32 (proved, not sampled,
    in tools/prove_kernels.py)."""
    qm, _ = load_model(_model(net))
    for i, qc in qm.convs.items():
        lo, hi = bounds.acc_range(qc.qW, qc.qb)
        assert bounds.fits_int32(lo, hi), f"L{i:02d}: [{lo.min()}, {hi.max()}]"


@pytest.mark.parametrize("net", NETS)
def test_output_stage_python_golden_hls(net, tmp_path, record_property):
    d = _model(net)
    qm, _ = load_model(d)
    t0 = time.time()
    blocks, want, exhaustive = [], [], 0
    for i, qc in sorted(qm.convs.items()):
        act = "leaky" if qc.act == "leaky" else "linear"
        for f, M0, acc in bounds.layer_sweep(qc):
            blocks.append((M0, qc.n, qc.qmax, act, acc))
            want.append(_python_out(M0, qc.n, qc.qmax, act, acc))
            lo, hi = acc[0], acc[-1]
            exhaustive += int(hi - lo + 1 == len(acc))
    want = np.concatenate(want)
    ops = tmp_path / "ops.bin"
    mask = bounds.write_ops(ops, blocks)
    n_values, n_channels = len(want), len(blocks)
    record_property("values", n_values)
    print(f"\n{net}: {n_values:,} accumulators, {n_channels} channels "
          f"({exhaustive} swept exhaustively), Python {time.time() - t0:.1f} s")
    ran = 0
    for name, cmd in (("golden", [str(GOLDEN_OPS), "requant", str(ops), str(tmp_path / "g.bin")]),
                      ("hls", [str(TB_OPS), "--ops", str(ops), "--out", str(tmp_path / "h.bin")])):
        if not Path(cmd[0]).exists():
            continue
        t0 = time.time()
        subprocess.run(cmd, check=True, capture_output=True)
        got = np.fromfile(cmd[-1], np.int8)[mask]
        bad = np.flatnonzero(got != want)
        assert bad.size == 0, f"{name}: {bad.size} differences, first at {bad[0]}"
        print(f"  {name} == Python ({time.time() - t0:.1f} s)")
        ran += 1
    if not ran:
        pytest.skip("golden_ops and tb_ops absent (make test-cpp, make csim-gcc)")


def test_sweep_covers_every_step():
    """The boundary sweep holds the two sides of every step of the output in the range."""
    M0, n, qmax = 1_234_567_891, 38 - 7, 127
    lo, hi = -40_000, 40_000
    s = bounds.sweep(lo, hi, M0, n, qmax, "leaky", dense=16, exhaustive=0)
    full = np.arange(lo, hi + 1)
    y = _python_out(M0, n, qmax, "leaky", full)
    steps = full[1:][np.diff(y.astype(int)) != 0]
    assert np.isin(steps, s).all() and np.isin(steps - 1, s).all()


def test_acc_range_is_reached():
    rng = np.random.default_rng(0)
    qW = rng.integers(-127, 128, (4, 3, 3, 3)).astype(np.int8)
    qb = rng.integers(-1000, 1000, 4).astype(np.int32)
    lo, hi = bounds.acc_range(qW, qb)
    for f in range(4):
        w = qW[f].astype(np.int64)
        xhi = np.where(w > 0, 127, -128)
        xlo = np.where(w > 0, -128, 127)
        assert (w * xhi).sum() + qb[f] == hi[f] and (w * xlo).sum() + qb[f] == lo[f]
        x = rng.integers(-128, 128, w.shape)
        assert lo[f] <= (w * x).sum() + qb[f] <= hi[f]


@pytest.mark.parametrize("net", NETS)
def test_luts_python_golden(net):
    d = _model(net)
    if not GOLDEN_OPS.exists():
        pytest.skip("golden_ops absent (make test-cpp)")
    m = json.loads((d / "manifest.json").read_text())
    scale = {e["id"]: e["scale"] for e in m["layers"] if "lut_offset" in e}
    out = subprocess.run([str(GOLDEN_OPS), "luts", str(d)], check=True, capture_output=True,
                         text=True).stdout
    rows = np.array([[int(v) for v in line.split()] for line in out.splitlines()], np.int64)
    assert len(rows) == 256 * len(scale)
    t = np.arange(-128, 128)
    for hid, s in scale.items():
        r = rows[rows[:, 0] == hid]
        assert (r[:, 1] == t).all()
        assert (r[:, 2] == lut.lookup(lut.sigmoid_lut(s), t)).all()
        assert (r[:, 3] == lut.lookup(lut.exp_lut(s), t)).all()
        assert (r[:, 4] == lut.lookup(lut.softmax_exp_lut(s), t - 127,
                                      lut.SOFTMAX_OFFSET)).all()


@pytest.mark.parametrize("seed", range(5))
def test_silu_python_golden_hls(seed, tmp_path):
    """T26.2: integer SiLU (requantization to the table scale, then the 256-entry table) on
    every accumulator next to each of the 256 table inputs and beyond both ends, for random
    (s_mid, s_out, M0, n): Python == golden_ops == tb_ops (`store_tile`)."""
    rng = np.random.default_rng(seed)
    s_mid = float(rng.uniform(0.01, 0.2))
    s_out = float(rng.uniform(0.01, 0.2))
    qmax = int(rng.choice([7, 127]))
    table = lut.silu_lut(s_mid, s_out, qmax)
    assert np.abs(table.astype(int)).max() <= qmax
    n = int(rng.integers(12, 31))
    M0 = int(rng.integers(1 << (n - 8), 1 << (n - 1)))
    lo, hi = -(1 << 30), (1 << 30) - 1
    acc = bounds.sweep(lo, hi, M0, n, qmax, "silu", dense=256, exhaustive=0)
    want = il.conv_out(acc[None, None, :, None], np.array([M0]), n, "silu", qmax,
                       lut=table).ravel().astype(np.int8)
    y_mid = il.requantize(acc[None, None, :, None], np.array([M0]), n).ravel()
    assert set(np.clip(y_mid, -128, 127)) == set(range(-128, 128))  # every table entry used
    ops = tmp_path / "ops.bin"
    mask = bounds.write_ops(ops, [(M0, n, qmax, "silu", acc)], lut=table)
    ran = 0
    for name, cmd in (("golden", [str(GOLDEN_OPS), "requant", str(ops), str(tmp_path / "g.bin")]),
                      ("hls", [str(TB_OPS), "--ops", str(ops), "--out", str(tmp_path / "h.bin")])):
        if not Path(cmd[0]).exists():
            continue
        subprocess.run(cmd, check=True, capture_output=True)
        got = np.fromfile(cmd[-1], np.int8)[mask]
        assert np.array_equal(got, want), name
        ran += 1
    if not ran:
        pytest.skip("golden_ops and tb_ops absent")
