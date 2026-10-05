"""T9.4.1 : modèle de l'architecture streaming (repliement, mémoire, budget DSP)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from stream_model import (divisors, line_buffer_bytes, plan, stage_cycles,  # noqa: E402
                          weight_bytes)

BOARD = {"name": "test", "dsp": 1248, "bram18": 288, "uram": 64, "freq_mhz": 200,
         "ddr_bw_gbps": 19.2, "ddr_efficiency": 0.7, "int8_macs_per_dsp": 1}
L = {"layer": 0, "k": 3, "pad": 1, "cin": 16, "cout": 32, "h": 52, "w": 52, "pool_k": 2,
     "pool_s": 2, "prepool": False}


def test_folding_cycles_and_buffers():
    assert stage_cycles(L, 1, 1) == 52 * 52 * 9 * 16 * 32
    assert stage_cycles(L, 4, 8) == 52 * 52 * 9 * 2 * 8
    assert line_buffer_bytes(L, 8) == 2 * 52 * 16 + 52 * 32  # (K−1)·W·C_in + ligne de pool
    assert weight_bytes(L, 4) == 9 * 16 * 32 / 2
    assert divisors(12) == [1, 2, 3, 4, 6, 12]


def test_tiny_yolov2_plan_respects_budgets():
    from perf_model import conv_layers

    man = ROOT / "model" / "tiny-yolov2-voc" / "manifest.json"
    if not man.exists():
        pytest.skip("export absent")
    layers = conv_layers(man)
    p = plan(layers, BOARD)
    assert p["dsp"] <= p["dsp_budget"] and p["onchip_bytes"] <= p["onchip_budget"]
    assert p["ii_cycles"] == max(s["cycles"] for s in p["stages"])
    for s, d in zip(p["stages"], layers):
        assert d["cin"] % s["simd"] == 0 and d["cout"] % s["pe"] == 0
    # 15,9 Mo de poids int8 > mémoire de la puce : L12-L13 en DDR (hybride)
    assert sum(weight_bytes(d, 8) for d in layers) > p["onchip_budget"]
    assert {s["layer"] for s in p["stages"] if s["weights"] == "ddr"} == {12, 13}


def test_csim_cycles_equal_model():
    """Cycles C-sim de hls/tb/tb_stream (itérations PE × SIMD par étage) == plan du modèle,
    et les repliements codés dans hls/stream/yolo_stream.hpp == ceux du plan."""
    import re
    import subprocess

    from perf_model import conv_layers

    tb = ROOT / "build" / "hls" / "tb_stream"
    man = ROOT / "model" / "tiny-yolov2-voc" / "manifest.json"
    if not tb.exists() or not man.exists():
        pytest.skip("tb_stream ou export absent (make csim-gcc)")
    out = subprocess.run([str(tb), "--image", "000001"], capture_output=True, text=True,
                         check=True).stdout
    got = [(int(a), int(b), int(c)) for a, b, c in
           re.findall(r"PE (\d+) × SIMD (\d+), (\d+) cycles, OK", out)]
    p = plan(conv_layers(man), BOARD)
    assert got == [(s["pe"], s["simd"], s["cycles"]) for s in p["stages"]]


def test_fifo_depths_and_rom_tables_match_model():
    """T10.8 : profondeurs des FIFO (table et `#pragma HLS STREAM`) et étages FRAME de
    hls/stream/yolo_stream.* == stream_model ; ROM rangée [og][ig][i][j][pe·SIMD + s]."""
    import re

    import numpy as np

    from gen_stream_rom import check_plan, hpp_table, rom_layout
    from perf_model import conv_layers
    from stream_model import fifo_depths

    man = ROOT / "model" / "tiny-yolov2-voc" / "manifest.json"
    if not man.exists():
        pytest.skip("export absent")
    layers = conv_layers(man)
    depths = fifo_depths(layers)
    assert hpp_table("FIFO_DEPTH") == depths
    cpp = (ROOT / "hls" / "stream" / "yolo_stream.cpp").read_text()
    pragmas = dict(re.findall(r"STREAM variable=(s\d) depth=(\d+)", cpp))
    assert [int(pragmas[f"s{k}"]) for k in range(len(depths))] == depths
    p = plan(layers, BOARD)
    assert check_plan(p) == [0, 1, 2, 3, 4, 5, 8]
    W = np.arange(4 * 6 * 3 * 3).reshape(4, 6, 3, 3)
    r = rom_layout(W, 2, 3).reshape(2, 2, 3, 3, 2, 3)  # [og][ig][i][j][pe][s]
    assert r[1, 1, 2, 0, 1, 2] == W[1 * 2 + 1, 1 * 3 + 2, 2, 0]
