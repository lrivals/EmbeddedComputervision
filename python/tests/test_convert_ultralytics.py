"""T26.1 / T33.0: torch-free converter of Ultralytics checkpoints.

Random state dicts check the folding and the parameter counts; with the official weights
(not in git, AGPL-3.0, see tools/convert_ultralytics.py), the output is compared with the
one of the former torch converter (`weights/m33/<net>_t261.*`, M33 step 0).
"""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from yolo.models import ul_graph, ul_net

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "weights" / "m33"
CHECKPOINTS = ROOT / "weights" / "ultralytics"
_spec = importlib.util.spec_from_file_location("convert_ultralytics",
                                               ROOT / "tools" / "convert_ultralytics.py")
cu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cu)


@pytest.mark.parametrize("net, params", [("yolov5n", 1_867_405), ("yolov8n", 3_151_888)])
def test_random_weights(net, params):
    full = cu.describe(net)
    sd = ul_net.random_state_dict(full, rng=3)
    desc, w = ul_net.fold(full, sd)
    assert sum(v.size for v in w.values()) == params  # published 1.87 M, 3.15 M
    assert all(v.dtype == np.float32 for v in w.values())
    x = np.random.default_rng(0).random((1, 3, 96, 96))
    want = ul_net.heads(ul_net.build(full, sd), x)
    got = ul_graph.forward(desc, w, x)
    no = 3 * 85 if net == "yolov5n" else 64 + 80
    assert [g.shape[1] for g in got] == [no] * 3
    # float32 folded weights: relative error of the float32 rounding
    assert max(float(np.abs(a - b).max() / np.abs(b).max()) for a, b in zip(got, want)) <= 1e-5


def test_cli_random(tmp_path):
    out = subprocess.run([sys.executable, str(ROOT / "tools" / "convert_ultralytics.py"),
                          "--net", "yolov5n", "--random", "--out", str(tmp_path), "--check", "0"],
                         capture_output=True, text=True, check=True)
    assert "1,867,405 parameters" in out.stdout
    desc = json.loads((tmp_path / "yolov5n.json").read_text())
    assert desc["anchors"][0] == [10, 13, 16, 30, 33, 23] and len(desc["layers"]) == 85


@pytest.mark.parametrize("net", ["yolov5n", "yolov8n"])
def test_official_equals_t261(net):
    """Same json and npz (bit for bit) as the torch converter of T26.1; anchors in pixels,
    where T26.1 wrote the stride units of the checkpoint."""
    pt, ref = CHECKPOINTS / f"{net}.pt", FIXTURES / f"{net}_t261"
    if not (pt.exists() and ref.with_suffix(".json").exists()):
        pytest.skip("official weights or fixtures absent (M33 step 0)")
    full, sd = cu.load(pt)
    desc, w = ul_net.fold(full, sd)
    want = json.loads(ref.with_suffix(".json").read_text())
    if net == "yolov5n":
        stride_units = np.array(want.pop("anchors"))
        assert np.array_equal(np.array(desc.pop("anchors")),
                              stride_units * np.array([8, 16, 32])[:, None])
    assert desc == want
    tw = np.load(ref.with_suffix(".npz"))
    assert sorted(w) == sorted(tw.files)
    assert all(w[k].tobytes() == tw[k].tobytes() for k in tw.files)
