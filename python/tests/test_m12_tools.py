"""M12 : outils des profils de test sur PC (tools/m12_report.py, détections par image)."""

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rep = _load("m12_report")


def test_dets_per_image():
    eq = _load("eval_quant")
    per = {0: (["a", "a", "b"], np.zeros(3), np.zeros((3, 4))), 1: (["c"], np.zeros(1),
                                                                    np.zeros((1, 4)))}
    assert eq.dets_per_image(per, 2) == 2.0
    assert eq.dets_per_image({}, 0) == 0.0


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def test_loss_summary(tmp_path):
    p = tmp_path / "loss.csv"
    rows = [[i, 1e-4, 416, 10 - i / 20, 0, 0, 0, 0, 0.5] for i in range(100)]
    _write_csv(p, ["it", "lr", "size", "loss", "coord", "obj", "noobj", "cls", "seconds"], rows)
    s = rep.loss_summary(p, window=50)
    assert s["iters"] == 100 and s["s_per_iter"] == pytest.approx(0.5)
    assert s["start"] == pytest.approx(np.mean([10 - i / 20 for i in range(50)]))
    assert s["end"] == pytest.approx(np.mean([10 - i / 20 for i in range(50, 100)]))
    assert s["change"] < 0
    assert rep.loss_summary(p, window=500)["start"] == pytest.approx(np.mean(
        [10 - i / 20 for i in range(100)]))


def test_admm_summary(tmp_path):
    p = tmp_path / "admm.csv"
    _write_csv(p, ["it", "rho", "res_0", "res_2"], [[50, 0.0013, 0.05, 0.06],
                                                    [100, 0.0017, 0.03, 0.008]])
    s = rep.admm_summary(p)
    assert s["updates"] == 2 and s["it"] == 100 and s["worst"] == "0"
    assert s["max_res"] == pytest.approx(0.03) and not s["converged"]
    assert s["mean_slope"] == pytest.approx(((0.03 - 0.05) + (0.008 - 0.06)) / 2)


def test_map_table(tmp_path):
    p = tmp_path / "e.json"
    p.write_text(json.dumps({"images": 500, "resize": "stretch", "conf": 0.005, "iou": 0.45,
                             "results": {"int": {"map": 0.5566, "aps": [],
                                                 "dets_per_image": 12.5},
                                         "int-hwpp": {"map": 0.5556, "aps": [],
                                                      "overflow": 3}}}))
    rows = rep.map_rows([p])
    assert [r["variant"] for r in rows] == ["int", "int-hwpp"]
    table = rep.map_table(rows)
    assert "| e | int | 500 | stretch | 0.005 | 0.45 | 55.66 | 12.5 | — |" in table
    assert "| 55.56 | — | 3 |" in table


def test_dets_compare(tmp_path):
    line = {"image": "000001", "boxes": [[0.1, 0.2, 0.3, 0.4]], "scores": [0.9], "labels": [3]}
    ref = tmp_path / "ref.jsonl"
    ref.write_text(json.dumps(line) + "\n" + json.dumps(dict(line, image="000002")) + "\n")
    same = tmp_path / "same.jsonl"
    same.write_text(json.dumps(line) + "\n")
    other = tmp_path / "other.jsonl"
    other.write_text(json.dumps(dict(line, scores=[0.8])) + "\n")
    assert rep.dets_compare(ref, [same])[:2] == (1, [])
    assert rep.dets_compare(ref, [other])[:2] == (1, ["000001"])


def test_m12_script_syntax():
    subprocess.run(["bash", "-n", str(ROOT / "tools" / "m12.sh")], check=True)
    subprocess.run(["bash", "-n", str(ROOT / "tools" / "bench_sim.sh")], check=True)
