"""T25.1: fault campaign tool (tools/fault_campaign.py) on a small random model."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.test_faults import _qm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fc = _load("fault_campaign")


def test_grids():
    convs = [0, 2, 4]
    ber = fc.grid("ber", convs, seeds=(0, 1))
    assert ber[0]["grid"] == "baseline"
    assert len(ber) == 1 + 2 * len(fc.SITES) * len(fc.BERS)
    layer = fc.grid("layer", convs, sites=("acc",), seeds=(0,))
    assert len(layer) == 1 + len(convs) * len(fc.LAYER_BERS)
    bit = fc.grid("bit", convs, seeds=(0,), flips=3)
    assert len(bit) == 1 + sum(len(fc.BITS[s]) for s in fc.SITES)
    assert len(fc.faults_of(bit[1])) == 3 and fc.faults_of(bit[0]) == []
    keys = [fc.spec_key(s) for s in ber + layer + bit]
    assert len(set(keys)) == len(keys) - 2  # the baseline appears in the three grids
    with pytest.raises(ValueError):
        fc.grid("all", convs)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from PIL import Image

    d = tmp_path_factory.mktemp("campaign")
    rng = np.random.default_rng(0)
    samples = []
    for k in range(3):
        path = d / f"{k}.png"
        Image.fromarray(rng.integers(0, 255, (60, 80, 3), dtype=np.uint8)).save(path)
        samples.append({"id": f"{k:06d}", "image": str(path), "width": 80, "height": 60,
                        "labels": np.array([k % 2]), "xyxy": np.array([[10., 10., 50., 40.]]),
                        "difficult": np.array([False])})
    qm, _ = _qm("tiny-yolov2-voc")
    cache = d / "cache"
    fc.prepare(cache, samples, qm, 64, jobs=1)
    meta = json.loads((cache / "meta.json").read_text())
    return d, samples, qm, cache, meta


def _serial(t):
    return [fc._image(x) for x in t]


def test_prepare_cache(setup):
    _, samples, qm, cache, meta = setup
    x = np.load(cache / "inputs.npy")
    assert x.shape == (3, 3, 64, 64) and x.dtype == np.int8
    assert meta["ids"] == [s["id"] for s in samples]
    for i in qm.outputs:
        assert np.load(cache / f"head_{i}.npy").shape[0] == 3


def test_campaign_baseline_resume_and_reproducible(setup):
    d, samples, qm, cache, meta = setup
    fc._setup(qm, cache, abft=True, conf=1e-4, iou=0.45)
    specs = fc.grid("ber", sorted(qm.convs), sites=("acc", "weight"), bers=(1e-3,))
    rows = fc.campaign(specs, samples, meta, 20, d / "a.jsonl", _serial)
    base, acc, weight = rows
    assert base["changed"] == 0 and base["detected"] == 0 and base["flips"] == 0
    assert acc["flips"] > 0 and acc["changed"] > 0
    assert acc["sdc"] == 0  # every accumulator fault is seen by the ABFT check
    assert weight["detected"] == weight["changed"] == 1.0
    # Resume: nothing left to run.
    assert fc.campaign(specs, samples, meta, 20, d / "a.jsonl", _serial) == []
    # Same seeds → same results, whatever was run before in the worker.
    again = fc.campaign(specs[::-1], samples, meta, 20, d / "b.jsonl", _serial)
    for r in again:
        twin = next(x for x in rows if x["key"] == r["key"])
        assert (r["map"], r["flips"], r["changed"]) == (twin["map"], twin["flips"],
                                                       twin["changed"])


def test_report_and_markdown():
    rows = [{"grid": "baseline", "site": None, "seed": 0, "map": 0.5, "images": 10,
             "changed": 0, "detected": 0, "sdc": 0, "flips": 0}]
    for seed, dm in ((0, 0.10), (1, 0.12)):
        for layer, f in ((0, 1.0), (2, 0.2)):
            rows.append({"grid": "layer", "site": "acc", "kind": "ber", "layer": layer,
                         "rate": 1e-5, "seed": seed, "map": 0.5 - dm * f, "images": 10,
                         "changed": 0.5, "detected": 0.5, "sdc": 0.0, "flips": 4})
        rows.append({"grid": "ber", "site": "act", "kind": "ber", "rate": 1e-6, "seed": seed,
                     "map": 0.5 - dm, "images": 10, "changed": 0.4, "detected": 0.1,
                     "sdc": 0.3, "flips": 2})
    rep = fc.report(rows)
    assert rep["baseline"] == 0.5
    assert [i for i, _ in rep["ranking"]] == [0, 2]
    assert rep["ranking"][0][1] == pytest.approx(0.11)
    act = rep["ber"][0]
    assert act["seeds"] == 2 and act["std"] == pytest.approx(np.std([0.4, 0.38], ddof=1))
    assert rep["abft"]["acc"]["detected_given_changed"] == 1.0
    assert rep["abft"]["act"]["detected_given_changed"] == pytest.approx(0.25)
    md = fc.markdown(rep)
    assert "| act | 39.00 ± 1.41 |" in md and "| L00 | -11.00 |" in md
    assert "Fault-free mAP: **50.00**" in md
