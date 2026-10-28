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


def test_person_match():
    ref = [[0.5, 0.5, 0.2, 0.4], [0.2, 0.2, 0.1, 0.1]]
    same = fc.person_match(ref, ref)
    assert same == (0, 0, 0)
    # Second person gone, a box slightly shifted (IoU ≈ 0.82: moved), one extra box.
    out = [[0.51, 0.51, 0.2, 0.4], [0.9, 0.9, 0.1, 0.1]]
    assert fc.person_match(ref, out) == (1, 1, 1)
    # Moved too far (IoU < 0.5): lost, and the faulty box is an extra.
    assert fc.person_match(ref[:1], [[0.6, 0.6, 0.2, 0.4]]) == (1, 1, 0)
    # One faulty box cannot cover two fault-free persons.
    assert fc.person_match([ref[0], ref[0]], ref[:1]) == (1, 0, 0)
    assert fc.person_match([], ref) == (0, 2, 0) and fc.person_match(ref, []) == (2, 0, 0)


def test_temporal_alarm():
    a, b = [0.5, 0.5, 0.2, 0.4], [0.2, 0.2, 0.1, 0.1]
    seen = [[a, b], [a, b], [a]]  # b missing in the last previous frame
    # A track present in the m previous frames and gone now raises the alarm.
    assert fc.temporal_alarm(seen, [b], 1) and fc.temporal_alarm(seen, [], 3)
    assert not fc.temporal_alarm(seen, [[0.51, 0.5, 0.2, 0.4]], 3)
    # b is not in the last previous frame: never a stable track, its absence is no alarm.
    assert len(fc.stable_tracks(seen, 1)) == 1
    # A track broken in an older frame is stable for small m only.
    broken = [[b], [a], [a]]
    assert fc.temporal_alarm(broken, [], 2) and not fc.temporal_alarm(broken, [], 3)
    assert not fc.temporal_alarm([[], [], []], [], 1)
    with pytest.raises(ValueError):
        fc.stable_tracks([[a]], 2)


def test_temporal_points():
    lab = lambda p: np.array([0] if p else [], int)  # noqa: E731
    samples = [{"id": f"s{i}", "sequence": "A", "frame": i, "labels": lab(i % 2)}
               for i in range(6)]
    samples += [{"id": "t0", "sequence": "B", "frame": 0, "labels": lab(1)}]
    pts = fc.temporal_points(samples, 0, 10, 3)
    # Frames 3 and 5 of A: a person and 2 contiguous previous frames; B is too short.
    assert pts == [[1, 2, 3], [3, 4, 5]]
    assert len(fc.temporal_points(samples, 0, 1, 3)) == 1


def test_campaign_person_criterion(setup):
    d, samples, qm, cache, meta = setup
    fc._setup(qm, cache, abft=False, conf=1e-4, iou=0.45, person=14, person_conf=1e-4)
    specs = fc.grid("ber", sorted(qm.convs), sites=("acc",), bers=(1e-2,))
    base, acc = fc.campaign(specs, samples, meta, 20, d / "p.jsonl", _serial)
    assert base["critical"] == base["lost"] == base["extra"] == 0
    assert base["person_images"] > 0
    assert 0 <= acc["critical"] <= 1 and acc["changed"] > 0
    rep = fc.report([base, acc])
    assert rep["person"] and "Critical images" in fc.markdown(rep)


def test_protect_grid(setup):
    d, samples, qm, cache, meta = setup
    convs = sorted(qm.convs)
    hit = fc.protected_hit("abft+dup-sel", convs)
    assert hit["weight"] == hit["bias"] == hit["acc"] == []
    assert hit["act"] == convs[fc.SENSITIVE:]
    assert fc.protected_hit("none", convs)["act"] == convs
    specs = fc.grid("protect", convs, bers=(1e-3,))
    assert len(specs) == 1 + len(fc.SCHEMES)
    none = next(s for s in specs if s.get("scheme") == "none")
    assert len(fc.faults_of(none)) == len(fc.SITES) * len(convs)
    fc._setup(qm, cache, abft=False, conf=1e-4, iou=0.45)
    keep = [specs[0], none, next(s for s in specs if s.get("scheme") == "abft+dup-sel")]
    keep.append({**keep[-1], "scheme": "all", "hit": {s: [] for s in fc.SITES}})
    base, _, _, masked = fc.campaign(keep, samples, meta, 20, d / "protect.jsonl", _serial)
    assert masked["flips"] == 0 and masked["map"] == base["map"]
    rows = [json.loads(line) for line in (d / "protect.jsonl").read_text().splitlines()]
    assert "| none |" in fc.markdown(fc.report(rows))
