"""M27: degradations, person-safety tool, dreaded events, failures per hour (T27.1-T27.7)."""

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import degrade as D  # noqa: E402
import dreaded_event as E  # noqa: E402
import person_safety as P  # noqa: E402

NET = "tiny-yolov3-coco"  # cfg only (80 classes, person = 0); weights for the end-to-end test


def _img(rng, h=60, w=80):
    return (rng.random((h, w, 3)) * 255).astype(np.uint8)


# ------------------------------------------------------------------------------ degrade
@pytest.mark.parametrize("spec", D.all_specs())
def test_degrade_shape_and_seed(spec):
    img = _img(np.random.default_rng(0))
    boxes = np.array([[10.0, 5, 40, 55]])
    a = D.apply(img, spec, D.rng_for(3, "x"), boxes)
    b = D.apply(img, spec, D.rng_for(3, "x"), boxes)
    assert a.shape == img.shape and a.dtype == np.uint8
    np.testing.assert_array_equal(a, b)  # same (seed, id): same image in any process
    assert not np.array_equal(a, img)


def test_degrade_severity_order():
    img = _img(np.random.default_rng(1))
    std = [D.apply(img, f"fog:{s}", D.rng_for(0, "a")).std() for s in D.SEVERITIES]
    assert std[0] > std[1] > std[2]  # contrast loss
    mean = [D.apply(img, f"low_light:{s}", D.rng_for(0, "a")).mean() for s in D.SEVERITIES]
    assert img.mean() > mean[0] > mean[1] > mean[2]
    grey = int(round(D.GREY * 255))
    for s, f in D.OCCLUSION.items():
        out = D.apply(img, f"occlusion:{s}", D.rng_for(0, "a"), [[20.0, 10, 60, 50]])
        covered = (out == grey).all(axis=2)[10:50, 20:60].mean()
        assert covered == pytest.approx(f, abs=0.03)
    out = D.apply(img, "occlusion:3", D.rng_for(0, "a"))  # no box: unchanged
    np.testing.assert_array_equal(out, img)


def test_degrade_parse():
    assert D.parse("jpeg:2") == ("jpeg", 2)
    for bad in ("jpeg", "jpeg:4", "rain:1"):
        with pytest.raises(ValueError):
            D.parse(bad)
    assert len(D.all_specs()) == 18


def test_contact_sheet():
    sheet = D.contact_sheet(_img(np.random.default_rng(2)), [[10.0, 5, 40, 55]], width=40)
    assert sheet.shape == (len(D.KINDS) * 30, 4 * 40, 3)


# ------------------------------------------------------------------ fake runs, report
def _industrial(root, n=10):
    """`n` labelled frames, one or two persons, every third frame empty."""
    from PIL import Image

    (root / "test" / "images").mkdir(parents=True)
    (root / "test" / "labels").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for i in range(n):
        Image.fromarray(_img(rng, 120, 160)).save(root / "test" / "images" / f"{i:03d}.jpg")
        lines = [] if i % 3 == 2 else [f"0 {0.2 + 0.05 * i:.3f} 0.5 0.15 {0.3 + 0.03 * k:.2f}"
                                       for k in range(1 + i % 2)]
        (root / "test" / "labels" / f"{i:03d}.txt").write_text("\n".join(lines))


def _fake_run(root, out, stage, miss=(), spec=None, seed=0):
    """Detections: each person found with a score in (0.5, 1), except the images of `miss`;
    one false alarm per image at a score in (0, 0.6)."""
    from yolo.data import datasets

    rng = np.random.default_rng(seed)
    path = P.dets_path("industrial", "test", NET, stage, spec, out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for s in datasets.load("industrial", root, "test"):
            boxes = [b for b in s["xyxy"].tolist()] if s["id"] not in miss else []
            scores = [float(rng.uniform(0.5, 1.0)) for _ in boxes]
            boxes.append([1.0, 1, 20, 20])
            scores.append(float(rng.uniform(0, 0.6)))
            f.write(json.dumps({"image": s["id"], "boxes": boxes, "scores": scores,
                                "labels": [0] * len(boxes)}) + "\n")


@pytest.fixture
def runs(tmp_path, monkeypatch):
    data, out = tmp_path / "industrial", tmp_path / "m27"
    _industrial(data)
    _fake_run(data, out, "float")
    _fake_run(data, out, "int", miss={"004"})
    _fake_run(data, out, "float", spec="fog:2", miss={"000", "001"})
    _fake_run(data, out, "int", spec="fog:2", miss={"000", "001", "004"})
    monkeypatch.setattr(P, "RESULTS", tmp_path / "person_safety.md")
    return data, out, tmp_path / "person_safety.md"


def _args(data, out, *rest):
    return ["--dataset", "industrial", "--data-root", str(data), "--split", "test", "--out",
            str(out), *rest]


def test_report_sections(runs):
    data, out, md = runs
    P.main(["report", *_args(data, out, "--nets", NET, "--stages", "float,int", "--write")])
    P.main(["oppoint", *_args(data, out, "--net", NET, "--block", "2", "--write")])
    P.main(["degrade-report", *_args(data, out, "--net", NET, "--specs", "fog:2", "--write")])
    P.main(["unflagged", *_args(data, out, "--net", NET, "--write")])
    text = md.read_text()
    for key in ("stages-industrial-test", f"oppoint-industrial-test-{NET}",
                f"degrade-industrial-test-{NET}", f"unflagged-industrial-test-{NET}"):
        assert f"<!-- m27:{key} -->" in text and f"<!-- /m27:{key} -->" in text
    assert text.startswith("# Person detection as a safety function")
    assert "quantization (integer − float)" in text
    # rewrite in place, no duplicate
    P.main(["report", *_args(data, out, "--nets", NET, "--stages", "float,int", "--write")])
    assert md.read_text().count("<!-- m27:stages-industrial-test -->") == 1
    op = json.loads((P.run_dir("industrial", "test", NET, out) / "oppoint.json").read_text())
    assert set(op["stages"]) == {"float", "int"} and "0.005" in op["stages"]["int"]["repo"]


def test_stage_metrics_values(runs):
    data, out, _ = runs
    m_f = P.stage_metrics("industrial", "test", NET, "float", data_root=data, out=out)
    m_i = P.stage_metrics("industrial", "test", NET, "int", data_root=data, out=out)
    # 10 frames, 3 empty, 4 with two persons → 11 persons; image 004 holds one person
    assert (m_f["npos"], m_f["n_images"]) == (11, 10)
    assert m_f["curve"]["recall"][-1] == 1.0
    assert m_i["curve"]["recall"][-1] == pytest.approx(10 / 11)
    m_e = P.stage_metrics("industrial", "test", NET, "float", data_root=data, out=out,
                          empty_only=True)
    assert m_e["n_images"] == 3


def test_unflagged_with_flags(runs, tmp_path):
    data, out, md = runs
    flags = tmp_path / "monitor.jsonl"
    flags.write_text("".join(json.dumps({"image": f"{i:03d}", "alarm": i == 4}) + "\n"
                             for i in range(10)))
    P.main(["unflagged", *_args(data, out, "--net", NET, "--fppi", "1.0", "--flags",
                                str(flags), "--write")])
    text = md.read_text()
    assert "| detector alone | person | 1 / 11 |" in text
    assert "| monitor | person | 0 / 11 |" in text  # the miss of 004 is flagged


def test_block_halves():
    s = [{"id": f"{c}{i}", "camera": c, "frame": i} for c in "AB" for i in range(5)]
    cal, chk = P.block_halves(s, 2)
    assert [x["id"] for x in cal] == ["A0", "A1", "A4", "B2", "B3"]
    assert len(cal) + len(chk) == 10 and not {x["id"] for x in cal} & {x["id"] for x in chk}


# --------------------------------------------------------------------- dreaded event
def _seq_samples(missing_frames, frames=10, gap_at=None, track=False):
    """One person per frame; frames listed in `missing_frames` get no detection."""
    samples, dets = [], {}
    for f in range(1, frames + 1):
        if f == gap_at:
            continue
        s = {"id": f"f{f}", "frame": f, "camera": "C", "xyxy": np.array([[1.0, 1, 20, 60]]),
             "labels": np.array([0]), "difficult": np.array([False]), "width": 100,
             "height": 100}
        if track:
            s["track"] = np.array([7])
        samples.append(s)
        found = f not in missing_frames
        dets[s["id"]] = (np.array([[1.0, 1, 20, 60]]) if found else np.zeros((0, 4)),
                         np.array([0.9]) if found else np.zeros(0),
                         np.array([0]) if found else np.zeros(0, np.int64))
    return samples, dets


@pytest.mark.parametrize("track", [False, True])
def test_dreaded_event_counts(track):
    samples, dets = _seq_samples({3, 4, 5, 8}, track=track)
    r = E.analyse(samples, dets, 0, 0.5, fps=10, min_height=0, max_n=4)
    assert r["mode"] == ("track" if track else "frame")
    assert r["counts"] == {"1": 2, "2": 1, "3": 1, "4": 0}
    assert r["frame_fn_rate"] == pytest.approx(0.4)
    assert r["hours"] == pytest.approx(10 / 10 / 3600)
    assert r["per_hour"]["1"] == pytest.approx(2 / r["hours"])
    assert r["events"]["3"][0]["images"] == ["f3", "f4", "f5"]
    assert r["dangerous"]["4"] == 1


def test_dreaded_event_gap_ends_run():
    # frame 4 absent: misses 3 and 5 are two runs, not one of length 3
    samples, dets = _seq_samples({3, 5}, gap_at=4)
    r = E.analyse(samples, dets, 0, 0.5, fps=10, min_height=0, max_n=3)
    assert r["counts"] == {"1": 2, "2": 0, "3": 0}
    assert [x["frames"] for x in r["runs"]] == [3, 6]


def test_dreaded_event_min_height():
    samples, dets = _seq_samples({3, 4})
    r = E.analyse(samples, dets, 0, 0.5, fps=10, min_height=100, max_n=2)
    assert r["units"] == 0 and r["counts"]["1"] == 0


@pytest.mark.parametrize("T,n", [(3, 1), (4, 2), (5, 3), (2, 3)])
def test_expected_runs_enumeration(T, n):
    p = 0.3
    exp = 0.0
    for seq in itertools.product([0, 1], repeat=T):
        prob = np.prod([p if x else 1 - p for x in seq])
        exp += prob * sum(L >= n for _, L in E.miss_runs(range(T), seq))
    assert E.expected_runs(T, p, n) == pytest.approx(exp)


# ------------------------------------------------------------------------------ DEEL
def test_failures_per_hour():
    r = P.failures_per_hour(fps=10, presence=0.5, p_frame=0.1, n_frames=3,
                            measured_per_hour=20.0, measured_presence=0.25)
    assert r["present_frames_per_hour"] == 18000
    assert r["independent"] == pytest.approx(18000 * 0.9 * 1e-3)
    assert r["measured"] == pytest.approx(40.0)
    g = P.sil_gap(1e-4)
    assert g[1] == pytest.approx(1.0) and g[3] == pytest.approx(3.0)


def test_deel_section(tmp_path, monkeypatch):
    monkeypatch.setattr(P, "RESULTS", tmp_path / "ps.md")
    samples, dets = _seq_samples({3, 4, 5, 8})
    r = E.analyse(samples, dets, 0, 0.5, fps=10, min_height=0, max_n=10)
    r.update(dataset="x", split="test", net=NET, stage="int", fps_source="argument")
    ev = tmp_path / "events.json"
    ev.write_text(json.dumps(r))
    P.main(["deel", "--events", str(ev), "--n", "3", "--presence", "0.2", "--write"])
    text = (tmp_path / "ps.md").read_text()
    assert "<!-- m27:deel-x -->" in text and "| independent |" in text and "| measured |" in text


# ------------------------------------------------------------------- end to end (T27.6)
@pytest.mark.skipif(not (ROOT / "weights" / "yolov3-tiny.weights").exists()
                    or not (ROOT / "build" / "quant" / NET / "calib.json").exists(),
                    reason="Tiny-YOLOv3 COCO weights or calibration absent")
def test_industrial_end_to_end(tmp_path, monkeypatch):
    """T27.6: a folder of 10 labelled images runs T27.1 (float, integer) and T27.3."""
    data, out = tmp_path / "industrial", tmp_path / "m27"
    _industrial(data)
    monkeypatch.setattr(P, "RESULTS", tmp_path / "ps.md")
    base = _args(data, out, "--net", NET, "--jobs", "2")
    P.main(["infer", *base])
    P.main(["infer", *base, "--degrade", "fog:2"])
    for st in ("float", "int"):
        assert len(P.read_dets(P.dets_path("industrial", "test", NET, st, None, out))) == 10
        assert P.dets_path("industrial", "test", NET, st, "fog:2", out).exists()
    P.main(["report", *_args(data, out, "--nets", NET, "--stages", "float,int", "--write")])
    P.main(["degrade-report", *_args(data, out, "--net", NET, "--specs", "fog:2", "--write")])
    text = (tmp_path / "ps.md").read_text()
    assert "stages-industrial-test" in text and "degrade-industrial-test" in text


def test_episode_outcomes():
    eps = [{"images": ["a", "b", "c", "d"], "runs": [[1, 3]]},   # run of 3 in b-d
           {"images": ["e", "f"], "runs": [[0, 2]]},              # too short for n = 3
           {"images": ["g", "h", "i"], "runs": [[0, 1]]}]         # no long run: success
    failed, flagged = P.episode_outcomes(eps, 3, {}, 1)
    assert (failed, flagged) == ([True, False], [False, False])
    _, flagged = P.episode_outcomes(eps, 3, {"b": True}, 1)
    assert flagged == [True, False]
    _, flagged = P.episode_outcomes(eps, 3, {"c": True}, 1)  # too late for deadline 1
    assert flagged == [False, False]
    _, flagged = P.episode_outcomes(eps, 3, {"c": True}, 2)
    assert flagged == [True, False]
