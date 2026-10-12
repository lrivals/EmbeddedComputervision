"""M18, M21 : prétraitements sur disque de tools/prep_datasets.py (DroneVehicle, UAVDT, ASL)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools import prep_datasets as P  # noqa: E402


def test_obb_to_yolo_border():
    # Losange de coins (200, 150), (300, 200), (200, 250), (100, 200) sur 840×712 : boîte
    # englobante [100, 300] × [150, 250], soit [0, 200] × [50, 150] une fois le cadre ôté.
    pts = [(200, 150), (300, 200), (200, 250), (100, 200)]
    line = "0 " + " ".join(f"{x / 840} {y / 712}" for x, y in pts)
    c, cx, cy, w, h = P.obb_to_yolo(line, 840, 712, 100).split()
    assert c == "0"
    np.testing.assert_allclose([float(v) for v in (cx, cy, w, h)],
                               [100 / 640, 100 / 512, 200 / 640, 100 / 512], atol=1e-6)
    # Entièrement dans le cadre : rien.
    assert P.obb_to_yolo("1 " + " ".join(["0.05"] * 8), 840, 712, 100) is None


def test_prep_dronevehicle(tmp_path):
    from PIL import Image

    src = tmp_path / "src"
    for sub in ("images", "labels"):
        (src / "train" / sub).mkdir(parents=True)
    for split in ("val", "test"):
        for sub in ("images", "labels"):
            (src / split / sub).mkdir(parents=True)
    Image.new("RGB", (840, 712), "white").save(src / "train" / "images" / "00007_jpg.rf.ab.jpg")
    (src / "train" / "labels" / "00007_jpg.rf.ab.txt").write_text(
        "0 0.2 0.2 0.3 0.2 0.3 0.3 0.2 0.3\n")
    out = tmp_path / "out"
    P.prep_dronevehicle(src, out, jobs=1)
    with Image.open(out / "train" / "images" / "00007.jpg") as img:
        assert img.size == (640, 512)
    assert (out / "train" / "labels" / "00007.txt").read_text().startswith("0 ")


def test_prep_uavdt(tmp_path):
    src = tmp_path / "src"
    for split in ("train", "test"):
        (src / split / "ann").mkdir(parents=True)
        (src / split / "img").mkdir(parents=True)
    ann = {"tags": [{"name": "sequence", "value": "M0203"}, {"name": "night"}],
           "size": {"width": 1024, "height": 540},
           "objects": [{"classTitle": "car", "points": {"exterior": [[10, 5], [29, 24]]},
                        "tags": [{"name": "small occlusion"}, {"name": "no out"}]}]}
    for name in ("M0203_img000001.jpg", "S0101_img000001.jpg"):
        (src / "test" / "img" / name).write_bytes(b"x")
        sot = dict(ann, objects=[dict(ann["objects"][0], classTitle="vehicle")])
        (src / "test" / "ann" / f"{name}.json").write_text(
            json.dumps(ann if name[0] == "M" else sot))
    out = tmp_path / "out"
    P.prep_uavdt(src, out)
    (r,) = json.loads((out / "annotations_test.json").read_text())  # séquence S écartée
    assert r["boxes"] == [[10, 5, 30, 25]] and r["labels"] == ["car"]
    assert r["sequence"] == "M0203" and r["tags"] == ["night"]
    assert r["occlusion"] == ["small"] and r["out"] == ["no"]
    assert (out / "images" / "test" / "M0203_img000001.jpg").exists()
    assert json.loads((out / "annotations_train.json").read_text()) == []


def test_landmarks_to_box():
    pts = [(50, 40), (150, 140), (100, 90)]
    np.testing.assert_allclose(P.landmarks_to_box(pts, 200, 200, margin=0.1),
                               [40, 30, 160, 150])
    # Clipped to the image.
    assert P.landmarks_to_box([(5, 0), (195, 200)], 200, 200) == [0.0, 0.0, 200.0, 200.0]


def test_asl_split():
    assert P.asl_frame("A1") == 1 and P.asl_frame("space3000") == 3000
    assert P.asl_frame("A_test") is None
    assert P.asl_split("del2550") == "train" and P.asl_split("del2551") == "val"


def test_asl_check():
    hand = [(50, 60), (100, 160)]
    assert P.asl_check(hand, 200, 200) is None
    assert P.asl_check([(190, 10), (230, 30), (240, 40)], 200, 200) == "outside"
    assert P.asl_check([(5, 5), (20, 40)], 200, 200) == "small"


def _fake_hands():
    """Fake detector: a hand everywhere except in image n° 3 (left out) ; one in `nothing`."""
    def detect(path):
        if path.stem.endswith("3") and not path.stem.startswith("nothing"):
            return None
        return np.array([[50.0, 60.0], [100.0, 160.0]]), "Right", 0.9, 0
    return detect


def test_prep_asl(tmp_path):
    from PIL import Image

    src = tmp_path / "ASL Alphabet"
    train = src / "asl_alphabet_train" / "asl_alphabet_train"
    test = src / "asl_alphabet_test" / "asl_alphabet_test"
    names = [train / c / f"{c}{n}.jpg" for c in ("A", "nothing") for n in (1, 3, 2551)]
    for p in names + [test / "A_test.jpg"]:
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (200, 200)).save(p)
    out = tmp_path / "asl"
    P.prep_asl(src, out, _fake_hands, jobs=1)
    tr = json.loads((out / "annotations_train.json").read_text())
    assert [e["image"] for e in tr] == ["images/train/A1.jpg", "images/train/nothing1.jpg",
                                        "images/train/nothing3.jpg"]
    a1 = tr[0]
    assert a1["labels"] == ["A"] and a1["frame"] == 1 and a1["hand"] == "Right"
    np.testing.assert_allclose(a1["boxes"][0], [42.5, 45, 107.5, 175])
    assert tr[1]["boxes"] == [] and tr[1]["hand"] == "Right"  # negative, hand counted
    val = json.loads((out / "annotations_val.json").read_text())
    assert [e["class"] for e in val] == ["A", "nothing"]
    (t,) = json.loads((out / "annotations_test.json").read_text())
    assert t["class"] == "A" and t["frame"] is None
    assert json.loads((out / "rejected.json").read_text()) == [
        {"image": "images/train/A3.jpg", "class": "A", "split": "train", "reason": "no hand"}]
    assert (out / "images" / "val" / "A2551.jpg").stat().st_ino == names[2].stat().st_ino
    # Detections cached: the detector is not called again, the annotations are rebuilt.
    (out / "annotations_train.json").unlink()
    P.prep_asl(src, out, lambda: None, jobs=1)
    assert json.loads((out / "annotations_train.json").read_text()) == tr
    # Loader on the output.
    from yolo.data import datasets as D

    assert len(D.load("asl", out, "train")) == 3
    P.asl_preview(out, tmp_path / "preview.png", per_class=1)
    assert (tmp_path / "preview.png").exists()
