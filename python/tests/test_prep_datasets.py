"""M18 : prétraitements sur disque de tools/prep_datasets.py (DroneVehicle, UAVDT)."""

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
