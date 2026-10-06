"""M11 : outils de préparation (tools/make_cfg.py, coco_subset.py, act_hist.py)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from yolo.models.tiny_yolo import CFG_DIR, CFG_FILES, build

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


make_cfg = _load("make_cfg")


@pytest.mark.parametrize("net", ["tiny-yolov2-voc", "tiny-yolov3-voc"])
def test_make_cfg_identity(net):
    text = (CFG_DIR / CFG_FILES[net]).read_text()
    assert make_cfg.make_cfg(text, 20) == text.rstrip("\n") + "\n"


@pytest.mark.parametrize("net,classes", [("tiny-yolov3-voc", 8), ("tiny-yolov2-voc", 1)])
def test_make_cfg_heads(tmp_path, net, classes):
    path = tmp_path / "n.cfg"
    path.write_text(make_cfg.make_cfg((CFG_DIR / CFG_FILES[net]).read_text(), classes))
    model = build(path)
    assert model.net["classes"] == classes
    out = model.forward(np.zeros((1, 3, 64, 64), np.float32))
    per_head = 3 if net == "tiny-yolov3-voc" else 5
    assert {v.shape[1] for v in out.values()} == {per_head * (5 + classes)}


def test_make_cfg_anchors(tmp_path):
    anchors = make_cfg.parse_anchors("8,20  15,40 30,60 50,90 90,120 150,200")
    path = tmp_path / "n.cfg"
    path.write_text(make_cfg.make_cfg((CFG_DIR / CFG_FILES["tiny-yolov3-voc"]).read_text(), 8,
                                      anchors))
    assert build(path).net["anchors"] == [list(map(int, a)) for a in anchors]


def test_coco_subset():
    sub = _load("coco_subset")
    data = {"images": [{"id": i} for i in range(10)],
            "annotations": [{"id": k, "image_id": k % 6} for k in range(12)],
            "categories": [{"id": 1}]}
    s = sub.subset(data, 4, seed=0)
    ids = {i["id"] for i in s["images"]}
    assert len(ids) == 4 and ids <= set(range(6))  # images annotées seulement
    assert all(a["image_id"] in ids for a in s["annotations"])
    assert len(s["annotations"]) == 2 * 4
    assert sub.subset(data, 4, seed=0)["images"] == s["images"]


def test_level_stats():
    ah = _load("act_hist")
    clip, levels = ah.level_stats(np.array([0.0, 0.1, 0.24, 1.0, 200.0]), 1.0)
    assert clip == pytest.approx(0.2)
    assert levels == 3  # 0, 1, 127
