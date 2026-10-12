"""M22: photo auto-labelling (tools/autolabel*.py): label I/O, NMS across prompts, alias
map, folder batch with a fake labeller, `--resume`, manual corrections.

A fake labeller returns fixed boxes: no weights, no network, no torch. The Gradio test is
skipped without the `autolabel` extra, the real-model test without its cached weights.
"""

import json
import sys
import zipfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yolo.data.datasets import _load_yolo_dir, parse_yolo_txt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

import autolabel as al  # noqa: E402
import autolabel_io as lio  # noqa: E402
import autolabel_models as lm  # noqa: E402


class FakeLabeller(lm.Labeller):
    """Two boxes per image (class 0 and the last class), plus a weak duplicate of the
    first under another class (removed by the agnostic NMS)."""
    name, version = "fake", "fake@1"

    def __init__(self):
        self.calls = 0

    def _predict(self, img, names, prompts, conf):
        self.calls += 1
        boxes = [[0.3, 0.4, 0.2, 0.3], [0.7, 0.6, 0.3, 0.2], [0.3, 0.4, 0.21, 0.3]]
        return np.array(boxes), np.array([0.9, 0.4, 0.2]), np.array([0, len(names) - 1, 1])


def _images(folder, names, size=(64, 48)):
    folder.mkdir(parents=True, exist_ok=True)
    for k, n in enumerate(names):
        p = folder / n
        p.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, (k * 40 % 256, 80, 120)).save(p)
    return lio.list_images(folder, recursive=True)


# ---------------------------------------------------------------- label I/O (T22.2)

def test_yolo_txt_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    wh = rng.uniform(0.05, 0.4, (7, 2))
    boxes = np.concatenate([rng.uniform(0.3, 0.7, (7, 2)), wh], axis=1)
    labels = rng.integers(0, 5, 7)
    lio.write_yolo_txt(tmp_path / "a.txt", boxes, labels)
    w, h = 500, 375
    corners, lab = parse_yolo_txt((tmp_path / "a.txt").read_text(), w, h, 5)
    c = np.array(corners) / [w, h, w, h]
    back = np.stack([(c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2,
                     c[:, 2] - c[:, 0], c[:, 3] - c[:, 1]], axis=1)
    np.testing.assert_allclose(back, boxes, atol=1e-6)
    assert lab == list(labels)
    b2, l2 = lio.read_yolo_txt(tmp_path / "a.txt")
    np.testing.assert_allclose(b2, boxes, atol=1e-6)
    assert list(l2) == list(labels)


def test_empty_result_writes_empty_file(tmp_path):
    lio.write_yolo_txt(tmp_path / "labels" / "neg.txt", np.zeros((0, 4)), [])
    assert (tmp_path / "labels" / "neg.txt").read_text() == ""


def test_labels_of_threshold_and_corrections():
    raw = lio.raw_record([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]],
                         [0.9, 0.3, 0.6], [0, 1, 1])
    rec = lio.image_record("x", (100, 100), raw)
    assert list(lio.labels_of(rec, 0.5)[3]) == [0, 2]
    assert list(lio.labels_of(rec, [0.95, 0.2])[3]) == [1, 2]  # one threshold per class
    rec.update(removed=[2], manual=[[0.1, 0.1, 0.05, 0.05, 1]])
    boxes, scores, labels, idx = lio.labels_of(rec, 0.5)
    assert list(idx) == [0, -1] and list(labels) == [0, 1] and scores[-1] == 1.0


def test_review_order_and_zip(tmp_path):
    images = {"a.jpg": {"n_boxes": 2, "best_score": 0.9},
              "b.jpg": {"n_boxes": 0, "best_score": 0.0},
              "c.jpg": {"n_boxes": 1, "best_score": 0.35}}
    assert lio.review_order(images) == ["b.jpg", "c.jpg", "a.jpg"]
    lio.write_yolo_txt(tmp_path / "r" / "labels" / "s" / "a.txt", [[0.5] * 4], [0])
    lio.write_classes(tmp_path / "r" / "classes.txt", ["cat"])
    lio.write_provenance(tmp_path / "r", {"images": {}})
    with zipfile.ZipFile(lio.zip_result(tmp_path / "r")) as z:
        assert sorted(z.namelist()) == ["autolabel.json", "classes.txt", "labels/s/a.txt"]


def test_exif_orientation_applied(tmp_path):
    img = Image.new("RGB", (40, 20))
    exif = img.getexif()
    exif[0x0112] = 6  # rotate 90° on display
    img.save(tmp_path / "r.jpg", exif=exif)
    assert lio.load_image(tmp_path / "r.jpg").size == (20, 40)


# ---------------------------------------------------------------- prompts and NMS (T22.1)

def test_parse_classes_and_alias_map():
    names, prompts = lm.parse_classes("person, tvmonitor, pottedplant=house plant, Person_sitting")
    assert names == ["person", "tvmonitor", "pottedplant", "Person_sitting"]
    assert prompts == ["person", "tv monitor", "house plant", "sitting person"]
    assert lm.same_class("tvmonitor", "tv monitor") and not lm.same_class("car", "cat")
    with pytest.raises(ValueError):
        lm.parse_classes("dog, dog")


def test_chunk_prompts():
    assert lm.chunk_prompts(["a"] * 5, [3, 3, 3, 3, 3], 8) == [[0, 1], [2, 3], [4]]
    assert lm.chunk_prompts(["a"], [50], 8) == [[0]]  # too long alone: still sent


def test_agnostic_nms_across_prompts():
    boxes = [[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.41, 0.4], [0.1, 0.1, 0.1, 0.1]]
    b, s, lab = lm.agnostic_nms(boxes, [0.6, 0.8, 0.3], [0, 1, 0])  # "dog" and "cat"
    assert list(s) == [0.8, 0.3] and list(lab) == [1, 0]


def test_predict_floor_and_nms():
    b, s, lab = FakeLabeller().predict(Image.new("RGB", (8, 8)), ["a", "b", "c"], conf=0.3)
    assert list(lab) == [0, 2] and len(b) == 2


def test_yolo_labeller_maps_classes_by_name():
    lab = lm.YoloLabeller.__new__(lm.YoloLabeller)
    lab.name = "yolo:x"
    lab.model = Namespace(names=("aeroplane", "car", "tvmonitor"),
                          infer=lambda img, conf: (np.eye(3, 4) + 0.1, np.array([0.9, 0.8, 0.7]),
                                                   np.array([0, 1, 2])))
    b, s, labels = lab.predict(Image.new("RGB", (8, 8)), ["tv monitor", "airplane"], 0.1)
    assert sorted(labels.tolist()) == [0, 1] and len(b) == 2  # "car" dropped


def test_missing_extra_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(lm.MissingExtra, match="autolabel"):
        lm.need_torch()


def test_list_runs(capsys):
    al.main(["--list"])
    out = capsys.readouterr().out
    assert "grounding-dino-tiny" in out and "device:" in out


# ---------------------------------------------------------------- folder batch (T22.3)

def test_folder_batch_and_resume(tmp_path):
    files = _images(tmp_path / "img", [f"{k}.jpg" for k in range(5)])
    (tmp_path / "img" / "bad.jpg").write_text("not an image")
    (tmp_path / "img" / "dup.jpg").write_bytes(files[0].read_bytes())
    files = lio.list_images(tmp_path / "img")
    lab, out = FakeLabeller(), tmp_path / "out"
    prov = al.run_folder(lab, files, tmp_path / "img", out, ["cat", "dog"], ["cat", "dog"],
                         conf=0.3)
    assert sorted(p.name for p in (out / "labels").iterdir()) == [f"{k}.txt" for k in range(5)] + ["dup.txt"]
    assert lio.read_classes(out / "classes.txt") == ["cat", "dog"]
    assert [u["path"] for u in prov["unreadable"]] == ["bad.jpg"]
    assert lab.calls == 5  # dup.jpg = 0.jpg: its raw predictions are reused (SHA-1 cache)
    saved = json.loads((out / "autolabel.json").read_text())
    assert saved["model"] == "fake" and len(saved["images"]) == 6
    assert len(saved["images"]["0.jpg"]["raw"]["scores"]) == 2  # floor 0.05, after NMS
    (tmp_path / "img" / "bad.jpg").unlink()  # the loader reads every file of the folder
    samples = _load_yolo_dir(tmp_path / "img", out / "labels", ["cat", "dog"])
    assert [len(s["labels"]) for s in samples] == [2] * 6  # read by the project loaders

    (tmp_path / "img" / "bad.jpg").write_text("not an image")
    lab2 = FakeLabeller()
    prov = al.run_folder(lab2, files, tmp_path / "img", out, ["cat", "dog"], ["cat", "dog"],
                         resume=True)
    assert lab2.calls == 0 and len(prov["unreadable"]) == 1
    with pytest.raises(ValueError, match="resume"):
        al.run_folder(lab2, files, tmp_path / "img", out, ["cat"], ["cat"], resume=True)


def test_recursive_names_do_not_collide(tmp_path):
    files = _images(tmp_path / "img", ["a/img.jpg", "b/img.jpg"])
    al.run_folder(FakeLabeller(), files, tmp_path / "img", tmp_path / "out", ["x"], ["x"])
    assert (tmp_path / "out/labels/a/img.txt").exists()
    assert (tmp_path / "out/labels/b/img.txt").exists()


def test_relabel_keeps_corrections(tmp_path):
    files = _images(tmp_path / "img", ["0.jpg"])
    out = tmp_path / "out"
    al.run_folder(FakeLabeller(), files, tmp_path / "img", out, ["a", "b"], ["a", "b"], conf=0.3)
    assert len(lio.read_yolo_txt(out / "labels/0.txt")[1]) == 2
    prov = lio.read_provenance(out)
    rec = prov["images"]["0.jpg"]
    shown = al.annotator_boxes(rec, prov["classes"], prov["conf"])
    boxes = shown[0][:1] + [{"xmin": 1, "ymin": 2, "xmax": 11, "ymax": 12, "label": "b"}]
    assert al.folder_edit(out, "0.jpg", shown, boxes)["edited"]
    assert al.folder_edit(out, "0.jpg", shown, boxes) is None  # idempotent
    al.relabel(out, 0.95)  # the 0.9 box goes, the hand box stays
    b, lab = lio.read_yolo_txt(out / "labels/0.txt")
    assert list(lab) == [1]
    np.testing.assert_allclose(b[0], [6 / 64, 7 / 48, 10 / 64, 10 / 48], atol=1e-6)


# ---------------------------------------------------------------- manual correction (T22.5)

def _record():
    raw = lio.raw_record([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]], [0.9, 0.8], [0, 1])
    return lio.image_record("x", (100, 50), raw)


def test_apply_edit_delete_draw_change_class():
    names = ["cat", "dog"]
    rec = _record()
    shown = al.annotator_boxes(rec, names, 0.3)
    boxes = [dict(b) for b in shown[0]]
    assert al.apply_edit(rec, names, shown, boxes)["edited"] is False  # value as written
    boxes[1]["label"] = "cat"                                          # change class
    boxes = boxes[1:] + [{"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10, "label": "dog"}]
    al.apply_edit(rec, names, shown, boxes)                            # delete + draw
    assert rec["removed"] == [0, 1] and rec["edited"]
    _, _, labels, idx = lio.labels_of(rec, 0.3)
    assert list(idx) == [-1, -1] and list(labels) == [0, 1]


def test_rerun_keeps_manual_boxes_and_deletions():
    names = ["cat", "dog"]
    rec = _record()
    shown = al.annotator_boxes(rec, names, 0.3)
    hand = {"xmin": 60, "ymin": 30, "xmax": 90, "ymax": 45, "label": "dog"}
    al.apply_edit(rec, names, shown, [shown[0][0], hand])  # dog box deleted, hand box drawn
    new = lio.raw_record([[0.2, 0.2, 0.1, 0.1], [0.75, 0.75, 0.3, 0.3], [0.5, 0.5, 0.2, 0.2]],
                         [0.7, 0.6, 0.95], [1, 0, 0])
    al.merge_manual(rec, new, names, names)
    assert rec["raw"]["scores"] == [0.95]  # deleted dog again, and the box under the hand one
    assert len(rec["manual"]) == 1 and rec["edited"]
    al.remap_manual(rec, names, ["dog"])
    assert rec["manual"][0][4] == 0


# ---------------------------------------------------------------- presets, app

def test_presets():
    assert al.preset_text("voc").startswith("aeroplane, bicycle")
    assert len(al.session_classes(preset="coco")[0]) == 80
    assert al.session_classes(preset="dataset:asl")[0][-1] == "space"
    with pytest.raises(ValueError):
        al.session_classes()


def test_gather_copies_inputs(tmp_path):
    files = _images(tmp_path / "src", ["a.jpg", "sub/b.png"])
    with zipfile.ZipFile(tmp_path / "z.zip", "w") as z:
        z.write(files[0], "x/a.jpg")
        z.writestr("../evil.jpg", b"x")
    n = al.gather([str(tmp_path / "z.zip"), str(files[0])], str(tmp_path / "src"), True,
                  tmp_path / "dest")
    assert n == 4  # zip (1), dropped file (1), folder (2); "../" refused
    assert sorted(p.name for p in (tmp_path / "src").rglob("*")) == ["a.jpg", "b.png", "sub"]


def test_build_app():
    pytest.importorskip("gradio")
    pytest.importorskip("gradio_image_annotation")
    args = Namespace(model="yolo:tiny-yolov2-voc", device="cpu", cfg=None, dataset=None,
                     classes="car", preset=None, conf=0.3)
    assert al.build_app(args) is not None


@pytest.mark.slow
def test_grounding_dino_finds_the_cars():
    """T22.1: on 000004.jpg with the prompt `car`, the VOC cars are found (IoU ≥ 0.5)."""
    pytest.importorskip("transformers")
    path = ROOT / "data/VOCdevkit/VOC2007/JPEGImages/000004.jpg"
    if not (path.exists() and lm.weights_ready("grounding-dino-tiny")):
        pytest.skip("VOC or the Grounding DINO weights missing")
    from yolo.data.voc import parse_annotation
    from yolo.infer.boxes import iou

    ann = parse_annotation(path.parents[1] / "Annotations" / "000004.xml")
    gt = ann["boxes"][ann["labels"] == 6]
    boxes, _, _ = lm.get_labeller("grounding-dino-tiny", "cpu").predict(
        lio.load_image(path), ["car"], 0.3)
    assert (iou(gt, boxes).max(axis=1) >= 0.5).sum() >= len(gt) - 1
