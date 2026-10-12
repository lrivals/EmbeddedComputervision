"""M20: live webcam demo (tools/webcam.py): model registry, frame slot, headless run.

The OpenCV tests are skipped when `cv2` is missing (pip install -e "python[webcam]").
"""

import importlib.util
import threading
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from yolo.data.datasets import COCO_CLASSES, VISDRONE_CLASSES
from yolo.data.voc import VOC_CLASSES
from yolo.infer.pipeline import detect, preprocess
from yolo.models.tiny_yolo import build

ROOT = Path(__file__).resolve().parents[2]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


webcam = _load("webcam")

CFG = """[net]
width=32
height=32
channels={channels}

[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
filters=25
size=1
stride=1
pad=1
activation=linear

[region]
anchors=1.0,1.5
classes=20
coords=4
num=1
softmax=1
"""


def _checkpoint(tmp_path, channels=3):
    """Tiny cfg and a `Trainer.save_checkpoint`-like file with random BN state."""
    cfg = tmp_path / f"tiny{channels}.cfg"
    cfg.write_text(CFG.format(channels=channels))
    net = build(cfg, rng=0)
    rng = np.random.default_rng(1)
    arrays = {"it": np.array(10), "seed": np.array(0), "velocity/0/W": np.zeros(1)}
    for i, (p, s) in enumerate(zip(net.params, net.state)):
        for k, v in p.items():
            p[k] = (v + rng.normal(0, 0.1, v.shape)).astype(v.dtype)
            arrays[f"param/{i}/{k}"] = p[k]
        for k, v in s.items():
            s[k] = (np.abs(v) + rng.uniform(0.1, 1, v.shape)).astype(v.dtype)
            arrays[f"state/{i}/{k}"] = s[k]
    path = tmp_path / f"ckpt{channels}.npz"
    np.savez(path, **arrays)
    return net, cfg, path


def _image(seed=0, w=48, h=40):
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


def test_class_names():
    assert webcam.class_names(20) == VOC_CLASSES
    assert webcam.class_names(80) == COCO_CLASSES
    assert webcam.class_names(len(VISDRONE_CLASSES), "visdrone") == VISDRONE_CLASSES
    assert webcam.class_names(3) == ("car", "truck", "bus")  # UAVDT, by count
    assert webcam.class_names(7) == tuple(f"class{c}" for c in range(7))


def test_latest_slot_keeps_newest():
    slot = webcam.LatestSlot()
    assert slot.get(0, timeout=0.01) is None
    assert [slot.put(k) for k in "abc"] == [1, 2, 3]
    assert slot.get(0) == (3, "c")
    assert slot.get(3, timeout=0.01) is None
    threading.Timer(0.05, slot.put, ["d"]).start()
    assert slot.get(3, timeout=2) == (4, "d")


def test_registry_files(tmp_path, monkeypatch):
    monkeypatch.setattr(webcam, "WEIGHTS", tmp_path / "weights")
    monkeypatch.setattr(webcam, "MODEL_DIR", tmp_path / "model")
    assert webcam.available() == []
    (tmp_path / "weights").mkdir()
    (tmp_path / "weights" / "yolov3-tiny.weights").write_bytes(b"")
    (tmp_path / "model" / "tiny-yolov2-voc").mkdir(parents=True)
    (tmp_path / "model" / "tiny-yolov2-voc" / "manifest.json").write_text("{}")
    assert webcam.available() == ["tiny-yolov3-coco", "int:tiny-yolov2-voc"]
    with pytest.raises(ValueError):
        webcam.model_files("yolov9")


@pytest.mark.parametrize("channels", [3, 1])
def test_checkpoint_model_matches_detect(tmp_path, channels):
    net, cfg, path = _checkpoint(tmp_path, channels)
    model = webcam.load_model(f"ckpt:{path}", cfg)
    assert (model.size, model.channels, model.names) == (32, channels, VOC_CLASSES)
    img = _image()
    x, wh = preprocess(img, 32, "letterbox", "pil", channels)
    ref = detect(net, x[None], [wh], "letterbox", 0.01, 0.45)[0]
    got = model.infer(img, 0.01, 0.45)
    assert len(ref[0]) > 0
    for a, b in zip(got, ref):
        np.testing.assert_array_equal(a, b)


def test_checkpoint_of_other_cfg_is_refused(tmp_path):
    _, _, path = _checkpoint(tmp_path, 3)
    cfg = tmp_path / "other.cfg"
    cfg.write_text(CFG.format(channels=3).replace("filters=8", "filters=16"))
    with pytest.raises(ValueError, match="does not match"):
        webcam.load_model(f"ckpt:{path}", cfg)


VOC_IMAGE = ROOT / "data" / "VOCdevkit" / "VOC2007" / "JPEGImages" / "000004.jpg"


@pytest.mark.parametrize("spec", ["tiny-yolov3-coco", "int:tiny-yolov2-voc"])
def test_pretrained_models(spec):
    missing = [p for p in webcam.model_files(spec) + [VOC_IMAGE] if not p.exists()]
    if missing:
        pytest.skip(f"missing {missing[0]}")
    model = webcam.load_model(spec)
    with Image.open(VOC_IMAGE) as img:
        boxes, scores, labels = model.infer(img)
    assert len(labels) > 0
    names = COCO_CLASSES if "coco" in spec else VOC_CLASSES
    assert model.names == names
    assert "car" in {names[c] for c in labels}  # 000004: cars


def test_draw_cv_matches_pixel_boxes():
    pytest.importorskip("cv2")
    img = np.zeros((100, 200, 3), np.uint8)
    boxes = np.array([[0.5, 0.5, 0.5, 0.4], [0.2, 0.05, 0.2, 0.1]])
    webcam.draw_cv(img, boxes, [0.9, 0.5], [6, 14], VOC_CLASSES)
    col = webcam.COLORS[6][::-1]
    # (x1, y1, x2, y2) = (50, 30, 150, 70): the box edges are drawn in the class colour.
    assert tuple(img[50, 50]) == col and tuple(img[50, 150]) == col
    assert tuple(img[70, 100]) == col
    assert img[50, 100].sum() == 0  # inside the box


def test_headless_run_on_folder(tmp_path):
    pytest.importorskip("cv2")
    _, cfg, path = _checkpoint(tmp_path, 3)
    src = tmp_path / "frames"
    src.mkdir()
    for k in range(3):
        _image(k, 64, 48).save(src / f"{k}.png")
    model = webcam.load_model(f"ckpt:{path}", cfg)
    out = tmp_path / "out"
    demo = webcam.run(webcam.FileSource(src), model, conf=0.01, window=False, save=out,
                      max_frames=2)
    assert demo.error is None and demo.n_infer == 2
    assert sorted(p.name for p in out.iterdir()) == ["frame_00000.jpg", "frame_00001.jpg"]
    assert demo.summary()["inferences"] == 2
