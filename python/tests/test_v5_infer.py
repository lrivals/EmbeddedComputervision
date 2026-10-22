"""YOLOv5-style inference (T33.4): letterbox, decoding, NMS settings."""

import numpy as np
import pytest

from yolo.data.letterbox import (GRAY_V5, boxes_from_letterbox, boxes_to_letterbox,
                                 letterbox_image, letterbox_params_v5)
from yolo.data.targets import anchors_frac, heads, is_v5, v5_levels
from yolo.infer.decode import decode, decode_head
from yolo.infer.nms import filter_and_nms
from yolo.layers.activations import sigmoid
from yolo.models.specs import NETWORKS
from yolo.models.tiny_yolo import load_cfg


@pytest.mark.parametrize("w, h, expected", [
    (640, 480, (640, 480, 0, 80)),
    (480, 640, (480, 640, 80, 0)),
    (640, 427, (640, 427, 0, 106)),   # margin 213: 106 on top, 107 at the bottom
    (500, 333, (640, 426, 0, 107)),   # scale 1.28, margin 214
    (320, 320, (640, 640, 0, 0)),
])
def test_letterbox_params_v5(w, h, expected):
    assert letterbox_params_v5(w, h, 640) == expected


def test_letterbox_v5_image_and_boxes():
    img = np.full((427, 640, 3), 255, dtype=np.uint8)
    x, (nw, nh, dx, dy) = letterbox_image(img, 640, mode="v5")
    assert x.shape == (640, 640, 3)
    assert np.allclose(x[:dy], GRAY_V5) and np.allclose(x[dy + nh:], GRAY_V5)
    assert (640 - nh - dy) == dy + 1
    assert np.allclose(x[dy:dy + nh], 1.0)
    boxes = np.array([[0.3, 0.6, 0.2, 0.5], [0.5, 0.5, 1.0, 1.0]])
    lb = boxes_to_letterbox(boxes, 640, 427, 640, mode="v5")
    assert np.allclose(lb[1], [0.5, (dy + nh / 2) / 640, 1.0, nh / 640])
    assert np.allclose(boxes_from_letterbox(lb, 640, 427, 640, mode="v5"), boxes)


def test_is_v5_and_levels():
    assert is_v5(NETWORKS["yolov5n-coco"]) and is_v5(load_cfg("tiny-yolov7-coco"))
    assert not is_v5(load_cfg("tiny-yolov4-coco"))
    for net in (NETWORKS["yolov5n-coco"], load_cfg("tiny-yolov7-coco")):
        lv = v5_levels(net)
        assert [s for _, s, _ in lv] == [8, 16, 32]
        assert np.allclose(lv[0][2], [[10, 13], [16, 30], [33, 23]])
        assert np.allclose(lv[2][2], [[116, 90], [156, 198], [373, 326]])


def test_decode_head_v5_formulas():
    rng = np.random.default_rng(0)
    c = 3
    out = rng.standard_normal((1, 2 * (5 + c), 4, 5))
    anchors = np.array([[0.1, 0.2], [0.3, 0.15]])
    boxes, obj, scores = decode_head(out, anchors, c, "v5")
    p = out.reshape(1, 2, 5 + c, 4, 5)
    a, i, j = 1, 2, 3
    s = sigmoid(p[0, a, :, i, j])
    k = (a * 4 + i) * 5 + j
    assert np.allclose(boxes[0, k], [(2 * s[0] - 0.5 + j) / 5, (2 * s[1] - 0.5 + i) / 4,
                                     (2 * s[2]) ** 2 * 0.3, (2 * s[3]) ** 2 * 0.15])
    assert np.isclose(obj[0, k], s[4])
    assert np.allclose(scores[0, k], s[4] * s[5:])


@pytest.mark.parametrize("size", [416, 640])
def test_decode_v5_anchors_in_input_pixels(size):
    """The anchors are pixels of the actual input: at 640 a (116, 90) anchor is 116/640."""
    net = NETWORKS["yolov5n-coco"]
    rng = np.random.default_rng(1)
    outs = {hid: rng.standard_normal((1, 255, size // s, size // s))
            for hid, s, _ in v5_levels(net)}
    boxes, _, _ = decode(outs, net)
    hid, stride, anchors = v5_levels(net)[2]
    ref, _, _ = decode_head(outs[hid], anchors / size, 80, "v5")
    assert np.allclose(boxes[0, -ref.shape[1]:], ref[0])


def test_decode_v7_416_unchanged():
    """At the cfg size, the v5 path gives the decoding of T26.0 (anchors / 416, α = 2)."""
    net = load_cfg("tiny-yolov7-coco")
    rng = np.random.default_rng(2)
    outs = {hid: rng.standard_normal((2, 255, 416 // s, 416 // s))
            for hid, s, _ in v5_levels(net)}
    got = decode(outs, net)
    anchors = anchors_frac(net["anchors"])
    parts = [decode_head(outs[hid], anchors[mask], 80, "v3", 2.0, True)
             for hid, mask in heads(net)]
    for g, r in zip(got, (np.concatenate(x, axis=1) for x in zip(*parts))):
        assert np.allclose(g, r)


def test_nms_multi_label_and_max_det():
    boxes = np.array([[0.5, 0.5, 0.2, 0.2], [0.51, 0.5, 0.2, 0.2], [0.1, 0.1, 0.05, 0.05]])
    scores = np.array([[0.9, 0.8], [0.7, 0.002], [0.3, 0.0]])
    b, s, lab = filter_and_nms(boxes, scores, 0.001, 0.65)
    # Box 0 is kept for both classes (multi-label); box 1 is suppressed by box 0 in each class.
    assert list(zip(lab, np.round(s, 3))) == [(0, 0.9), (1, 0.8), (0, 0.3)]
    b, s, lab = filter_and_nms(boxes, scores, 0.001, 0.65, max_det=2)
    assert list(s) == [0.9, 0.8]
