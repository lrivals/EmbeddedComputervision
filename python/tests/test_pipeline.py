"""T3.4 : prétraitement et détection de bout en bout (§8)."""

import numpy as np
import pytest

from yolo.infer.pipeline import detect, preprocess, resize_darknet
from yolo.models.tiny_yolo import build

Image = pytest.importorskip("PIL.Image")


def test_square_image_modes_agree():
    # Image carrée : letterbox sans bandes == redimensionnement direct.
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 256, (64, 64, 3), dtype=np.uint8))
    xl, wh = preprocess(img, 32, "letterbox")
    xs, _ = preprocess(img, 32, "stretch")
    assert xl.shape == (3, 32, 32) and xl.dtype == np.float32 and wh == (64, 64)
    np.testing.assert_array_equal(xl, xs)


def test_detect_letterbox_to_original():
    net = build("tiny-yolov2-voc", rng=0)
    # Biais de tête : toutes les ancres confiantes, pour avoir des détections.
    net.params[14]["b"][:] = 2.0
    rng = np.random.default_rng(1)
    img = Image.fromarray(rng.integers(0, 256, (200, 400, 3), dtype=np.uint8))
    x, wh = preprocess(img, 416, "letterbox")
    out = detect(net, x[None], [wh], "letterbox", conf_thr=0.0, iou_thr=1.0)
    raw = detect(net, x[None], [(416, 416)], "stretch", conf_thr=0.0, iou_thr=1.0)
    (b, s, lab), (rb, rs, _) = out[0], raw[0]
    assert len(b) == len(s) == len(lab) > 0
    np.testing.assert_array_equal(s, rs)
    # Image 400×200 → 416×208 centrée verticalement (dy = 104) : y' = (y·416 − 104)/208.
    np.testing.assert_allclose(b[:, 0], rb[:, 0], atol=1e-12)
    np.testing.assert_allclose(b[:, 1], (rb[:, 1] * 416 - 104) / 208, atol=1e-12)
    np.testing.assert_allclose(b[:, 3], rb[:, 3] * 2, atol=1e-12)


def _resize_darknet_ref(x, w, h):
    # resize_image de Darknet (image.c), boucle par boucle.
    H, W, _ = x.shape
    part, out = np.zeros((H, w, 3)), np.zeros((h, w, 3))
    ws, hs = (W - 1) / (w - 1), (H - 1) / (h - 1)
    for r in range(H):
        for c in range(w):
            if c == w - 1 or W == 1:
                part[r, c] = x[r, W - 1]
            else:
                sx = c * ws
                ix = int(sx)
                part[r, c] = (1 - (sx - ix)) * x[r, ix] + (sx - ix) * x[r, ix + 1]
    for r in range(h):
        sy = r * hs
        iy = int(sy)
        out[r] = (1 - (sy - iy)) * part[iy]
        if r == h - 1 or H == 1:
            continue
        out[r] += (sy - iy) * part[iy + 1]
    return out


@pytest.mark.parametrize("w,h", [(20, 15), (80, 60), (53, 37)])
def test_resize_darknet(w, h):
    x = np.random.default_rng(2).random((37, 53, 3))
    np.testing.assert_allclose(resize_darknet(x, w, h), _resize_darknet_ref(x, w, h),
                               atol=1e-6)
