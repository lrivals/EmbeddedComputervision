"""T2.6 : letterbox, augmentation, chargeur."""

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from yolo.data.augment import (  # noqa: E402
    IDENTITY,
    adjust_hsv,
    affine,
    augment,
    hsv_to_rgb,
    random_params,
    rgb_to_hsv,
)
from yolo.data.letterbox import (  # noqa: E402
    boxes_from_letterbox,
    boxes_to_letterbox,
    letterbox_image,
    letterbox_params,
)
from yolo.data.loader import DataLoader, VOCDataset  # noqa: E402


def _rect_image(w, h, box):
    """Image noire avec un rectangle blanc à la boîte normalisée `box` (cx, cy, bw, bh)."""
    a = np.zeros((h, w, 3), np.uint8)
    x1, x2 = round((box[0] - box[2] / 2) * w), round((box[0] + box[2] / 2) * w)
    y1, y2 = round((box[1] - box[3] / 2) * h), round((box[1] + box[3] / 2) * h)
    a[y1:y2, x1:x2] = 255
    exact = [(x1 + x2) / 2 / w, (y1 + y2) / 2 / h, (x2 - x1) / w, (y2 - y1) / h]
    return Image.fromarray(a), np.array(exact)


def _white_bbox(img, size):
    ys, xs = np.nonzero(img.mean(axis=2) > 0.75)
    return np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]) / size


@pytest.mark.parametrize("w,h", [(500, 333), (333, 500), (416, 416), (640, 100)])
def test_letterbox_reversible(w, h):
    rng = np.random.default_rng(0)
    boxes = np.concatenate([rng.uniform(0.1, 0.9, (20, 2)), rng.uniform(0.01, 0.5, (20, 2))], 1)
    for size in (320, 416, 608):
        lb = boxes_to_letterbox(boxes, w, h, size)
        assert np.all(lb[:, :2] >= 0) and np.all(lb[:, :2] <= 1)
        np.testing.assert_allclose(boxes_from_letterbox(lb, w, h, size), boxes, atol=1e-6)


def test_letterbox_image():
    img, box = _rect_image(200, 100, [0.5, 0.5, 0.4, 0.4])
    out, (nw, nh, dx, dy) = letterbox_image(img, 128)
    assert out.shape == (128, 128, 3) and out.dtype == np.float32
    assert (nw, nh, dx, dy) == letterbox_params(200, 100, 128) == (128, 64, 0, 32)
    assert np.all(out[:dy] == 0.5) and np.all(out[dy + nh:] == 0.5)
    lb = boxes_to_letterbox(box, 200, 100, 128)[0]
    x1, y1, x2, y2 = _white_bbox(out, 128)
    np.testing.assert_allclose([x1, y1, x2, y2],
                               [lb[0] - lb[2] / 2, lb[1] - lb[3] / 2,
                                lb[0] + lb[2] / 2, lb[1] + lb[3] / 2], atol=1.5 / 128)


def test_identity_augment_is_letterbox():
    img, box = _rect_image(300, 200, [0.4, 0.6, 0.3, 0.2])
    m = affine(300, 200, 416, IDENTITY)
    nw, nh, dx, dy = letterbox_params(300, 200, 416)
    np.testing.assert_allclose(m, [[nw / 300, 0, dx], [0, nh / 200, dy]])
    _, b, _ = augment(img, box[None], [3], 416, IDENTITY)
    np.testing.assert_allclose(b, boxes_to_letterbox(box, 300, 200, 416), atol=1e-12)


def test_geometric_augment_applied_to_boxes():
    # Le rectangle blanc de l'image transformée coïncide avec la boîte transformée (±1,5 px).
    rng = np.random.default_rng(1)
    size = 256
    checked = 0
    for _ in range(30):
        img, box = _rect_image(300, 180, [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7),
                                          rng.uniform(0.1, 0.3), rng.uniform(0.1, 0.3)])
        params = random_params(rng)
        params["sat"] = params["val"] = 1.0
        out, b, labels = augment(img, box[None], [5], size, params)
        if len(b) == 0:
            continue
        checked += 1
        assert list(labels) == [5]
        cx, cy, bw, bh = b[0]
        np.testing.assert_allclose(_white_bbox(out, size),
                                   [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                                   atol=1.5 / size)
    assert checked >= 25


def test_random_params_ranges():
    rng = np.random.default_rng(2)
    ps = [random_params(rng) for _ in range(2000)]
    scale = np.array([p["scale"] for p in ps])
    shift = np.array([p["shift"] for p in ps])
    sat = np.array([p["sat"] for p in ps])
    assert 0.8 <= scale.min() and scale.max() <= 1.2  # §7.1 : ±20 %
    assert np.abs(shift).max() <= 0.2
    assert 1 / 1.5 <= sat.min() and sat.max() <= 1.5   # §7.1 : facteur 1,5
    assert 0.4 < np.mean([p["flip"] for p in ps]) < 0.6


def test_out_of_frame_box_dropped():
    img, box = _rect_image(200, 200, [0.95, 0.5, 0.1, 0.1])
    params = dict(IDENTITY, shift=np.array([0.2, 0.0]))
    _, b, labels = augment(img, box[None], [1], 128, params)
    assert len(b) == 0 and len(labels) == 0


def test_hsv_roundtrip_and_factors():
    rgb = np.random.default_rng(3).uniform(0, 1, (50, 3))
    np.testing.assert_allclose(hsv_to_rgb(rgb_to_hsv(rgb)), rgb, atol=1e-12)
    import colorsys
    np.testing.assert_allclose(rgb_to_hsv(rgb), [colorsys.rgb_to_hsv(*c) for c in rgb],
                               atol=1e-12)
    img = rgb.reshape(5, 10, 3).astype(np.float32)
    out = adjust_hsv(img, 1.0, 0.5)
    np.testing.assert_allclose(rgb_to_hsv(out)[..., 2], rgb_to_hsv(img)[..., 2] * 0.5, atol=1e-6)


@pytest.fixture
def samples(tmp_path):
    rng = np.random.default_rng(4)
    out = []
    for k in range(5):
        box = [rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7), 0.2, 0.3]
        img, exact = _rect_image(int(rng.integers(150, 300)), int(rng.integers(150, 300)), box)
        path = tmp_path / f"{k}.png"
        img.save(path)
        out.append({"image": path, "boxes": np.array([exact, [0.5, 0.5, 0.1, 0.1]]),
                    "labels": np.array([k, 9]), "difficult": np.array([False, True])})
    return out


def test_loader_batches(samples):
    loader = DataLoader(VOCDataset(samples), batch_size=2, seed=7)
    assert len(loader) == 2
    batches = list(loader.epoch(0, size_fn=lambda it: 64 + 32 * it))
    assert [b[0].shape for b in batches] == [(2, 3, 64, 64), (2, 3, 96, 96)]
    assert batches[0][0].dtype == np.float32
    for _, boxes, labels in batches:
        for lab in labels:
            assert 9 not in lab  # difficult retiré


def test_loader_parallel_is_deterministic(samples):
    ds = VOCDataset(samples)
    seq = list(DataLoader(ds, batch_size=2, seed=7).epoch(3, size_fn=lambda it: 96))
    par_loader = DataLoader(ds, batch_size=2, seed=7, workers=2, prefetch=1)
    par = list(par_loader.epoch(3, size_fn=lambda it: 96))
    par_loader.close()
    for (xa, ba, la), (xb, bb, lb) in zip(seq, par):
        np.testing.assert_array_equal(xa, xb)
        for u, v in zip(ba, bb):
            np.testing.assert_array_equal(u, v)
    # Reprise : sauter le premier lot donne le second.
    resumed = list(DataLoader(ds, batch_size=2, seed=7).epoch(3, lambda it: 96, skip=1))
    np.testing.assert_array_equal(resumed[0][0], seq[1][0])
