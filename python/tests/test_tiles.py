"""M15 (VisDrone) : entraînement en stretch et par découpes, inférence par tuiles."""

import numpy as np

pytest = __import__("pytest")
pytest.importorskip("PIL")
from PIL import Image  # noqa: E402

from yolo.data.augment import IDENTITY, affine, augment, random_crop  # noqa: E402
from yolo.data.loader import VOCDataset  # noqa: E402
from yolo.infer.tiles import merge, tile_grid, tile_to_image  # noqa: E402


def _rect_image(w, h, box):
    a = np.zeros((h, w, 3), np.uint8)
    x1, x2 = round((box[0] - box[2] / 2) * w), round((box[0] + box[2] / 2) * w)
    y1, y2 = round((box[1] - box[3] / 2) * h), round((box[1] + box[3] / 2) * h)
    a[y1:y2, x1:x2] = 255
    return Image.fromarray(a), np.array([(x1 + x2) / 2 / w, (y1 + y2) / 2 / h,
                                         (x2 - x1) / w, (y2 - y1) / h])


def test_stretch_identity_keeps_normalized_boxes():
    # En stretch, l'entrée est l'image étirée : le repère normalisé est celui d'origine,
    # comme `detect(..., mode="stretch")` à l'éval.
    img, box = _rect_image(320, 180, [0.4, 0.6, 0.2, 0.3])
    m = affine(320, 180, 416, IDENTITY, "stretch")
    np.testing.assert_allclose(m, [[416 / 320, 0, 0], [0, 416 / 180, 0]])
    out, b, _ = augment(img, box[None], [1], 416, IDENTITY, mode="stretch")
    np.testing.assert_allclose(b[0], box, atol=1e-12)
    ys, xs = np.nonzero(out.mean(axis=2) > 0.75)
    np.testing.assert_allclose([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                               [(box[0] - box[2] / 2) * 416, (box[1] - box[3] / 2) * 416,
                                (box[0] + box[2] / 2) * 416, (box[1] + box[3] / 2) * 416],
                               atol=1.5)


def test_random_crop_boxes_follow_pixels():
    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(40):
        img, box = _rect_image(800, 600, [rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8),
                                          0.05, 0.08])
        crop, b, labels = random_crop(img, box[None], [2], 300, rng)
        assert crop.size == (300, 300)
        a = np.asarray(crop).mean(axis=2) > 127
        if len(b) == 0:
            continue
        checked += 1
        assert labels.tolist() == [2]
        ys, xs = np.nonzero(a)
        cx, cy, bw, bh = b[0] * 300
        np.testing.assert_allclose([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1],
                                   [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2],
                                   atol=1.0)
    assert checked >= 5


def test_random_crop_drops_mostly_cut_boxes():
    img = Image.new("RGB", (200, 100))
    # boîte à cheval sur le bord droit d'une découpe calée à gauche : 25 % visible
    boxes = np.array([[(90 + 20) / 200, 0.5, 40 / 200, 0.2], [0.25, 0.5, 0.1, 0.2]])

    class Left:
        def integers(self, lo, hi):
            return 0

    _, b, labels = random_crop(img, boxes, [0, 1], 100, Left())
    assert labels.tolist() == [1]
    np.testing.assert_allclose(b[0], [0.5, 0.5, 0.2, 0.2])


def test_crop_larger_than_image_is_whole_image():
    img = Image.new("RGB", (120, 80))
    box = np.array([[0.5, 0.5, 0.2, 0.2]])
    out, b, _ = random_crop(img, box, [0], 640, np.random.default_rng(0))
    assert out.size == (120, 80)
    np.testing.assert_allclose(b, box)


def test_dataset_crop_only_in_training(tmp_path):
    img, box = _rect_image(800, 600, [0.5, 0.5, 0.05, 0.05])
    path = tmp_path / "a.jpg"
    img.save(path)
    s = {"image": path, "boxes": box[None], "labels": np.array([0]),
         "difficult": np.array([False])}
    ds = VOCDataset([s], train=False, crop=200, resize="stretch")
    _, b, _ = ds.load(0, 416)
    np.testing.assert_allclose(b[0], box, atol=1e-9)


@pytest.mark.parametrize("w,h,tile", [(1360, 765, 640), (2000, 1500, 832), (500, 300, 640)])
def test_tile_grid_covers_image(w, h, tile):
    grid = tile_grid(w, h, tile, 0.2)
    cover = np.zeros((h, w), bool)
    for x0, y0, tw, th in grid:
        assert 0 <= x0 and x0 + tw <= w and 0 <= y0 and y0 + th <= h
        assert tw == min(tile, w) and th == min(tile, h)
        cover[y0:y0 + th, x0:x0 + tw] = True
    assert cover.all()
    if w <= tile and h <= tile:
        assert grid == [(0, 0, w, h)]


def test_tile_to_image_and_merge():
    # même objet vu par deux tuiles qui se recouvrent → une détection après fusion
    w, h = 1000, 500
    t1, t2 = (0, 0, 500, 500), (400, 0, 500, 500)
    obj = np.array([[450, 250, 40, 40]], float)                 # pixels image (cx, cy, w, h)
    in1 = (obj - [t1[0], t1[1], 0, 0]) / [500, 500, 500, 500]
    in2 = (obj - [t2[0], t2[1], 0, 0]) / [500, 500, 500, 500]
    b1, b2 = tile_to_image(in1, t1, w, h), tile_to_image(in2, t2, w, h)
    np.testing.assert_allclose(b1, obj / [w, h, w, h])
    np.testing.assert_allclose(b2, obj / [w, h, w, h])
    other = tile_to_image([[0.5, 0.5, 0.1, 0.1]], t2, w, h)
    boxes, scores, labels = merge([(b1, [0.9], [3]), (b2, [0.7], [3]), (other, [0.5], [1])])
    assert labels.tolist() == [3, 1] and scores.tolist() == [0.9, 0.5]
    # deux classes différentes au même endroit : pas de suppression croisée
    _, _, labels = merge([(b1, [0.9], [3]), (b2, [0.7], [4])])
    assert sorted(labels.tolist()) == [3, 4]
    assert len(merge([])[0]) == 0
