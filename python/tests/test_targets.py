"""T2.3 : assignation cellule / ancre et cibles de régression (§5.1)."""

import numpy as np
import pytest

from yolo.data.targets import anchors_frac, build_targets, heads
from yolo.models.specs import TINY_YOLOV3_VOC

ANCHORS = TINY_YOLOV3_VOC["anchors"]  # 10,14 23,27 37,58 81,82 135,169 344,319
HEADS = heads(TINY_YOLOV3_VOC)
GRIDS = {16: 13, 23: 26}


def test_heads():
    assert HEADS == [(16, [3, 4, 5]), (23, [0, 1, 2])]
    region = {"layers": [{"type": "conv"}, {"type": "region", "num": 5}]}
    assert heads(region) == [(1, [0, 1, 2, 3, 4])]


def test_hand_built_boxes():
    # Boîtes taillées exactement comme une ancre : celle-ci est choisie sans ambiguïté.
    a = np.array(ANCHORS, dtype=np.float64) / 416
    boxes0 = [
        [0.50, 0.50, *a[4]],   # ancre 4 → tête 13×13, position 1
        [0.10, 0.90, *a[0]],   # ancre 0 → tête 26×26, position 0
    ]
    boxes1 = [
        [0.333, 0.777, a[2][0] * 1.1, a[2][1] * 0.9],  # ancre 2 → tête 26×26, position 2
    ]
    t = build_targets([boxes0, boxes1], [[3, 7], [12]], ANCHORS, HEADS, GRIDS)
    t13, t26 = t[16], t[23]
    assert t13["obj"].shape == (2, 3, 13, 13) and t26["obj"].shape == (2, 3, 26, 26)
    assert t13["obj"].sum() == 1 and t26["obj"].sum() == 2

    # Objet 0 : cellule ⌊0.5·13⌋ = 6.
    assert t13["obj"][0, 1, 6, 6]
    assert t13["x"][0, 1, 6, 6] == pytest.approx(0.5 * 13 - 6)
    assert t13["y"][0, 1, 6, 6] == pytest.approx(0.5 * 13 - 6)
    assert t13["tw"][0, 1, 6, 6] == pytest.approx(0.0, abs=1e-15)
    assert t13["th"][0, 1, 6, 6] == pytest.approx(0.0, abs=1e-15)
    assert t13["cls"][0, 1, 6, 6] == 3
    assert t13["scale"][0, 1, 6, 6] == pytest.approx(2 - a[4][0] * a[4][1])

    # Objet 1 : j = ⌊0.1·26⌋ = 2, i = ⌊0.9·26⌋ = 23.
    assert t26["obj"][0, 0, 23, 2]
    assert t26["x"][0, 0, 23, 2] == pytest.approx(0.1 * 26 - 2)
    assert t26["y"][0, 0, 23, 2] == pytest.approx(0.9 * 26 - 23)
    assert t26["cls"][0, 0, 23, 2] == 7

    # Objet 2 (image 1) : j = ⌊0.333·26⌋ = 8, i = ⌊0.777·26⌋ = 20.
    assert t26["obj"][1, 2, 20, 8]
    assert t26["tw"][1, 2, 20, 8] == pytest.approx(np.log(1.1))
    assert t26["th"][1, 2, 20, 8] == pytest.approx(np.log(0.9))
    assert t26["cls"][1, 2, 20, 8] == 12

    # Hors obj : cibles nulles, classe −1.
    for tt in (t13, t26):
        assert np.all(tt["cls"][~tt["obj"]] == -1)
        assert np.all(tt["scale"][~tt["obj"]] == 0)


def test_best_anchor_over_all_scales():
    # Pour chaque forme, l'ancre de meilleure IoU de forme est calculée à la main.
    a = anchors_frac(ANCHORS)
    rng = np.random.default_rng(0)
    for _ in range(50):
        w, h = rng.uniform(0.01, 0.9, 2)
        ious = [min(w, pw) * min(h, ph) / (w * h + pw * ph - min(w, pw) * min(h, ph))
                for pw, ph in a]
        k = int(np.argmax(ious))
        t = build_targets([[[0.5, 0.5, w, h]]], [[0]], ANCHORS, HEADS, GRIDS)
        hid, pos = (16, k - 3) if k >= 3 else (23, k)
        assert t[hid]["obj"][0, pos].sum() == 1
        other = 23 if hid == 16 else 16
        assert t[other]["obj"].sum() == 0


def test_targets_decode_back():
    # §5.1 inverse le décodage du §8.1 : (x* + j)/S = g_x, p_w e^{t_w*} = g_w.
    rng = np.random.default_rng(1)
    boxes = np.concatenate([rng.uniform(0, 1, (8, 2)), rng.uniform(0.02, 0.8, (8, 2))], axis=1)
    t = build_targets([boxes], [np.arange(8)], ANCHORS, HEADS, GRIDS)
    a = anchors_frac(ANCHORS)
    found = []
    for hid, mask in HEADS:
        tt, s = t[hid], GRIDS[hid]
        for _, pos, i, j in zip(*np.nonzero(tt["obj"])):
            pw, ph = a[mask[pos]]
            found.append([(tt["x"][0, pos, i, j] + j) / s, (tt["y"][0, pos, i, j] + i) / s,
                          pw * np.exp(tt["tw"][0, pos, i, j]), ph * np.exp(tt["th"][0, pos, i, j]),
                          tt["cls"][0, pos, i, j]])
    found = np.array(sorted(found, key=lambda r: r[4]))
    keep = found[:, 4].astype(int)  # collisions éventuelles
    np.testing.assert_allclose(found[:, :4], boxes[keep], atol=1e-12)


def test_edge_and_collision():
    a = anchors_frac(ANCHORS)
    # g_x = 1 : dernière cellule, x* = 1.
    t = build_targets([[[1.0, 1.0, *a[5]]]], [[0]], ANCHORS, HEADS, GRIDS)
    assert t[16]["obj"][0, 2, 12, 12]
    assert t[16]["x"][0, 2, 12, 12] == pytest.approx(1.0)
    # Deux objets, même cellule et même ancre : le dernier gagne.
    boxes = [[0.51, 0.51, *a[5]], [0.52, 0.52, *(a[5] * 0.95)]]
    t = build_targets([boxes], [[1, 2]], ANCHORS, HEADS, GRIDS)
    assert t[16]["obj"].sum() == 1
    assert t[16]["cls"][0, 2, 6, 6] == 2


def test_empty_image():
    t = build_targets([np.zeros((0, 4))], [np.zeros(0)], ANCHORS, HEADS, GRIDS)
    assert t[16]["obj"].sum() == 0 and t[23]["obj"].sum() == 0
