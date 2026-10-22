"""YOLOv5 targets (T33.5): anchor ratio test and neighbour cells."""

import numpy as np

from yolo.data.targets import build_targets_v5

# One level of stride 8 on a 64 × 64 input (8 × 8 grid), anchors 2 × 2 and 4 × 1 cells.
LEVEL = [(0, 8, np.array([[16.0, 16.0], [32.0, 8.0]]))]
GRID = {0: 8}


def cells(t):
    return sorted(zip(t["b"].tolist(), t["a"].tolist(), t["gj"].tolist(), t["gi"].tolist()))


def test_three_cells_and_offsets():
    # Centre (2.4, 5.6) cells: own (gi 2, gj 5), left (frac x 0.4), down (frac y 0.6).
    box = [[2.4 / 8, 5.6 / 8, 2 / 8, 2 / 8]]
    t = build_targets_v5([box], [[3]], LEVEL, GRID)[0]
    # The 2 × 2 box matches anchor 0 (r = 1) and anchor 1 (r = 0.5 and 2).
    assert cells(t) == [(0, a, gj, gi) for a in (0, 1) for gj, gi in ((5, 1), (5, 2), (6, 2))]
    sel = t["a"] == 0
    got = sorted(map(tuple, np.round(t["tbox"][sel], 6).tolist()))
    assert got == [(0.4, -0.4, 2, 2), (0.4, 0.6, 2, 2), (1.4, 0.6, 2, 2)]
    assert set(t["cls"]) == {3}
    assert np.allclose(t["anch"][sel], [2, 2])


def test_anchor_ratio_test():
    # w = 9 cells: r = 4.5 against anchor 0 (rejected) and 2.25 against anchor 1 (kept, h 1).
    box = [[0.5, 0.5, 9 / 8, 1 / 8]]
    t = build_targets_v5([box], [[0]], LEVEL, GRID)[0]
    assert set(t["a"]) == {1}
    # w = 0.4 cells: 1/r = 5 for anchor 0, 10 for anchor 1: no match at all.
    t = build_targets_v5([[[0.5, 0.5, 0.4 / 8, 0.4 / 8]]], [[0]], LEVEL, GRID)[0]
    assert len(t["b"]) == 0


def test_no_neighbour_outside_the_grid():
    # Corner boxes: x = 0.4 has frac < 0.5 but x < 1; x = 7.6 has frac > 0.5 but S − x < 1.
    boxes = [[0.4 / 8, 0.4 / 8, 2 / 8, 2 / 8], [7.6 / 8, 7.6 / 8, 2 / 8, 2 / 8]]
    t = build_targets_v5([boxes], [[0, 1]], [(0, 8, LEVEL[0][2][:1])], GRID)[0]
    assert cells(t) == [(0, 0, 0, 0), (0, 0, 7, 7)]
    # x = 8 exactly (right border): cell clamped to 7, offset 1.
    t = build_targets_v5([[[1.0, 0.5, 2 / 8, 2 / 8]]], [[0]], [(0, 8, LEVEL[0][2][:1])], GRID)[0]
    assert t["gi"].max() == 7 and np.isclose(t["tbox"][t["gi"] == 7, 0].max(), 1.0)


def test_random_boxes_stay_in_grid():
    rng = np.random.default_rng(0)
    levels = [(10, 8, np.array([[10, 13], [16, 30], [33, 23]], float)),
              (20, 16, np.array([[30, 61], [62, 45], [59, 119]], float))]
    grids = {10: (8, 12), 20: (4, 6)}
    boxes = [np.column_stack([rng.random((20, 2)), rng.random((20, 2)) * 0.6 + 0.01])
             for _ in range(3)]
    labels = [rng.integers(0, 5, 20) for _ in range(3)]
    t = build_targets_v5(boxes, labels, levels, grids)
    for hid, (sh, sw) in grids.items():
        d = t[hid]
        assert len(d["b"]) > 0
        assert d["gi"].min() >= 0 and d["gi"].max() < sw
        assert d["gj"].min() >= 0 and d["gj"].max() < sh
        assert d["tbox"][:, :2].min() >= -0.5 and d["tbox"][:, :2].max() <= 1.5
        # Each match comes from one box: the target width is its width in grid units.
        assert np.all(np.isin(np.round(d["tbox"][:, 2] / sw, 9),
                              np.round(np.concatenate(boxes)[:, 2], 9)))


def test_multi_level_match():
    levels = [(1, 8, np.array([[16.0, 16.0]])), (2, 16, np.array([[32.0, 32.0]]))]
    # 24 px box: r = 1.5 at stride 8, 0.75 at stride 16, matches both levels.
    t = build_targets_v5([[[0.45, 0.45, 24 / 64, 24 / 64]]], [[0]], levels, {1: 8, 2: 4})
    assert len(t[1]["b"]) == 3 and len(t[2]["b"]) == 3
    # Centre on a cell corner (x = y = 4): frac 0 is < 0.5 on both sides, so the reference
    # takes 5 matches; right and down floor back to the own cell (4 + 0.5 → 4).
    t = build_targets_v5([[[0.5, 0.5, 24 / 64, 24 / 64]]], [[0]], levels, {1: 8, 2: 4})
    assert cells(t[1]) == [(0, 0, 3, 4), (0, 0, 4, 3)] + [(0, 0, 4, 4)] * 3
