"""T2.1 : conversions de boîtes et IoU (§5.2, §5.3)."""

import numpy as np
import pytest

from yolo.infer.boxes import cxcywh_to_xyxy, iou, iou_wh, iou_xyxy, xyxy_to_cxcywh


def _iou_scalar(a, b):
    # §5.3 écrit élément par élément, boîtes (cx, cy, w, h).
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def test_conversions_roundtrip():
    b = np.random.default_rng(0).uniform(0.1, 1, (7, 4))
    np.testing.assert_allclose(xyxy_to_cxcywh(cxcywh_to_xyxy(b)), b, atol=1e-15)
    np.testing.assert_allclose(cxcywh_to_xyxy([0.5, 0.5, 0.2, 0.4]), [0.4, 0.3, 0.6, 0.7])


@pytest.mark.parametrize("a,b,expected", [
    ([0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1], 0.0),        # disjointes
    ([0.2, 0.2, 0.2, 0.2], [0.4, 0.2, 0.2, 0.2], 0.0),        # contact sur un côté
    ([0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.2, 0.2], 0.0),        # contact par un coin
    ([0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.2, 0.2], 0.25),       # incluse : rapport d'aires
    ([0.3, 0.6, 0.2, 0.5], [0.3, 0.6, 0.2, 0.5], 1.0),        # identiques
    ([0.5, 0.5, 0.2, 0.2], [0.6, 0.5, 0.2, 0.2], 1 / 3),      # moitié recouverte
])
def test_iou_edge_cases(a, b, expected):
    assert iou(a, b)[0, 0] == pytest.approx(expected, abs=1e-15)
    assert iou(b, a)[0, 0] == pytest.approx(expected, abs=1e-15)


def test_iou_vectorized_matches_scalar():
    rng = np.random.default_rng(1)
    a = np.concatenate([rng.uniform(0, 1, (6, 2)), rng.uniform(0.05, 0.6, (6, 2))], axis=1)
    b = np.concatenate([rng.uniform(0, 1, (4, 2)), rng.uniform(0.05, 0.6, (4, 2))], axis=1)
    m = iou(a, b)
    assert m.shape == (6, 4)
    expected = [[_iou_scalar(x, y) for y in b] for x in a]
    np.testing.assert_allclose(m, expected, atol=1e-15)
    np.testing.assert_allclose(iou_xyxy(cxcywh_to_xyxy(a), cxcywh_to_xyxy(b)), m)
    assert (m > 0).any() and (m == 0).any()


def test_iou_zero_area():
    assert iou([0.5, 0.5, 0, 0], [0.5, 0.5, 0, 0])[0, 0] == 0.0


def test_iou_wh():
    # §5.2 : centres alignés.
    m = iou_wh([[2, 2], [1, 4]], [[1, 1], [2, 2], [4, 1]])
    np.testing.assert_allclose(m, [[0.25, 1.0, 2 / 6], [0.25, 2 / 6, 1 / 7]])
    wh = np.random.default_rng(2).uniform(0.1, 1, (5, 2))
    full = np.concatenate([np.full((5, 2), 0.5), wh], axis=1)
    np.testing.assert_allclose(iou_wh(wh, wh), iou(full, full), atol=1e-15)
