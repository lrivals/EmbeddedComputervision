"""T3.3 : mAP PASCAL VOC (§8.3), comparée au port Python du devkit officiel."""

import numpy as np
import pytest

from tests.voc_eval_ref import voc_eval
from yolo.data.voc import VOC_CLASSES, load_split, xyxy_to_cxcywh
from yolo.infer.metrics import (eval_class, evaluate, read_detections, to_voc_pixels, voc_ap,
                                write_detections)

CLASSES = ("cat", "dog", "person")


def _xml(name, w, h, objects):
    objs = "".join(
        f"<object><name>{c}</name><pose>Unspecified</pose><truncated>0</truncated>"
        f"<difficult>{int(d)}</difficult><bndbox><xmin>{b[0]}</xmin><ymin>{b[1]}</ymin>"
        f"<xmax>{b[2]}</xmax><ymax>{b[3]}</ymax></bndbox></object>"
        for c, b, d in objects)
    return (f"<annotation><filename>{name}.jpg</filename><size><width>{w}</width>"
            f"<height>{h}</height><depth>3</depth></size>{objs}</annotation>")


def _random_box(rng, w, h):
    x1, y1 = int(rng.integers(1, w - 20)), int(rng.integers(1, h - 20))
    return [x1, y1, int(rng.integers(x1 + 10, w + 1)), int(rng.integers(y1 + 10, h + 1))]


@pytest.fixture
def dataset(tmp_path):
    """Jeu de détections de test : 50 images, vérités dont `difficult`, détections bruitées,
    doublons, fausses alarmes et vérités manquées.
    """
    rng = np.random.default_rng(0)
    base = tmp_path / "VOC2007"
    (base / "Annotations").mkdir(parents=True)
    (base / "ImageSets" / "Main").mkdir(parents=True)
    ids = [f"{k:06d}" for k in range(50)]
    dets = {c: ([], [], []) for c in range(len(CLASSES))}

    def add(c, i, s, b):
        dets[c][0].append(i)
        dets[c][1].append(s)
        dets[c][2].append(b)

    for i in ids:
        w, h = int(rng.integers(200, 500)), int(rng.integers(200, 500))
        objects = []
        for _ in range(int(rng.integers(0, 5))):
            c = int(rng.integers(0, len(CLASSES)))
            b = _random_box(rng, w, h)
            objects.append((CLASSES[c], b, rng.random() < 0.15))
            r = rng.random()
            if r < 0.7:  # détection bruitée (parfois assez loin pour être un FP)
                add(c, i, rng.random(), np.array(b) + rng.normal(0, 8, 4))
            if r < 0.2:  # doublon
                add(c, i, rng.random(), np.array(b) + rng.normal(0, 4, 4))
        for _ in range(int(rng.integers(0, 3))):  # fausses alarmes
            add(int(rng.integers(0, len(CLASSES))), i, rng.random() * 0.5,
                np.array(_random_box(rng, w, h), dtype=float))
        (base / "Annotations" / f"{i}.xml").write_text(_xml(i, w, h, objects))
    (base / "ImageSets" / "Main" / "test.txt").write_text("\n".join(ids) + "\n")
    dets = {c: (v[0], np.array(v[1]), np.array(v[2]).reshape(-1, 4)) for c, v in dets.items()}
    write_detections(dets, CLASSES, tmp_path / "results")
    return tmp_path


@pytest.mark.parametrize("use_07", [True, False])
def test_matches_official_devkit(dataset, use_07):
    anns = _reindex(load_split(dataset, 2007, "test"))
    assert len(anns) == 50
    dets = read_detections(CLASSES, dataset / "results")
    aps, m = evaluate(dets, anns, len(CLASSES), use_07=use_07)
    ref = []
    for name in CLASSES:
        _, _, ap = voc_eval(str(dataset / "results" / "comp4_det_test_{}.txt"),
                            str(dataset / "VOC2007" / "Annotations" / "{}.xml"),
                            str(dataset / "VOC2007" / "ImageSets" / "Main" / "test.txt"),
                            name, ovthresh=0.5, use_07_metric=use_07)
        ref.append(ap)
    np.testing.assert_allclose(aps, ref, rtol=0, atol=1e-12)
    assert abs(m - np.mean(ref)) < 1e-12
    assert 0.1 < m < 0.9  # jeu non trivial


def _reindex(anns):
    """Indices de classe VOC → indices dans CLASSES (le jeu n'utilise que 3 classes)."""
    remap = np.array([CLASSES.index(n) if n in CLASSES else -1 for n in VOC_CLASSES])
    return [dict(a, labels=remap[a["labels"]]) for a in anns]


def _gts(*boxes, difficult=None):
    b = np.array(boxes, dtype=float).reshape(-1, 4)
    d = np.zeros(len(b), bool) if difficult is None else np.array(difficult)
    return {"a": (b, d)}


def test_perfect_is_one():
    gts = _gts([1, 1, 10, 10], [20, 20, 40, 40])
    _, _, ap = eval_class(["a", "a"], [0.9, 0.8], [[1, 1, 10, 10], [20, 20, 40, 40]], gts)
    assert ap == pytest.approx(1.0)


def test_duplicate_is_false_positive():
    # Une seule vérité, appariée une seule fois : la seconde détection est un FP.
    gts = _gts([1, 1, 10, 10])
    rec, prec, _ = eval_class(["a", "a"], [0.9, 0.8], [[1, 1, 10, 10], [1, 1, 10, 10]], gts)
    np.testing.assert_allclose(rec, [1, 1])
    np.testing.assert_allclose(prec, [1, 0.5])


def test_difficult_ignored():
    # Détection sur une vérité difficile : ni TP ni FP ; npos = 1.
    gts = _gts([1, 1, 10, 10], [50, 50, 80, 80], difficult=[False, True])
    rec, prec, ap = eval_class(["a", "a"], [0.9, 0.8], [[50, 50, 80, 80], [1, 1, 10, 10]], gts)
    np.testing.assert_allclose(rec, [0, 1])
    np.testing.assert_allclose(prec[1], 1.0)
    assert ap == pytest.approx(1.0)


def test_iou_threshold_inclusive():
    # Vérité 10×10 px, détection 10×5 px incluse : IoU = 50/100 = 0,5 exactement → TP.
    gts = _gts([1, 1, 10, 10])
    rec, _, _ = eval_class(["a"], [0.9], [[1, 1, 10, 5]], gts)
    assert rec[-1] == 1.0


def test_ap11():
    # Rappel 0,5 avec précision 1 : 6 niveaux (0 à 0,5) sur 11.
    assert voc_ap([0.5], [1.0]) == pytest.approx(6 / 11)


def test_voc_pixels_roundtrip():
    xyxy = np.array([[1, 1, 100, 50], [11, 6, 30, 25]], dtype=float)
    back = to_voc_pixels(xyxy_to_cxcywh(xyxy, 100, 50), 100, 50)
    np.testing.assert_allclose(back, xyxy, atol=1e-12)
