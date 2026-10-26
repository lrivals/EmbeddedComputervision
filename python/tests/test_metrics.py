"""T3.3 : mAP PASCAL VOC (§8.3), comparée au port Python du devkit officiel."""

import math

import numpy as np
import pytest

from tests import mr_ref
from tests.voc_eval_ref import voc_eval
from yolo.data.voc import VOC_CLASSES, load_split, xyxy_to_cxcywh
from yolo.infer.metrics import (clopper_pearson, eval_class, evaluate, log_average_miss_rate,
                                match_class, mr_fppi, person_metrics, point_at_threshold,
                                read_detections, recall_at_fppi, threshold_for_recall,
                                to_voc_pixels, unflagged_miss_rate, voc_ap, write_detections)

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


def test_ignore_regions_neither_tp_nor_fp():
    # T15.10 : une détection non appariée centrée dans une région ignorée ne compte pas.
    gts = {"a": (np.array([[1, 1, 10, 10]]), np.array([False]))}
    dets = (["a", "a"], [0.9, 0.8], [[50, 50, 60, 60], [1, 1, 10, 10]])
    _, prec, ap = eval_class(*dets, gts)
    assert ap < 0.99 and prec[-1] == 0.5
    rec, prec, ap = eval_class(*dets, gts, ignore={"a": np.array([[45, 45, 70, 70]])})
    assert ap == pytest.approx(1.0) and prec[-1] == 1.0 and rec[-1] == 1.0
    # image sans vérité de la classe : même règle
    rec, prec, _ = eval_class(["b"], [0.9], [[50, 50, 60, 60]], gts,
                              ignore={"b": np.array([[45, 45, 70, 70]])})
    assert prec[-1] == 0.0 and rec[-1] == 0.0


def test_evaluate_reads_ignore_regions():
    ann = {"id": "a", "xyxy": np.array([[1.0, 1, 10, 10]]), "labels": np.array([0]),
           "difficult": np.array([False]), "ignore_xyxy": np.array([[45.0, 45, 70, 70]])}
    dets = {0: (["a", "a"], np.array([0.9, 0.8]), np.array([[50.0, 50, 60, 60],
                                                             [1, 1, 10, 10]]))}
    aps, _ = evaluate(dets, [ann], 1)
    assert aps[0] == pytest.approx(1.0)
    aps, _ = evaluate(dets, [ann], 1, use_ignore=False)
    assert aps[0] < 0.99


# ---------------------------------------------------------------------- T27.0 person metrics
def _sample(i, boxes, difficult=None, visible=None, w=100, h=100):
    b = np.array(boxes, dtype=float).reshape(-1, 4)
    s = {"id": i, "width": w, "height": h, "xyxy": b, "labels": np.zeros(len(b), np.int64),
         "difficult": np.zeros(len(b), bool) if difficult is None else np.array(difficult)}
    if visible is not None:
        s["visible"] = np.array(visible, float)
    return s


def _toy():
    """3 images. A: g1 (height 100), g2 (height 20); B: g3 (height 50), g4 difficult; C: empty.
    Detections by score: .9 g1 TP, .8 on C FP, .7 g3 TP, .6 g4 ignored, .5 g1 again FP,
    .4 g2 TP."""
    samples = [_sample("A", [[1, 1, 50, 100], [60, 1, 80, 20]], visible=[1.0, 0.5]),
               _sample("B", [[1, 1, 40, 50], [50, 50, 90, 90]], [False, True], [0.2, 1.0]),
               _sample("C", [])]
    dets = (["A", "C", "B", "B", "A", "A"], np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4]),
            np.array([[1, 1, 50, 100], [10, 10, 30, 30], [1, 1, 40, 50], [50, 50, 90, 90],
                      [2, 2, 50, 100], [60, 1, 80, 20]], float))
    return samples, dets


def test_toy_curve_by_hand():
    samples, dets = _toy()
    m = person_metrics(dets, samples)
    c = m["curve"]
    assert m["npos"] == 3 and m["n_images"] == 3
    np.testing.assert_allclose(c["thr"], [np.inf, .9, .8, .7, .6, .5, .4])
    np.testing.assert_allclose(c["fppi"], [0, 0, 1 / 3, 1 / 3, 1 / 3, 2 / 3, 2 / 3])
    np.testing.assert_allclose(c["recall"], [0, 1 / 3, 1 / 3, 2 / 3, 2 / 3, 2 / 3, 1])
    # operating point at 0.1 and 0.01 FPPI: threshold .9, recall 1/3
    for f in (0.1, 0.01):
        assert m["at"][f]["recall"] == pytest.approx(1 / 3) and m["at"][f]["thr"] == 0.9
    assert recall_at_fppi(c, 0.5) == (pytest.approx(2 / 3), 0.6)
    assert recall_at_fppi(c, 1.0) == (1.0, 0.4)
    # LAMR: 7 references below 1/3 → miss 2/3, 0.562 → 1/3, 1.0 → 0 (floor 1e-10)
    ref = math.exp((7 * math.log(2 / 3) + math.log(1 / 3) + math.log(1e-10)) / 9)
    assert m["lamr"] == pytest.approx(ref, rel=1e-12)
    # per height at .9: far (g2) 0/1, medium (g3) 0/1, near (g1) 1/1; g4 difficult left out
    assert m["at"][0.1]["height"] == [(0, 1), (0, 1), (1, 1)]
    # per occlusion (1 − visible): g1 0 → bin 0, g3 0.8 → bin 3, g2 0.5 → bin 2
    assert m["at"][0.1]["occlusion"] == [(1, 1), (0, 0), (0, 1), (0, 1)]
    assert point_at_threshold(c, 0.5) == (pytest.approx(2 / 3), pytest.approx(2 / 3))
    assert point_at_threshold(c, 0.005) == (1.0, pytest.approx(2 / 3))
    assert threshold_for_recall(c, 0.95) == 0.4 and threshold_for_recall(c, 0.5) == 0.7


def test_toy_empty_only_and_ignore():
    samples, dets = _toy()
    # FPPI on the empty frames alone: only the FP on C counts, over 1 image.
    m = person_metrics(dets, samples, empty_only=True)
    assert m["n_images"] == 1
    np.testing.assert_allclose(m["curve"]["fppi"], [0, 0, 1, 1, 1, 1, 1])
    # an ignore region over C's detection removes that FP (as in the mAP, T15.10)
    samples[2]["ignore_xyxy"] = np.array([[1.0, 1, 50, 50]])
    m = person_metrics(dets, samples)
    np.testing.assert_allclose(m["curve"]["fppi"], [0, 0, 0, 0, 0, 1 / 3, 1 / 3])
    assert m["at"][0.1]["recall"] == pytest.approx(2 / 3)


def test_match_class_gt_scores():
    samples, dets = _toy()
    gts = {s["id"]: (s["xyxy"], s["difficult"]) for s in samples}
    order, tp, fp, gs = match_class(*dets, gts)
    assert tp.tolist() == [1, 0, 1, 0, 0, 1] and fp.tolist() == [0, 1, 0, 0, 1, 0]
    assert gs["A"].tolist() == [0.9, 0.4] and gs["B"][0] == 0.7 and np.isnan(gs["B"][1])


def test_mr_fppi_ties():
    # two detections with the same score form a single point
    c = mr_fppi([0.5, 0.5, 0.2], np.array([1, 0, 1]), np.array([0, 1, 0]), 2, 1)
    np.testing.assert_allclose(c["fppi"], [0, 1, 1])
    np.testing.assert_allclose(c["recall"], [0, 0.5, 1])


def test_lamr_matches_brute_force_reference():
    """The incremental curve and the Caltech averaging against `mr_ref` (re-matching at every
    threshold), on random scenes with duplicates, difficult boxes and false alarms."""
    rng = np.random.default_rng(1)
    for _ in range(5):
        samples, ids, scores, boxes, gts = [], [], [], [], {}
        for i in range(30):
            objs = []
            for _ in range(int(rng.integers(0, 4))):
                x, y = rng.integers(1, 150, 2)
                b = [float(x), float(y), float(x + rng.integers(10, 60)),
                     float(y + rng.integers(20, 90))]
                objs.append((b, bool(rng.random() < 0.1)))
                if rng.random() < 0.8:
                    for _ in range(1 + int(rng.random() < 0.3)):  # sometimes a duplicate
                        ids.append(str(i))
                        scores.append(round(float(rng.random()), 2))  # ties
                        boxes.append(np.array(b) + rng.normal(0, 4, 4))
            for _ in range(int(rng.integers(0, 3))):
                x, y = rng.integers(1, 150, 2)
                ids.append(str(i))
                scores.append(round(float(rng.random()) * 0.6, 2))
                boxes.append(np.array([x, y, x + 30.0, y + 50.0]))
            samples.append(_sample(str(i), [o[0] for o in objs], [o[1] for o in objs],
                                   w=250, h=250))
            gts[str(i)] = objs
        dets = (ids, np.array(scores), np.array(boxes).reshape(-1, 4))
        m = person_metrics(dets, samples)
        ref = mr_ref.lamr(list(zip(ids, scores, [list(b) for b in boxes])), gts, len(samples))
        assert m["lamr"] == pytest.approx(ref, rel=1e-12)
        assert 0.05 < ref < 1.0


def test_lamr_perfect_and_empty():
    c = {"fppi": np.array([0.0, 0.0]), "mr": np.array([1.0, 0.0])}
    assert log_average_miss_rate(c) == pytest.approx(1e-10)
    c = {"fppi": np.array([0.0]), "mr": np.array([1.0])}
    assert log_average_miss_rate(c) == 1.0


def test_clopper_pearson():
    assert clopper_pearson(0, 10)[0] == 0.0
    assert clopper_pearson(0, 10)[1] == pytest.approx(1 - 0.025 ** (1 / 10), rel=1e-9)
    assert clopper_pearson(10, 10) == (pytest.approx(0.025 ** (1 / 10), rel=1e-9), 1.0)
    lo, hi = clopper_pearson(5, 100)
    assert lo == pytest.approx(0.01643, abs=1e-5) and hi == pytest.approx(0.11283, abs=1e-5)
    assert clopper_pearson(0, 0) == (0.0, 1.0)


def test_unflagged_miss_rate():
    r = unflagged_miss_rate([True, True, False, False], [True, False, True, False])
    assert (r["k"], r["n"], r["rate"]) == (1, 4, 0.25)
    assert r["ci"] == clopper_pearson(1, 4)
