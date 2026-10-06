"""T11.1 : métrique COCO en NumPy, égale à pycocotools (utilisé seulement ici)."""

import contextlib
import io

import numpy as np
import pytest

from yolo.data.datasets import parse_coco
from yolo.infer.coco_eval import STATS, evaluate, voc_to_xywh


def _coco(rng, n_images=12, n_classes=3):
    """Vérités COCO aléatoires : tailles variées (petits, moyens, grands), quelques foules,
    des images sans objet."""
    images, anns = [], []
    for i in range(1, n_images + 1):
        w, h = int(rng.integers(200, 700)), int(rng.integers(200, 600))
        images.append({"id": i, "file_name": f"{i:06d}.jpg", "width": w, "height": h})
        for _ in range(int(rng.integers(0, 7))):
            bw = float(rng.choice([rng.uniform(4, 30), rng.uniform(30, 95), rng.uniform(96, 180)]))
            bh = float(rng.uniform(0.5, 1.5) * bw)
            x, y = float(rng.uniform(0, w - 20)), float(rng.uniform(0, h - 20))
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(rng.integers(1, n_classes + 1)) * 2,
                         "bbox": [x, y, bw, bh], "area": bw * bh * float(rng.uniform(0.6, 1.0)),
                         "iscrowd": int(rng.random() < 0.08)})
    cats = [{"id": 2 * (k + 1), "name": f"c{k}"} for k in range(n_classes)]
    return {"images": images, "annotations": anns, "categories": cats}


def _dets(rng, data, n_classes=3):
    """Détections : vérités bruitées (rang d'IoU large), doublons et fausses alarmes ; plus
    de 100 par image et par classe pour certaines images (maxDets)."""
    out = []
    for a in data["annotations"]:
        for _ in range(int(rng.integers(0, 3))):
            x, y, w, h = a["bbox"]
            j = rng.normal(0, rng.choice([0.02, 0.06, 0.2]), 4) * [w, h, w, h]
            out.append({"image_id": a["image_id"], "category_id": a["category_id"],
                        "bbox": [x + j[0], y + j[1], max(w + j[2], 1.0), max(h + j[3], 1.0)],
                        "score": float(rng.random())})
    for img in data["images"]:
        n = 130 if img["id"] == 3 else int(rng.integers(0, 8))
        for _ in range(n):
            w, h = float(rng.uniform(5, 150)), float(rng.uniform(5, 150))
            out.append({"image_id": img["id"], "category_id": int(rng.integers(1, n_classes + 1)) * 2,
                        "bbox": [float(rng.uniform(0, img["width"] - 5)),
                                 float(rng.uniform(0, img["height"] - 5)), w, h],
                        "score": float(rng.random())})
    return out


def _ours(data, dets, n_classes=3):
    samples = parse_coco(data, "/tmp")
    cat_index = {c["id"]: k for k, c in enumerate(sorted(data["categories"], key=lambda c: c["id"]))}
    per = {c: ([], [], []) for c in range(n_classes)}
    for d in dets:
        x, y, w, h = d["bbox"]
        per[cat_index[d["category_id"]]][0].append(str(d["image_id"]))
        per[cat_index[d["category_id"]]][1].append(d["score"])
        per[cat_index[d["category_id"]]][2].append([x + 1, y + 1, x + w, y + h])
    per = {c: (v[0], np.array(v[1]), np.array(v[2]).reshape(-1, 4)) for c, v in per.items()}
    return evaluate(per, samples, n_classes)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools(seed):
    pytest.importorskip("pycocotools")
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    rng = np.random.default_rng(seed)
    data = _coco(rng)
    dets = _dets(rng, data)
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = data
        gt.createIndex()
        ev = COCOeval(gt, gt.loadRes(dets), "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    ours = _ours(data, dets)
    np.testing.assert_allclose([ours[k] for k in STATS], ev.stats, atol=1e-4)


def test_perfect():
    data = {"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40],
                             "area": 2000.0, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "x"}]}
    res = _ours(data, [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 50, 40],
                        "score": 0.9}], n_classes=1)
    assert res["AP"] == pytest.approx(1.0) and res["AP50"] == pytest.approx(1.0)
    assert res["APs"] == -1.0 and res["APm"] == pytest.approx(1.0)


def test_voc_to_xywh():
    np.testing.assert_allclose(voc_to_xywh([[11, 6, 30, 25]]), [[10, 5, 20, 20]])
