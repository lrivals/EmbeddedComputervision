"""T11.0 : chargeurs génériques (annotations factices de chaque format) et correspondances."""

import json

import numpy as np
import pytest

from yolo.data import datasets as D
from yolo.data.voc import VOC_CLASSES, parse_annotation

from .test_voc import XML

KEYS = {"id", "image", "width", "height", "boxes", "xyxy", "labels", "difficult"}


def _image(path, w, h):
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h)).save(path)


def _check(s):
    """Mêmes clés et types que `voc.parse_annotation`, boîtes cohérentes avec `xyxy`."""
    assert KEYS <= s.keys()
    n = len(s["labels"])
    assert s["boxes"].shape == (n, 4) and s["boxes"].dtype == np.float64
    assert s["xyxy"].shape == (n, 4) and s["labels"].dtype == np.int64
    assert s["difficult"].dtype == bool and s["difficult"].shape == (n,)
    if n:
        np.testing.assert_allclose(
            s["boxes"], D.xyxy_to_cxcywh(s["xyxy"], s["width"], s["height"]))


def test_make_sample_voc_convention():
    # Boîte continue [10, 30] × [5, 25] = coins VOC 11..30 × 6..25 (cf. test_voc).
    s = D.make_sample("a", "a.jpg", 100, 50, [[10, 5, 30, 25]], [3], [False])
    np.testing.assert_allclose(s["xyxy"], [[11, 6, 30, 25]])
    np.testing.assert_allclose(s["boxes"], [[0.2, 0.3, 0.2, 0.4]])
    _check(s)


def test_make_sample_clip_and_empty():
    s = D.make_sample("a", "a.jpg", 100, 50, [[-10, -5, 50, 60], [120, 0, 130, 10]], [0, 1],
                      [False, True])
    assert s["labels"].tolist() == [0]
    np.testing.assert_allclose(s["xyxy"], [[1, 1, 50, 50]])


COCO = {
    "images": [{"id": 2, "file_name": "000002.jpg", "width": 100, "height": 80},
               {"id": 1, "file_name": "000001.jpg", "width": 60, "height": 40}],
    "annotations": [
        {"id": 1, "image_id": 2, "category_id": 3, "bbox": [10, 5, 20, 20], "area": 300.0,
         "iscrowd": 0},
        {"id": 2, "image_id": 2, "category_id": 1, "bbox": [0, 0, 100, 80], "area": 7000.0,
         "iscrowd": 1}],
    "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "car"}],
}


def test_coco(tmp_path):
    samples = D.parse_coco(COCO, tmp_path)
    assert [s["id"] for s in samples] == ["1", "2"]  # ordre des id, image vide gardée
    assert len(samples[0]["labels"]) == 0
    s = samples[1]
    _check(s)
    assert s["labels"].tolist() == [1, 0]  # rang du category_id
    assert s["difficult"].tolist() == [False, True] and s["crowd"].tolist() == [False, True]
    np.testing.assert_allclose(s["xyxy"][0], [11, 6, 30, 25])
    assert s["area"].tolist() == [300.0, 7000.0]
    assert s["image"] == tmp_path / "000002.jpg"
    # Par nom : une catégorie absente de `classes` est ignorée.
    by_name = D.parse_coco(COCO, tmp_path, classes=("car",))
    assert by_name[1]["labels"].tolist() == [0]


def test_coco_loader(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_val2017.json").write_text(json.dumps(COCO))
    samples = D.load("coco", tmp_path, "val2017")
    assert samples[1]["image"] == tmp_path / "val2017" / "000002.jpg"


def test_coco_classes():
    assert len(D.COCO_CLASSES) == 80 and len(set(D.COCO_CLASSES)) == 80
    assert set(VOC_CLASSES) <= set(D.COCO_CLASSES)


def test_kitti(tmp_path):
    lab = tmp_path / "training" / "label_2"
    lab.mkdir(parents=True)
    for i in range(10):
        _image(tmp_path / "training" / "image_2" / f"{i:06d}.png", 124, 37)
        (lab / f"{i:06d}.txt").write_text(
            "Car 0.00 0 -1.58 10.00 5.00 30.00 25.00 1.6 1.6 3.9 -3.0 1.6 13.0 -1.8\n"
            "DontCare -1 -1 -10 50.0 10.0 60.0 20.0 -1 -1 -1 -1000 -1000 -1000 -10\n"
            "Pedestrian 0.00 0 0.2 40.5 2.0 50.5 30.0 1.8 0.5 0.8 1.0 1.6 9.0 0.3\n")
    train, val = D.load("kitti", tmp_path, "train"), D.load("kitti", tmp_path, "val")
    assert len(val) == 2 and len(train) == 8
    assert not {s["id"] for s in train} & {s["id"] for s in val}
    assert [s["id"] for s in D.load("kitti", tmp_path, "train")] == [s["id"] for s in train]
    s = train[0]
    _check(s)
    assert (s["width"], s["height"]) == (124, 37)
    assert s["labels"].tolist() == [D.KITTI_CLASSES.index("Car"),
                                    D.KITTI_CLASSES.index("Pedestrian")]
    np.testing.assert_allclose(s["xyxy"][0], [11, 6, 30, 25])
    assert len(D.load("kitti", tmp_path, "trainval")) == 10


def test_visdrone(tmp_path):
    d = tmp_path / "VisDrone2019-DET-val"
    _image(d / "images" / "0000001_00000_d_0000001.jpg", 200, 100)
    (d / "annotations").mkdir()
    (d / "annotations" / "0000001_00000_d_0000001.txt").write_text(
        "10,5,20,20,1,4,0,0\n"      # car
        "50,50,10,10,0,0,0,0\n"     # région ignorée
        "60,20,5,8,1,11,0,1\n"      # others
        "70,20,5,8,1,1,0,1,\n")     # pedestrian, virgule finale
    (s,) = D.load("visdrone", tmp_path, "val")
    _check(s)
    assert s["labels"].tolist() == [D.VISDRONE_CLASSES.index("car"),
                                    D.VISDRONE_CLASSES.index("pedestrian")]
    np.testing.assert_allclose(s["xyxy"][0], [11, 6, 30, 25])


def test_crowdhuman(tmp_path):
    _image(tmp_path / "Images" / "273271,1a0d6000b9e1f5b7.jpg", 100, 50)
    line = {"ID": "273271,1a0d6000b9e1f5b7", "gtboxes": [
        {"tag": "person", "fbox": [10, 5, 20, 20], "vbox": [10, 5, 20, 10], "extra": {}},
        {"tag": "person", "fbox": [90, 0, 30, 60], "extra": {"ignore": 1}},
        {"tag": "mask", "fbox": [40, 10, 10, 10], "extra": {"ignore": 1}}]}
    (tmp_path / "annotation_val.odgt").write_text(json.dumps(line) + "\n")
    (s,) = D.load("crowdhuman", tmp_path, "val")
    _check(s)
    assert s["labels"].tolist() == [0, 0, 0]
    assert s["difficult"].tolist() == [False, True, True]
    np.testing.assert_allclose(s["xyxy"][1], [91, 1, 100, 50])  # fbox bornée à l'image


def test_crowdhuman_images_val(tmp_path):
    # Version Kaggle : images de val dans Images_val/.
    _image(tmp_path / "Images_val" / "1,ab.jpg", 100, 50)
    line = {"ID": "1,ab", "gtboxes": [{"tag": "person", "fbox": [10, 5, 20, 20]}]}
    (tmp_path / "annotation_val.odgt").write_text(json.dumps(line) + "\n")
    (s,) = D.load("crowdhuman", tmp_path, "val")
    assert s["image"] == tmp_path / "Images_val" / "1,ab.jpg"


def test_exdark(tmp_path):
    _image(tmp_path / "ExDark" / "Bicycle" / "2015_00001.PNG", 300, 200)
    _image(tmp_path / "ExDark" / "Car" / "2015_00002.jpg", 300, 200)
    ann = tmp_path / "ExDark_Annno" / "Bicycle"
    ann.mkdir(parents=True)
    (ann / "2015_00001.png.txt").write_text(
        "% bbGt version=3\nBicycle 10 5 20 20 0 0 0 0 0 0 0\nPeople 100 50 30 60 0 0 0 0 0 0 0\n")
    (tmp_path / "ExDark_Annno" / "Car").mkdir()
    (tmp_path / "ExDark_Annno" / "Car" / "2015_00002.jpg.txt").write_text("% bbGt version=3\n")
    (tmp_path / "imageclasslist.txt").write_text(
        "Name | Class | Light | In/Out | Train/Val/Test\n"
        "2015_00001.png 1 2 1 1\n2015_00002.jpg 5 1 2 3\n")
    (s,) = D.load("exdark", tmp_path, "train")
    _check(s)
    assert s["id"] == "2015_00001"
    assert s["labels"].tolist() == [D.EXDARK_CLASSES.index("Bicycle"),
                                    D.EXDARK_CLASSES.index("People")]
    np.testing.assert_allclose(s["xyxy"][0], [11, 6, 30, 25])
    (t,) = D.load("exdark", tmp_path, "test")
    assert len(t["labels"]) == 0


def test_flir(tmp_path):
    d = tmp_path / "images_thermal_val"
    d.mkdir()
    data = {"images": [{"id": 0, "file_name": "data/a.jpg", "width": 640, "height": 512}],
            "annotations": [{"id": 1, "image_id": 0, "category_id": 3, "bbox": [1, 2, 3, 4],
                             "area": 12, "iscrowd": 0},
                            {"id": 2, "image_id": 0, "category_id": 99, "bbox": [1, 2, 3, 4],
                             "area": 12, "iscrowd": 0}],
            "categories": [{"id": 3, "name": "car"}, {"id": 99, "name": "inconnu"}]}
    (d / "coco.json").write_text(json.dumps(data))
    (s,) = D.load("flir", tmp_path, "val")
    assert s["labels"].tolist() == [D.FLIR_CLASSES.index("car")]
    assert s["image"] == d / "data" / "a.jpg"


def test_voc_wrapper(tmp_path):
    base = tmp_path / "VOC2007"
    for sub in ("Annotations", "ImageSets/Main"):
        (base / sub).mkdir(parents=True)
    (base / "Annotations" / "000001.xml").write_text(XML)
    (base / "ImageSets" / "Main" / "test.txt").write_text("000001\n")
    (s,) = D.load("voc", tmp_path)  # split par défaut : 2007:test
    ref = parse_annotation(base / "Annotations" / "000001.xml")
    for k in ("boxes", "xyxy", "labels", "difficult"):
        np.testing.assert_array_equal(s[k], ref[k])
    assert len(D.load("voc", tmp_path, "2007:test,2007:test")) == 2


# ---------------------------------------------------------------- correspondances

def test_voc_identity():
    v = D.eval_view("voc", 20)
    assert v.names == VOC_CLASSES
    np.testing.assert_array_equal(v.gt_lut, np.arange(20))
    np.testing.assert_array_equal(v.det_lut, np.arange(20))


def test_self_identity():
    v = D.eval_view("kitti", len(D.KITTI_CLASSES))
    assert v.names == D.KITTI_CLASSES


def test_coco_on_voc():
    v = D.eval_view("voc", 80)
    assert v.names == tuple(c for c in D.COCO_CLASSES if c in VOC_CLASSES)
    for c, name in enumerate(VOC_CLASSES):
        assert v.names[v.gt_lut[c]] == name
        assert v.det_lut[D.COCO_CLASSES.index(name)] == v.gt_lut[c]
    assert (v.det_lut >= 0).sum() == 20


def test_kitti_on_voc():
    v = D.eval_view("kitti", 20)
    assert set(v.names) == {"car", "person", "train"}
    k = D.KITTI_CLASSES
    assert v.gt_lut[k.index("Van")] == v.gt_lut[k.index("Car")]
    assert v.gt_lut[k.index("Cyclist")] == -1 and v.gt_lut[k.index("Truck")] == -1
    assert v.det_lut[VOC_CLASSES.index("dog")] == -1


@pytest.mark.parametrize("name", sorted(D.DATASETS))
@pytest.mark.parametrize("n", [20, 80])
def test_mappings_valid(name, n):
    """Toute correspondance vise des classes existantes et ses sources sont des classes du jeu."""
    v = D.eval_view(name, n)
    assert len(v.names) >= 1 and len(v.gt_lut) == len(D.DATASETS[name].classes)
    m = D.MAPPINGS.get((name, "voc" if n == 20 else "coco"))
    if m:
        assert set(m) <= set(D.DATASETS[name].classes)


def test_model_family_error():
    with pytest.raises(ValueError):
        D.model_family("voc", 7)


def test_remap():
    s = D.make_sample("a", "a.jpg", 100, 50, [[0, 0, 10, 10], [5, 5, 20, 20], [1, 1, 2, 2]],
                      [0, 1, 2], [False, True, False], area=[1.0, 2.0, 3.0])
    (r,) = D.remap([s], [-1, 4, 0])
    assert r["labels"].tolist() == [4, 0]
    assert r["difficult"].tolist() == [True, False] and r["area"].tolist() == [2.0, 3.0]
    assert s["labels"].tolist() == [0, 1, 2]  # original intact


def test_remap_detections():
    per = {0: "a", 5: "b", 7: "c"}
    assert D.remap_detections(per, np.array([1, -1, -1, -1, -1, 0, -1, -1])) == {1: "a", 0: "b"}


def test_tag():
    assert D.tag("voc", "2007:test") == ""
    assert D.tag("coco", "val2017") == "coco-val2017"
