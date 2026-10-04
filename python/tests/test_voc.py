"""T0.6 : parsing des annotations VOC sur un devkit synthétique."""

import numpy as np
import pytest

from yolo.data.voc import VOC_CLASSES, load_split, parse_annotation, xyxy_to_cxcywh

XML = """<annotation>
  <folder>VOC2007</folder>
  <filename>000001.jpg</filename>
  <size><width>100</width><height>50</height><depth>3</depth></size>
  <object>
    <name>dog</name><pose>Left</pose><truncated>0</truncated><difficult>0</difficult>
    <bndbox><xmin>1</xmin><ymin>1</ymin><xmax>100</xmax><ymax>50</ymax></bndbox>
  </object>
  <object>
    <name>person</name><difficult>1</difficult>
    <bndbox><xmin>11</xmin><ymin>6</ymin><xmax>30</xmax><ymax>25</ymax></bndbox>
  </object>
</annotation>
"""


@pytest.fixture
def devkit(tmp_path):
    base = tmp_path / "VOC2007"
    for d in ("Annotations", "ImageSets/Main", "JPEGImages"):
        (base / d).mkdir(parents=True)
    (base / "Annotations" / "000001.xml").write_text(XML)
    (base / "ImageSets" / "Main" / "trainval.txt").write_text("000001\n")
    return tmp_path


def test_classes():
    assert len(VOC_CLASSES) == 20
    assert VOC_CLASSES[0] == "aeroplane" and VOC_CLASSES[-1] == "tvmonitor"


def test_full_image_box():
    # Boîte 1..W × 1..H : toute l'image.
    np.testing.assert_allclose(xyxy_to_cxcywh([1, 1, 100, 50], 100, 50), [[0.5, 0.5, 1, 1]])


def test_parse(devkit):
    ann = parse_annotation(devkit / "VOC2007" / "Annotations" / "000001.xml")
    assert (ann["filename"], ann["width"], ann["height"]) == ("000001.jpg", 100, 50)
    assert ann["labels"].tolist() == [VOC_CLASSES.index("dog"), VOC_CLASSES.index("person")]
    assert ann["difficult"].tolist() == [False, True]
    # person : x ∈ [10, 30], y ∈ [5, 25] en coordonnées continues.
    np.testing.assert_allclose(ann["boxes"][1], [20 / 100, 15 / 50, 20 / 100, 20 / 50])
    assert ann["boxes"].dtype == np.float64


def test_load_split(devkit):
    samples = load_split(devkit, 2007, "trainval")
    assert [s["id"] for s in samples] == ["000001"]
    assert samples[0]["image"].name == "000001.jpg"


def test_no_object(tmp_path):
    path = tmp_path / "a.xml"
    path.write_text("<annotation><filename>a.jpg</filename>"
                    "<size><width>10</width><height>10</height></size></annotation>")
    ann = parse_annotation(path)
    assert ann["boxes"].shape == (0, 4) and ann["labels"].shape == (0,)
