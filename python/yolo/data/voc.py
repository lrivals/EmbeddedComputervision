"""PASCAL VOC : annotations XML → boîtes normalisées (§1, §8.3).

Les boîtes sont rendues en `(cx, cy, w, h)` relatifs à l'image d'origine, dans [0, 1] ; le
passage au repère de l'image après letterbox se fait au prétraitement (docs/conventions.md).
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

# Ordre officiel du devkit VOC (indices de classe 0-19).
VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
)
CLASS_INDEX = {name: i for i, name in enumerate(VOC_CLASSES)}


def xyxy_to_cxcywh(xyxy, width, height):
    """Coins VOC en pixels (1-indexés, bornes incluses) → `(cx, cy, w, h)` normalisés.

    Le pixel i couvre l'intervalle continu [i-1, i] : la boîte xmin..xmax couvre
    [xmin-1, xmax], soit une largeur de xmax - xmin + 1 pixels.
    """
    xyxy = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    x0, y0 = xyxy[:, 0] - 1, xyxy[:, 1] - 1
    x1, y1 = xyxy[:, 2], xyxy[:, 3]
    boxes = np.stack([(x0 + x1) / 2 / width, (y0 + y1) / 2 / height,
                      (x1 - x0) / width, (y1 - y0) / height], axis=1)
    return np.clip(boxes, 0.0, 1.0)


def parse_annotation(xml_path):
    """Lit un fichier Annotations/*.xml.

    Rend un dict : `filename`, `width`, `height`, `boxes` (n, 4) float64 `(cx, cy, w, h)`
    normalisés, `labels` (n,) int64, `difficult` (n,) bool. Les objets `difficult` sont gardés :
    c'est l'évaluation VOC qui les ignore (§8.3).
    """
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    xyxy, labels, difficult = [], [], []
    for obj in root.iter("object"):
        bb = obj.find("bndbox")
        xyxy.append([float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")])
        labels.append(CLASS_INDEX[obj.findtext("name").strip()])
        difficult.append(obj.findtext("difficult", "0").strip() == "1")
    return {
        "filename": root.findtext("filename"),
        "width": width,
        "height": height,
        "boxes": xyxy_to_cxcywh(xyxy, width, height) if xyxy else np.zeros((0, 4)),
        "labels": np.array(labels, dtype=np.int64),
        "difficult": np.array(difficult, dtype=bool),
    }


def split_ids(devkit, year, split):
    """Identifiants d'images de ImageSets/Main/<split>.txt (split : train, val, trainval, test)."""
    path = Path(devkit) / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt"
    return path.read_text().split()


def load_split(devkit, year, split):
    """Annotations d'un split ; chaque entrée reçoit aussi `id` et `image` (chemin du JPEG)."""
    base = Path(devkit) / f"VOC{year}"
    samples = []
    for image_id in split_ids(devkit, year, split):
        ann = parse_annotation(base / "Annotations" / f"{image_id}.xml")
        ann["id"] = image_id
        ann["image"] = base / "JPEGImages" / f"{image_id}.jpg"
        samples.append(ann)
    return samples
