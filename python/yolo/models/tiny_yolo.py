"""Construction de Tiny-YOLOv2 et Tiny-YOLOv3 à partir des `.cfg` Darknet (§3.1, §3.2).

Les `.cfg` de `cfg/` sont hors base (Darknet) ; le réseau construit est vérifié contre
`yolo.models.specs`, lui-même testé contre les tableaux du §3.
"""

from pathlib import Path

import numpy as np

from yolo.models.cfg import parse_cfg
from yolo.models.graph import Network

CFG_DIR = Path(__file__).parent / "cfg"

# Nom du réseau (celui de `specs.NETWORKS`) -> fichier .cfg.
CFG_FILES = {
    "tiny-yolov2-voc": "yolov2-tiny-voc.cfg",
    "tiny-yolov3-voc": "yolov3-tiny-voc.cfg",
    "tiny-yolov3-coco": "yolov3-tiny.cfg",
}


def load_cfg(name_or_path):
    """Description du réseau au format de `specs`, depuis un nom de `CFG_FILES` ou un chemin."""
    if name_or_path in CFG_FILES:
        return parse_cfg(CFG_DIR / CFG_FILES[name_or_path], name=name_or_path)
    return parse_cfg(Path(name_or_path))


def build(name_or_path, dtype=np.float32, rng=None, **kwargs):
    """Réseau initialisé à la He (§7.1)."""
    return Network(load_cfg(name_or_path), dtype=dtype, rng=rng, **kwargs)


def tiny_yolov2_voc(dtype=np.float32, rng=None, **kwargs):
    return build("tiny-yolov2-voc", dtype=dtype, rng=rng, **kwargs)


def tiny_yolov3_voc(dtype=np.float32, rng=None, **kwargs):
    return build("tiny-yolov3-voc", dtype=dtype, rng=rng, **kwargs)


# Poids Darknet pré-entraînés (`make get-weights`), relatifs à `weights/`.
PRETRAINED = {
    "tiny-yolov2-voc": "yolov2-tiny-voc.weights",
    "tiny-yolov3-coco": "yolov3-tiny.weights",
}
