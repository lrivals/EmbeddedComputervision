"""Démo : image → image annotée (T3.4, §8).

    python tools/detect.py --net tiny-yolov2-voc --weights weights/yolov2-tiny-voc.weights \\
        data/VOCdevkit/VOC2007/JPEGImages/000001.jpg -o build/detect/000001.jpg
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from yolo.data.datasets import COCO_CLASSES, DATASETS, FAMILIES, model_family  # noqa: E402
from yolo.data.letterbox import input_size, parse_size  # noqa: E402
from yolo.infer.boxes import cxcywh_to_xyxy  # noqa: E402
from yolo.infer.nms import CONF_THR, IOU_THR  # noqa: E402
from yolo.infer.pipeline import INTERPS, MODES, detect, preprocess  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import build  # noqa: E402

# One colour per class, seeded by the class index: the 20 VOC colours stay the same with 80.
COLORS = [tuple(int(v) for v in np.random.default_rng(c).integers(64, 256, 3))
          for c in range(len(COCO_CLASSES))]


def class_names(n_classes, dataset=None):
    """Output class names of a model with `n_classes` classes (T20.1): those of `dataset`
    (fine-tuned weights, VOC or COCO otherwise), else VOC (20) or COCO (80) by count."""
    if dataset is not None:
        return model_family(dataset, n_classes)[1]
    for names in FAMILIES.values():
        if len(names) == n_classes:
            return names
    for d in DATASETS.values():
        if len(d.classes) == n_classes:
            return d.classes
    return tuple(f"class{c}" for c in range(n_classes))


def pixel_boxes(boxes, w, h):
    """Normalised `(cx, cy, w, h)` → `(x1, y1, x2, y2)` pixels clipped to a w × h image,
    shared by `draw` and the OpenCV drawing of tools/webcam.py (T20.4)."""
    xyxy = cxcywh_to_xyxy(np.asarray(boxes, dtype=np.float64).reshape(-1, 4)) * [w, h, w, h]
    return np.clip(xyxy, 0, [w - 1, h - 1, w - 1, h - 1])


def draw(img, boxes, scores, labels, names):
    """Boîtes `(cx, cy, w, h)` normalisées sur `img` ; `scores` None : vérités terrain,
    étiquette sans score (galeries de M16)."""
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    if scores is None:
        scores = [None] * len(labels)
    for (x1, y1, x2, y2), s, c in zip(pixel_boxes(boxes, w, h), scores, labels):
        col = COLORS[c % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        text = names[c] if s is None else f"{names[c]} {s:.2f}"
        tx, ty, tx2, ty2 = d.textbbox((x1, y1), text)
        d.rectangle([tx - 1, ty - 1, tx2 + 1, ty2 + 1], fill=col)
        d.text((x1, y1), text, fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov2-tiny-voc.weights")
    ap.add_argument("--size", type=parse_size, default=None,
                    help="S ou LxH ; défaut : entrée de la cfg")
    ap.add_argument("--resize", choices=MODES, default="letterbox")
    ap.add_argument("--interp", choices=INTERPS, default="pil")
    ap.add_argument("--conf", type=float, default=CONF_THR)
    ap.add_argument("--iou", type=float, default=IOU_THR)
    ap.add_argument("--dataset", choices=sorted(DATASETS), default=None,
                    help="class names of this dataset (fine-tuned weights); default: VOC "
                         "or COCO from the class count of the cfg")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="image de sortie (une seule image) ou dossier")
    args = ap.parse_args()

    net = build(args.net)
    load_darknet_weights(net, args.weights)
    size, channels = input_size(net.net, args.size), net.net["input"][0]
    names = class_names(net.net["classes"], args.dataset)
    out_dir = args.out if args.out and (len(args.images) > 1 or args.out.suffix == "") else None
    for path in args.images:
        with Image.open(path) as img:
            x, wh = preprocess(img, size, args.resize, args.interp, channels)
            boxes, scores, labels = detect(net, x[None], [wh], args.resize, args.conf,
                                           args.iou)[0]
            for b, s, c in zip(boxes, scores, labels):
                print(f"{path.name}: {names[c]:12s} {s:.3f}  cxcywh {np.round(b, 3)}")
            dest = (out_dir / path.name if out_dir else args.out
                    or ROOT / "build" / "detect" / path.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            draw(img, boxes, scores, labels, names).save(dest)
            print(f"→ {dest}")


if __name__ == "__main__":
    main()
