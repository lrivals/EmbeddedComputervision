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

from yolo.data.voc import VOC_CLASSES  # noqa: E402
from yolo.infer.boxes import cxcywh_to_xyxy  # noqa: E402
from yolo.infer.nms import CONF_THR, IOU_THR  # noqa: E402
from yolo.infer.pipeline import INTERPS, MODES, detect, preprocess  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import build  # noqa: E402

COLORS = [tuple(int(v) for v in np.random.default_rng(c).integers(64, 256, 3))
          for c in range(len(VOC_CLASSES))]


def draw(img, boxes, scores, labels, names):
    img = img.convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for b, s, c in zip(cxcywh_to_xyxy(boxes) * [w, h, w, h], scores, labels):
        x1, y1, x2, y2 = np.clip(b, 0, [w - 1, h - 1, w - 1, h - 1])
        col = COLORS[c % len(COLORS)]
        d.rectangle([x1, y1, x2, y2], outline=col, width=3)
        text = f"{names[c]} {s:.2f}"
        tx, ty, tx2, ty2 = d.textbbox((x1, y1), text)
        d.rectangle([tx - 1, ty - 1, tx2 + 1, ty2 + 1], fill=col)
        d.text((x1, y1), text, fill=(0, 0, 0))
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("images", nargs="+", type=Path)
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--weights", type=Path, default=ROOT / "weights" / "yolov2-tiny-voc.weights")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--resize", choices=MODES, default="letterbox")
    ap.add_argument("--interp", choices=INTERPS, default="pil")
    ap.add_argument("--conf", type=float, default=CONF_THR)
    ap.add_argument("--iou", type=float, default=IOU_THR)
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="image de sortie (une seule image) ou dossier")
    args = ap.parse_args()

    net = build(args.net)
    load_darknet_weights(net, args.weights)
    out_dir = args.out if args.out and (len(args.images) > 1 or args.out.suffix == "") else None
    for path in args.images:
        with Image.open(path) as img:
            x, wh = preprocess(img, args.size, args.resize, args.interp)
            boxes, scores, labels = detect(net, x[None], [wh], args.resize, args.conf,
                                           args.iou)[0]
            for b, s, c in zip(boxes, scores, labels):
                print(f"{path.name}: {VOC_CLASSES[c]:12s} {s:.3f}  cxcywh {np.round(b, 3)}")
            dest = (out_dir / path.name if out_dir else args.out
                    or ROOT / "build" / "detect" / path.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            draw(img, boxes, scores, labels, VOC_CLASSES).save(dest)
            print(f"→ {dest}")


if __name__ == "__main__":
    main()
