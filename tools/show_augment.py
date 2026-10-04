"""Contrôle visuel de l'augmentation (T2.6, §7.1) : images VOC augmentées et leurs boîtes.

    python tools/show_augment.py --n 8               # build/augment_samples/*.png
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.loader import VOCDataset  # noqa: E402
from yolo.data.voc import VOC_CLASSES, load_split  # noqa: E402


def draw(chw, boxes, labels, path):
    from PIL import Image, ImageDraw

    img = Image.fromarray((chw.transpose(1, 2, 0) * 255).round().astype(np.uint8))
    d = ImageDraw.Draw(img)
    size = img.width
    for (cx, cy, w, h), label in zip(boxes * size, labels):
        d.rectangle([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], outline=(0, 255, 0), width=2)
        d.text((cx - w / 2 + 3, cy - h / 2 + 2), VOC_CLASSES[label], fill=(0, 255, 0))
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    samples = load_split(args.devkit, 2007, "trainval")
    rng = np.random.default_rng(args.seed)
    ds = VOCDataset(samples)
    out = ROOT / "build" / "augment_samples"
    out.mkdir(parents=True, exist_ok=True)
    for k in rng.choice(len(samples), args.n, replace=False):
        for variant in range(2):
            chw, boxes, labels = ds.load(int(k), args.size, np.random.default_rng([k, variant]))
            path = out / f"{samples[k]['id']}_{variant}.png"
            draw(chw, boxes, labels, path)
            print(path)


if __name__ == "__main__":
    main()
