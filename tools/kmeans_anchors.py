"""Ancres par k-means sur VOC, comparées aux ancres Darknet (T2.2, §5.2).

    python tools/kmeans_anchors.py                    # écrit results/anchors.md

Boîtes de VOC2007 + VOC2012 trainval (objets non-difficult), mesurées en pixels dans
l'image letterbox 416 (repère des ancres, docs/conventions.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.anchors import kmeans_anchors, mean_best_iou  # noqa: E402
from yolo.data.letterbox import boxes_to_letterbox  # noqa: E402
from yolo.data.voc import load_split  # noqa: E402
from yolo.models.tiny_yolo import load_cfg  # noqa: E402

SIZE = 416
SPLITS = [(2007, "trainval"), (2012, "trainval")]


def voc_wh(devkit):
    wh = []
    for year, split in SPLITS:
        for s in load_split(devkit, year, split):
            b = s["boxes"][~s["difficult"]]
            if len(b):
                wh.append(boxes_to_letterbox(b, s["width"], s["height"], SIZE)[:, 2:] * SIZE)
    return np.concatenate(wh)


def fmt(anchors):
    return "  ".join(f"{w:.0f},{h:.0f}" for w, h in anchors)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "anchors.md")
    args = ap.parse_args()

    wh = voc_wh(args.devkit)
    darknet = {5: np.array(load_cfg("tiny-yolov2-voc")["anchors"]),
               6: np.array(load_cfg("tiny-yolov3-voc")["anchors"])}
    rows = []
    for k, ref_name in ((5, "yolov2-tiny-voc.cfg"), (6, "yolov3-tiny.cfg")):
        ours = kmeans_anchors(wh, k, rng=args.seed)
        rows.append((k, fmt(ours), mean_best_iou(wh, ours), ref_name, fmt(darknet[k]),
                     mean_best_iou(wh, darknet[k])))

    lines = [
        "# Ancres par k-means sur VOC (T2.2)",
        "",
        f"`python tools/kmeans_anchors.py --seed {args.seed}` — {len(wh)} boîtes non-difficult "
        "de VOC2007 + VOC2012 trainval, en pixels de l'image letterbox 416 ; distance "
        "1 − IoU (§5.2), initialisation k-means++.",
        "",
        "| k | k-means (w,h px à 416) | IoU moyenne | Darknet | ancres Darknet | IoU moyenne |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {k} | `{a}` | {i:.4f} | {n} | `{d}` | {j:.4f} |" for k, a, i, n, d, j in rows]
    text = "\n".join(lines) + "\n"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
