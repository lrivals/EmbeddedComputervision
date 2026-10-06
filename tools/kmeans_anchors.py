"""Ancres par k-means, comparées aux ancres Darknet (T2.2, §5.2 ; autres jeux : T11.4, T11.5).

    python tools/kmeans_anchors.py                    # écrit results/anchors.md
    python tools/kmeans_anchors.py --dataset kitti    # → results/anchors_kitti.md

Boîtes du split d'entraînement du jeu (VOC : VOC2007 + VOC2012 trainval), objets
non-difficult, mesurées en pixels dans l'image letterbox `--size` (repère des ancres,
docs/conventions.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.anchors import kmeans_anchors, mean_best_iou  # noqa: E402
from yolo.data.letterbox import boxes_to_letterbox  # noqa: E402
from yolo.data import datasets  # noqa: E402
from yolo.models.tiny_yolo import load_cfg  # noqa: E402

SIZE = 416


def dataset_wh(samples, size=SIZE):
    """(n, 2) largeurs et hauteurs en pixels de l'image letterbox `size`."""
    wh = []
    for s in samples:
        b = s["boxes"][~s["difficult"]]
        if len(b):
            wh.append(boxes_to_letterbox(b, s["width"], s["height"], size)[:, 2:] * size)
    return np.concatenate(wh)


def fmt(anchors):
    return "  ".join(f"{w:.0f},{h:.0f}" for w, h in anchors)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    datasets.add_args(ap, "split(s) ; défaut : split d'entraînement du jeu")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="défaut results/anchors.md (VOC), results/anchors_<jeu>.md")
    args = ap.parse_args()
    split = datasets.split_of(args, "train")
    out = args.out or ROOT / "results" / (
        "anchors.md" if args.dataset == "voc" else f"anchors_{args.dataset}.md")

    wh = dataset_wh(datasets.load_args(args, "train"), args.size)
    where = ("VOC2007 + VOC2012 trainval" if split == "2007:trainval,2012:trainval"
             else f"{args.dataset} {split}")
    opt = "" if args.dataset == "voc" else f" --dataset {args.dataset}"
    darknet = {5: np.array(load_cfg("tiny-yolov2-voc")["anchors"]),
               6: np.array(load_cfg("tiny-yolov3-voc")["anchors"])}
    rows = []
    for k, ref_name in ((5, "yolov2-tiny-voc.cfg"), (6, "yolov3-tiny.cfg")):
        ours = kmeans_anchors(wh, k, rng=args.seed)
        rows.append((k, fmt(ours), mean_best_iou(wh, ours), ref_name, fmt(darknet[k]),
                     mean_best_iou(wh, darknet[k])))

    lines = [
        "# Ancres par k-means sur VOC (T2.2)" if args.dataset == "voc"
        else f"# Ancres par k-means sur {args.dataset} (T11)",
        "",
        f"`python tools/kmeans_anchors.py{opt} --seed {args.seed}` — {len(wh)} boîtes "
        "non-difficult "
        f"de {where}, en pixels de l'image letterbox {args.size} ; distance "
        "1 − IoU (§5.2), initialisation k-means++.",
        "",
        f"| k | k-means (w,h px à {args.size}) | IoU moyenne | Darknet | ancres Darknet "
        "| IoU moyenne |",
        "|---|---|---|---|---|---|",
    ]
    lines += [f"| {k} | `{a}` | {i:.4f} | {n} | `{d}` | {j:.4f} |" for k, a, i, n, d, j in rows]
    text = "\n".join(lines) + "\n"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
