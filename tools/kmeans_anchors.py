"""Ancres par k-means, comparées aux ancres Darknet (T2.2, §5.2 ; autres jeux : T11.4, T11.5).

    python tools/kmeans_anchors.py                    # écrit results/anchors.md
    python tools/kmeans_anchors.py --dataset kitti    # → results/anchors_kitti.md
    # entrée non carrée (T11.4) : ancres en pixels de l'entrée 640 × 192
    python tools/kmeans_anchors.py --dataset kitti --size 640x192 \
        --out results/anchors_kitti_640x192.md

Boîtes du split d'entraînement du jeu (VOC : VOC2007 + VOC2012 trainval), objets
non-difficult, mesurées en pixels dans l'image letterbox `--size` (S, ou LxH pour un réseau
non carré : repère des ancres, docs/conventions.md).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.anchors import kmeans_anchors, mean_best_iou  # noqa: E402
from yolo.data.letterbox import as_hw, boxes_to_letterbox, parse_size, size_label  # noqa: E402
from yolo.data import datasets  # noqa: E402
from yolo.models.tiny_yolo import load_cfg  # noqa: E402

SIZE = 416


def dataset_wh(samples, size=SIZE, mode="letterbox", crop=0):
    """(n, 2) largeurs et hauteurs en pixels de l'image letterbox (ou étirée) `size` (S ou
    (H, W)). `crop` > 0 : image d'origine ramenée à une découpe `crop` × `crop` (tuiles)."""
    sh, sw = as_hw(size)
    wh = []
    for s in samples:
        b = s["boxes"][~s["difficult"]]
        if len(b):
            w, h = s["width"], s["height"]
            if crop:
                cw, ch = min(crop, w), min(crop, h)
                b = b * (w / cw, h / ch, w / cw, h / ch)
                w, h = cw, ch
            wh.append(boxes_to_letterbox(b, w, h, size, mode)[:, 2:] * (sw, sh))
    return np.concatenate(wh)


def fmt(anchors):
    return "  ".join(f"{w:.0f},{h:.0f}" for w, h in anchors)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    datasets.add_args(ap, "split(s) ; défaut : split d'entraînement du jeu")
    ap.add_argument("--size", type=parse_size, default=SIZE, help="S ou LxH (ex. 640x192)")
    ap.add_argument("--resize", choices=("letterbox", "stretch"), default="letterbox",
                    help="géométrie de l'entrée, celle de l'entraînement")
    ap.add_argument("--crop", type=int, default=0,
                    help="tuiles N×N px de l'image d'origine (train.py --crop)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None,
                    help="défaut results/anchors.md (VOC), results/anchors_<jeu>.md")
    args = ap.parse_args()
    split = datasets.split_of(args, "train")
    out = args.out or ROOT / "results" / (
        "anchors.md" if args.dataset == "voc" else f"anchors_{args.dataset}.md")

    wh = dataset_wh(datasets.load_args(args, "train"), args.size, args.resize, args.crop)
    where = ("VOC2007 + VOC2012 trainval" if split == "2007:trainval,2012:trainval"
             else f"{args.dataset} {split}")
    opt = "" if args.dataset == "voc" else f" --dataset {args.dataset}"
    if args.size != SIZE:
        opt += f" --size {size_label(args.size).replace('×', 'x')}"
    if args.resize != "letterbox":
        opt += f" --resize {args.resize}"
    if args.crop:
        opt += f" --crop {args.crop}"
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
        f"de {where}, en pixels de l'image {args.resize} {size_label(args.size)}"
        + (f" (tuiles {args.crop} px)" if args.crop else "") + " ; distance "
        "1 − IoU (§5.2), initialisation k-means++.",
        "",
        f"| k | k-means (w,h px à {size_label(args.size)}) | IoU moyenne | Darknet "
        "| ancres Darknet "
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
