"""Comptes d'images et d'objets des splits VOC, et rendu d'images annotées (T0.6).

    python tools/voc_stats.py                 # compare aux chiffres officiels
    python tools/voc_stats.py --show 10       # + 10 images au hasard dans build/voc_samples/
"""

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.voc import VOC_CLASSES, load_split  # noqa: E402

# Chiffres du devkit (tableaux « Main ») : objets non-difficult, ceux de l'évaluation.
OFFICIAL = {
    (2007, "trainval"): (5011, 12608),
    (2007, "test"): (4952, 12032),
    (2012, "trainval"): (11540, 27450),
}


def draw(sample, path):
    from PIL import Image, ImageDraw

    img = Image.open(sample["image"]).convert("RGB")
    d = ImageDraw.Draw(img)
    w, h = img.size
    for (cx, cy, bw, bh), label, diff in zip(sample["boxes"], sample["labels"],
                                             sample["difficult"]):
        x0, y0, x1, y1 = (cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h
        color = (255, 160, 0) if diff else (0, 255, 0)
        d.rectangle([x0, y0, x1, y1], outline=color, width=2)
        name = VOC_CLASSES[label] + (" (difficult)" if diff else "")
        d.text((x0 + 3, y0 + 2), name, fill=color)
    img.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ok = True
    pool = []
    print("| split | images | objets (non-difficult) | objets (tous) | officiel | |")
    print("|---|---|---|---|---|---|")
    for (year, split), (n_img, n_obj) in OFFICIAL.items():
        samples = load_split(args.devkit, year, split)
        pool += samples
        easy = sum(int((~s["difficult"]).sum()) for s in samples)
        total = sum(len(s["labels"]) for s in samples)
        match = (len(samples), easy) == (n_img, n_obj)
        ok &= match
        print(f"| VOC{year} {split} | {len(samples)} | {easy} | {total} | {n_img} / {n_obj} | "
              f"{'ok' if match else 'ÉCART'} |")

    if args.show:
        out = ROOT / "build" / "voc_samples"
        out.mkdir(parents=True, exist_ok=True)
        for s in random.Random(args.seed).sample(pool, args.show):
            name = f"{s['image'].parent.parent.name}_{s['id']}.png"
            draw(s, out / name)
            print(out / name)

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
