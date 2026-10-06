"""Sous-ensemble de COCO train2017 pour la calibration INT8 (T11.1), sans les 18 Go du split.

    python tools/coco_subset.py --images 500        # → data/coco/annotations/instances_calib2017.json
                                                    #   + data/coco/calib2017/*.jpg

Tire `--images` images de `annotations/instances_train2017.json` (graine fixe, images
annotées seulement), écrit leurs annotations comme un split `calib2017` et télécharge les
images depuis leur `coco_url`. Idempotent : une image déjà présente n'est pas reprise.
Prérequis : `tools/get_datasets.sh coco` (annotations_trainval2017.zip).
"""

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402


def subset(data, n, seed=0, name="calib2017"):
    """Annotations COCO réduites à `n` images annotées tirées sans remise."""
    annotated = sorted({a["image_id"] for a in data["annotations"]})
    pick = set(np.asarray(annotated)[np.random.default_rng(seed).choice(
        len(annotated), size=min(n, len(annotated)), replace=False)].tolist())
    images = [i for i in data["images"] if i["id"] in pick]
    return {"info": {**data.get("info", {}), "description": f"{name} : {len(images)} images "
                     f"de train2017, graine {seed} (tools/coco_subset.py)"},
            "licenses": data.get("licenses", []),
            "images": images,
            "annotations": [a for a in data["annotations"] if a["image_id"] in pick],
            "categories": data["categories"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, default=ROOT / "data" / "coco")
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--name", default="calib2017")
    args = ap.parse_args()

    src = args.root / "annotations" / "instances_train2017.json"
    if not src.exists():
        sys.exit(f"{src} absent : lancer d'abord tools/get_datasets.sh coco")
    sub = subset(json.loads(src.read_text()), args.images, args.seed, args.name)
    dst = args.root / "annotations" / f"instances_{args.name}.json"
    dst.write_text(json.dumps(sub))
    print(f"{len(sub['images'])} images, {len(sub['annotations'])} annotations → {dst}")

    out = args.root / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for k, img in enumerate(sub["images"], 1):
        path = out / img["file_name"]
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(img["coco_url"], tmp)
            tmp.rename(path)
        if k % 50 == 0 or k == len(sub["images"]):
            print(f"{k}/{len(sub['images'])} images  {time.time() - t0:.0f} s", flush=True)
    print(f"images : {out}")


if __name__ == "__main__":
    main()
