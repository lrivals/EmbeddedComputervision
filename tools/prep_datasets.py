"""Prétraitements sur disque des jeux de M18 (docs/tasks/M18-jeux-drone.md#prétraitements).

    python tools/prep_datasets.py dronevehicle [--keep-border]   # → data/dronevehicle/
    python tools/prep_datasets.py uavdt                          # → data/uavdt/

- **dronevehicle** (version Roboflow YOLO-OBB, `data/DroneVehicle Dataset/`) : le cadre
  blanc de 100 px autour de l'image 640×512 est recadré ; chaque boîte orientée (4 coins)
  devient sa boîte englobante, décalée puis rognée à l'image. Sortie
  `data/dronevehicle/<split>/{images,labels}/<n°>.{jpg,txt}` en YOLO `classe cx cy w h`
  (nom sans le suffixe Roboflow `_jpg.rf.<hash>`). `--keep-border` garde les images
  entières (copiées) et ne convertit que les boîtes.
- **uavdt** (export Supervisely de DatasetNinja, `data/UAVDT Dataset/`) : seules les
  séquences M (benchmark DET) sont gardées ; les séquences S du test (suivi d'un objet,
  classe `vehicle`) sont écartées. Sortie `data/uavdt/annotations_<split>.json` (liste
  compacte : image, taille, coins continus, classes, séquence, tags) et
  `data/uavdt/images/<split>/` (liens physiques vers les images brutes, copie à défaut).

Idempotent : un fichier de sortie présent n'est pas refait. AU-AIR et HIT-UAV n'ont pas de
prétraitement sur disque (liens de `tools/get_datasets.sh local`).
"""

import argparse
import json
import os
import shutil
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.datasets import DRONEVEHICLE_CLASSES, UAVDT_CLASSES  # noqa: E402

DATA = ROOT / "data"
BORDER = 100  # px de cadre blanc de DroneVehicle de chaque côté (840×712 → 640×512)


# ------------------------------------------------------------------------- DroneVehicle

def obb_to_yolo(line, width, height, border):
    """Ligne YOLO-OBB (`classe x1 y1 … x4 y4` normalisés sur l'image d'origine) → ligne YOLO
    `classe cx cy w h` normalisée sur l'image recadrée de `border` px ; None si la boîte sort
    de l'image recadrée."""
    f = line.split()
    if len(f) < 9:
        return None
    c = int(f[0])
    xs = [float(v) * width - border for v in f[1:9:2]]
    ys = [float(v) * height - border for v in f[2:9:2]]
    w, h = width - 2 * border, height - 2 * border
    x0, x1 = max(min(xs), 0.0), min(max(xs), w)
    y0, y1 = max(min(ys), 0.0), min(max(ys), h)
    if x1 <= x0 or y1 <= y0:
        return None
    return (f"{c} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
            f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")


def _dronevehicle_one(job):
    image, label, out_img, out_lab, keep_border = job
    from PIL import Image

    if out_img.exists() and out_lab.exists():
        return 0
    with Image.open(image) as img:
        width, height = img.size
        border = 0 if keep_border or (width, height) != (840, 712) else BORDER
        if border:
            img.crop((border, border, width - border, height - border)).save(out_img,
                                                                           quality=95)
        else:
            shutil.copyfile(image, out_img)
    lines = label.read_text().splitlines() if label.exists() else []
    yolo = [y for y in (obb_to_yolo(l, width, height, border) for l in lines) if y]
    out_lab.write_text("".join(y + "\n" for y in yolo))
    return 1


def prep_dronevehicle(src, out, keep_border=False, jobs=None):
    jobs_list = []
    for split in ("train", "val", "test"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        for image in sorted((src / split / "images").iterdir()):
            n = image.name.split("_")[0]
            jobs_list.append((image, src / split / "labels" / f"{image.stem}.txt",
                              out / split / "images" / f"{n}.jpg",
                              out / split / "labels" / f"{n}.txt", keep_border))
    with Pool(jobs) as pool:
        done = sum(pool.imap_unordered(_dronevehicle_one, jobs_list, chunksize=64))
    print(f"dronevehicle : {done} images écrites, {len(jobs_list) - done} déjà là → {out}")
    (out / "classes.txt").write_text("".join(c + "\n" for c in DRONEVEHICLE_CLASSES))


# -------------------------------------------------------------------------------- UAVDT

def uavdt_record(ann, image_rel):
    """Annotation Supervisely d'une image → entrée compacte (coins continus : le coin bas
    droit de `exterior` est un pixel inclus, d'où + 1)."""
    tags = [t["name"] for t in ann.get("tags", []) if t["name"] != "sequence"]
    boxes, labels, occl, out = [], [], [], []
    for o in ann["objects"]:
        if o["classTitle"] not in UAVDT_CLASSES:
            continue
        (x0, y0), (x1, y1) = o["points"]["exterior"]
        boxes.append([min(x0, x1), min(y0, y1), max(x0, x1) + 1, max(y0, y1) + 1])
        labels.append(o["classTitle"])
        names = [t["name"] for t in o.get("tags", [])]
        occl.append(next((n.split()[0] for n in names if n.endswith("occlusion")), ""))
        out.append(next((n.split()[0] for n in names if n.endswith(" out")), ""))
    return {"image": image_rel, "width": ann["size"]["width"],
            "height": ann["size"]["height"], "sequence": Path(image_rel).name.split("_")[0],
            "tags": tags, "boxes": boxes, "labels": labels, "occlusion": occl, "out": out}


def _link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def prep_uavdt(src, out):
    for split in ("train", "test"):
        dest = out / f"annotations_{split}.json"
        imgs = out / "images" / split
        imgs.mkdir(parents=True, exist_ok=True)
        records, skipped = [], 0
        for ann in sorted((src / split / "ann").glob("*.json")):
            name = ann.name[:-len(".json")]
            if not name.startswith("M"):  # séquences S : suivi d'un objet, pas DET
                skipped += 1
                continue
            _link(src / split / "img" / name, imgs / name)
            if not dest.exists():
                records.append(uavdt_record(json.loads(ann.read_text()),
                                            f"images/{split}/{name}"))
        if not dest.exists():
            dest.write_text(json.dumps(records, separators=(",", ":")))
            print(f"uavdt {split} : {len(records)} images (séquences M), "
                  f"{skipped} écartées (séquences S) → {dest}")
        else:
            print(f"uavdt {split} : {dest} déjà là")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+", choices=("dronevehicle", "uavdt"))
    ap.add_argument("--src", type=Path, default=None,
                    help="dossier brut (défaut : data/DroneVehicle Dataset/"
                         "DroneVehiclesDatasetYOLO, data/UAVDT Dataset)")
    ap.add_argument("--out", type=Path, default=None,
                    help="sortie (défaut : data/dronevehicle, data/uavdt)")
    ap.add_argument("--keep-border", action="store_true",
                    help="DroneVehicle : garder le cadre blanc (boîtes seules converties)")
    ap.add_argument("--jobs", type=int, default=None)
    args = ap.parse_args()
    for ds in args.datasets:
        if ds == "dronevehicle":
            src = args.src or DATA / "DroneVehicle Dataset" / "DroneVehiclesDatasetYOLO"
            prep_dronevehicle(src, args.out or DATA / "dronevehicle", args.keep_border,
                              args.jobs)
        else:
            prep_uavdt(args.src or DATA / "UAVDT Dataset", args.out or DATA / "uavdt")


if __name__ == "__main__":
    main()
