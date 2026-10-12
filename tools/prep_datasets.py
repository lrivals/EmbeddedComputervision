"""Prétraitements sur disque des jeux de M18 (docs/tasks/M18-jeux-drone.md#prétraitements).

    python tools/prep_datasets.py dronevehicle [--keep-border]   # → data/dronevehicle/
    python tools/prep_datasets.py uavdt                          # → data/uavdt/
    python tools/prep_datasets.py asl [--preview build/asl.png]  # → data/asl/ (M21)

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

- **asl** (ASL Alphabet, Kaggle grassknoted/asl-alphabet, `data/ASL Alphabet/`, M21):
  classification images without boxes. A MediaPipe hand landmarker (`pip install -e
  "python[asl]"`, model downloaded to `weights/hand_landmarker.task`) finds the hand; its
  box is the landmarks' min/max plus a margin. Letter images with no hand found, or a
  doubtful one (landmarks out of the frame, tiny box), are left out
  (`data/asl/rejected.json`); `nothing` images are negatives. Split by frame number
  (`ASL_VAL_FROM`). Output `data/asl/annotations_{train,val,test}.json` and
  `data/asl/images/<split>/` (hard links, as for UAVDT); detections cached in
  `data/asl/detections.json` (the annotations are rebuilt from it at each run). See
  docs/tasks/M21-asl-alphabet.md.

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

from yolo.data.datasets import (  # noqa: E402
    ASL_CLASSES, ASL_VAL_FROM, DRONEVEHICLE_CLASSES, UAVDT_CLASSES)

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
    yolo = [y for y in (obb_to_yolo(t, width, height, border) for t in lines) if y]
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


# --------------------------------------------------------------------------- ASL Alphabet

ASL_RAW = "ASL Alphabet"
ASL_MODEL = ROOT / "weights" / "hand_landmarker.task"
ASL_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                 "hand_landmarker/float16/latest/hand_landmarker.task")
# Detection attempts, first hit wins: (upscale, edge padding in px of the upscaled image,
# mirror). On 560 sampled letter images: 87 % found by the first, 92 % by both. A third,
# mirrored try found 1 % more but its hands were mostly wrong (boxes under 30 px).
ASL_STEPS = ((2, 0, False), (2, 100, False))
ASL_CONF = 0.1     # min_hand_detection_confidence and min_hand_presence_confidence
ASL_MARGIN = 0.15  # margin per side, as a fraction of the landmarks' extent
# Sanity checks of a detection (a hand on the background or out of the frame gives a tiny
# box against the border): share of landmarks inside the image, smaller box side in px.
ASL_MIN_INSIDE = 0.9
ASL_MIN_SIDE = 24
ASL_NEGATIVE = "nothing"


def landmarks_to_box(points, width, height, margin=ASL_MARGIN):
    """Hand landmarks (n, 2) in pixels → continuous corners [x0, y0, x1, y1]: their min/max
    widened by `margin` × extent on each side (landmarks are joint centres, the skin goes
    beyond), clipped to the image."""
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    (x0, y0), (x1, y1) = p.min(0), p.max(0)
    mx, my = margin * (x1 - x0), margin * (y1 - y0)
    return [max(x0 - mx, 0.0), max(y0 - my, 0.0), min(x1 + mx, float(width)),
            min(y1 + my, float(height))]


def asl_frame(stem):
    """`A1234` → 1234 ; None for the official test images (`A_test`)."""
    digits = stem[len(stem.rstrip("0123456789")):]
    return int(digits) if digits else None


def asl_split(stem):
    """Split of a train image: frames n° ≥ ASL_VAL_FROM go to val (consecutive frames are
    near duplicates, so no random split)."""
    return "val" if asl_frame(stem) >= ASL_VAL_FROM else "train"


class HandDetector:
    """MediaPipe hand landmarker, tried with each of `steps` until a hand is found.
    `detector(path)` → (landmarks (21, 2) in pixels of the original image, handedness,
    handedness score, step index) or None."""

    def __init__(self, model=ASL_MODEL, conf=ASL_CONF, steps=ASL_STEPS):
        from mediapipe.tasks.python import BaseOptions, vision

        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=conf, min_hand_presence_confidence=conf)
        self.landmarker = vision.HandLandmarker.create_from_options(opts)
        self.steps = steps

    def __call__(self, path):
        import mediapipe as mp
        import numpy as np
        from PIL import Image, ImageOps

        with Image.open(path) as img:
            img = img.convert("RGB")
        w, h = img.size
        for i, (scale, pad, mirror) in enumerate(self.steps):
            im = ImageOps.mirror(img) if mirror else img
            a = np.asarray(im.resize((w * scale, h * scale), Image.BILINEAR))
            if pad:
                a = np.pad(a, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            r = self.landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                                data=np.ascontiguousarray(a)))
            if not r.hand_landmarks:
                continue
            pts = np.array([(lm.x * a.shape[1], lm.y * a.shape[0])
                            for lm in r.hand_landmarks[0]])
            pts = (pts - pad) / scale
            if mirror:
                pts[:, 0] = w - pts[:, 0]
            hand = r.handedness[0][0]
            return pts, hand.category_name, float(hand.score), i
        return None


def ensure_hand_model(model=ASL_MODEL):
    if not model.exists():
        import urllib.request

        model.parent.mkdir(parents=True, exist_ok=True)
        print(f"download {ASL_MODEL_URL} → {model}")
        urllib.request.urlretrieve(ASL_MODEL_URL, model)
    return model


_DETECTOR = None


def _asl_init(factory):
    global _DETECTOR
    _DETECTOR = factory()


def _asl_detect(path):
    found = _DETECTOR(path)
    if found is None:
        return None
    pts, hand, score, step = found
    return [[round(float(x), 2), round(float(y), 2)] for x, y in pts], hand, round(score, 4), step


def asl_jobs(src):
    """(image, class, split) of every raw image: train/<class>/ by frame number, test/."""
    jobs = []
    train = src / "asl_alphabet_train" / "asl_alphabet_train"
    for d in sorted(p for p in train.iterdir() if p.is_dir()):
        for img in sorted(d.glob("*.jpg"), key=lambda p: asl_frame(p.stem) or 0):
            jobs.append((img, d.name, asl_split(img.stem)))
    for img in sorted((src / "asl_alphabet_test" / "asl_alphabet_test").glob("*.jpg")):
        jobs.append((img, img.stem[:-len("_test")], "test"))
    for _, cls, _ in jobs:
        if cls != ASL_NEGATIVE and cls not in ASL_CLASSES:
            raise ValueError(f"unknown ASL class {cls}")
    return jobs


def asl_detections(work, cache, factory, jobs=None):
    """{image name: [landmarks (21, 2) px, handedness, score, cascade step] or None}, read
    from `cache` if there, else computed (`factory()` builds a detector per worker) and
    written there. The slow part (≈ 6 min on 16 cores): the filters below reuse it."""
    if cache.exists():
        return json.loads(cache.read_text())
    paths = [w[0] for w in work]
    if jobs == 1:
        _asl_init(factory)
        found = [_asl_detect(p) for p in paths]
    else:
        with Pool(jobs, initializer=_asl_init, initargs=(factory,)) as pool:
            found = pool.map(_asl_detect, paths, chunksize=64)
    det = {p.name: f for p, f in zip(paths, found)}
    cache.write_text(json.dumps(det, separators=(",", ":")))
    return det


def asl_check(points, width, height):
    """Reason to drop a detection (`outside`, `small`) or None if it looks like a hand."""
    import numpy as np

    p = np.asarray(points, dtype=np.float64)
    inside = ((p[:, 0] >= 0) & (p[:, 0] <= width) & (p[:, 1] >= 0) & (p[:, 1] <= height))
    if inside.mean() < ASL_MIN_INSIDE:
        return "outside"
    x0, y0, x1, y1 = landmarks_to_box(p, width, height)
    if min(x1 - x0, y1 - y0) < ASL_MIN_SIDE:
        return "small"
    return None


def asl_record(path, cls, split, found, width=200, height=200):
    """Entry of `annotations_<split>.json`, and the reason it is dropped (None: kept)."""
    entry = {"image": f"images/{split}/{path.name}", "width": width, "height": height,
             "class": cls, "frame": asl_frame(path.stem), "boxes": [], "labels": [],
             "hand": None, "hand_score": None, "step": None}
    if found is None:
        return entry, (None if cls == ASL_NEGATIVE else "no hand")
    pts, hand, score, step = found
    entry.update(hand=hand, hand_score=score, step=step)
    if cls == ASL_NEGATIVE:  # stays a negative image whatever the detector says
        return entry, None
    reason = asl_check(pts, width, height)
    if reason is None:
        entry["boxes"] = [[round(v, 2) for v in landmarks_to_box(pts, width, height)]]
        entry["labels"] = [cls]
    return entry, reason


def prep_asl(src, out, factory=HandDetector, jobs=None):
    """Hand pseudo-boxes for every image. Detections are cached in `detections.json`
    (delete it to detect again); the annotation files are rebuilt from it at each run."""
    from PIL import Image

    splits = ("train", "val", "test")
    work = asl_jobs(src)
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
    for path, _, split in work:
        _link(path, out / "images" / split / path.name)
    det = asl_detections(work, out / "detections.json", factory, jobs)
    records = {s: [] for s in splits}
    rejected = []
    for path, cls, split in work:
        with Image.open(path) as img:  # header only
            w, h = img.size
        entry, reason = asl_record(path, cls, split, det[path.name], w, h)
        if reason is None:
            records[split].append(entry)
        else:
            rejected.append({"image": entry["image"], "class": cls, "split": split,
                             "reason": reason})
    (out / "rejected.json").write_text(json.dumps(rejected, indent=0))
    (out / "classes.txt").write_text("".join(c + "\n" for c in ASL_CLASSES))
    for split in splits:  # val last: it is the marker of tools/get_datasets.sh
        (out / f"annotations_{split}.json").write_text(
            json.dumps(records[split], separators=(",", ":")))
    asl_report(records, rejected)


def asl_report(records, rejected):
    from collections import Counter

    kept = Counter(e["class"] for r in records.values() for e in r)
    lost = Counter(r["class"] for r in rejected)
    for split, r in records.items():
        boxes = sum(len(e["boxes"]) for e in r)
        print(f"asl {split} : {len(r)} images, {boxes} boxes")
    letters = [c for c in sorted(set(kept) | set(lost)) if c != ASL_NEGATIVE]
    total = sum(kept[c] for c in letters) + sum(lost.values())
    why = Counter(r["reason"] for r in rejected)
    print(f"asl : {sum(lost.values())} letter images left out "
          f"({100 * sum(lost.values()) / max(total, 1):.1f} %: "
          + ", ".join(f"{k} {n}" for k, n in why.most_common()) + "), worst classes: "
          + ", ".join(f"{c} {100 * lost[c] / (kept[c] + lost[c]):.1f} %"
                      for c in sorted(letters, key=lambda c: -lost[c] / (kept[c] + lost[c]))[:5]))
    neg = [e for r in records.values() for e in r if e["class"] == ASL_NEGATIVE]
    print(f"asl : {sum(e['hand'] is not None for e in neg)} of {len(neg)} `nothing` images "
          "with a hand found (kept as negatives)")


def asl_preview(out, dest, per_class=2, split="val", seed=0):
    """Grid of `per_class` images per class of `split` with their pseudo-box (T21.3)."""
    import random

    from PIL import Image, ImageDraw

    entries = json.loads((out / f"annotations_{split}.json").read_text())
    by_class = {}
    for e in entries:
        by_class.setdefault(e["class"], []).append(e)
    rng = random.Random(seed)
    picks = [e for c in sorted(by_class) for e in rng.sample(by_class[c],
                                                             min(per_class, len(by_class[c])))]
    cols = 8
    grid = Image.new("RGB", (cols * 200, -(-len(picks) // cols) * 200), "white")
    for i, e in enumerate(picks):
        with Image.open(out / e["image"]) as img:
            img = img.convert("RGB").resize((200, 200))
        d = ImageDraw.Draw(img)
        for b in e["boxes"]:
            d.rectangle(b, outline=(255, 0, 0), width=2)
        d.text((4, 4), f"{e['class']} {e['frame']}", fill=(255, 255, 0))
        grid.paste(img, (i % cols * 200, i // cols * 200))
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid.save(dest)
    print(f"asl preview : {len(picks)} images → {dest}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+", choices=("dronevehicle", "uavdt", "asl"))
    ap.add_argument("--src", type=Path, default=None,
                    help="dossier brut (défaut : data/DroneVehicle Dataset/"
                         "DroneVehiclesDatasetYOLO, data/UAVDT Dataset, data/ASL Alphabet)")
    ap.add_argument("--out", type=Path, default=None,
                    help="sortie (défaut : data/dronevehicle, data/uavdt, data/asl)")
    ap.add_argument("--keep-border", action="store_true",
                    help="DroneVehicle : garder le cadre blanc (boîtes seules converties)")
    ap.add_argument("--jobs", type=int, default=None)
    ap.add_argument("--preview", type=Path, default=None,
                    help="ASL: also write a grid of pseudo-boxes (2 val images per class)")
    args = ap.parse_args()
    for ds in args.datasets:
        if ds == "dronevehicle":
            src = args.src or DATA / "DroneVehicle Dataset" / "DroneVehiclesDatasetYOLO"
            prep_dronevehicle(src, args.out or DATA / "dronevehicle", args.keep_border,
                              args.jobs)
        elif ds == "uavdt":
            prep_uavdt(args.src or DATA / "UAVDT Dataset", args.out or DATA / "uavdt")
        else:
            out = args.out or DATA / "asl"
            ensure_hand_model()
            prep_asl(args.src or DATA / ASL_RAW, out, jobs=args.jobs)
            if args.preview:
                asl_preview(out, args.preview)


if __name__ == "__main__":
    main()
