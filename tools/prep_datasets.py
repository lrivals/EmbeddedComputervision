"""Prétraitements sur disque des jeux de M18 (docs/tasks/M18-jeux-drone.md#prétraitements).

    python tools/prep_datasets.py dronevehicle [--keep-border]   # → data/dronevehicle/
    python tools/prep_datasets.py uavdt                          # → data/uavdt/
    python tools/prep_datasets.py asl [--preview build/asl.png]  # → data/asl/ (M21)
    python tools/prep_datasets.py pets [--preview build/pets_preview.png]          # M32
    python tools/prep_datasets.py trashnet [--preview build/trashnet_preview.png]  # M32

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

- **pets** (Oxford-IIIT Pet, `data/Oxford-IIIT Pet/{images,annotations}/`, M32): the
  whole-body box comes from the hand-made trimap (pixels 1 pet and 3 boundary): the min /
  max of the mask without its specks (an occluder often splits one animal; the rare images
  with two animals get one box over both). Images with a
  mask under 2 % of the image or a head box (`annotations/xmls/`) outside the body box are
  left out (`data/pets/rejected.json`). Split: `test.txt` → test, `trainval.txt` → train /
  val (90 / 10, stratified by breed). Output `data/pets/annotations_{train,val,test}.json`
  and `data/pets/images/<split>/` (hard links).
- **trashnet** (TrashNet, `data/TrashNet/dataset-resized/<class>/`, M32): one object per
  image, boxed by the M22 labeller (Grounding DINO tiny, `pip install -e
  "python[autolabel]"`) with the prompt of its class folder plus `object`; the best box is
  kept if its score is ≥ `TRASHNET_CONF` and it does not cover the whole photo. The raw
  predictions are cached in `data/trashnet/autolabel.json` (format of
  tools/autolabel_io.py); the annotations are rebuilt from it at each run. Split 70 / 15 /
  15 stratified by class. See docs/tasks/M32-trashnet-pets.md.

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

sys.path.insert(0, str(ROOT / "tools"))

from yolo.data.datasets import (  # noqa: E402
    ASL_CLASSES, ASL_VAL_FROM, DRONEVEHICLE_CLASSES, PETS_CATS, PETS_CLASSES, TRASHNET_CLASSES,
    UAVDT_CLASSES)

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

# ------------------------------------------------------------------------ Oxford-IIIT Pet

PETS_RAW = "Oxford-IIIT Pet"
# The trimap holds every animal in one mask, and an occluder (coat, hand, collar, leash)
# often splits one animal in several components: on the 128 images with several components
# over 1 % of the image, almost all are one animal cut apart, a handful two animals. So the
# body box is the union of the mask (specks under PETS_SPECK ignored); `components` counts
# the parts over PETS_PART, for the stats.
PETS_SPECK = 0.002    # mask components under this share of the image are ignored
PETS_PART = 0.01      # components over this share of the image are counted
PETS_SMALL = 0.02     # mask under this share of the image: reason `small`
PETS_HEAD_IN = 0.9    # share of the head box that must lie inside a body box
PETS_VAL = 0.1        # share of trainval.txt held out as `val`, stratified by breed
SEED = 0


def pets_breed(stem):
    """`Abyssinian_12` → `Abyssinian`, `great_pyrenees_3` → `great_pyrenees`."""
    return stem.rsplit("_", 1)[0]


def trimap_box(trimap, speck=PETS_SPECK, part=PETS_PART):
    """Trimap (h, w) (1 pet, 2 background, 3 boundary) → (continuous corners of the union of
    the mask {1, 3} without its specks, or None; number of components over `part` × image;
    mask share of the image)."""
    import numpy as np
    from scipy import ndimage

    mask = (trimap == 1) | (trimap == 3)
    lab, n = ndimage.label(mask)
    if not n:
        return None, 0, 0.0
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    keep = np.nonzero(sizes[1:] >= speck * mask.size)[0] + 1
    if not len(keep):
        return None, 0, float(mask.mean())
    ys, xs = np.nonzero(np.isin(lab, keep))
    box = [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
    return box, int((sizes[1:] >= part * mask.size).sum()), float(mask.mean())


def pets_head(xml):
    """Head box of `annotations/xmls/<stem>.xml` (VOC 1-indexed pixels) → continuous corners,
    or None if the file is missing (test images)."""
    import xml.etree.ElementTree as ET

    if not xml.exists():
        return None
    b = ET.parse(xml).find("object/bndbox")
    v = [float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax")]
    return [v[0] - 1, v[1] - 1, v[2], v[3]]


def _inside(inner, outer):
    """Share of the area of `inner` inside `outer` (corners)."""
    w = min(inner[2], outer[2]) - max(inner[0], outer[0])
    h = min(inner[3], outer[3]) - max(inner[1], outer[1])
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return max(w, 0) * max(h, 0) / area if area > 0 else 0.0


def pets_check(box, mask_share, head):
    """Reason to drop an image (`small`, `head outside`) or None."""
    if box is None or mask_share < PETS_SMALL:
        return "small"
    if head is not None and _inside(head, box) < PETS_HEAD_IN:
        return "head outside"
    return None


def stratified_split(items, key, shares, names, seed=SEED):
    """{item: split}: within each `key(item)` group (sorted, then shuffled with `seed`), the
    first round(share × n) items go to each split of `names[1:]`, the rest to `names[0]`."""
    import random

    groups = {}
    for it in sorted(items):
        groups.setdefault(key(it), []).append(it)
    out = {}
    for g in sorted(groups):
        its = groups[g]
        random.Random(f"{seed}-{g}").shuffle(its)
        start = 0
        for name, share in zip(names[1:], shares[1:]):
            n = round(share * len(its))
            out.update((it, name) for it in its[start:start + n])
            start += n
        out.update((it, names[0]) for it in its[start:])
    return out


def pets_jobs(src):
    """(stem, split): test.txt → test; trainval.txt → train / val by breed."""
    ann = src / "annotations"

    def read(f):
        return [ln.split()[0] for ln in (ann / f).read_text().splitlines() if ln.strip()]

    trainval = read("trainval.txt")
    split = stratified_split(trainval, pets_breed, (1 - PETS_VAL, PETS_VAL), ("train", "val"))
    jobs = [(s, split[s]) for s in trainval] + [(s, "test") for s in read("test.txt")]
    for stem, _ in jobs:
        if pets_breed(stem) not in PETS_CLASSES:
            raise ValueError(f"unknown Pet breed {stem}")
    return jobs


def _pets_one(job):
    """(src, stem, split) → (entry, reason or None)."""
    import numpy as np
    from PIL import Image

    src, stem, split = job
    breed = pets_breed(stem)
    entry = {"image": f"images/{split}/{stem}.jpg", "width": 0, "height": 0, "class": breed,
             "species": "cat" if breed in PETS_CATS else "dog", "boxes": [], "labels": [],
             "head": None, "components": 0, "mask": 0.0}
    try:
        with Image.open(src / "images" / f"{stem}.jpg") as img:
            entry["width"], entry["height"] = img.size
    except OSError:
        return entry, "unreadable"
    tri = src / "annotations" / "trimaps" / f"{stem}.png"
    if not tri.exists():
        return entry, "no trimap"
    with Image.open(tri) as t:
        trimap = np.asarray(t)
    if trimap.shape != (entry["height"], entry["width"]):
        return entry, "size mismatch"
    box, parts, share = trimap_box(trimap)
    head = pets_head(src / "annotations" / "xmls" / f"{stem}.xml")
    entry.update(head=head, components=parts, mask=round(share, 4))
    reason = pets_check(box, share, head)
    if reason is None:
        entry["boxes"] = [box]
        entry["labels"] = [breed]
    return entry, reason


def prep_pets(src, out, jobs=None):
    """Whole-body boxes of every image of trainval.txt and test.txt from its trimap."""
    splits = ("train", "val", "test")
    work = pets_jobs(src)
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
    for stem, split in work:
        _link(src / "images" / f"{stem}.jpg", out / "images" / split / f"{stem}.jpg")
    with Pool(jobs) as pool:
        done = pool.map(_pets_one, [(src, s, sp) for s, sp in work], chunksize=32)
    records = {s: [] for s in splits}
    rejected = []
    for (stem, split), (entry, reason) in zip(work, done):
        if reason is None:
            records[split].append(entry)
        else:
            rejected.append({"image": entry["image"], "class": entry["class"], "split": split,
                             "reason": reason})
    write_split_files(out, records, rejected, PETS_CLASSES)
    pets_report(records, rejected)


def write_split_files(out, records, rejected, classes):
    (out / "rejected.json").write_text(json.dumps(rejected, indent=0))
    (out / "classes.txt").write_text("".join(c + "\n" for c in classes))
    for split in ("train", "test", "val"):  # val last: it is the marker of get_datasets.sh
        (out / f"annotations_{split}.json").write_text(
            json.dumps(records[split], separators=(",", ":")))


def pets_report(records, rejected):
    from collections import Counter

    for split, r in records.items():
        print(f"pets {split} : {len(r)} images, {sum(len(e['boxes']) for e in r)} boxes")
    kept = sum(len(r) for r in records.values())
    multi = sum(e["components"] > 1 for r in records.values() for e in r)
    why = Counter(r["reason"] for r in rejected)
    print(f"pets : {kept} of {kept + len(rejected)} images kept "
          f"({100 * kept / max(kept + len(rejected), 1):.1f} %), multi {multi}; left out: "
          + (", ".join(f"{k} {n}" for k, n in why.most_common()) or "none"))


def grid_preview(out, dest, picks, label, extra=None, cell=256, cols=8):
    """Grid of `picks` (annotation entries) with their boxes in red, `extra(entry)` boxes in
    green, `label(entry)` written on each."""
    from PIL import Image, ImageDraw

    grid = Image.new("RGB", (cols * cell, -(-len(picks) // cols) * cell), "white")
    for i, e in enumerate(picks):
        with Image.open(out / e["image"]) as img:
            img = img.convert("RGB")
        sx, sy = cell / img.width, cell / img.height
        img = img.resize((cell, cell))
        d = ImageDraw.Draw(img)
        for boxes, color in ((e["boxes"], (255, 0, 0)), (extra(e) if extra else [], (0, 255, 0))):
            for b in boxes:
                d.rectangle([b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy], outline=color, width=2)
        d.text((4, 4), label(e), fill=(255, 255, 0))
        grid.paste(img, (i % cols * cell, i // cols * cell))
    dest.parent.mkdir(parents=True, exist_ok=True)
    grid.save(dest)


def pick_per_class(out, split, per_class, seed=SEED, first=None):
    import random

    entries = json.loads((out / f"annotations_{split}.json").read_text())
    by_class = {}
    for e in entries:
        by_class.setdefault(e["class"], []).append(e)
    rng = random.Random(seed)
    picks = [e for c in sorted(by_class) for e in rng.sample(by_class[c],
                                                             min(per_class, len(by_class[c])))]
    if first:  # images to show first (e.g. several components)
        picks = [e for e in entries if first(e)][:8] + picks
    return picks


def pets_preview(out, dest, per_class=1, split="val"):
    """Body boxes (red) and head boxes (green) of `per_class` images per breed (T32.5); the
    first row shows images with several components."""
    picks = pick_per_class(out, split, per_class, first=lambda e: e["components"] > 1)
    grid_preview(out, dest, picks, lambda e: e["class"][:18],
                 extra=lambda e: [e["head"]] if e["head"] else [])
    print(f"pets preview : {len(picks)} images → {dest}")


# ------------------------------------------------------------------------------ TrashNet

TRASHNET_RAW = "TrashNet"
# One image = one object of a known class: the labeller only places the box. Prompt of
# each class folder, plus a generic one sent with every image (`TRASHNET_GENERIC`).
TRASHNET_PROMPTS = {"cardboard": "cardboard", "glass": "glass bottle or jar",
                    "metal": "metal can", "paper": "paper",
                    "plastic": "plastic bottle or container", "trash": "waste"}
TRASHNET_GENERIC = "object"
TRASHNET_LABELLER = "grounding-dino-tiny"
TRASHNET_CONF = 0.3     # best box kept if its score is ≥ this
TRASHNET_WHOLE = 0.98   # a box over this share of the image boxed the whole photo
TRASHNET_SPLIT = (0.7, 0.15, 0.15)  # train, val, test, stratified by class
TRASHNET_THREADS = 6    # torch threads per labeller process


def trashnet_files(src):
    """(image, class) of every raw image: `dataset-resized/<class>/*.jpg`."""
    base = src / "dataset-resized"
    files = [(p, d.name) for d in sorted(base.iterdir()) if d.is_dir()
             for p in sorted(d.glob("*.jpg"))]
    for _, cls in files:
        if cls not in TRASHNET_CLASSES:
            raise ValueError(f"unknown TrashNet class {cls}")
    return files


def trashnet_prompts():
    """Global prompt list of the cache: the class prompts in class order, then the generic."""
    return [TRASHNET_PROMPTS[c] for c in TRASHNET_CLASSES] + [TRASHNET_GENERIC]


def trashnet_labeller():
    import autolabel_models as lm

    try:
        import torch
        torch.set_num_threads(TRASHNET_THREADS)
    except ImportError:
        pass
    return lm.get_labeller(TRASHNET_LABELLER, device="cpu")


_LABELLER = None


def _trashnet_init(factory):
    global _LABELLER
    _LABELLER = factory()


def _trashnet_label(job):
    """(path, class) → raw predictions with global prompt indices, image size."""
    import autolabel_io as lio
    import numpy as np

    path, cls = job
    img = lio.load_image(path)
    k = TRASHNET_CLASSES.index(cls)
    names = [cls, TRASHNET_GENERIC]
    boxes, scores, labels = _LABELLER.predict(
        img, (names, [TRASHNET_PROMPTS[cls], TRASHNET_GENERIC]))
    glob = np.array([k, len(TRASHNET_CLASSES)], dtype=np.int64)
    labels = glob[np.asarray(labels, dtype=np.int64)] if len(labels) else labels
    return lio.raw_record(boxes, scores, labels), img.size


def trashnet_autolabel(files, cache, factory, jobs=None, every=100):
    """Provenance of `tools/autolabel_io.py` ({sha1: record}), read from `cache` and
    completed for the images it lacks (`factory()` builds a labeller per worker); written
    back every `every` images, so a stopped run resumes. Returns (provenance, sha1 by path)."""
    import autolabel_io as lio

    prov = lio.read_provenance(cache.parent) if cache.exists() else None
    if prov is None:
        prov = lio.new_provenance(TRASHNET_LABELLER, None, list(TRASHNET_CLASSES)
                                  + [TRASHNET_GENERIC], trashnet_prompts(), TRASHNET_CONF)
    digests = {p: lio.sha1(p) for p, _ in files}
    if len(set(digests.values())) < len(digests):  # one prediction per distinct image
        print(f"trashnet : {len(digests) - len(set(digests.values()))} duplicate images")
    todo = [(p, c) for p, c in files if digests[p] not in prov["images"]]
    if todo:
        print(f"trashnet : labelling {len(todo)} images ({len(files) - len(todo)} cached)",
              flush=True)

        def store(job, res):
            raw, size = res
            prov["images"][digests[job[0]]] = lio.image_record(digests[job[0]], size, raw)

        if jobs == 1:
            _trashnet_init(factory)
            for i, job in enumerate(todo, 1):
                store(job, _trashnet_label(job))
                if i % every == 0:
                    lio.write_provenance(cache.parent, prov)
        else:
            with Pool(jobs or 3, initializer=_trashnet_init, initargs=(factory,)) as pool:
                for i, (job, res) in enumerate(zip(todo, pool.imap(_trashnet_label, todo)), 1):
                    store(job, res)
                    if i % every == 0:
                        lio.write_provenance(cache.parent, prov)
                        print(f"trashnet : {i} / {len(todo)}", flush=True)
        if prov["model_version"] is None and getattr(_LABELLER, "version", None):
            prov["model_version"] = _LABELLER.version
        lio.write_provenance(cache.parent, prov)
    return prov, digests


def trashnet_box(record, conf=TRASHNET_CONF):
    """Best raw box of an image over all its prompts → (continuous corners, score) and the
    reason it is dropped (`no box`, `whole image`) or None."""
    raw, w, h = record["raw"], record["width"], record["height"]
    if not raw["scores"]:
        return None, None, "no box"
    i = max(range(len(raw["scores"])), key=lambda j: raw["scores"][j])
    score = raw["scores"][i]
    cx, cy, bw, bh = raw["boxes"][i]
    box = [max((cx - bw / 2) * w, 0.0), max((cy - bh / 2) * h, 0.0),
           min((cx + bw / 2) * w, float(w)), min((cy + bh / 2) * h, float(h))]
    box = [round(v, 2) for v in box]
    if score < conf:
        return box, score, "no box"
    if (box[2] - box[0]) * (box[3] - box[1]) > TRASHNET_WHOLE * w * h:
        return box, score, "whole image"
    return box, score, None


def prep_trashnet(src, out, factory=trashnet_labeller, jobs=None, conf=TRASHNET_CONF):
    """One auto-labelled box per image. The raw predictions are cached in
    `autolabel.json` (delete it to label again); the annotation files are rebuilt from it at
    each run."""
    splits = ("train", "val", "test")
    files = trashnet_files(src)
    split = stratified_split([p for p, _ in files], lambda p: p.parent.name,
                             TRASHNET_SPLIT, splits)
    for s in splits:
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
    for p, _ in files:
        _link(p, out / "images" / split[p] / p.name)
    prov, digests = trashnet_autolabel(files, out / "autolabel.json", factory, jobs)
    records = {s: [] for s in splits}
    rejected = []
    for p, cls in files:
        rec = prov["images"][digests[p]]
        box, score, reason = trashnet_box(rec, conf)
        entry = {"image": f"images/{split[p]}/{p.name}", "width": rec["width"],
                 "height": rec["height"], "class": cls, "boxes": [], "labels": [],
                 "score": score}
        if reason is None:
            entry.update(boxes=[box], labels=[cls])
            records[split[p]].append(entry)
        else:
            rejected.append({"image": entry["image"], "class": cls, "split": split[p],
                             "reason": reason, "score": score})
    write_split_files(out, records, rejected, TRASHNET_CLASSES)
    trashnet_report(records, rejected)


def trashnet_report(records, rejected):
    from collections import Counter

    for split, r in records.items():
        print(f"trashnet {split} : {len(r)} images")
    kept = Counter(e["class"] for r in records.values() for e in r)
    lost = Counter(r["class"] for r in rejected)
    why = Counter(r["reason"] for r in rejected)
    total = sum(kept.values()) + sum(lost.values())
    print(f"trashnet : {sum(kept.values())} of {total} images kept "
          f"({100 * sum(kept.values()) / max(total, 1):.1f} %); left out: "
          + (", ".join(f"{k} {n}" for k, n in why.most_common()) or "none"))
    print("trashnet kept by class : " + ", ".join(
        f"{c} {100 * kept[c] / max(kept[c] + lost[c], 1):.1f} %" for c in TRASHNET_CLASSES))


def trashnet_preview(out, dest, per_class=2, split="val"):
    """Auto-label boxes of `per_class` images per class (T32.3)."""
    picks = pick_per_class(out, split, per_class)
    grid_preview(out, dest, picks, lambda e: f"{e['class']} {e['score']:.2f}", cols=6)
    print(f"trashnet preview : {len(picks)} images → {dest}")


TRASHNET_HANDCHECK = 100


def trashnet_handcheck(out, dest, n=TRASHNET_HANDCHECK):
    """`n` val images, stratified by class, copied to `dest/` for the "Folder" tab of the M22
    editor (`make autolabel`): label them there with the classes of `dest/classes.txt`,
    correct every box by hand, then "Download all (zip)" (T32.3)."""
    entries = json.loads((out / "annotations_val.json").read_text())
    cls = {e["image"]: e["class"] for e in entries}
    pick = stratified_split(list(cls), cls.get, (1 - n / len(cls), n / len(cls)),
                            ("rest", "check"))
    dest.mkdir(parents=True, exist_ok=True)
    chosen = sorted(i for i in cls if pick[i] == "check")
    for i in chosen:
        _link(out / i, dest / Path(i).name)
    (dest.parent / f"{dest.name}_classes.txt").write_text(
        ", ".join(f"{c}={TRASHNET_PROMPTS[c]}" for c in TRASHNET_CLASSES) + "\n")
    print(f"trashnet hand check : {len(chosen)} val images → {dest} (classes to type: "
          f"{dest.parent / (dest.name + '_classes.txt')})")


def _read_result_labels(result):
    """{image stem: (n, 4) normalised cxcywh} of an editor result: a folder holding
    `labels/` or the zip of "Download all"."""
    import zipfile

    import numpy as np

    out = {}
    if Path(result).suffix == ".zip":
        with zipfile.ZipFile(result) as z:
            for m in z.namelist():
                if m.endswith(".txt") and "/labels/" in f"/{m}":
                    rows = [ln.split() for ln in z.read(m).decode().splitlines() if ln.strip()]
                    out[Path(m).stem] = np.array([[float(v) for v in r[1:5]] for r in rows])
    else:
        for f in Path(result).rglob("labels/**/*.txt"):
            rows = [ln.split() for ln in f.read_text().splitlines() if ln.strip()]
            out[f.stem] = np.array([[float(v) for v in r[1:5]] for r in rows])
    return out


def box_iou(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(w, 0) * max(h, 0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def trashnet_handcheck_iou(out, result):
    """IoU of the auto-labels against the boxes corrected by hand in the editor (`result`:
    zip or folder); the corrected boxes (class: the folder's) are saved as the hand-checked
    val subset `out/handcheck_val.json`. Returns the IoUs (0: no box kept or drawn)."""
    import numpy as np

    hand = _read_result_labels(result)
    entries = {Path(e["image"]).stem: e
               for s in ("train", "val", "test")
               for e in json.loads((out / f"annotations_{s}.json").read_text())}
    rejected = {Path(r["image"]).stem: r for r in json.loads((out / "rejected.json").read_text())}
    ious, ref = [], []
    for stem, b in sorted(hand.items()):
        e = entries.get(stem)
        if e is None and stem not in rejected:
            continue
        w, h = (e["width"], e["height"]) if e else (512, 384)
        corners = [[(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]
                   for x, y, bw, bh in b]
        cls = e["class"] if e else rejected[stem]["class"]
        ref.append({"image": e["image"] if e else rejected[stem]["image"], "width": w,
                    "height": h, "class": cls, "boxes": [[round(v, 2) for v in c]
                                                         for c in corners],
                    "labels": [cls] * len(corners)})
        auto = e["boxes"][0] if e and e["boxes"] else None
        ious.append(max((box_iou(auto, c) for c in corners), default=0.0) if auto else 0.0)
    (out / "handcheck_val.json").write_text(json.dumps(ref, separators=(",", ":")))
    ious = np.array(ious)
    if len(ious):
        print(f"trashnet hand check : {len(ious)} images, IoU median {np.median(ious):.3f}, "
              f"q10 {np.quantile(ious, 0.1):.3f}, < 0.5 : {int((ious < 0.5).sum())}")
    return ious


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("datasets", nargs="+",
                    choices=("dronevehicle", "uavdt", "asl", "trashnet", "pets"))
    ap.add_argument("--src", type=Path, default=None,
                    help="dossier brut (défaut : data/DroneVehicle Dataset/"
                         "DroneVehiclesDatasetYOLO, data/UAVDT Dataset, data/ASL Alphabet, "
                         "data/TrashNet, data/Oxford-IIIT Pet)")
    ap.add_argument("--out", type=Path, default=None,
                    help="sortie (défaut : data/<dataset>)")
    ap.add_argument("--keep-border", action="store_true",
                    help="DroneVehicle : garder le cadre blanc (boîtes seules converties)")
    ap.add_argument("--jobs", type=int, default=None,
                    help="processes (TrashNet: labeller processes of "
                         f"{TRASHNET_THREADS} threads, default 3)")
    ap.add_argument("--preview", type=Path, default=None,
                    help="ASL, TrashNet, Pet: also write a grid of boxes (val images)")
    ap.add_argument("--handcheck", type=Path, default=None,
                    help="TrashNet: copy 100 val images there for a hand check in the M22 "
                         "editor (T32.3)")
    ap.add_argument("--handcheck-iou", type=Path, default=None,
                    help="TrashNet: zip or folder of the corrected labels → IoU of the "
                         "auto-labels, data/trashnet/handcheck_val.json")
    args = ap.parse_args()
    for ds in args.datasets:
        if ds == "dronevehicle":
            src = args.src or DATA / "DroneVehicle Dataset" / "DroneVehiclesDatasetYOLO"
            prep_dronevehicle(src, args.out or DATA / "dronevehicle", args.keep_border,
                              args.jobs)
        elif ds == "uavdt":
            prep_uavdt(args.src or DATA / "UAVDT Dataset", args.out or DATA / "uavdt")
        elif ds == "asl":
            out = args.out or DATA / "asl"
            ensure_hand_model()
            prep_asl(args.src or DATA / ASL_RAW, out, jobs=args.jobs)
            if args.preview:
                asl_preview(out, args.preview)
        elif ds == "pets":
            out = args.out or DATA / "pets"
            prep_pets(args.src or DATA / PETS_RAW, out, jobs=args.jobs)
            if args.preview:
                pets_preview(out, args.preview)
        else:
            out = args.out or DATA / "trashnet"
            prep_trashnet(args.src or DATA / TRASHNET_RAW, out, jobs=args.jobs)
            if args.preview:
                trashnet_preview(out, args.preview)
            if args.handcheck:
                trashnet_handcheck(out, args.handcheck)
            if args.handcheck_iou:
                trashnet_handcheck_iou(out, args.handcheck_iou)


if __name__ == "__main__":
    main()
