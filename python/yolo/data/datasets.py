"""Jeux de données au-delà de VOC : registre, chargeurs et correspondances de classes (T11.0).

Chaque chargeur rend la même liste de dicts que `yolo.data.voc.load_split` : `id`, `image`,
`width`, `height`, `boxes` (n, 4) `(cx, cy, w, h)` normalisés, `xyxy` (n, 4) coins en
convention pixel VOC (1-indexés, bornes incluses : la boîte continue [x0, x1] s'écrit
x0 + 1 .. x1), `labels` (n,) int64 indices dans les classes **du jeu**, `difficult` (n,)
bool. Le format COCO ajoute `area` (aire de la segmentation) et `crowd` (`iscrowd`), lus
par la métrique COCO (`yolo.infer.coco_eval`).

Les régions sans classe (`DontCare` de KITTI, régions ignorées et « others » de VisDrone)
sont retirées des objets. Les régions ignorées de VisDrone (catégorie 0) restent dans
`ignore_xyxy` (k, 4), par image : l'évaluation VOC n'y compte pas les détections non
appariées (`yolo.infer.metrics.eval_class`, T15.10). Les objets à ignorer
d'une classe connue (`iscrowd`, `mask` et `ignore` de CrowdHuman) restent, en `difficult`.

Correspondances (`MAPPINGS`) : `{classe du jeu: classe du modèle}`, explicites, pour évaluer
des poids VOC (20 classes) ou COCO (80) sur un autre jeu sans réentraînement (T11.2). Les
classes sans équivalent sont absentes ; les choix discutables sont commentés.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from yolo.data import voc
from yolo.data.voc import VOC_CLASSES, xyxy_to_cxcywh

DATA_DIR = Path(__file__).resolve().parents[3] / "data"

# Ordre de `coco.names` (Darknet) : celui des 80 sorties de yolov3-tiny.weights, qui est
# aussi l'ordre croissant des `category_id` de COCO (noms Darknet, proches de ceux de VOC).
COCO_CLASSES = (
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
    "toothbrush",
)
KITTI_CLASSES = ("Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram",
                 "Misc")
VISDRONE_CLASSES = ("pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle",
                    "awning-tricycle", "bus", "motor")
CROWDHUMAN_CLASSES = ("person",)
EXDARK_CLASSES = ("Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog",
                  "Motorbike", "People", "Table")
# FLIR ADAS v2 : catégories du jeu thermique présentes en val (à vérifier au téléchargement).
FLIR_CLASSES = ("person", "bike", "car", "motor", "bus", "train", "truck", "light",
                "hydrant", "sign", "dog", "skateboard", "stroller", "scooter", "other vehicle")
# Jeux drone et thermiques (M18, docs/tasks/M18-jeux-drone.md).
AUAIR_CLASSES = ("Human", "Car", "Truck", "Van", "Motorbike", "Bicycle", "Bus", "Trailer")
DRONEVEHICLE_CLASSES = ("small-vehicle", "large-vehicle")
HITUAV_CLASSES = ("Person", "Car", "Bicycle", "OtherVehicle")  # DontCare (4) retiré
UAVDT_CLASSES = ("car", "truck", "bus")
# ASL Alphabet (M21, docs/tasks/M21-asl-alphabet.md): one hand pseudo-box per image;
# `nothing` images are negatives (no box), so it is not a class.
ASL_CLASSES = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ("del", "space")
# TrashNet (M32, docs/tasks/M32-trashnet-pets.md): one box per image from the M22 labeller;
# the class is the folder of the raw image.
TRASHNET_CLASSES = ("cardboard", "glass", "metal", "paper", "plastic", "trash")
# Oxford-IIIT Pet (M32): 37 breeds in the CLASS-ID order of `annotations/list.txt`
# (alphabetical, case ignored); a capitalised name is a cat breed. Whole-body boxes from
# the trimaps.
PETS_CLASSES = (
    "Abyssinian", "american_bulldog", "american_pit_bull_terrier", "basset_hound", "beagle",
    "Bengal", "Birman", "Bombay", "boxer", "British_Shorthair", "chihuahua", "Egyptian_Mau",
    "english_cocker_spaniel", "english_setter", "german_shorthaired", "great_pyrenees",
    "havanese", "japanese_chin", "keeshond", "leonberger", "Maine_Coon", "miniature_pinscher",
    "newfoundland", "Persian", "pomeranian", "pug", "Ragdoll", "Russian_Blue", "saint_bernard",
    "samoyed", "scottish_terrier", "shiba_inu", "Siamese", "Sphynx",
    "staffordshire_bull_terrier", "wheaten_terrier", "yorkshire_terrier",
)
PETS_CATS = tuple(c for c in PETS_CLASSES if c[0].isupper())
# PREWID (M38, docs/tasks/M38-prewid.md): workers seen by fixed cameras in legacy plants;
# empty frames are kept (no box), they are where false alarms are counted.
PREWID_CLASSES = ("person",)
# Construction-PPE (M43, docs/tasks/M43-construction-ppe.md, AGPL-3.0): ids of the
# Ultralytics YAML. The person loader keeps `Person` only; the PPE classes serve T43.6.
CONSTRUCTION_PPE_CLASSES = ("helmet", "gloves", "vest", "boots", "goggles", "none", "Person",
                            "no_helmet", "no_goggle", "no_gloves", "no_boots")
CONSTRUCTION_PPE_PERSON_CLASSES = ("person",)

# Objets per-objet d'un échantillon (filtrés ensemble par `select`).
# `visible` (visible fraction of the full box, CrowdHuman, MOT17) and `track` (identity,
# MOT17) are optional per-object fields of M27.
PER_OBJECT = ("boxes", "xyxy", "labels", "difficult", "area", "crowd", "visible", "track")


def _image_size(path):
    from PIL import Image

    with Image.open(path) as img:  # en-tête seulement
        return img.size


def make_sample(image_id, image, width, height, x0y0x1y1, labels, difficult, clip=True,
                **extra):
    """Échantillon au format de `voc.load_split` depuis des coins **continus** (x0, y0, x1, y1)
    en pixels. `clip` borne les boîtes à l'image et retire celles qui deviennent vides ;
    sans `clip` (COCO), les boîtes sont gardées telles quelles, comme dans `COCOeval`.
    """
    b = np.asarray(x0y0x1y1, dtype=np.float64).reshape(-1, 4)
    keep = np.ones(len(b), bool)
    if clip:
        b = np.clip(b, 0, [width, height, width, height])
        keep = (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    xyxy = np.stack([b[:, 0] + 1, b[:, 1] + 1, b[:, 2], b[:, 3]], axis=1)
    s = {
        "id": str(image_id),
        "image": Path(image),
        "filename": Path(image).name,
        "width": int(width),
        "height": int(height),
        "boxes": xyxy_to_cxcywh(xyxy, width, height) if len(b) else np.zeros((0, 4)),
        "xyxy": xyxy,
        "labels": np.asarray(labels, dtype=np.int64).reshape(-1),
        "difficult": np.asarray(difficult, dtype=bool).reshape(-1),
    }
    for k, v in extra.items():
        s[k] = np.asarray(v).reshape(-1)
    return select(s, keep)


def select(sample, keep):
    """Copie de `sample` réduite aux objets `keep` (masque ou indices)."""
    out = dict(sample)
    for k in PER_OBJECT:
        if k in sample:
            out[k] = sample[k][keep]
    return out


# ---------------------------------------------------------------------------- formats

def parse_coco(data, image_dir, classes=None):
    """Annotations COCO JSON (dict) → échantillons. `classes` : noms retenus (catégories
    associées par nom, les autres ignorées) ; `None` : toutes, dans l'ordre croissant des
    `category_id` (COCO : ordre de `COCO_CLASSES`). Toutes les images sont rendues, y
    compris celles sans objet (comme `COCOeval`).
    """
    cats = sorted(data["categories"], key=lambda c: c["id"])
    if classes is None:
        cat_index = {c["id"]: i for i, c in enumerate(cats)}
    else:
        names = {n: i for i, n in enumerate(classes)}
        cat_index = {c["id"]: names[c["name"]] for c in cats if c["name"] in names}
    per_image = {}
    for a in data["annotations"]:
        if a["category_id"] in cat_index:
            per_image.setdefault(a["image_id"], []).append(a)
    samples = []
    for img in sorted(data["images"], key=lambda i: i["id"]):
        anns = per_image.get(img["id"], [])
        x0y0x1y1 = [[x, y, x + w, y + h] for x, y, w, h in (a["bbox"] for a in anns)]
        crowd = [bool(a.get("iscrowd", 0)) for a in anns]
        samples.append(make_sample(
            img["id"], Path(image_dir) / img["file_name"], img["width"], img["height"],
            x0y0x1y1, [cat_index[a["category_id"]] for a in anns], crowd, clip=False,
            area=np.array([a.get("area", a["bbox"][2] * a["bbox"][3]) for a in anns],
                          dtype=np.float64),
            crowd=np.array(crowd, dtype=bool)))
    return samples


def parse_kitti(text):
    """label_2/*.txt → (coins continus, labels). `DontCare` retiré."""
    boxes, labels = [], []
    for line in text.splitlines():
        f = line.split()
        if not f or f[0] == "DontCare":
            continue
        labels.append(KITTI_CLASSES.index(f[0]))
        boxes.append([float(v) for v in f[4:8]])
    return boxes, labels


def parse_visdrone(text, with_ignore=False):
    """annotations/*.txt (`x,y,w,h,score,catégorie,troncature,occlusion`) → (coins, labels).
    Catégorie 0 (région ignorée) et 11 (« others ») retirées des objets ; `with_ignore` rend
    aussi les coins des régions ignorées (catégorie 0), pour l'évaluation (T15.10).
    """
    boxes, labels, ignore = [], [], []
    for line in text.splitlines():
        f = [v for v in line.strip().split(",") if v != ""]
        if len(f) < 6:
            continue
        x, y, w, h = (float(v) for v in f[:4])
        cat = int(f[5])
        if 1 <= cat <= 10:
            boxes.append([x, y, x + w, y + h])
            labels.append(cat - 1)
        elif cat == 0:
            ignore.append([x, y, x + w, y + h])
    return (boxes, labels, ignore) if with_ignore else (boxes, labels)


def parse_crowdhuman(line, box="fbox"):
    """Une ligne odgt → (ID, coins, difficult). `mask` et `extra.ignore` → difficult."""
    d = json.loads(line)
    boxes, difficult = [], []
    for g in d.get("gtboxes", []):
        if g.get("tag") not in ("person", "mask"):
            continue
        x, y, w, h = g[box]
        boxes.append([x, y, x + w, y + h])
        difficult.append(g["tag"] == "mask" or bool(g.get("extra", {}).get("ignore", 0)))
    return d["ID"], boxes, difficult


def crowdhuman_visible(line):
    """Visible fraction |vbox ∩ fbox| / |fbox| of each box kept by `parse_crowdhuman` (NaN
    without `vbox`): the occlusion level of T27.0."""
    out = []
    for g in json.loads(line).get("gtboxes", []):
        if g.get("tag") not in ("person", "mask"):
            continue
        if "vbox" not in g:
            out.append(np.nan)
            continue
        fx, fy, fw, fh = g["fbox"]
        vx, vy, vw, vh = g["vbox"]
        iw = max(0.0, min(fx + fw, vx + vw) - max(fx, vx))
        ih = max(0.0, min(fy + fh, vy + vh) - max(fy, vy))
        out.append(iw * ih / (fw * fh) if fw * fh > 0 else np.nan)
    return out


def parse_bbgt(text):
    """ExDark, format bbGt v3 (`classe l t w h …`, en-tête `% bbGt`) → (coins, labels)."""
    boxes, labels = [], []
    for line in text.splitlines():
        f = line.split()
        if not f or f[0].startswith("%"):
            continue
        x, y, w, h = (float(v) for v in f[1:5])
        labels.append(EXDARK_CLASSES.index(f[0]))
        boxes.append([x, y, x + w, y + h])
    return boxes, labels


def parse_yolo_txt(text, width, height, n_classes, drop=()):
    """Étiquettes YOLO (`classe cx cy w h` normalisés) → (coins continus en pixels, labels).
    Les classes de `drop` (DontCare de HIT-UAV) et celles hors `n_classes` sont retirées.
    """
    boxes, labels = [], []
    for line in text.splitlines():
        f = line.split()
        if len(f) < 5:
            continue
        c = int(f[0])
        if c in drop or c >= n_classes:
            continue
        cx, cy, w, h = (float(v) for v in f[1:5])
        boxes.append([(cx - w / 2) * width, (cy - h / 2) * height,
                      (cx + w / 2) * width, (cy + h / 2) * height])
        labels.append(c)
    return boxes, labels


# ---------------------------------------------------------------------------- chargeurs

def _splits(load):
    """Plusieurs splits séparés par des virgules → échantillons concaténés."""
    def wrapped(root, split):
        out = []
        for s in split.split(","):
            out += load(Path(root), s.strip())
        return out
    return wrapped


def load_voc(root, split):
    """`split` : « année:nom » (2007:test, 2012:trainval…)."""
    year, name = split.split(":")
    return voc.load_split(root, int(year), name)


def load_coco(root, split):
    """`root/annotations/instances_<split>.json`, images dans `root/<split>/`."""
    data = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
    return parse_coco(data, root / split)


def load_flir(root, split):
    """FLIR ADAS v2 : `root/images_thermal_<split>/coco.json` (chemins relatifs au dossier)."""
    d = root / f"images_thermal_{split}"
    return parse_coco(json.loads((d / "coco.json").read_text()), d, FLIR_CLASSES)


def kitti_ids(root, split, val_fraction=0.2, seed=0):
    """Découpage train/val de KITTI training/ (le test n'a pas d'annotations publiques) :
    permutation à graine fixe des identifiants triés ; `trainval` : tout.
    """
    if split not in ("train", "val", "trainval"):
        raise ValueError(f"split KITTI inconnu : {split}")
    ids = sorted(p.stem for p in (root / "training" / "label_2").glob("*.txt"))
    if split == "trainval":
        return ids
    perm = np.random.default_rng(seed).permutation(len(ids))
    n_val = int(round(len(ids) * val_fraction))
    pick = perm[:n_val] if split == "val" else perm[n_val:]
    return [ids[i] for i in sorted(pick)]


def load_kitti(root, split):
    out = []
    for i in kitti_ids(root, split):
        image = root / "training" / "image_2" / f"{i}.png"
        boxes, labels = parse_kitti((root / "training" / "label_2" / f"{i}.txt").read_text())
        out.append(make_sample(i, image, *_image_size(image), boxes, labels,
                               np.zeros(len(labels), bool)))
    return out


def load_visdrone(root, split):
    """`root/VisDrone2019-DET-<split>/{images,annotations}` (split : train, val, test-dev)."""
    d = root / f"VisDrone2019-DET-{split}"
    out = []
    for ann in sorted((d / "annotations").glob("*.txt")):
        image = d / "images" / f"{ann.stem}.jpg"
        boxes, labels, ignore = parse_visdrone(ann.read_text(), with_ignore=True)
        s = make_sample(ann.stem, image, *_image_size(image), boxes, labels,
                        np.zeros(len(labels), bool))
        # Régions ignorées, par image (hors PER_OBJECT) : coins pixels VOC comme `xyxy`.
        r = np.asarray(ignore, dtype=np.float64).reshape(-1, 4)
        s["ignore_xyxy"] = np.stack([r[:, 0] + 1, r[:, 1] + 1, r[:, 2], r[:, 3]], axis=1)
        out.append(s)
    return out


def load_crowdhuman(root, split):
    """`root/annotation_<split>.odgt`, images dans `root/Images/<ID>.jpg` (ou
    `root/Images_val/`, comme dans la version Kaggle).
    """
    out = []
    for line in (root / f"annotation_{split}.odgt").read_text().splitlines():
        if not line.strip():
            continue
        i, boxes, difficult = parse_crowdhuman(line)
        image = root / "Images" / f"{i}.jpg"
        if not image.exists():
            image = root / "Images_val" / f"{i}.jpg"
        out.append(make_sample(i, image, *_image_size(image), boxes, [0] * len(boxes),
                               difficult, visible=crowdhuman_visible(line)))
    return out


EXDARK_SPLITS = {"train": 1, "val": 2, "test": 3}


def load_exdark(root, split):
    """`root/ExDark/<Classe>/<image>`, `root/ExDark_Annno/<Classe>/<image>.txt` (sic),
    découpage officiel de `root/imageclasslist.txt` (5e colonne : 1 train, 2 val, 3 test).
    Les noms de fichiers sont appariés sans tenir compte de la casse.
    """
    images = {p.name.lower(): p for p in (root / "ExDark").rglob("*") if p.is_file()}
    anns = {p.name.lower(): p for p in (root / "ExDark_Annno").rglob("*.txt")}
    want = EXDARK_SPLITS[split]
    out = []
    for line in (root / "imageclasslist.txt").read_text().splitlines():
        f = line.split()
        if len(f) < 5 or not f[4].isdigit() or int(f[4]) != want:
            continue
        name = f[0].lower()
        image = images[name]
        boxes, labels = parse_bbgt(anns[name + ".txt"].read_text())
        out.append(make_sample(Path(f[0]).stem, image, *_image_size(image), boxes, labels,
                               np.zeros(len(labels), bool)))
    return out


# AU-AIR n'a pas de découpage officiel : val = 3 vidéos entières (≈ 15 % des images), pour
# qu'aucune trame de val n'ait sa voisine en train.
AUAIR_VAL = ("frame_20190829091111", "frame_20190905111947", "frame_20190905143505")


def auair_sequence(image_name):
    """Vidéo d'une trame : `frame_<date>_x_<n>.jpg` (ou `_xx_`) → `frame_<date>`."""
    return "_".join(image_name.split("_")[:2])


def load_auair(root, split):
    """`root/annotations.json` (liste `annotations`, boîtes top/left/width/height en pixels),
    images dans `root/images/` ; split `train` ou `val` (`AUAIR_VAL`), `all` : tout.
    """
    if split not in ("train", "val", "all"):
        raise ValueError(f"split AU-AIR inconnu : {split}")
    data = json.loads((root / "annotations.json").read_text())
    out = []
    for a in sorted(data["annotations"], key=lambda a: a["image_name"]):
        name = a["image_name"]
        in_val = auair_sequence(name) in AUAIR_VAL
        if split != "all" and in_val != (split == "val"):
            continue
        w = a.get("image_width", a.get("image_width:"))  # clé « image_width: » (sic)
        h = a["image_height"]
        boxes = [[b["left"], b["top"], b["left"] + b["width"], b["top"] + b["height"]]
                 for b in a["bbox"]]
        labels = [b["class"] for b in a["bbox"]]
        out.append(make_sample(Path(name).stem, root / "images" / name, w, h, boxes, labels,
                               np.zeros(len(labels), bool)))
    return out


def _load_yolo_dir(images, labels, classes, drop=()):
    out = []
    for image in sorted(p for p in images.iterdir() if p.is_file()):
        w, h = _image_size(image)
        ann = labels / f"{image.stem}.txt"
        text = ann.read_text() if ann.exists() else ""
        boxes, lab = parse_yolo_txt(text, w, h, len(classes) + len(drop), drop)
        out.append(make_sample(image.stem, image, w, h, boxes, lab,
                               np.zeros(len(lab), bool)))
    return out


def load_dronevehicle(root, split):
    """Sortie de `tools/prep_datasets.py dronevehicle` : `root/<split>/{images,labels}`,
    cadre blanc recadré, boîtes orientées ramenées à leur boîte englobante."""
    return _load_yolo_dir(root / split / "images", root / split / "labels",
                          DRONEVEHICLE_CLASSES)


def load_hituav(root, split):
    """`root/{images,labels}/<split>/` (YOLO txt), images thermiques à un canal."""
    return _load_yolo_dir(root / "images" / split, root / "labels" / split, HITUAV_CLASSES,
                          drop=(4,))


def load_uavdt(root, split):
    """Sortie de `tools/prep_datasets.py uavdt` : `root/annotations_<split>.json` (séquences
    M du benchmark DET seules), chemins d'images relatifs à `root`."""
    out = []
    for a in json.loads((root / f"annotations_{split}.json").read_text()):
        labels = [UAVDT_CLASSES.index(c) for c in a["labels"]]
        out.append(make_sample(Path(a["image"]).stem, root / a["image"], a["width"],
                               a["height"], a["boxes"], labels, np.zeros(len(labels), bool)))
    return out


ASL_VAL_FROM = 2551  # frames n° 2551-3000 of each class: val split (M21, no random split)


def _load_json_split(root, split, classes, keep=None, meta=()):
    """`root/annotations_<split>.json` written by tools/prep_datasets.py (continuous corners,
    class names, image paths relative to `root`) → samples; `keep(entry)` filters them;
    the entry keys `meta` are copied to the sample (PREWID `camera`, `frame`)."""
    out = []
    for a in json.loads((root / f"annotations_{split}.json").read_text()):
        if keep and not keep(a):
            continue
        labels = [classes.index(c) for c in a["labels"]]
        s = make_sample(Path(a["image"]).stem, root / a["image"], a["width"], a["height"],
                        a["boxes"], labels, np.zeros(len(labels), bool))
        out.append(s | {k: a[k] for k in meta if k in a})
    return out


def load_asl(root, split):
    """Output of `tools/prep_datasets.py asl`: `root/annotations_<split>.json` (hand
    pseudo-boxes; letter images where no hand was found are left out), image paths relative
    to `root`."""
    return _load_json_split(root, split, ASL_CLASSES)


def load_trashnet(root, split):
    """Output of `tools/prep_datasets.py trashnet`: one auto-labelled box per image (images
    the labeller missed are left out), stratified random split."""
    return _load_json_split(root, split, TRASHNET_CLASSES)


def load_pets(root, split):
    """Output of `tools/prep_datasets.py pets`: whole-body boxes from the trimaps (one per
    mask component), `test` = official test.txt, `train` / `val` from trainval.txt."""
    return _load_json_split(root, split, PETS_CLASSES)


def load_prewid(root, split):
    """Output of `tools/prep_datasets.py prewid`: `train` / `val` / `test` = contiguous time
    blocks of each plant camera, `lab` = the lab sequence, `loco_<camera>_<split>` = leave
    one camera out. `<split>:<camera>` keeps the frames of one camera (`test:CAM2`). The
    samples carry `camera` and `frame` (sequences of T27.4)."""
    name, _, camera = split.partition(":")
    return _load_json_split(root, name, PREWID_CLASSES,
                            (lambda a: a["camera"] == camera) if camera else None,
                            meta=("camera", "frame"))


def load_construction_ppe(root, split):
    """Output of `tools/prep_datasets.py construction_ppe`: `Person` boxes of the official
    split (images with no person kept, with no box)."""
    return _load_json_split(root, split, CONSTRUCTION_PPE_PERSON_CLASSES)


# MOT17 (T27.4): ground-level pedestrian videos. gt.txt classes: 1 pedestrian (kept), 2
# person on vehicle, 7 static person, 8 distractor, 12 reflection (kept as `difficult`);
# vehicles and occluders are dropped. Every sequence comes three times (DPM, FRCNN, SDP
# public detections) with the same images and ground truth: the FRCNN copy is read.
MOT_CLASSES = ("person",)
MOT_DIFFICULT = (2, 7, 8, 12)


def mot_seqinfo(seq_dir):
    """`seqinfo.ini` → {name, frameRate, seqLength, imWidth, imHeight, imDir, imExt}."""
    info = {}
    for line in (Path(seq_dir) / "seqinfo.ini").read_text().splitlines():
        k, eq, v = line.partition("=")
        if eq:
            info[k.strip()] = int(v) if v.strip().isdigit() else v.strip()
    return info


def load_mot17(root, split):
    """`root/<train|test>/<SEQ>-FRCNN/{img1,gt/gt.txt,seqinfo.ini}`; `train:MOT17-02` keeps
    one sequence. Every frame is a sample (empty frames kept) with `sequence`, `frame` and
    `fps`; objects carry `track` and `visible`. A box with the `consider` flag 0 is
    `difficult`."""
    name, _, seq = split.partition(":")
    dirs = sorted((root / name).glob("MOT17-*-FRCNN"))
    if seq:
        dirs = [d for d in dirs if d.name.startswith(seq + "-")]
    if not dirs:
        raise FileNotFoundError(f"no MOT17-*-FRCNN sequence under {root / name}"
                                + (f" for {seq}" if seq else ""))
    out = []
    for d in dirs:
        info = mot_seqinfo(d)
        per = {}
        for line in (d / "gt" / "gt.txt").read_text().splitlines():
            f = line.split(",")
            if len(f) < 9:
                continue
            frame, track, x, y, w, h = int(f[0]), int(f[1]), *map(float, f[2:6])
            cls, consider, vis = int(f[7]), int(f[6]), float(f[8])
            if cls != 1 and cls not in MOT_DIFFICULT:
                continue
            per.setdefault(frame, []).append(
                ([x - 1, y - 1, x - 1 + w, y - 1 + h], cls != 1 or consider == 0, track, vis))
        seq_name = d.name.rsplit("-", 1)[0]
        for frame in range(1, info["seqLength"] + 1):
            objs = per.get(frame, [])
            image = d / info.get("imDir", "img1") / f"{frame:06d}{info.get('imExt', '.jpg')}"
            s = make_sample(f"{seq_name}_{frame:06d}", image, info["imWidth"],
                            info["imHeight"], [o[0] for o in objs], [0] * len(objs),
                            [o[1] for o in objs], track=[o[2] for o in objs],
                            visible=[o[3] for o in objs])
            out.append(s | {"sequence": seq_name, "frame": frame, "fps": info["frameRate"]})
    return out


# T27.6: industrial-site images (ARCYS), until then any folder of images + YOLO txt.
INDUSTRIAL_CLASSES = ("person",)
INDUSTRIAL_HELP = (
    "expected layout: <root>/<split>/images/*.jpg|png and <root>/<split>/labels/<stem>.txt "
    "(YOLO `0 cx cy w h` normalised, an empty txt = a frame without person); procedure in "
    "docs/tasks/M27-safety-metrics.md (T27.6): autolabel (M22), then manual correction")


def load_industrial(root, split):
    """`root/<split>/{images,labels}`: images and YOLO txt, class 0 = person; an image without
    txt is an error (an empty frame has an empty txt). A missing or empty folder raises a
    `FileNotFoundError` that names the expected layout."""
    images = root / split / "images"
    found = sorted(p for p in images.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
        if images.is_dir() else []
    if not found:
        raise FileNotFoundError(f"industrial: no image in {images}; {INDUSTRIAL_HELP}")
    out = []
    for img in found:
        txt = root / split / "labels" / f"{img.stem}.txt"
        if not txt.exists():
            raise FileNotFoundError(f"industrial: no label {txt} for {img.name}; "
                                    f"{INDUSTRIAL_HELP}")
        w, h = _image_size(img)
        boxes, labels = parse_yolo_txt(txt.read_text(), w, h, len(INDUSTRIAL_CLASSES))
        out.append(make_sample(img.stem, img, w, h, boxes, labels, np.zeros(len(labels), bool)))
    return out


@dataclass(frozen=True)
class Dataset:
    classes: tuple
    load: object      # (racine, split) → échantillons ; splits séparés par des virgules
    root: str         # dossier par défaut, relatif à data/
    test: str         # split d'évaluation
    calib: str        # split de calibration INT8
    train: str        # split(s) d'affinage
    metric: str = "voc"


DATASETS = {
    "voc": Dataset(VOC_CLASSES, _splits(load_voc), "VOCdevkit", "2007:test", "2007:trainval",
                   "2007:trainval,2012:trainval"),
    # calib2017 : sous-ensemble de train2017 de tools/coco_subset.py (T11.1).
    "coco": Dataset(COCO_CLASSES, _splits(load_coco), "coco", "val2017", "calib2017",
                    "train2017", metric="coco"),
    "kitti": Dataset(KITTI_CLASSES, _splits(load_kitti), "kitti", "val", "train", "train"),
    "visdrone": Dataset(VISDRONE_CLASSES, _splits(load_visdrone), "visdrone", "val", "train",
                        "train"),
    "crowdhuman": Dataset(CROWDHUMAN_CLASSES, _splits(load_crowdhuman), "crowdhuman", "val",
                          "train", "train"),
    "exdark": Dataset(EXDARK_CLASSES, _splits(load_exdark), "exdark", "test", "train",
                      "train"),
    "flir": Dataset(FLIR_CLASSES, _splits(load_flir), "flir", "val", "train", "train",
                    metric="coco"),
    "auair": Dataset(AUAIR_CLASSES, _splits(load_auair), "auair", "val", "train", "train"),
    "dronevehicle": Dataset(DRONEVEHICLE_CLASSES, _splits(load_dronevehicle), "dronevehicle",
                            "test", "train", "train"),
    "hituav": Dataset(HITUAV_CLASSES, _splits(load_hituav), "hituav", "test", "train",
                      "train"),
    "uavdt": Dataset(UAVDT_CLASSES, _splits(load_uavdt), "uavdt", "test", "train", "train"),
    # No VOC or COCO mapping (no hand class): fine-tuned models only.
    "asl": Dataset(ASL_CLASSES, _splits(load_asl), "asl", "val", "train", "train"),
    # M32: TrashNet has no mapping (no waste class worth it in VOC or COCO); Pet breeds map to
    # cat / dog.
    "trashnet": Dataset(TRASHNET_CLASSES, _splits(load_trashnet), "trashnet", "test", "train",
                        "train"),
    "pets": Dataset(PETS_CLASSES, _splits(load_pets), "pets", "test", "train", "train"),
    # M38: person only, so every pre-trained network gets a score (MAPPINGS).
    "prewid": Dataset(PREWID_CLASSES, _splits(load_prewid), "prewid", "test", "train",
                      "train"),
}

# M27: sets read by the safety tools only (no datasheet, notebooks or stats run): video
# sequences (T27.4) and the industrial slot (T27.6). `load`, `eval_view` and `split_of`
# read both registries.
EVAL_ONLY = {
    "mot17": Dataset(MOT_CLASSES, _splits(load_mot17), "mot17", "train", "train", "train"),
    "industrial": Dataset(INDUSTRIAL_CLASSES, _splits(load_industrial), "industrial", "test",
                          "calib", "train"),
    # M43: here until its datasheet and notebooks exist (T43.3), then moved to DATASETS.
    "construction_ppe": Dataset(CONSTRUCTION_PPE_PERSON_CLASSES, _splits(load_construction_ppe),
                                "construction_ppe", "test", "train", "train"),
}


def info(name):
    """`Dataset` entry of `name`, in `DATASETS` or `EVAL_ONLY`."""
    return DATASETS[name] if name in DATASETS else EVAL_ONLY[name]


# ---------------------------------------------------------------- correspondances

_VOC_TO_COCO = {c: c for c in VOC_CLASSES}  # les 20 noms VOC sont dans coco.names

MAPPINGS = {
    ("voc", "coco"): _VOC_TO_COCO,
    ("coco", "voc"): _VOC_TO_COCO,
    # Cyclist (personne + vélo dans une seule boîte) n'a pas d'équivalent ; Van compté en car
    # (KITTI ne pénalise pas une voiture détectée sur un van) ; Truck, Misc sans équivalent VOC.
    ("kitti", "voc"): {"Car": "car", "Van": "car", "Pedestrian": "person",
                       "Person_sitting": "person", "Tram": "train"},
    ("kitti", "coco"): {"Car": "car", "Van": "car", "Truck": "truck", "Pedestrian": "person",
                        "Person_sitting": "person", "Tram": "train"},
    # people (personnes assises, non debout) et pedestrian confondus en person.
    ("visdrone", "voc"): {"pedestrian": "person", "people": "person", "bicycle": "bicycle",
                          "car": "car", "van": "car", "bus": "bus", "motor": "motorbike"},
    ("visdrone", "coco"): {"pedestrian": "person", "people": "person", "bicycle": "bicycle",
                           "car": "car", "van": "car", "truck": "truck", "bus": "bus",
                           "motor": "motorbike"},
    ("crowdhuman", "voc"): {"person": "person"},
    ("crowdhuman", "coco"): {"person": "person"},
    ("exdark", "voc"): {"Bicycle": "bicycle", "Boat": "boat", "Bottle": "bottle", "Bus": "bus",
                        "Car": "car", "Cat": "cat", "Chair": "chair", "Dog": "dog",
                        "Motorbike": "motorbike", "People": "person", "Table": "diningtable"},
    ("exdark", "coco"): {"Bicycle": "bicycle", "Boat": "boat", "Bottle": "bottle",
                         "Bus": "bus", "Car": "car", "Cat": "cat", "Chair": "chair",
                         "Cup": "cup", "Dog": "dog", "Motorbike": "motorbike",
                         "People": "person", "Table": "diningtable"},
    # sign : tout panneau, pas seulement « stop sign » ; light : feu de circulation.
    ("flir", "voc"): {"person": "person", "bike": "bicycle", "car": "car",
                      "motor": "motorbike", "bus": "bus", "train": "train", "dog": "dog"},
    ("flir", "coco"): {"person": "person", "bike": "bicycle", "car": "car",
                       "motor": "motorbike", "bus": "bus", "train": "train", "truck": "truck",
                       "light": "traffic light", "hydrant": "fire hydrant", "dog": "dog",
                       "skateboard": "skateboard"},
    # Van compté en car (comme KITTI et VisDrone) ; Trailer sans équivalent.
    ("auair", "voc"): {"Human": "person", "Car": "car", "Van": "car", "Motorbike": "motorbike",
                       "Bicycle": "bicycle", "Bus": "bus"},
    ("auair", "coco"): {"Human": "person", "Car": "car", "Van": "car", "Truck": "truck",
                        "Motorbike": "motorbike", "Bicycle": "bicycle", "Bus": "bus"},
    # small-vehicle : voitures et vans ; large-vehicle confond bus, camions et fourgons :
    # compté en truck pour COCO (le bus y est faux), sans équivalent VOC.
    ("dronevehicle", "voc"): {"small-vehicle": "car"},
    ("dronevehicle", "coco"): {"small-vehicle": "car", "large-vehicle": "truck"},
    # OtherVehicle (tout véhicule hors voiture et vélo) sans équivalent.
    ("hituav", "voc"): {"Person": "person", "Car": "car", "Bicycle": "bicycle"},
    ("hituav", "coco"): {"Person": "person", "Car": "car", "Bicycle": "bicycle"},
    ("uavdt", "voc"): {"car": "car", "bus": "bus"},
    ("uavdt", "coco"): {"car": "car", "truck": "truck", "bus": "bus"},
    # M32: each breed to its species.
    ("pets", "voc"): {c: "cat" if c in PETS_CATS else "dog" for c in PETS_CLASSES},
    ("pets", "coco"): {c: "cat" if c in PETS_CATS else "dog" for c in PETS_CLASSES},
    ("prewid", "voc"): {"person": "person"},
    ("prewid", "coco"): {"person": "person"},
    ("mot17", "voc"): {"person": "person"},
    ("mot17", "coco"): {"person": "person"},
    ("industrial", "voc"): {"person": "person"},
    ("industrial", "coco"): {"person": "person"},
    ("construction_ppe", "voc"): {"person": "person"},
    ("construction_ppe", "coco"): {"person": "person"},
}

FAMILIES = {"voc": VOC_CLASSES, "coco": COCO_CLASSES}


def model_family(dataset, n_model_classes):
    """Classes de sortie du modèle : celles du jeu si le compte concorde (poids affinés sur
    ce jeu), sinon VOC (20) ou COCO (80). Rend (nom de famille, noms des classes).
    """
    if n_model_classes == len(info(dataset).classes):
        return dataset, info(dataset).classes
    for fam, names in FAMILIES.items():
        if n_model_classes == len(names):
            return fam, names
    raise ValueError(f"{n_model_classes} classes de sortie : ni {dataset}, ni VOC, ni COCO")


def class_lut(dataset, n_model_classes):
    """Indice de classe du jeu → indice de classe du modèle (−1 : sans équivalent)."""
    fam, names = model_family(dataset, n_model_classes)
    src = info(dataset).classes
    if fam == dataset:
        return np.arange(len(src), dtype=np.int64)
    if (dataset, fam) not in MAPPINGS:
        raise ValueError(f"pas de correspondance {dataset} → {fam}")
    m = MAPPINGS[(dataset, fam)]
    return np.array([names.index(m[c]) if c in m else -1 for c in src], dtype=np.int64)


@dataclass(frozen=True)
class EvalView:
    """Classes évaluées et passages vers leurs indices (−1 : hors évaluation)."""
    names: tuple
    gt_lut: np.ndarray    # classe du jeu → classe évaluée
    det_lut: np.ndarray   # classe du modèle → classe évaluée


def eval_view(dataset, n_model_classes):
    """Évaluation d'un modèle à `n_model_classes` sorties sur `dataset` : les classes
    évaluées sont les classes du modèle atteintes par la correspondance, dans l'ordre du
    modèle (identité pour VOC et des poids VOC).
    """
    _, names = model_family(dataset, n_model_classes)
    to_model = class_lut(dataset, n_model_classes)
    used = sorted(set(int(c) for c in to_model if c >= 0))
    det_lut = np.full(n_model_classes, -1, dtype=np.int64)
    det_lut[used] = np.arange(len(used))
    gt_lut = np.array([det_lut[c] if c >= 0 else -1 for c in to_model], dtype=np.int64)
    return EvalView(tuple(names[c] for c in used), gt_lut, det_lut)


def remap(samples, lut):
    """Labels passés par `lut` ; les objets à −1 sont retirés (les images restent)."""
    lut = np.asarray(lut, dtype=np.int64)
    out = []
    for s in samples:
        new = lut[s["labels"]] if len(s["labels"]) else s["labels"]
        r = select(s, new >= 0)
        r["labels"] = new[new >= 0]
        out.append(r)
    return out


def remap_detections(per_class, det_lut):
    """{classe modèle: dets} → {classe évaluée: dets} (correspondance injective)."""
    return {int(det_lut[c]): d for c, d in per_class.items()
            if c < len(det_lut) and det_lut[c] >= 0}


# ------------------------------------------------------------------------- outils CLI

def load(name, root=None, split=None, role="test"):
    """Échantillons du jeu `name` ; `split` par défaut selon `role` (test, calib, train)."""
    ds = info(name)
    return ds.load(Path(root) if root else DATA_DIR / ds.root, split or getattr(ds, role))


def add_args(ap, split_help="split(s), séparés par des virgules ; défaut selon le jeu"):
    """Options communes `--dataset`, `--data-root` (alias `--devkit`), `--split`."""
    ap.add_argument("--dataset", choices=sorted(DATASETS), default="voc")
    ap.add_argument("--data-root", "--devkit", dest="data_root", type=Path, default=None,
                    help="racine du jeu (défaut data/<dossier du jeu>)")
    ap.add_argument("--split", default=None, help=split_help)


def split_of(args, role="test"):
    return args.split or getattr(info(args.dataset), role)


def load_args(args, role="test"):
    """Échantillons désignés par les options de `add_args`."""
    return load(args.dataset, args.data_root, split_of(args, role), role)


def tag(dataset, split):
    """Suffixe des sorties : vide pour VOC (noms historiques), sinon `<jeu>-<split>`."""
    return "" if dataset == "voc" else f"{dataset}-{split.replace(',', '+').replace(':', '')}"
