"""Statistiques d'un jeu de données, vues par Tiny-YOLO et la cible embarquée (M16).

    python tools/data_stats.py --dataset kitti --split train,val --size 416 \\
        --out build/notebooks/kitti/stats [--sample 500] [--markdown] [--figures]
    python tools/data_stats.py --dataset voc --only comptes          # comptes seuls
    python tools/data_stats.py --report docs/tasks/stats-jeux.md     # synthèse des jeux

Une fonction pure par analyse (docs/tasks/M16-presentation-jeux.md, section B), qui prend la
liste d'échantillons des chargeurs (`DATASETS[jeu].load`) et rend un dict sérialisable :
`comptes` (T16.5), `geometrie` (T16.6), `densite` (T16.7), `images` (T16.8), `ancres`
(T16.9), `qualite` (T16.10), `ecart` (T16.11). Chaque analyse est donnée par split, pour
les groupes `train` et `test` du `Dataset` et pour l'ensemble. Tout ce qui vient des
annotations porte sur le split entier ; seule `images` lit des pixels, sur `--sample`
images tirées avec `--seed`.

Les grandeurs « vues par le réseau » sont calculées après `letterbox` ou `stretch` à
`--size` (`yolo.data.letterbox`), les collisions par `yolo.data.targets.build_targets`
avec les ancres de la cfg `--net`. NumPy pur, sans pandas (ADR 0001).

Sorties dans `--out` : `stats.json` (une clé par analyse), `stats.md` (tables), `galerie/`
(`--gallery`), `figures/` (`--figures`, `tools/figures/donnees.py`).
"""

import argparse
import json
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from yolo.data import datasets as D  # noqa: E402
from yolo.data.anchors import kmeans_anchors, mean_best_iou  # noqa: E402
from yolo.data.letterbox import as_hw, boxes_to_letterbox, parse_size, size_label  # noqa: E402
from yolo.data.targets import anchor_ref, anchors_frac, build_targets, heads  # noqa: E402
from yolo.infer.boxes import cxcywh_to_xyxy, iou_wh, iou_xyxy  # noqa: E402
from yolo.models.specs import infer_shapes  # noqa: E402
from yolo.models.tiny_yolo import load_cfg  # noqa: E402

ANALYSES = ("comptes", "geometrie", "densite", "images", "qualite", "ancres", "ecart")
SIZE = 416
SAMPLE = 200
SLOTS = (256, 1024)        # emplacements de yolo_post (T10.3), et seuil de T11.6
AREA_BINS = (32 ** 2, 96 ** 2)  # petit / moyen / grand (convention COCO, en pixels)
H_UNDER = (4, 8, 16, 32)   # hauteurs de boîte (px d'entrée) relevées (question KITTI)
DEGENERATE = 2             # côté minimal (px) d'une boîte non dégénérée
DUPLICATE_IOU = 0.95
SCATTER = 5000             # boîtes gardées pour les nuages (w, h)
CENTER_BINS = 32
K_RANGE = tuple(range(1, 13))
BATCH = 64                 # images par appel de build_targets
SUSPECTS = 10

# Bacs fixes des histogrammes de stats.json (les figures les relisent tels quels).
BINS = {
    "side": np.geomspace(1, 4096, 61),          # largeur, hauteur (px)
    "area": np.geomspace(1, 4096 ** 2, 61),     # aire (px²)
    "area_rel": np.geomspace(1e-6, 1, 61),      # aire / aire de l'image
    "aspect": np.linspace(-4, 4, 41),           # log2(w / h)
    "image_aspect": np.linspace(-2, 2, 41),     # log2(W / H) des images
}

# Types d'éclairage d'ExDark (3e colonne de imageclasslist.txt, article d'ExDark).
EXDARK_LIGHT = {1: "Low", 2: "Ambient", 3: "Object", 4: "Single", 5: "Weak", 6: "Strong",
                7: "Screen", 8: "Window", 9: "Shadow", 10: "Twilight"}

# Réseau dont les ancres servent aux collisions quand la cfg du jeu manque.
FALLBACK_NET = "tiny-yolov3-coco"
REF_ANCHORS = {"VOC (tiny-yolov2-voc)": "tiny-yolov2-voc",
               "VOC (tiny-yolov3-voc)": "tiny-yolov3-voc",
               "COCO (tiny-yolov3-coco)": "tiny-yolov3-coco"}


# ---------------------------------------------------------------------------- fiches (T16.2)

# Seule partie écrite à la main, comme `OFFICIAL` de tools/voc_stats.py. `official` :
# {split du dépôt: (images, objets)} publiés par le jeu (None : non publié) ; `objects`
# précise ce que compte le chiffre officiel. `notes` : particularités du chargeur, qui
# expliquent les écarts entre comptes mesurés et officiels.
FICHES = {
    "voc": {
        "name": "PASCAL VOC 2007 + 2012",
        "source": "http://host.robots.ox.ac.uk/pascal/VOC/",
        "version": "VOC2007 (test, trainval), VOC2012 (trainval)",
        "license": "images Flickr (conditions de Flickr), annotations pour la recherche",
        "sensor": "photos grand public de Flickr, scènes de la vie courante, couleur",
        "resolution": "≈ 500×375",
        "classes": D.VOC_CLASSES,
        "official": {"2007:test": (4952, 12032), "2007:trainval": (5011, 12608),
                     "2012:trainval": (11540, 27450)},
        "objects": "objets non-difficult (tableaux « Main » du devkit)",
        "notes": ["objets `difficult` gardés, ignorés par l'évaluation VOC",
                  "test : 2007:test ; calibration : 2007:trainval ; affinage : 07+12 trainval"],
    },
    "coco": {
        "name": "MS COCO 2017",
        "source": "https://cocodataset.org",
        "version": "2017 (val2017, train2017 ; calib2017 de tools/coco_subset.py)",
        "license": "annotations CC BY 4.0, images Flickr (conditions de Flickr)",
        "sensor": "photos Flickr, scènes encombrées, couleur",
        "resolution": "≈ 640×480",
        "classes": D.COCO_CLASSES,
        "official": {"val2017": (5000, 36781), "train2017": (118287, 860001),
                     "calib2017": (None, None)},
        "objects": "toutes les annotations, `iscrowd` compris",
        "notes": ["boîtes gardées sans rognage (`clip=False`, comme COCOeval)",
                  "`iscrowd` en `difficult` et `crowd`",
                  "calib2017 : 500 images annotées de train2017 (tools/coco_subset.py)"],
    },
    "kitti": {
        "name": "KITTI 2D object",
        "source": "https://www.cvlibs.net/datasets/kitti/eval_object.php",
        "version": "object 2012, training/ (le test n'a pas d'annotations publiques)",
        "license": "CC BY-NC-SA 3.0",
        "sensor": "caméra couleur stéréo (Point Grey Flea 2) sur voiture, Karlsruhe, de jour",
        "resolution": "≈ 1242×375",
        "classes": D.KITTI_CLASSES,
        "official": {"train": (5985, None), "val": (1496, None)},
        "objects": "découpage maison de 7 481 images (`kitti_ids`, 20 % en val, graine 0)",
        "notes": ["`DontCare` retiré par le chargeur",
                  "train/val : permutation à graine fixe des identifiants (`kitti_ids`)"],
    },
    "visdrone": {
        "name": "VisDrone2019-DET",
        "source": "https://github.com/VisDrone/VisDrone-Dataset",
        "version": "VisDrone2019-DET (train, val, test-dev)",
        "license": "recherche, non commerciale",
        "sensor": "caméras de drones, 14 villes de Chine, altitudes et angles variés",
        "resolution": "960×540 à 2000×1500",
        "classes": D.VISDRONE_CLASSES,
        "official": {"train": (6471, None), "val": (548, None)},
        "objects": "objets des 10 classes",
        "notes": ["régions ignorées (catégorie 0) et « others » (11) retirées par le chargeur"],
    },
    "crowdhuman": {
        "name": "CrowdHuman",
        "source": "https://www.crowdhuman.org",
        "version": "2018 (annotation_train.odgt, annotation_val.odgt), boîtes `fbox`",
        "license": "recherche, non commerciale",
        "sensor": "images web de foules, couleur",
        "resolution": "variable (≈ 800 à 1400 px de large)",
        "classes": D.CROWDHUMAN_CLASSES,
        "official": {"train": (15000, 339565), "val": (4370, 99481)},
        "objects": "personnes annotées hors `mask` et `ignore`",
        "notes": ["`mask` (régions de foule) et `extra.ignore` gardés en `difficult`",
                  "boîte entière (`fbox`), qui peut déborder de l'image : rognée"],
    },
    "exdark": {
        "name": "ExDark (Exclusively Dark)",
        "source": "https://github.com/cs-chan/Exclusively-Dark-Image-Dataset",
        "version": "2019, découpage officiel de imageclasslist.txt",
        "license": "BSD 3-Clause (dépôt), recherche",
        "sensor": "photos en faible luminosité, 10 types d'éclairage, couleur",
        "resolution": "variable",
        "classes": D.EXDARK_CLASSES,
        "official": {"train": (3000, None), "val": (1800, None), "test": (2563, None)},
        "objects": "7 363 images au total, 12 classes",
        "notes": ["type d'éclairage (3e colonne de imageclasslist.txt) lu par data_stats, "
                  "pas par `load_exdark`"],
    },
    "flir": {
        "name": "Teledyne FLIR ADAS Thermal v2",
        "source": "https://www.flir.com/oem/adas/adas-dataset-form/",
        "version": "v2 (2022), images_thermal_train et images_thermal_val",
        "license": "licence Teledyne FLIR du jeu ADAS, recherche non commerciale",
        "sensor": "caméra thermique Boson/Tau2 sur voiture, jour et nuit, 8 bits",
        "resolution": "640×512",
        "classes": D.FLIR_CLASSES,
        "official": {"train": (10742, None), "val": (1144, None)},
        "objects": "15 catégories retenues (`FLIR_CLASSES`), les autres ignorées",
        "notes": ["catégories hors `FLIR_CLASSES` ignorées par `parse_coco`",
                  "images 8 bits du jeu ; analyse 16 bits hors périmètre"],
    },
    # Jeux drone et thermiques de M18 (docs/tasks/M18-jeux-drone.md).
    "auair": {
        "name": "AU-AIR",
        "source": "https://bozcani.github.io/auairdataset",
        "version": "2019 (annotations.json, 8 vidéos) ; découpage maison par vidéo",
        "license": "CC BY-NC-SA 2.0 (champ `licenses` du JSON)",
        "sensor": "drone Parrot Bebop 2, caméra couleur, Aarhus (Danemark), vues obliques "
                  "et verticales, altitude et attitude par trame",
        "resolution": "1920×1080",
        "classes": D.AUAIR_CLASSES,
        "official": {"train": (27880, None), "val": (4943, None)},
        "objects": "32 823 trames annotées, 132 031 boîtes (8 classes)",
        "notes": ["pas de découpage officiel : val = 3 vidéos entières (`AUAIR_VAL`)",
                  "clé `image_width:` (sic) et noms `_xx_` du JSON tolérés",
                  "boîtes vides après rognage retirées (54)"],
    },
    "dronevehicle": {
        "name": "DroneVehicle (RGB)",
        "source": "https://github.com/VisDrone/DroneVehicle",
        "version": "version YOLO-OBB de Roboflow (M. Mandal), re-découpée depuis le train "
                   "d'origine ; tools/prep_datasets.py dronevehicle",
        "license": "CC BY-NC-SA 4.0",
        "sensor": "drone, caméra couleur (la moitié infrarouge du jeu d'origine est absente), "
                  "jour et nuit",
        "resolution": "640×512 (840×712 avec le cadre blanc d'origine)",
        "classes": D.DRONEVEHICLE_CLASSES,
        "official": {"train": (12118, 193940), "val": (2599, 42220), "test": (2608, 41918)},
        "objects": "boîtes orientées ramenées à leur boîte englobante",
        "notes": ["cadre blanc de 100 px recadré par `prep_datasets.py`",
                  "small-vehicle : voiture et van ; large-vehicle : bus, camion, fourgon"],
    },
    "hituav": {
        "name": "HIT-UAV (infrarouge thermique)",
        "source": "https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset",
        "version": "2022, version YOLO (train, val, test)",
        "license": "à vérifier (page du jeu)",
        "sensor": "caméra thermique sur drone, 60 à 130 m d'altitude, jour et nuit, 8 bits, "
                  "un canal",
        "resolution": "640×512",
        "classes": D.HITUAV_CLASSES,
        "official": {"train": (2008, 17518), "val": (287, 2453), "test": (571, 4780)},
        "objects": "4 classes ; DontCare retiré",
        "notes": ["`DontCare` (classe 4) retiré par le chargeur",
                  "nom de fichier : `<période>_<altitude>_<angle>_…` (jour ou nuit, altitude 60 à "
                  "130 m, angle 30 à 90°)"],
    },
    "uavdt": {
        "name": "UAVDT-DET",
        "source": "https://sites.google.com/view/grli-uavdt",
        "version": "export Supervisely de DatasetNinja ; séquences M seules "
                   "(tools/prep_datasets.py uavdt)",
        "license": "recherche seulement",
        "sensor": "drone, caméra couleur, scènes urbaines, altitudes et météo variées "
                  "(tags jour, nuit, brouillard)",
        "resolution": "1024×540",
        "classes": D.UAVDT_CLASSES,
        "official": {"train": (24143, 422911), "test": (16592, 375884)},
        "objects": "30 séquences en train, 20 en test, trames consécutives",
        "notes": ["séquences S du test (suivi d'un objet) écartées par `prep_datasets.py`",
                  "régions ignorées du jeu d'origine absentes de l'export"],
    },
}


# ---------------------------------------------------------------------------- contexte

@dataclass
class Ctx:
    dataset: str
    size: object = SIZE        # entier ou (H, W)
    resize: str = "letterbox"
    net: str = FALLBACK_NET
    seed: int = 0
    sample: int = 0
    root: Path = None          # racine du jeu
    splits: tuple = ()         # splits des échantillons analysés (fichiers bruts)
    _spec: dict = field(default=None, repr=False)

    @property
    def hw(self):
        return as_hw(self.size)

    def spec(self):
        """(cfg, têtes, {id: (S_h, S_w)}, {id: pas en px}) du réseau à `size`."""
        if self._spec is None:
            cfg = load_cfg(self.net)
            sh, sw = self.hw
            shapes = infer_shapes(dict(cfg, input=(cfg["input"][0], sh, sw)))
            hl = heads(cfg)
            grids = {hid: tuple(shapes[hid][1][1:]) for hid, _ in hl}
            strides = {hid: sw / g[1] for hid, g in grids.items()}
            self._spec = {"cfg": cfg, "heads": hl, "grids": grids, "strides": strides}
        return self._spec


def _hist(values, edges):
    return np.histogram(np.asarray(values, dtype=np.float64), bins=edges)[0].tolist()


def _py(x):
    """Objet NumPy → types JSON."""
    if isinstance(x, dict):
        return {str(k): _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_py(v) for v in x]
    if isinstance(x, np.ndarray):
        return _py(x.tolist())
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.floating):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float) and not np.isfinite(x):
        return None
    if isinstance(x, Path):
        return str(x)
    return x


def _frac(mask):
    mask = np.asarray(mask)
    return float(mask.mean()) if mask.size else 0.0


def _easy(s):
    return ~s["difficult"]


def input_boxes(s, size, resize):
    """Boîtes normalisées dans l'entrée `size` : letterbox par `boxes_to_letterbox`, stretch
    = repère d'origine (redimensionnement direct)."""
    if resize == "letterbox":
        return boxes_to_letterbox(s["boxes"], s["width"], s["height"], size)
    return np.asarray(s["boxes"], dtype=np.float64).reshape(-1, 4)


def wh_pixels(samples, size=None, resize="letterbox", easy=True):
    """(n, 2) largeurs et hauteurs des boîtes en pixels d'origine (`size` None) ou d'entrée."""
    out = []
    for s in samples:
        keep = _easy(s) if easy else np.ones(len(s["labels"]), bool)
        if size is None:
            b = s["boxes"][keep][:, 2:] * (s["width"], s["height"])
        else:
            sh, sw = as_hw(size)
            b = input_boxes(s, size, resize)[keep][:, 2:] * (sw, sh)
        out.append(b)
    return np.concatenate(out) if out else np.zeros((0, 2))


# ---------------------------------------------------------------------------- T16.5 comptes

def coverage(samples, dataset):
    """Part des objets du jeu qui ont une classe VOC ou COCO (`MAPPINGS`), et objets
    évalués hors domaine (non-difficult après `remap` par `eval_view`, comme eval_voc.py)."""
    out = {}
    total = sum(len(s["labels"]) for s in samples)
    for fam, names in D.FAMILIES.items():
        if fam != dataset and (dataset, fam) not in D.MAPPINGS:
            continue
        view = D.eval_view(dataset, len(names))
        kept = D.remap(samples, view.gt_lut)
        n = sum(len(s["labels"]) for s in kept)
        out[fam] = {"objects": n, "share": n / total if total else 0.0,
                    "evaluated": sum(int(_easy(s).sum()) for s in kept),
                    "classes": list(view.names)}
    return out


def comptes(samples, ctx):
    classes = D.DATASETS[ctx.dataset].classes
    c = len(classes)
    per_class = np.zeros(c, np.int64)
    per_class_easy = np.zeros(c, np.int64)
    images_with = np.zeros(c, np.int64)
    cooc = np.zeros((c, c), np.int64)
    difficult = crowd = empty = 0
    for s in samples:
        lab = s["labels"]
        np.add.at(per_class, lab, 1)
        np.add.at(per_class_easy, lab[_easy(s)], 1)
        present = np.unique(lab)
        images_with[present] += 1
        cooc[np.ix_(present, present)] += 1
        difficult += int(s["difficult"].sum())
        crowd += int(s["crowd"].sum()) if "crowd" in s else 0
        empty += int(len(lab) == 0)
    nz = per_class[per_class > 0]
    return {
        "images": len(samples), "objects": int(per_class.sum()),
        "objects_easy": int(per_class_easy.sum()), "difficult": difficult, "crowd": crowd,
        "empty_images": empty, "per_class": per_class, "per_class_easy": per_class_easy,
        "images_with": images_with, "cooc": cooc,
        "imbalance": float(nz.max() / nz.min()) if len(nz) else None,
        "absent_classes": [classes[i] for i in np.flatnonzero(per_class == 0)],
        "coverage": coverage(samples, ctx.dataset),
    }


# ---------------------------------------------------------------------------- T16.6 géométrie

def _coco_cats(area):
    a = np.asarray(area)
    return [int((a < AREA_BINS[0]).sum()), int(((a >= AREA_BINS[0]) & (a < AREA_BINS[1])).sum()),
            int((a >= AREA_BINS[1]).sum())]


def _sizes(wh):
    area = wh[:, 0] * wh[:, 1]
    return {"w": _hist(wh[:, 0], BINS["side"]), "h": _hist(wh[:, 1], BINS["side"]),
            "area": _hist(area, BINS["area"]), "coco": _coco_cats(area),
            "median_side": float(np.median(np.sqrt(area))) if len(area) else None}


def geometrie(samples, ctx):
    """Objets non-difficult (ceux de l'entraînement et de l'évaluation)."""
    sp = ctx.spec()
    classes = D.DATASETS[ctx.dataset].classes
    orig = wh_pixels(samples)
    rel = np.concatenate([s["boxes"][_easy(s)][:, 2] * s["boxes"][_easy(s)][:, 3]
                          for s in samples]) if samples else np.zeros(0)
    out = {"n": len(orig), "size": list(ctx.hw), "resize": ctx.resize,
           "orig": {**_sizes(orig), "area_rel": _hist(rel, BINS["area_rel"]),
                    "aspect": _hist(np.log2(np.maximum(orig[:, 0], 1e-6)
                                            / np.maximum(orig[:, 1], 1e-6)), BINS["aspect"])}}
    for mode in ("letterbox", "stretch"):
        wh = wh_pixels(samples, ctx.size, mode)
        d = _sizes(wh)
        d["fit_cell"] = {f"{g[0]}x{g[1]}": _frac(wh.max(axis=1) < sp["strides"][hid])
                         if len(wh) else 0.0 for hid, g in sp["grids"].items()}
        d["h_under"] = {str(t): _frac(wh[:, 1] < t) for t in H_UNDER}
        d["min_side_under"] = {str(t): _frac(wh.min(axis=1) < t) for t in H_UNDER}
        out[mode] = d
    out["strides"] = {f"{g[0]}x{g[1]}": sp["strides"][hid] for hid, g in sp["grids"].items()}
    centers = np.zeros((CENTER_BINS, CENTER_BINS), np.int64)
    side_in = [[] for _ in classes]
    main = []
    for s in samples:
        b = s["boxes"][_easy(s)]
        if len(b):
            i = np.minimum((b[:, 1] * CENTER_BINS).astype(int), CENTER_BINS - 1)
            j = np.minimum((b[:, 0] * CENTER_BINS).astype(int), CENTER_BINS - 1)
            np.add.at(centers, (np.clip(i, 0, None), np.clip(j, 0, None)), 1)
        sh, sw = ctx.hw
        wh = input_boxes(s, ctx.size, ctx.resize)[_easy(s)][:, 2:] * (sw, sh)
        for lab, (w, h) in zip(s["labels"][_easy(s)], wh):
            side_in[lab].append(np.sqrt(w * h))
        main.append(wh)
    out["centers"] = centers
    out["per_class"] = {
        "n": [len(v) for v in side_in],
        "median_side_in": [float(np.median(v)) if v else None for v in side_in],
        "small_in": [_frac(np.square(v) < AREA_BINS[0]) if v else None for v in side_in],
    }
    main = np.concatenate(main) if main else np.zeros((0, 2))
    rng = np.random.default_rng(ctx.seed)
    pick = np.sort(rng.choice(len(main), min(SCATTER, len(main)), replace=False))
    out["scatter_in"] = np.round(main[pick], 2)
    return out


# ---------------------------------------------------------------------------- T16.7 densité

def collisions(samples, ctx):
    """Objets non-difficult (cibles de l'entraînement) qui tombent sur la même cellule et la
    même ancre qu'un autre : `build_targets` n'en garde qu'un (le dernier). Perdus = objets −
    cibles positives, par tête et par classe."""
    sp = ctx.spec()
    cfg, hl, grids = sp["cfg"], sp["heads"], sp["grids"]
    ref = anchor_ref(cfg)
    af = anchors_frac(cfg["anchors"], ref)
    owner = {k: hid for hid, mask in hl for k in mask}
    classes = D.DATASETS[ctx.dataset].classes
    obj_head = Counter()
    pos_head = Counter()
    obj_cls = np.zeros(len(classes), np.int64)
    pos_cls = np.zeros(len(classes), np.int64)
    lost_img = []
    for k in range(0, len(samples), BATCH):
        chunk = samples[k:k + BATCH]
        bl, ll = [], []
        for s in chunk:
            b = input_boxes(s, ctx.size, ctx.resize)
            keep = _easy(s) & (b[:, 2] > 0) & (b[:, 3] > 0)
            bl.append(b[keep])
            ll.append(s["labels"][keep])
            if keep.any():
                for kk in np.argmax(iou_wh(b[keep][:, 2:4], af), axis=1):
                    obj_head[owner.get(int(kk))] += 1
            np.add.at(obj_cls, s["labels"][keep], 1)
        with np.errstate(divide="ignore"):
            t = build_targets(bl, ll, cfg["anchors"], hl, grids, ref)
        per_img = np.zeros(len(chunk), np.int64)
        for hid, _ in hl:
            obj = t[hid]["obj"]
            pos_head[hid] += int(obj.sum())
            np.add.at(pos_cls, t[hid]["cls"][obj], 1)
            per_img += obj.reshape(len(chunk), -1).sum(axis=1)
        lost_img += [len(b) - int(p) for b, p in zip(bl, per_img)]
    lost_img = np.array(lost_img, np.int64)
    n_obj = int(obj_cls.sum())
    return {
        "net": str(ctx.net), "anchors": cfg["anchors"],
        "objects": n_obj, "positives": int(pos_cls.sum()),
        "lost": n_obj - int(pos_cls.sum()),
        "lost_share": (n_obj - int(pos_cls.sum())) / n_obj if n_obj else 0.0,
        "per_head": {f"{grids[hid][0]}x{grids[hid][1]}": {
            "objects": obj_head[hid], "positives": pos_head[hid],
            "lost": obj_head[hid] - pos_head[hid]} for hid, _ in hl},
        "per_class": {"objects": obj_cls, "lost": obj_cls - pos_cls},
        "images_with_loss": int((lost_img > 0).sum()),
        "max_lost_image": int(lost_img.max()) if len(lost_img) else 0,
    }


def densite(samples, ctx):
    per = np.array([len(s["labels"]) for s in samples], np.int64)
    easy = np.array([int(_easy(s).sum()) for s in samples], np.int64)
    out = {"per_image": np.bincount(per) if len(per) else [],
           "per_image_easy": np.bincount(easy) if len(easy) else [],
           "median": float(np.median(per)) if len(per) else 0.0,
           "mean": float(per.mean()) if len(per) else 0.0,
           "max": int(per.max()) if len(per) else 0,
           "over": {str(n): int((per > n).sum()) for n in SLOTS},
           "over_easy": {str(n): int((easy > n).sum()) for n in SLOTS}}
    out["collisions"] = collisions(samples, ctx)
    return out


# ---------------------------------------------------------------------------- T16.8 images

GRAY_MODES = ("1", "L", "LA", "I", "I;16", "F", "P;L")


def pick(n, k, seed):
    """`k` indices parmi `n`, tirés avec `seed`, triés (même tirage d'une exécution à l'autre)."""
    return np.sort(np.random.default_rng(seed).choice(n, min(k, n), replace=False))


def pixel_stats(samples, sample, seed):
    """Sur `sample` images tirées avec `seed` : canaux, histogrammes 8 bits par canal,
    moyenne et écart type, luminance moyenne par image, niveaux gardés par l'INT8 d'entrée."""
    from PIL import Image

    from yolo.quant.quantize import INPUT_SCALE, QMAX

    idx = pick(len(samples), sample, seed)
    modes = Counter()
    for i in idx:
        with Image.open(samples[i]["image"]) as img:
            modes[img.mode] += 1
    channels = 1 if all(m in GRAY_MODES for m in modes) else 3
    hist = np.zeros((channels, 256), np.int64)
    lum_hist = np.zeros(256, np.int64)
    lum, ids = [], []
    for i in idx:
        with Image.open(samples[i]["image"]) as img:
            x = np.asarray(img.convert("L" if channels == 1 else "RGB"), dtype=np.uint8)
            y = x if channels == 1 else np.asarray(img.convert("L"), dtype=np.uint8)
        x = x.reshape(-1, channels)
        for c in range(channels):
            hist[c] += np.bincount(x[:, c], minlength=256)
        lum_hist += np.bincount(y.reshape(-1), minlength=256)
        lum.append(float(y.mean()))
        ids.append(samples[i]["id"])
    v = np.arange(256)
    tot = hist.sum(axis=1)
    mean = (hist * v).sum(axis=1) / np.maximum(tot, 1)
    std = np.sqrt((hist * v ** 2).sum(axis=1) / np.maximum(tot, 1) - mean ** 2)
    used = v[lum_hist > 0]
    q = np.clip(np.floor(used / 255.0 / INPUT_SCALE + 0.5), -QMAX, QMAX)
    # Masse couverte : niveaux 8 bits qui portent 99 % des pixels.
    order = np.sort(lum_hist)[::-1]
    n99 = int(np.searchsorted(np.cumsum(order), 0.99 * order.sum()) + 1) if order.sum() else 0
    return {"images": len(idx), "seed": seed, "modes": dict(modes), "channels": channels,
            "hist": hist, "mean": mean, "std": std, "lum_hist": lum_hist,
            "lum": np.round(lum, 2), "ids": ids,
            "lum_mean": float(np.mean(lum)) if lum else None,
            "levels": {"8bit": len(used), "int8": len(np.unique(q)), "8bit_99": n99,
                       "input_scale": INPUT_SCALE}}


def images(samples, ctx):
    wh = np.array([(s["width"], s["height"]) for s in samples], np.int64).reshape(-1, 2)
    res = Counter(map(tuple, wh.tolist()))
    out = {"resolutions": [[w, h, n] for (w, h), n in res.most_common(10)],
           "distinct": len(res),
           "width": [int(wh[:, 0].min()), float(np.median(wh[:, 0])), int(wh[:, 0].max())]
           if len(wh) else [], "height": [int(wh[:, 1].min()), float(np.median(wh[:, 1])),
                                          int(wh[:, 1].max())] if len(wh) else [],
           "aspect": _hist(np.log2(wh[:, 0] / wh[:, 1]), BINS["image_aspect"]) if len(wh) else [],
           "square": _frac(wh[:, 0] == wh[:, 1]) if len(wh) else 0.0}
    if ctx.sample:
        out["pixels"] = pixel_stats(samples, ctx.sample, ctx.seed)
    return out


# ---------------------------------------------------------------------------- T16.9 ancres

def ancres(samples, ctx):
    """Boîtes non-difficult en pixels de l'image letterbox `size` (`dataset_wh` de
    tools/kmeans_anchors.py) : mêmes ancres que cet outil aux mêmes `--size` et `--seed`."""
    from tools.kmeans_anchors import dataset_wh

    if not any(_easy(s).any() for s in samples):
        return {}
    wh = dataset_wh(samples, ctx.size)
    cfg_anchors = np.array(ctx.spec()["cfg"]["anchors"], dtype=np.float64)
    kmeans = {k: kmeans_anchors(wh, k, rng=ctx.seed) for k in K_RANGE}
    refs = {name: np.array(load_cfg(net)["anchors"], dtype=np.float64)
            for name, net in REF_ANCHORS.items()}
    refs[f"cfg du jeu ({Path(str(ctx.net)).stem})"] = cfg_anchors
    rng = np.random.default_rng(ctx.seed)
    sel = np.sort(rng.choice(len(wh), min(SCATTER, len(wh)), replace=False))
    return {"boxes": len(wh), "size": list(ctx.hw), "seed": ctx.seed,
            "kmeans": {str(k): np.round(a, 2) for k, a in kmeans.items()},
            "kmeans_iou": {str(k): mean_best_iou(wh, a) for k, a in kmeans.items()},
            "refs": {n: {"anchors": a, "iou": mean_best_iou(wh, a)} for n, a in refs.items()},
            "scatter": np.round(wh[sel], 2)}


# ---------------------------------------------------------------------------- T16.10 qualité

def _corners(s):
    """Coins continus (x0, y0, x1, y1) en pixels d'origine."""
    return cxcywh_to_xyxy(s["boxes"]) * (s["width"], s["height"], s["width"], s["height"])


def raw_annotations(dataset, root, samples, splits):
    """{id: (coins bruts (n, 4), {région retirée: nombre})} relus dans les fichiers du jeu,
    par les `parse_*` des chargeurs ; None pour un jeu sans retrait (VOC, COCO, FLIR,
    DroneVehicle et UAVDT, déjà prétraités par tools/prep_datasets.py)."""
    root = Path(root) if root is not None else None
    out = {}
    if dataset == "kitti":
        for s in samples:
            text = (Path(s["image"]).parents[1] / "label_2" / f"{s['id']}.txt").read_text()
            boxes, _ = D.parse_kitti(text)
            dc = sum(1 for line in text.splitlines() if line.split()[:1] == ["DontCare"])
            out[s["id"]] = (boxes, {"DontCare": dc})
    elif dataset == "visdrone":
        for s in samples:
            text = (Path(s["image"]).parents[1] / "annotations" / f"{s['id']}.txt").read_text()
            boxes, _ = D.parse_visdrone(text)
            cats = Counter()
            for line in text.splitlines():
                f = [v for v in line.strip().split(",") if v != ""]
                if len(f) >= 6 and int(f[5]) in (0, 11):
                    cats["ignorée (0)" if int(f[5]) == 0 else "others (11)"] += 1
            out[s["id"]] = (boxes, dict(cats))
    elif dataset == "crowdhuman":
        lines = {}
        for sp in splits:
            for line in (root / f"annotation_{sp}.odgt").read_text().splitlines():
                if line.strip():
                    lines[json.loads(line)["ID"]] = line
        for s in samples:
            d = json.loads(lines[s["id"]])
            _, boxes, _ = D.parse_crowdhuman(lines[s["id"]])
            other = Counter(g.get("tag") for g in d.get("gtboxes", [])
                            if g.get("tag") not in ("person", "mask"))
            out[s["id"]] = (boxes, dict(other))
    elif dataset == "auair":
        data = json.loads((root / "annotations.json").read_text())
        by_id = {Path(a["image_name"]).stem: a for a in data["annotations"]}
        for s in samples:
            b = by_id[s["id"]]["bbox"]
            out[s["id"]] = ([[x["left"], x["top"], x["left"] + x["width"],
                              x["top"] + x["height"]] for x in b], {})
    elif dataset == "hituav":
        for s in samples:
            img = Path(s["image"])
            ann = img.parents[2] / "labels" / img.parent.name / f"{img.stem}.txt"
            text = ann.read_text() if ann.exists() else ""
            boxes, _ = D.parse_yolo_txt(text, s["width"], s["height"], 5, drop=(4,))
            dc = sum(1 for line in text.splitlines() if line.split()[:1] == ["4"])
            out[s["id"]] = (boxes, {"DontCare": dc})
    elif dataset == "exdark":
        anns = {p.name.lower(): p for p in (root / "ExDark_Annno").rglob("*.txt")}
        for s in samples:
            boxes, _ = D.parse_bbgt(anns[Path(s["image"]).name.lower() + ".txt"].read_text())
            out[s["id"]] = (boxes, {})
    else:
        return None
    return out


def qualite(samples, ctx):
    raw = raw_annotations(ctx.dataset, ctx.root, samples, ctx.splits)
    sh, sw = ctx.hw
    removed = clipped = deg_orig = deg_in = dup = 0
    regions = Counter()
    scored = []
    for s in samples:
        n = len(s["labels"])
        if raw is not None:
            rb, reg = raw[s["id"]]
            rb = np.asarray(rb, dtype=np.float64).reshape(-1, 4)
            regions.update(reg)
        else:
            rb = _corners(s)
        W, H = s["width"], s["height"]
        out_img = (rb[:, 0] < 0) | (rb[:, 1] < 0) | (rb[:, 2] > W) | (rb[:, 3] > H)
        r = len(rb) - n if raw is not None else 0
        c = int(out_img.sum()) - r if raw is not None else int(out_img.sum())
        c = max(c, 0)
        wh = s["boxes"][:, 2:] * (W, H)
        win = input_boxes(s, ctx.size, ctx.resize)[:, 2:] * (sw, sh)
        do = int((wh.min(axis=1) < DEGENERATE).sum()) if n else 0
        di = int((win.min(axis=1) < DEGENERATE).sum()) if n else 0
        dp = 0
        if n > 1:
            corners = _corners(s)
            for lab in np.unique(s["labels"]):
                m = np.flatnonzero(s["labels"] == lab)
                if len(m) > 1:
                    iou = np.triu(iou_xyxy(corners[m], corners[m]), 1)
                    dp += int((iou > DUPLICATE_IOU).sum())
        removed += r
        clipped += c
        deg_orig += do
        deg_in += di
        dup += dp
        score = r + c + do + 2 * dp
        if score:
            scored.append((score, s["id"], str(s["image"]),
                           {"retirées": r, "rognées": c, "dégénérées": do, "doublons": dp}))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return {"raw_checked": raw is not None, "removed": removed, "clipped": clipped,
            "degenerate_orig": deg_orig, "degenerate_in": deg_in, "duplicates": dup,
            "regions": dict(regions),
            "suspects": [{"id": i, "image": p, "score": sc, **why}
                         for sc, i, p, why in scored[:SUSPECTS]]}


# ---------------------------------------------------------------------------- T16.11 écart

def _tv(p, q):
    """Distance en variation totale entre deux histogrammes (normalisés ici), dans [0, 1]."""
    p, q = np.asarray(p, np.float64), np.asarray(q, np.float64)
    n = max(len(p), len(q))
    p, q = np.pad(p, (0, n - len(p))), np.pad(q, (0, n - len(q)))
    if p.sum() == 0 or q.sum() == 0:
        return None
    return float(0.5 * np.abs(p / p.sum() - q / q.sum()).sum())


def ecart(train, test):
    """Écart entre les résultats des groupes `train` et `test` (dicts d'analyses)."""
    out = {}
    if "comptes" in train and "comptes" in test:
        p = np.asarray(train["comptes"]["per_class"], np.float64)
        q = np.asarray(test["comptes"]["per_class"], np.float64)
        out["classes"] = _tv(p, q)
        out["share_train"] = (p / max(p.sum(), 1)).tolist()
        out["share_test"] = (q / max(q.sum(), 1)).tolist()
    if "geometrie" in train and "geometrie" in test:
        mode = train["geometrie"]["resize"]
        for k in ("area", "w", "h"):
            out[f"{k}_in"] = _tv(train["geometrie"][mode][k], test["geometrie"][mode][k])
        out["aspect"] = _tv(train["geometrie"]["orig"]["aspect"],
                            test["geometrie"]["orig"]["aspect"])
    if "densite" in train and "densite" in test:
        out["per_image"] = _tv(train["densite"]["per_image"], test["densite"]["per_image"])
    return out


# ---------------------------------------------------------------------------- extras par jeu

def exdark_light(root, lums, ids):
    """Luminance moyenne par type d'éclairage (imageclasslist.txt, colonne 3)."""
    light = {}
    for line in (Path(root) / "imageclasslist.txt").read_text().splitlines():
        f = line.split()
        if len(f) >= 5 and f[2].isdigit():
            light[Path(f[0]).stem] = int(f[2])
    per = {}
    for i, lum in zip(ids, lums):
        t = light.get(i)
        if t is not None:
            per.setdefault(EXDARK_LIGHT.get(t, str(t)), []).append(lum)
    return {k: {"images": len(v), "lum_mean": float(np.mean(v))} for k, v in
            sorted(per.items(), key=lambda kv: np.mean(kv[1]))}


# ---------------------------------------------------------------------------- orchestration

def default_splits(dataset):
    """Éléments de `Dataset.train` puis de `Dataset.test` (sans doublon)."""
    ds = D.DATASETS[dataset]
    return list(dict.fromkeys(ds.train.split(",") + ds.test.split(",")))


def groups_of(dataset, splits):
    ds = D.DATASETS[dataset]
    out = {}
    for g in ("train", "test"):
        members = [s for s in getattr(ds, g).split(",") if s in splits]
        if members:
            out[g] = members
    return out


def default_net(dataset):
    """Cfg du jeu : celle de l'affinage (tools/m11.sh <jeu>-prep) si elle existe, sinon
    tiny-yolov3-voc pour VOC et tiny-yolov3-coco ailleurs (ancres COCO)."""
    if dataset == "voc":
        return "tiny-yolov3-voc"
    try:
        from tools.notebooks.matrice import TRAINABLE, finetuned
    except ImportError:
        return FALLBACK_NET
    if dataset in TRAINABLE:
        cfg = finetuned(dataset)[1]
        if (ROOT / cfg).exists():
            return cfg
    return FALLBACK_NET


def analyse(samples, ctx, only):
    out = {}
    for name, fn in (("comptes", comptes), ("geometrie", geometrie), ("densite", densite),
                     ("images", images), ("qualite", qualite)):
        if name in only:
            out[name] = fn(samples, ctx)
    return out


def compute(dataset, splits=None, data_root=None, size=SIZE, resize="letterbox", net=None,
            sample=0, seed=0, only=ANALYSES, log=print):
    """Toutes les analyses du jeu → dict de stats.json (avant `_py`) et échantillons."""
    ds = D.DATASETS[dataset]
    splits = list(splits or default_splits(dataset))
    root = Path(data_root) if data_root else D.DATA_DIR / ds.root
    net = net or default_net(dataset)
    parts = {}
    for sp in splits:
        log(f"chargement {dataset} {sp}…")
        parts[sp] = ds.load(root, sp)
    groups = groups_of(dataset, splits)
    stats = {"dataset": dataset, "classes": list(ds.classes), "fiche": FICHES.get(dataset),
             "params": {"splits": splits, "size": list(as_hw(size)), "resize": resize,
                        "net": str(net), "seed": seed, "sample": sample,
                        "revision": revision()},
             "groups": groups, "bins": {k: v.tolist() for k, v in BINS.items()},
             "splits": {}, "groupes": {}}

    def ctx(sps):
        return Ctx(dataset, size, resize, net, seed, sample, root, tuple(sps))

    shared = ctx(splits)
    for sp, samples in parts.items():
        log(f"analyses {sp} ({len(samples)} images)…")
        c = ctx([sp])
        c._spec = shared.spec()
        stats["splits"][sp] = analyse(samples, c, only)
    for g, members in groups.items():
        if len(members) == 1:
            stats["groupes"][g] = members[0]
        else:
            log(f"analyses groupe {g}…")
            c = ctx(members)
            c._spec = shared.spec()
            stats["groupes"][g] = analyse([s for m in members for s in parts[m]], c, only)
    every = [s for sp in splits for s in parts[sp]]
    if len(splits) > 1:
        log("analyses ensemble…")
        shared_all = ctx(splits)
        shared_all._spec = shared.spec()
        stats["ensemble"] = analyse(every, shared_all, only)
    if "ancres" in only:
        train = groups.get("train") or splits
        log("ancres…")
        c = ctx(train)
        c._spec = shared.spec()
        stats["ancres"] = ancres([s for m in train for s in parts[m]], c)
    if "ecart" in only and {"train", "test"} <= set(groups):
        stats["ecart"] = ecart(group(stats, "train"), group(stats, "test"))
    if sample and "images" in only:
        if dataset != "voc" and (D.DATA_DIR / "VOCdevkit").exists():
            log("référence VOC2007 test (pixels)…")
            stats["voc_ref"] = pixel_stats(D.load("voc", split="2007:test"), sample, seed)
        if dataset == "exdark":
            px = whole(stats).get("images", {}).get("pixels")
            if px:
                stats["exdark_light"] = exdark_light(root, px["lum"], px["ids"])
    return stats, parts


def group(stats, g):
    """Analyses du groupe `g` (un nom de split est résolu)."""
    v = stats["groupes"].get(g)
    return stats["splits"][v] if isinstance(v, str) else (v or {})


def whole(stats):
    """Analyses de l'ensemble (le split unique s'il n'y en a qu'un)."""
    return stats.get("ensemble") or next(iter(stats["splits"].values()), {})


def revision():
    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------- galerie (T16.4)

def _save_drawn(s, classes, dest, max_side=640):
    from PIL import Image

    from tools.detect import draw

    with Image.open(s["image"]) as img:
        shown = draw(img, s["boxes"], None, s["labels"], classes)
    shown.thumbnail((max_side, max_side))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shown.save(dest, quality=85)
    return dest


def contact_sheet(paths, dest, cols=4, cell=320):
    """Planche-contact des vignettes `paths` (une image par notebook, pas une par vignette)."""
    from PIL import Image

    paths = [Path(p) for p in paths]
    if not paths:
        return None
    rows = (len(paths) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * cell, rows * cell), "white")
    for k, p in enumerate(paths):
        with Image.open(p) as img:
            img = img.convert("RGB")
            img.thumbnail((cell - 4, cell - 4))
            x = (k % cols) * cell + (cell - img.width) // 2
            y = (k // cols) * cell + (cell - img.height) // 2
            sheet.paste(img, (x, y))
    dest.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(dest, quality=85)
    return dest


def gallery(stats, parts, out, n, seed):
    """`n` images par split tirées avec `seed`, une image par classe, et les images
    suspectes de T16.10 ; boîtes dessinées par `tools/detect.py:draw`. Rend {groupe: [chemins
    relatifs à out]}."""
    classes = D.DATASETS[stats["dataset"]].classes
    out = Path(out)
    index = {}
    for sp, samples in parts.items():
        tag = sp.replace(":", "-").replace(",", "+")
        index[sp] = [_save_drawn(samples[i], classes, out / "galerie" / tag
                                 / f"{samples[i]['id']}.jpg").relative_to(out).as_posix()
                     for i in pick(len(samples), n, seed)]
    every = [s for samples in parts.values() for s in samples]
    order = np.random.default_rng(seed).permutation(len(every))
    per_class = {}
    for i in order:
        for lab in np.unique(every[i]["labels"]):
            per_class.setdefault(int(lab), every[i])
        if len(per_class) == len(classes):
            break
    index["classes"] = [_save_drawn(per_class[c], classes, out / "galerie" / "classes"
                                    / f"{c:02d}-{classes[c]}.jpg").relative_to(out).as_posix()
                        for c in sorted(per_class)]
    by_id = {s["id"]: s for s in every}
    sus = whole(stats).get("qualite", {}).get("suspects", [])
    index["suspectes"] = [_save_drawn(by_id[d["id"]], classes, out / "galerie" / "suspectes"
                                      / f"{d['id']}.jpg").relative_to(out).as_posix()
                          for d in sus if d["id"] in by_id]
    sheets = {}
    for g, paths in index.items():
        tag = g.replace(":", "-").replace(",", "+")
        dest = contact_sheet([out / p for p in paths], out / "galerie" / f"planche_{tag}.jpg")
        if dest:
            sheets[g] = dest.relative_to(out).as_posix()
    index["planches"] = sheets
    return index


# ---------------------------------------------------------------------------- markdown

def _f(x, d=1):
    """Nombre à la française (virgule décimale, espace des milliers)."""
    if x is None:
        return "—"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}".replace(",", " ")
    return f"{x:,.{d}f}".replace(",", " ").replace(".", ",")


def _pct(x, d=1):
    return "—" if x is None else f"{_f(100 * x, d)} %"


def table(head, rows):
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def _parts(stats):
    """[(nom, analyses)] : splits, groupes de plusieurs splits, ensemble."""
    out = list(stats["splits"].items())
    out += [(f"groupe {g}", v) for g, v in stats["groupes"].items() if isinstance(v, dict)]
    if stats.get("ensemble"):
        out.append(("ensemble", stats["ensemble"]))
    return out


def md_fiche(stats):
    f = stats.get("fiche") or {}
    rows = [("jeu", f.get("name", stats["dataset"])), ("source", f.get("source", "—")),
            ("version", f.get("version", "—")), ("licence", f.get("license", "—")),
            ("capteur", f.get("sensor", "—")), ("résolution typique", f.get("resolution", "—")),
            ("classes", f"{len(stats['classes'])} : {', '.join(stats['classes'])}")]
    ds = D.DATASETS[stats["dataset"]]
    rows.append(("splits du dépôt", f"test `{ds.test}`, calibration `{ds.calib}`, "
                                    f"affinage `{ds.train}`"))
    for sp, (i, o) in (f.get("official") or {}).items():
        rows.append((f"officiel {sp}", f"{_f(i) if i else '—'} images, "
                                       f"{_f(o) if o else '—'} objets"))
    if f.get("objects"):
        rows.append(("objets officiels", f["objects"]))
    for n in f.get("notes", []):
        rows.append(("chargeur", n))
    return table(["", ""], rows)


def check_official(stats):
    """[(split, images, objets comptés, officiels, ok)] : comptes mesurés face à la fiche.
    Les objets officiels se comparent aux objets non-difficult (VOC, CrowdHuman) ou à
    tous les objets (les autres jeux, qui n'ont pas de difficult)."""
    f = stats.get("fiche") or {}
    out = []
    for sp, (i, o) in (f.get("official") or {}).items():
        a = stats["splits"].get(sp, {}).get("comptes")
        if not a:
            continue
        objs = a["objects_easy"] if stats["dataset"] in ("voc", "crowdhuman") else a["objects"]
        ok = (i is None or i == a["images"]) and (o is None or o == objs)
        out.append((sp, a["images"], objs, (i, o), ok))
    return out


def md_synthese(stats):
    rows = []
    for name, a in _parts(stats):
        c, g, d = a.get("comptes"), a.get("geometrie"), a.get("densite")
        if not c:
            continue
        mode = g["resize"] if g else None
        rows.append([name, _f(c["images"]), _f(c["objects"]), _f(c["difficult"]),
                     _f(c["empty_images"]),
                     _f(d["median"], 0) + " / " + _f(d["max"]) if d else "—",
                     _pct(g[mode]["coco"][0] / max(g["n"], 1)) if g else "—",
                     " · ".join(f"{k} {_pct(v)}" for k, v in g[mode]["fit_cell"].items())
                     if g else "—",
                     _pct(d["collisions"]["lost_share"], 2) if d else "—",
                     f"{d['over']['256']} / {d['over']['1024']}" if d else "—"])
    p = stats["params"]
    head = ["partie", "images", "objets", "difficult", "images vides",
            "objets/image (méd. / max)", f"petits < 32² à {size_label(p['size'])}",
            "plus petits qu'une cellule", "cibles perdues", "images > 256 / > 1 024"]
    return table(head, rows)


def md_classes(stats):
    classes = stats["classes"]
    parts = [(n, a) for n, a in stats["splits"].items() if "comptes" in a]
    if not parts:
        return ""
    w = whole(stats)
    g = w.get("geometrie")
    head = ["classe"] + [f"{n} : objets (images)" for n, _ in parts]
    if g:
        head += [f"côté médian (px, {g['resize']})", "petits"]
    rows = []
    for k, c in enumerate(classes):
        r = [c] + [f"{_f(a['comptes']['per_class'][k])} ({_f(a['comptes']['images_with'][k])})"
                   for _, a in parts]
        if g:
            r += [_f(g["per_class"]["median_side_in"][k], 0), _pct(g["per_class"]["small_in"][k], 0)]
        rows.append(r)
    imb = w.get("comptes", {}).get("imbalance")
    return table(head, rows) + (f"\n\nRapport classe la plus / la moins fréquente : "
                                f"{_f(imb, 0)}." if imb else "")


AP_SOURCES = ("results/map_stades.md", "results/map_float.md")


def voc_ap(paths=AP_SOURCES):
    """{classe: AP} de la première colonne numérique des tables par classe de M3/T13.13
    (Tiny-YOLOv2 VOC, flottant, VOC2007 test) ; {} si aucune table n'est là."""
    for rel in paths:
        p = ROOT / rel
        if not p.exists():
            continue
        out = {}
        for line in p.read_text().splitlines():
            f = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(f) >= 2 and f[0] in D.VOC_CLASSES and f[0] not in out:
                try:
                    out[f[0]] = float(f[1].replace(",", "."))
                except ValueError:
                    pass
        if len(out) == len(D.VOC_CLASSES):
            return {"source": rel, "ap": out}
    return {}


def md_par_classe(stats, ap=None):
    """Par classe, sur le groupe test : objets, côté médian et part de petits à l'entrée,
    cibles perdues par collision ; AP en regard si `ap` ({classe: AP}) est donné."""
    t = group(stats, "test") or whole(stats)
    c, g, d = t.get("comptes"), t.get("geometrie"), t.get("densite")
    if not (c and g and d):
        return ""
    col = d["collisions"]["per_class"]
    rows = []
    for k, name in enumerate(stats["classes"]):
        n = col["objects"][k]
        r = [name, _f(c["per_class_easy"][k]), _pct(c["per_class_easy"][k]
                                                    / max(c["objects_easy"], 1)),
             _f(g["per_class"]["median_side_in"][k], 0), _pct(g["per_class"]["small_in"][k], 0),
             _pct(col["lost"][k] / n, 2) if n else "—"]
        if ap is not None:
            r.append(_f(ap.get(name), 1))
        rows.append(r)
    head = ["classe", "objets (test, non-difficult)", "part", "côté médian (px)", "petits",
            "cibles perdues"] + (["AP"] if ap is not None else [])
    return table(head, rows)


def md_geometrie(stats):
    w = whole(stats).get("geometrie")
    if not w:
        return ""
    lab = size_label(w["size"])
    rows = []
    for mode, name in (("orig", "image d'origine"), ("letterbox", f"letterbox {lab}"),
                       ("stretch", f"stretch {lab}")):
        c = w[mode]["coco"]
        n = max(sum(c), 1)
        rows.append([name, _pct(c[0] / n), _pct(c[1] / n), _pct(c[2] / n),
                     _f(w[mode]["median_side"], 0),
                     " · ".join(f"{k} {_pct(v)}" for k, v in w[mode].get("fit_cell", {}).items())
                     or "—",
                     _pct(w[mode]["h_under"]["8"]) if "h_under" in w[mode] else "—"])
    return table(["repère", "petits", "moyens", "grands", "côté médian (px)",
                  "plus petits qu'une cellule", "hauteur < 8 px"], rows) + \
        f"\n\n{_f(w['n'])} objets non-difficult ; pas des têtes : " + \
        ", ".join(f"{k} → {_f(v, 0)} px" for k, v in w["strides"].items()) + "."


def md_densite(stats):
    rows = []
    for name, a in _parts(stats):
        d = a.get("densite")
        if not d:
            continue
        col = d["collisions"]
        heads_ = " · ".join(f"{k} : {_f(v['lost'])} / {_f(v['objects'])}"
                            for k, v in col["per_head"].items())
        rows.append([name, _f(d["mean"], 1), _f(d["median"], 0), _f(d["max"]),
                     f"{d['over']['256']} / {d['over']['1024']}",
                     f"{_f(col['lost'])} / {_f(col['objects'])} ({_pct(col['lost_share'], 2)})",
                     heads_, _f(col["images_with_loss"])])
    if not rows:
        return ""
    net = whole(stats)["densite"]["collisions"]["net"]
    return table(["partie", "objets/image moy.", "méd.", "max", "> 256 / > 1 024",
                  "cibles perdues", "par tête (perdues / objets)", "images touchées"], rows) + \
        f"\n\nAncres de `{net}` ; objets non-difficult, comme à l'entraînement."


def md_images(stats):
    rows = []
    for name, a in _parts(stats):
        im = a.get("images")
        if not im:
            continue
        px = im.get("pixels")
        res = ", ".join(f"{w}×{h} ({_f(n)})" for w, h, n in im["resolutions"][:3])
        rows.append([name, res, _f(im["distinct"]),
                     f"{px['images']} images" if px else "—",
                     str(px["channels"]) if px else "—",
                     " / ".join(_f(m, 1) for m in px["mean"]) if px else "—",
                     " / ".join(_f(s, 1) for s in px["std"]) if px else "—",
                     _f(px["lum_mean"], 1) if px else "—",
                     f"{px['levels']['8bit']} → {px['levels']['int8']}" if px else "—"])
    if not rows:
        return ""
    text = table(["partie", "résolutions (3 premières)", "distinctes", "pixels sur",
                  "canaux", "moyenne par canal", "écart type par canal", "luminance moy.",
                  "niveaux 8 bits → INT8"], rows)
    if stats.get("voc_ref"):
        v = stats["voc_ref"]
        text += (f"\n\nRéférence VOC2007 test ({v['images']} images, graine {v['seed']}) : "
                 f"luminance moyenne {_f(v['lum_mean'], 1)}, moyenne par canal "
                 f"{' / '.join(_f(m, 1) for m in v['mean'])}.")
    return text


def md_ancres(stats):
    a = stats.get("ancres")
    if not a:
        return ""
    rows = [[n, "`" + "  ".join(f"{w:.0f},{h:.0f}" for w, h in r["anchors"]) + "`",
             _f(r["iou"], 4)] for n, r in a["refs"].items()]
    for k in ("5", "6"):
        rows.append([f"k-means k = {k} (graine {a['seed']})",
                     "`" + "  ".join(f"{w:.0f},{h:.0f}" for w, h in a["kmeans"][k]) + "`",
                     _f(a["kmeans_iou"][k], 4)])
    return table(["ancres", f"w,h (px à {size_label(a['size'])})", "IoU moyenne"], rows) + \
        f"\n\n{_f(a['boxes'])} boîtes non-difficult du groupe train, letterbox " \
        f"{size_label(a['size'])}."


def md_qualite(stats):
    rows = []
    for name, a in _parts(stats):
        q = a.get("qualite")
        if not q:
            continue
        reg = ", ".join(f"{k} {_f(v)}" for k, v in q["regions"].items()) or "—"
        rows.append([name, _f(q["removed"]) if q["raw_checked"] else "—", _f(q["clipped"]),
                     f"{_f(q['degenerate_orig'])} / {_f(q['degenerate_in'])}",
                     _f(q["duplicates"]), reg])
    if not rows:
        return ""
    return table(["partie", "retirées (make_sample)", "rognées", "côté < 2 px (orig. / entrée)",
                  "doublons (IoU > 0,95)", "régions retirées par le chargeur"], rows)


def md_ecart(stats):
    e = stats.get("ecart")
    rows = [[k, _f(v, 3)] for k, v in (e or {}).items() if not k.startswith("share")]
    text = table(["grandeur (train ↔ test)", "variation totale"], rows) if rows else ""
    cov = []
    for name, a in stats["splits"].items():
        for fam, c in a.get("comptes", {}).get("coverage", {}).items():
            cov.append([name, fam.upper(), _f(c["objects"]), _pct(c["share"]),
                        _f(c["evaluated"])])
    if cov:
        text += ("\n\n" if text else "") + table(
            ["split", "classes du modèle", "objets couverts", "part", "évalués (non-difficult)"],
            cov)
    return text


def markdown(stats):
    p = stats["params"]
    parts = [f"# Statistiques de {stats['dataset']}", "",
             f"Révision `{p['revision']}` · splits {', '.join(f'`{s}`' for s in p['splits'])} · "
             f"entrée {size_label(p['size'])} ({p['resize']}) · cfg `{p['net']}` · pixels sur "
             f"{p['sample']} images par partie (graine {p['seed']}).", "",
             "## Fiche", "", md_fiche(stats), "", "## Synthèse", "", md_synthese(stats)]
    chk = check_official(stats)
    if chk:
        parts += ["", table(["split", "images", "objets", "officiel (images / objets)", ""],
                            [[sp, _f(i), _f(o), f"{_f(a) if a else '—'} / {_f(b) if b else '—'}",
                              "ok" if ok else "écart"] for sp, i, o, (a, b), ok in chk])]
    for title, fn in (("Classes", md_classes), ("Géométrie des boîtes", md_geometrie),
                      ("Densité et collisions", md_densite), ("Images", md_images),
                      ("Ancres", md_ancres), ("Qualité des annotations", md_qualite),
                      ("Écart entre splits et couverture", md_ecart)):
        body = fn(stats)
        if body:
            parts += ["", f"## {title}", "", body]
    if stats.get("exdark_light"):
        parts += ["", "## Luminance par type d'éclairage", "", table(
            ["éclairage", "images", "luminance moy."],
            [[k, v["images"], _f(v["lum_mean"], 1)] for k, v in stats["exdark_light"].items()])]
    return "\n".join(parts) + "\n"


# ---------------------------------------------------------------------------- synthèse

STATS_GLOB = "build/notebooks/*/stats/stats.json"


def comparatif(all_stats):
    rows = []
    for st in all_stats:
        w = whole(st)
        c, g, d, im = (w.get(k) for k in ("comptes", "geometrie", "densite", "images"))
        px = (im or {}).get("pixels")
        a = st.get("ancres", {}).get("refs", {})
        cfg_iou = next((v["iou"] for k, v in a.items() if k.startswith("cfg")), None)
        mode = g["resize"] if g else None
        rows.append([st["dataset"], len(st["classes"]), _f(c["images"]) if c else "—",
                     _f(c["objects"]) if c else "—",
                     _f(d["median"], 0) if d else "—",
                     _pct(g[mode]["coco"][0] / max(g["n"], 1)) if g else "—",
                     _pct(list(g[mode]["fit_cell"].values())[-1]) if g else "—",
                     _pct(d["collisions"]["lost_share"], 2) if d else "—",
                     _f(d["over"]["256"]) if d else "—",
                     _f(px["lum_mean"], 0) if px else "—", _f(cfg_iou, 3)])
    return table(["jeu", "classes", "images", "objets", "objets/image (méd.)",
                  "petits à l'entrée", "plus petits qu'une cellule fine", "cibles perdues",
                  "images > 256", "luminance moy.", "IoU ancres de la cfg"], rows)


def _replace_block(text, key, body):
    start, end = f"<!-- data_stats:{key} -->", f"<!-- /data_stats:{key} -->"
    block = f"{start}\n{body.strip()}\n{end}"
    if start in text and end in text:
        a = text.index(start)
        b = text.index(end) + len(end)
        return text[:a] + block + text[b:]
    return text.rstrip("\n") + f"\n\n{block}\n"


def report(doc, paths=None):
    """Remplit les blocs générés de `doc` depuis les stats.json trouvés : un par jeu (fiche
    courte, synthèse, comptes face à la fiche) et la table comparative. Le texte hors des
    marqueurs (réponses aux questions) est gardé."""
    paths = sorted(paths or ROOT.glob(STATS_GLOB))
    all_stats = [json.loads(Path(p).read_text()) for p in paths]
    order = list(D.DATASETS)
    all_stats.sort(key=lambda s: order.index(s["dataset"]) if s["dataset"] in order else 99)
    doc = Path(doc)
    text = doc.read_text() if doc.exists() else "# Statistiques des jeux (M16)\n"
    text = _replace_block(text, "comparatif", comparatif(all_stats))
    for st in all_stats:
        p = st["params"]
        body = [f"Révision `{p['revision']}`, entrée {size_label(p['size'])} ({p['resize']}), "
                f"cfg `{p['net']}`, pixels sur {p['sample']} images (graine {p['seed']}). "
                f"Commande : `python tools/data_stats.py --dataset {st['dataset']} "
                f"--split {','.join(p['splits'])} --sample {p['sample']}`.", "",
                md_synthese(st)]
        chk = check_official(st)
        if chk:
            body += ["", table(["split", "images", "objets", "officiel", ""],
                               [[sp, _f(i), _f(o), f"{_f(a) if a else '—'} / "
                                 f"{_f(b) if b else '—'}", "ok" if ok else "écart"]
                                for sp, i, o, (a, b), ok in chk])]
        text = _replace_block(text, st["dataset"], "\n".join(body))
    doc.write_text(text)
    return doc


# ---------------------------------------------------------------------------- CLI

def write(stats, out, md=True):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "stats.json").write_text(json.dumps(_py(stats), ensure_ascii=False) + "\n")
    if md:
        (out / "stats.md").write_text(markdown(_py(stats)))
    return out / "stats.json"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    D.add_args(ap, "splits séparés par des virgules (VOC : 2007:test,…) ; défaut : splits "
                   "train puis test du jeu")
    ap.add_argument("--size", type=parse_size, default=SIZE, help="S ou LxH (ex. 640x192)")
    ap.add_argument("--resize", choices=("letterbox", "stretch"), default="letterbox")
    ap.add_argument("--net", default=None,
                    help="cfg dont les ancres servent aux collisions ; défaut : cfg du jeu")
    ap.add_argument("--sample", type=int, default=0, help="images lues pour les pixels (0 : aucune)")
    ap.add_argument("--gallery", type=int, default=0, help="images de la galerie par split")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=",".join(ANALYSES), help="analyses, séparées par des virgules")
    ap.add_argument("--out", type=Path, default=None,
                    help="défaut build/notebooks/<jeu>/stats")
    ap.add_argument("--markdown", action="store_true", help="affiche stats.md")
    ap.add_argument("--figures", action="store_true", help="figures dans <out>/figures/")
    ap.add_argument("--report", type=Path, default=None,
                    help="remplit ce document depuis les stats.json de build/notebooks/")
    args = ap.parse_args(argv)
    if args.report:
        print(f"→ {report(args.report)}")
        return 0
    only = tuple(a for a in args.only.split(",") if a)
    bad = set(only) - set(ANALYSES)
    if bad:
        ap.error(f"analyses inconnues : {', '.join(sorted(bad))}")
    splits = args.split.split(",") if args.split else None
    out = args.out or ROOT / "build" / "notebooks" / args.dataset / "stats"
    stats, parts = compute(args.dataset, splits, args.data_root, args.size, args.resize,
                           args.net, args.sample, args.seed, only)
    if args.gallery:
        stats["galerie"] = gallery(stats, parts, out, args.gallery, args.seed)
    path = write(stats, out)
    print(f"→ {path}")
    print(f"→ {out / 'stats.md'}")
    if args.figures:
        from tools.figures import donnees

        for p in donnees.plot_all(json.loads(path.read_text()), Path(out) / "figures"):
            print(f"→ {p}")
    if args.markdown:
        print((Path(out) / "stats.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
