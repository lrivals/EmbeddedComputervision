"""Registre des notebooks (T14.2) : couples modèle × jeu × rôle, construits depuis le code.

- Inférence, poids pré-entraînés (`PRETRAINED`) : chaque réseau sur son jeu d'origine et
  sur tout jeu qui a une correspondance vers ses classes (`MAPPINGS`, hors domaine, T11.2).
  COCO n'évalue pas les poids VOC : il a son réseau à 80 classes (T11.1).
- Inférence, poids affinés : `tiny-yolov3-<jeu>` pour chaque jeu de `TRAINABLE`, avec les
  poids `final.weights` du notebook d'entraînement correspondant.
- Statistiques (M16) : un notebook `<jeu>_stats` par jeu de `DATASETS`, sans modèle, en tête
  des notebooks du jeu ; la cfg du jeu ne sert qu'aux ancres et aux grilles (collisions).
- Entraînement : `TRAINABLE`, seule liste écrite à la main (COCO train2017 est trop grand en
  NumPy ; ExDark et CrowdHuman ajoutés pour T15.14) : un notebook `train`
  (affinage unique de tools/m11.sh) et un notebook `sweep` (balayage lot × sous-ensemble,
  T14.10) par jeu. L'inférence affinée lit l'un ou l'autre de leurs runs (`runs.py`).

Ajouter un jeu à `DATASETS` (avec ses `MAPPINGS`) ajoute ses notebooks d'inférence hors
domaine sans toucher au générateur.
"""

from tools.notebooks import Notebook
from tools.notebooks.commandes import cfg_path
from yolo.data.datasets import DATASETS, FAMILIES, MAPPINGS
from yolo.models.tiny_yolo import PRETRAINED, load_cfg

TRAINABLE = ("voc", "kitti", "visdrone", "flir", "exdark", "crowdhuman", "auair",
             "dronevehicle", "hituav", "uavdt")

# Entrée des cfg d'affinage (options SIZE et CH de tools/m11.sh) : FLIR et HIT-UAV thermiques
# à un canal.
TRAIN_INPUT = {"flir": {"channels": 1}, "hituav": {"channels": 1}}

# Images du split d'évaluation (`Dataset.test`), comptées par tools/data_stats.py (M16,
# docs/tasks/stats-jeux.md ; KITTI : 20 % des 7 481 images de training, `kitti_ids` ; COCO :
# chiffre officiel de val2017).
SPLIT_IMAGES = {"voc": 4952, "coco": 5000, "kitti": 1496, "visdrone": 548,
                "crowdhuman": 4370, "exdark": 2563, "flir": 1144, "auair": 4943,
                "dronevehicle": 2608, "hituav": 571, "uavdt": 16592}

# Fichier témoin de chaque jeu (celui de `tools/get_datasets.sh check`, VOC : get_voc.sh).
DATA_MARKERS = {"voc": "data/VOCdevkit/VOC2007/ImageSets/Main/test.txt",
                "coco": "data/coco/annotations/instances_val2017.json",
                "kitti": "data/kitti/training/image_2/000000.png",
                "visdrone": "data/visdrone/VisDrone2019-DET-val/annotations",
                "crowdhuman": "data/crowdhuman/annotation_val.odgt",
                "exdark": "data/exdark/imageclasslist.txt",
                "flir": "data/flir/images_thermal_val/coco.json",
                "auair": "data/auair/annotations.json",
                "dronevehicle": "data/dronevehicle/test/labels",
                "hituav": "data/hituav/labels/test",
                "uavdt": "data/uavdt/annotations_test.json"}

# Jeux à inscription : ni get_voc.sh ni get_datasets.sh ne les téléchargent.
REGISTRATION = ("visdrone", "crowdhuman", "exdark", "flir", "auair", "dronevehicle", "hituav",
                "uavdt")

# Tâches sources, pour l'en-tête des notebooks.
TASKS = {("voc", "infer"): ("T3.4", "T4.5", "T11.0"), ("coco", "infer"): ("T11.1",),
         ("hors-domaine", "infer"): ("T11.2",), ("voc", "train"): ("T2.9", "T9.2", "T9.3"),
         ("kitti", "train"): ("T11.4",), ("visdrone", "train"): ("T11.5",),
         ("flir", "train"): ("T11.7",), ("exdark", "train"): ("T11.3", "T15.14"),
         ("crowdhuman", "train"): ("T11.6", "T15.14"), ("exdark", "infer"): ("T11.3",),
         ("crowdhuman", "infer"): ("T11.6",), ("sweep", "sweep"): ("T14.10",),
         ("auair", "train"): ("T18.1", "T18.10"), ("dronevehicle", "train"): ("T18.2", "T18.10"),
         ("hituav", "train"): ("T18.3", "T18.10"), ("uavdt", "train"): ("T18.4", "T18.10"),
         ("auair", "infer"): ("T18.10",), ("dronevehicle", "infer"): ("T18.10",),
         ("hituav", "infer"): ("T18.10",), ("uavdt", "infer"): ("T18.10",)}

# Tâche M11 de chaque jeu et tâche d'exécution M16 (en-tête des notebooks `stats`).
STATS_TASKS = {"voc": ("T13.7", "T16.12"), "coco": ("T11.1", "T16.13"),
               "kitti": ("T11.4", "T16.14"), "visdrone": ("T11.5", "T16.15"),
               "flir": ("T11.7", "T16.16"), "exdark": ("T11.3", "T16.17"),
               "crowdhuman": ("T11.6", "T16.18"), "auair": ("T18.1", "T18.8"),
               "dronevehicle": ("T18.2", "T18.8"), "hituav": ("T18.3", "T18.8"),
               "uavdt": ("T18.4", "T18.8")}

# Jeux dont le chargeur lit l'en-tête de chaque image (taille) : palier M pour les
# annotations seules (docs/tasks/M16-presentation-jeux.md#règles).
READS_HEADERS = ("kitti", "visdrone", "crowdhuman", "exdark", "dronevehicle", "hituav")


def family_of(net):
    """Famille de classes d'un réseau pré-entraîné : voc (20) ou coco (80)."""
    n = load_cfg(net)["classes"]
    return next(f for f, names in FAMILIES.items() if len(names) == n)


def _tasks(dataset, role, out_of_domain=False):
    key = ("hors-domaine", role) if out_of_domain else (dataset, role)
    own = TASKS.get((dataset, role), ()) if out_of_domain else ()
    return TASKS.get(key, ()) + own


def finetuned(dataset):
    """Modèle affiné sur `dataset` : (nom, cfg, entrée de la cfg)."""
    if dataset == "voc":
        return "tiny-yolov3-voc", "tiny-yolov3-voc", {}
    inp = TRAIN_INPUT.get(dataset, {})
    return f"tiny-yolov3-{dataset}", cfg_path("tiny-yolov3", dataset, **inp), inp


def stats_net(dataset):
    """Cfg des notebooks `stats` : celle de l'affinage du jeu (ancres et grilles des
    collisions), tiny-yolov3-voc pour VOC, tiny-yolov3-coco pour COCO."""
    if dataset == "voc":
        return "tiny-yolov3-voc"
    if dataset in TRAINABLE:
        return finetuned(dataset)[1]
    return "tiny-yolov3-coco"


def build_registry():
    out = []
    for ds in DATASETS:
        out.append(Notebook(ds, ds, "stats", stats_net(ds), "", ds,
                            STATS_TASKS.get(ds, ()) + ("T16.4",)))
        for net, weights in PRETRAINED.items():
            fam = family_of(net)
            if ds == fam or ((ds, fam) in MAPPINGS and ds != "coco"):
                out.append(Notebook(ds, net, "infer", net, f"weights/{weights}", fam,
                                    _tasks(ds, "infer", ds != fam)))
        if ds in TRAINABLE:
            model, cfg, inp = finetuned(ds)
            out.append(Notebook(ds, model, "infer", cfg,
                                f"build/notebooks/{ds}/{model}/final.weights", ds,
                                _tasks(ds, "train") + ("T14.6", "T14.11"), **inp))
    init = f"weights/{PRETRAINED['tiny-yolov3-coco']}"
    for role in ("train", "sweep"):
        for ds in TRAINABLE:
            model, cfg, inp = finetuned(ds)
            tasks = _tasks(ds, "train") + (TASKS[("sweep", "sweep")] if role == "sweep" else ())
            out.append(Notebook(ds, model, role, cfg, init, ds, tasks, **inp))
    return {f"{nb.dataset}/{nb.name}": nb for nb in out}


NOTEBOOKS = build_registry()


# --------------------------------------------------------------------------- paliers

def palier(images=0, iters=0):
    """Palier de coût M12 (docs/tasks/M12-profils-pc.md#paliers-de-coût) : R jusqu'à
    `--subset 100`, M jusqu'à `--subset 500` ou un entraînement court (< 600 itérations),
    N au-delà (split complet, ≥ 600 itérations). `images` None : split non relevé, N."""
    if images is None or images > 500 or iters >= 600:
        return "N"
    if images > 100 or iters > 0:
        return "M"
    return "R"


def palier_stats(dataset, sample):
    """(annotations, pixels) : R ou M pour les annotations (M si le chargeur lit les en-têtes
    d'image), palier de `sample` images lues pour les pixels."""
    return ("M" if dataset in READS_HEADERS else "R"), palier(sample)


def palier_of(nb, subset=0, iters=0):
    """Palier d'un passage du notebook : `subset` images (0 : split complet)."""
    images = subset or SPLIT_IMAGES.get(nb.dataset)
    if nb.trains:
        return palier(0, iters)
    return palier(images)


def prerequis(nb):
    """{type: chemin ou commande} : ce que le notebook demande avant de tourner."""
    req = {"données": DATA_MARKERS[nb.dataset]}
    if nb.role == "stats":  # sans poids ; la cfg du jeu est remplacée si elle manque
        return req
    # Poids affinés : le run choisi (RUN) parmi ceux du notebook _train ou _sweep.
    req["poids"] = (f"{nb.train_dir}/[runs/<run>/]final.weights" if nb.finetuned
                    else nb.weights)
    if nb.net.endswith(".cfg"):
        req["cfg"] = nb.net
    return req


def how_to_get(kind, nb):
    """Commande qui fournit le prérequis `kind` de `nb`."""
    if kind == "données":
        if nb.dataset == "voc":
            return "tools/get_voc.sh"
        if nb.dataset in REGISTRATION:
            return f"tools/get_datasets.sh {nb.dataset} (inscription : source affichée)"
        return f"tools/get_datasets.sh {nb.dataset}"
    if kind == "poids":
        return (f"notebooks/{nb.dataset}/{nb.model}_train.ipynb ou _sweep.ipynb"
                if nb.finetuned else "tools/get_weights.sh")
    return f"notebooks/{nb.dataset}/{nb.model}_train.ipynb (préparation)"


# Secondes par image d'entraînement à 416 × 416 (M12, paliers de coût) ; GPU : Colab,
# 624 s pour 600 itérations au lot 32 (M15, T15.3).
S_PER_IMAGE = {"cpu": 0.45, "gpu": 0.033}


def estimate(iters, batch, device, loss_csv=None):
    """Durée estimée d'un entraînement (s) : images × itérations × s/image du backend,
    mesurée dans `loss_csv` (colonne `seconds`, par itération) s'il existe ; None sinon."""
    import csv
    from pathlib import Path

    if loss_csv and Path(loss_csv).exists():
        with open(loss_csv) as f:
            rows = list(csv.DictReader(f))[-50:]
        if rows:
            return iters * sum(float(r["seconds"]) for r in rows) / len(rows)
    s = S_PER_IMAGE.get(device)
    return iters * batch * s if s else None
