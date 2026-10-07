"""Registre des notebooks (T14.2) : couples modèle × jeu × rôle, construits depuis le code.

- Inférence, poids pré-entraînés (`PRETRAINED`) : chaque réseau sur son jeu d'origine et
  sur tout jeu qui a une correspondance vers ses classes (`MAPPINGS`, hors domaine, T11.2).
  COCO n'évalue pas les poids VOC : il a son réseau à 80 classes (T11.1).
- Inférence, poids affinés : `tiny-yolov3-<jeu>` pour chaque jeu de `TRAINABLE`, avec les
  poids `final.weights` du notebook d'entraînement correspondant.
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

TRAINABLE = ("voc", "kitti", "visdrone", "flir", "exdark", "crowdhuman")

# Entrée des cfg d'affinage (options SIZE et CH de tools/m11.sh) : FLIR thermique à un canal.
TRAIN_INPUT = {"flir": {"channels": 1}}

# Images du split d'évaluation (`Dataset.test`), d'après docs/tasks/M11-jeux-de-donnees.md
# (KITTI : 20 % des 7 481 images de training, `kitti_ids`). None : non relevé.
SPLIT_IMAGES = {"voc": 4952, "coco": 5000, "kitti": 1496, "visdrone": 548,
                "crowdhuman": 4370, "exdark": None, "flir": None}

# Fichier témoin de chaque jeu (celui de `tools/get_datasets.sh check`, VOC : get_voc.sh).
DATA_MARKERS = {"voc": "data/VOCdevkit/VOC2007/ImageSets/Main/test.txt",
                "coco": "data/coco/annotations/instances_val2017.json",
                "kitti": "data/kitti/training/image_2/000000.png",
                "visdrone": "data/visdrone/VisDrone2019-DET-val/annotations",
                "crowdhuman": "data/crowdhuman/annotation_val.odgt",
                "exdark": "data/exdark/imageclasslist.txt",
                "flir": "data/flir/images_thermal_val/coco.json"}

# Jeux à inscription : ni get_voc.sh ni get_datasets.sh ne les téléchargent.
REGISTRATION = ("visdrone", "crowdhuman", "exdark", "flir")

# Tâches sources, pour l'en-tête des notebooks.
TASKS = {("voc", "infer"): ("T3.4", "T4.5", "T11.0"), ("coco", "infer"): ("T11.1",),
         ("hors-domaine", "infer"): ("T11.2",), ("voc", "train"): ("T2.9", "T9.2", "T9.3"),
         ("kitti", "train"): ("T11.4",), ("visdrone", "train"): ("T11.5",),
         ("flir", "train"): ("T11.7",), ("exdark", "train"): ("T11.3", "T15.14"),
         ("crowdhuman", "train"): ("T11.6", "T15.14"), ("exdark", "infer"): ("T11.3",),
         ("crowdhuman", "infer"): ("T11.6",), ("sweep", "sweep"): ("T14.10",)}


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


def build_registry():
    out = []
    for ds in DATASETS:
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


def palier_of(nb, subset=0, iters=0):
    """Palier d'un passage du notebook : `subset` images (0 : split complet)."""
    images = subset or SPLIT_IMAGES.get(nb.dataset)
    if nb.trains:
        return palier(0, iters)
    return palier(images)


def prerequis(nb):
    """{type: chemin ou commande} : ce que le notebook demande avant de tourner."""
    req = {"données": DATA_MARKERS[nb.dataset]}
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


# Secondes par image d'entraînement à 416 × 416 (M12, paliers de coût) ; GPU : T12.11-e.
S_PER_IMAGE = {"cpu": 0.45}


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
