"""Cellule d'environnement des notebooks (T14.1), une fois le dépôt en place.

En local, rien n'est installé ni téléchargé : `prepare` vérifie les données, les poids et
la cfg, et lève `Prerequis` avec les commandes à lancer. Sur Colab, il lance
`tools/get_weights.sh`, puis `colab.prepare_data` (témoin, archive `<DRIVE_DIR>/data/<jeu>.tar`
ou téléchargement direct, docs/tasks/M14-notebooks.md#données-sur-colab). Drive n'est monté
que si le jeu ne se télécharge pas : un notebook VOC ou KITTI part sans intervention.
"""

import platform
import subprocess
import sys
from pathlib import Path

from tools.notebooks import ROOT, colab
from tools.notebooks.matrice import DATA_MARKERS, REGISTRATION


class Prerequis(RuntimeError):
    """Donnée, poids ou cfg absents : le notebook s'arrête avec la commande à lancer."""


def _sh(*cmd, check=True):
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=check)


def mount_drive(*paths):
    """Monte Google Drive si l'un des chemins y pointe et qu'il ne l'est pas déjà (Colab)."""
    drive = colab.DRIVE_MOUNT
    if any(str(p or "").startswith(drive) for p in paths) and not Path(drive, "MyDrive").exists():
        colab.mount_drive()


def data_ok(dataset, data_root=None):
    if data_root:
        return Path(data_root).exists()
    return (ROOT / DATA_MARKERS[dataset]).exists()


def prepare(dataset, weights=(), cfg=None, data_root=None, in_colab=False, hint="",
            drive_dir=None):
    """Données de `dataset`, poids `weights` (chemins relatifs à la racine) et `cfg`.

    `hint` : commande qui produit les poids ou la cfg s'ils ne sont pas téléchargeables
    (notebook d'entraînement correspondant). `drive_dir` : dossier Drive des archives
    `data/<jeu>.tar` (Colab)."""
    weights = [w for w in ([weights] if isinstance(weights, str) else weights) if w]
    missing = []
    if in_colab:
        mount_drive(data_root)
        if any(w.startswith("weights/") and not (ROOT / w).exists() for w in weights):
            _sh("bash", "tools/get_weights.sh")
        if not data_root and not data_ok(dataset):
            if dataset not in colab.DOWNLOAD:  # ExDark, FLIR : seulement par Drive
                mount_drive(drive_dir)
            colab.prepare_data(dataset, drive_dir)
    if not data_ok(dataset, data_root):
        where = data_root or DATA_MARKERS[dataset]
        cmd = "tools/get_voc.sh" if dataset == "voc" else f"tools/get_datasets.sh {dataset}"
        if dataset in REGISTRATION:
            cmd += (" (inscription ; pour Colab : tools/get_datasets.sh pack "
                    f"{dataset}, archive sur Drive)")
        missing.append(f"données {dataset} absentes ({where}) : {cmd}")
    for w in weights:
        if not (ROOT / w).exists():
            missing.append(f"poids {w} absents : "
                           + ("tools/get_weights.sh" if w.startswith("weights/") else hint))
    if cfg and str(cfg).endswith(".cfg") and not (ROOT / cfg).exists():
        missing.append(f"cfg {cfg} absente : {hint}")
    if missing:
        raise Prerequis("à lancer avant ce notebook :\n  " + "\n  ".join(missing))
    print(f"données {dataset} : {data_root or DATA_MARKERS[dataset]} ; "
          f"poids : {', '.join(weights) or '—'}")


def check_device(device):
    """Contrôle de T12.11 sans repli : `yolo.backend.use` lève avec son message. Le noyau
    revient ensuite au CPU (l'entraînement tourne dans un sous-processus)."""
    from yolo import backend

    backend.use(device)
    if device == "gpu":
        import cupy

        print(f"GPU : {cupy.cuda.runtime.getDeviceCount()} visible(s), CuPy {cupy.__version__}")
        backend.use("cpu")


def trace(device):
    """Versions et révision, pour la traçabilité des résultats."""
    import numpy as np

    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip() or "?"
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    print(f"dépôt {ROOT} @ {rev}{' (modifié)' if dirty else ''} ; Python "
          f"{platform.python_version()} ; NumPy {np.__version__} ; backend {device}")
    return {"rev": rev, "dirty": bool(dirty), "python": sys.version.split()[0],
            "numpy": np.__version__, "device": device}
