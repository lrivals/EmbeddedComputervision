"""Cellule d'environnement des notebooks (T14.1), une fois le dépôt en place.

En local, rien n'est installé : un jeu présent dans data/ est utilisé tel quel ; absent, son
archive est rapatriée de Google Drive par rclone (`tools/get_datasets.sh pull <jeu>`,
docs/tasks/donnees-drive.md) ; sinon `prepare` lève `Prerequis` avec les commandes à lancer,
comme pour les poids et la cfg. Sur Colab, il lance
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


def restore_cfg(cfg):
    """Cfg d'affinage `cfg` absente (runtime Colab neuf : build/m11/cfg/ n'est ni dans le
    dépôt ni sur Drive) : reconstruite par `tools/make_cfg.py` depuis la table d'ancres que
    la préparation des notebooks _train et _sweep laisse dans leur dossier (copié sur Drive
    avec les runs), ou depuis celle de `tools/m11.sh <jeu>-prep` (results/). Mêmes ancres,
    même cfg (BASE et ANCHORS_K par défaut). Rend True si la cfg existe à la fin."""
    from tools.notebooks import commandes as C
    from tools.notebooks.matrice import NOTEBOOKS

    if (ROOT / cfg).exists():
        return True
    nb = next((n for n in NOTEBOOKS.values() if n.trains and n.net == str(cfg)), None)
    if nb is None:
        return False
    name = f"anchors_{nb.dataset}{'_' + str(nb.size) if nb.size else ''}.md"
    for anchors in (Path(nb.train_dir) / name, Path("results") / name):
        if (ROOT / anchors).exists():
            print(f"cfg {cfg} absente : reconstruite depuis {anchors}")
            C.run(C.cmd_make_cfg("tiny-yolov3-voc", nb.dataset,
                                 C.anchors_k(ROOT / anchors, 6), cfg, nb.size, nb.channels))
            return True
    return False


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
            mount_drive(drive_dir)  # archive sur Drive avant le téléchargement direct
            colab.prepare_data(dataset, drive_dir)
    elif not data_root and not data_ok(dataset):
        try:  # local : archive sur Drive par rclone, jamais de téléchargement implicite
            colab.prepare_data(dataset, drive_dir=None, download=False)
        except RuntimeError as e:
            print(e)
    if not data_ok(dataset, data_root):
        where = data_root or DATA_MARKERS[dataset]
        cmd = "tools/get_voc.sh" if dataset == "voc" else f"tools/get_datasets.sh {dataset}"
        if dataset in REGISTRATION:
            cmd += " (inscription)"
        cmd += (f", ou tools/get_datasets.sh pull {dataset} (archive sur Drive, "
                "docs/tasks/donnees-drive.md)")
        missing.append(f"données {dataset} absentes ({where}) : {cmd}")
    for w in weights:
        if not (ROOT / w).exists():
            missing.append(f"poids {w} absents : "
                           + ("tools/get_weights.sh" if w.startswith("weights/") else hint))
    if cfg and str(cfg).endswith(".cfg") and not restore_cfg(cfg):
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
