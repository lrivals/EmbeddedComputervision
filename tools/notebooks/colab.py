"""Données et sorties des notebooks sur Colab (M14, T14.1 et T14.8).

Les jeux ne sont pas versionnés (33 Go). `prepare_data` les met dans `data/` (sur Colab, le
disque local de la session) en essayant, dans l'ordre :

1. le témoin de `tools/get_datasets.sh` (déjà prêt : utilisé tel quel) ;
2. l'archive `<drive_dir>/data/<jeu>.tar` sur le Drive monté (Colab), envoyée depuis le PC
   par `tools/get_datasets.sh push <jeu>` (seule voie pour ExDark) ;
3. la même archive par rclone (`get_datasets.sh pull <jeu>`, remote `remote`) : le PC
   sans le jeu, si rclone est installé ;
4. le téléchargement direct : `get_voc.sh`, `get_datasets.sh coco|kitti`, ou l'API Kaggle
   pour CrowdHuman, VisDrone et FLIR (jeton : `<drive_dir>/kaggle.json`, ou les secrets Colab
   KAGGLE_USERNAME, KAGGLE_KEY, illisibles depuis un noyau Colab sous VS Code).

Mise en place de rclone et export des jeux : docs/tasks/donnees-drive.md.

Les images ne sont jamais lues à travers le montage de Drive (lent pour des milliers de
petits fichiers) : seul le tar y est lu, d'une traite.

Pas de dépendance hors bibliothèque standard : ce module est importé par les notebooks.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GET_DATASETS = ROOT / "tools" / "get_datasets.sh"
DRIVE_MOUNT = "/content/drive"
DRIVE_DIR = "/content/drive/MyDrive/EmbeddedCV"  # data/*.tar, runs/
DRIVE_REMOTE = "gdrive:EmbeddedCV/data"  # le même dossier data/, vu par rclone

# Téléchargement direct de chaque jeu (3e voie), relatif à la racine du dépôt.
DOWNLOAD = {
    "voc": ["tools/get_voc.sh"],
    "coco": ["tools/get_datasets.sh", "coco"],
    "kitti": ["tools/get_datasets.sh", "kitti"],
    "crowdhuman": ["tools/get_datasets.sh", "kaggle-download", "crowdhuman"],
    "visdrone": ["tools/get_datasets.sh", "kaggle-download", "visdrone"],
    "flir": ["tools/get_datasets.sh", "kaggle-download", "flir"],
}


def in_colab():
    return "google.colab" in sys.modules


def mount_drive():
    """Monte Google Drive sur /content/drive (demande l'autorisation dans le notebook)."""
    from google.colab import drive
    drive.mount(DRIVE_MOUNT)


def kaggle_credentials(drive_dir=DRIVE_DIR):
    """Jeton de la CLI Kaggle : variables d'environnement, ~/.kaggle/kaggle.json,
    `<drive_dir>/kaggle.json` (Drive monté), puis secrets Colab KAGGLE_USERNAME et KAGGLE_KEY.

    Sous VS Code, les secrets Colab ne répondent pas (TimeoutException : pas de page pour
    autoriser l'accès) ; le fichier sur le Drive est la voie qui marche partout."""
    if os.environ.get("KAGGLE_KEY") or Path("~/.kaggle/kaggle.json").expanduser().is_file():
        return True
    on_drive = Path(drive_dir) / "kaggle.json" if drive_dir else None
    if on_drive and on_drive.is_file():
        token = json.loads(on_drive.read_text())
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = token["username"], token["key"]
        print(f"jeton Kaggle : {on_drive}")
        return True
    try:
        from google.colab import userdata
        for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
            os.environ[k] = userdata.get(k)
    except Exception as e:  # pas sur Colab, ou secret absent / non partagé avec le notebook
        print(f"jeton Kaggle indisponible ({e.__class__.__name__}) : copier kaggle.json "
              f"(kaggle.com, Settings, Create New Token) dans {on_drive or '~/.kaggle/'}")
        return False
    return True


def _sh(cmd, data_dir=None, **extra):
    env = dict(os.environ, **({"DATA_DIR": str(data_dir)} if data_dir else {}), **extra)
    print("$ " + " ".join(cmd), flush=True)
    return subprocess.run(["bash", *cmd], cwd=ROOT, env=env).returncode


def _check(dataset, data_dir=None):
    """Témoin du jeu présent (`get_datasets.sh ready`) ?"""
    env = dict(os.environ, **({"DATA_DIR": str(data_dir)} if data_dir else {}))
    return subprocess.run(["bash", str(GET_DATASETS), "ready", dataset], cwd=ROOT, env=env,
                          capture_output=True).returncode == 0


def prepare_data(dataset, drive_dir=DRIVE_DIR, data_dir=None, download=True,
                 remote=DRIVE_REMOTE):
    """Met `dataset` dans data/ (ou `data_dir`) ; rend la voie prise : ready, drive, rclone,
    download. `drive_dir` (Drive monté) ou `remote` (rclone) à None : voie désactivée.

    Lève RuntimeError avec la marche à suivre si aucune voie n'aboutit.
    """
    if _check(dataset, data_dir):
        print(f"{dataset} : prêt")
        return "ready"
    archives = Path(drive_dir) / "data" if drive_dir else None
    if archives and (archives / f"{dataset}.tar").is_file():
        if _sh([str(GET_DATASETS), "unpack", str(archives), dataset], data_dir) == 0:
            return "drive"
    if remote and shutil.which("rclone"):
        if _sh([str(GET_DATASETS), "pull", dataset], data_dir, DRIVE_REMOTE=remote) == 0:
            return "rclone"
    if download and dataset in DOWNLOAD:
        if "kaggle-download" in DOWNLOAD[dataset]:
            kaggle_credentials(drive_dir)
            if not shutil.which("kaggle"):
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kaggle"])
        if _sh(DOWNLOAD[dataset], data_dir) == 0 and _check(dataset, data_dir):
            return "download"
    raise RuntimeError(
        f"{dataset} introuvable. Sur le PC qui l'a : tools/get_datasets.sh push {dataset} "
        f"(pack {dataset}, puis rclone vers {remote or DRIVE_REMOTE}) ; voir "
        f"docs/tasks/donnees-drive.md.")


def sync_outputs(src="build/notebooks", drive_dir=DRIVE_DIR):
    """Copie les sorties (checkpoints compris) dans `<drive_dir>/runs/`, pour survivre à la
    fin de la session ; rend le dossier de destination."""
    dst = Path(drive_dir) / "runs" / src
    shutil.copytree(ROOT / src, dst, dirs_exist_ok=True)
    print(f"{src} → {dst}")
    return dst


def restore_outputs(src="build/notebooks", drive_dir=DRIVE_DIR):
    """Inverse de `sync_outputs` en début de session : reprise depuis le dernier checkpoint."""
    saved = Path(drive_dir) / "runs" / src
    if saved.is_dir():
        shutil.copytree(saved, ROOT / src, dirs_exist_ok=True)
        print(f"{saved} → {src}")
    return saved.is_dir()
