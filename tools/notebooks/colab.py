"""Données et sorties des notebooks sur Colab (M14, T14.1 et T14.8).

Les jeux ne sont pas versionnés (33 Go). `prepare_data` les met dans `data/` (sur Colab, le
disque local de la session) en essayant, dans l'ordre :

1. le témoin de `tools/get_datasets.sh` (déjà prêt : utilisé tel quel) ;
2. l'archive `<drive_dir>/data/<jeu>.tar` sur le Drive monté (Colab), envoyée depuis le PC
   par `tools/get_datasets.sh push <jeu>` (seule voie pour ExDark) ;
3. la même archive par rclone (`get_datasets.sh pull <jeu>`, remote `remote`) : le PC
   sans le jeu, si rclone est installé ;
4. le téléchargement direct : `get_voc.sh`, `get_datasets.sh coco|kitti`, ou l'API Kaggle
   pour CrowdHuman, VisDrone et FLIR (jeton : `<drive_dir>/access_token` ou `kaggle.json`,
   ou les secrets Colab, illisibles depuis un noyau Colab sous VS Code).

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
    """Jeton de la CLI Kaggle, dans l'ordre : variables d'environnement (KAGGLE_API_TOKEN,
    ou KAGGLE_USERNAME et KAGGLE_KEY), ~/.kaggle/, `<drive_dir>/access_token` (jeton KGAT_…
    de kaggle.com, Settings, API Tokens) ou `<drive_dir>/kaggle.json` (ancien format) sur le
    Drive monté, puis secrets Colab KAGGLE_API_TOKEN ou KAGGLE_USERNAME et KAGGLE_KEY.

    Sous VS Code, les secrets Colab ne répondent pas (TimeoutException : pas de page pour
    autoriser l'accès) ; le fichier sur le Drive est la voie qui marche partout."""
    home = Path("~/.kaggle").expanduser()
    if (os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY")
            or (home / "access_token").is_file() or (home / "kaggle.json").is_file()):
        return True
    drive = Path(drive_dir) if drive_dir else None
    if drive and (drive / "access_token").is_file():
        os.environ["KAGGLE_API_TOKEN"] = (drive / "access_token").read_text().strip()
        print(f"jeton Kaggle : {drive / 'access_token'}")
        return True
    if drive and (drive / "kaggle.json").is_file():
        token = json.loads((drive / "kaggle.json").read_text())
        os.environ["KAGGLE_USERNAME"], os.environ["KAGGLE_KEY"] = token["username"], token["key"]
        print(f"jeton Kaggle : {drive / 'kaggle.json'}")
        return True
    try:
        from google.colab import userdata
        try:
            os.environ["KAGGLE_API_TOKEN"] = userdata.get("KAGGLE_API_TOKEN")
        except userdata.SecretNotFoundError:
            for k in ("KAGGLE_USERNAME", "KAGGLE_KEY"):
                os.environ[k] = userdata.get(k)
    except Exception as e:  # pas sur Colab, secret absent / non partagé, ou noyau sous VS Code
        print(f"jeton Kaggle indisponible ({e.__class__.__name__}) : mettre le jeton KGAT_… "
              f"(kaggle.com, Settings, API Tokens) dans {drive or home}/access_token")
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
            # Jeton KGAT_… : CLI 1.8 ou plus (celle de Colab est plus ancienne).
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle>=1.8"])
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


# Commandes dont les sorties partent sur le Drive dès leur fin (T14.8) : `--out` et
# `--markdown` (dossier parent) sous build/.
AUTOSYNC_TOOLS = ("tools/train.py", "tools/eval_voc.py", "tools/eval_quant.py")


def _drive_ready(drive_dir=DRIVE_DIR):
    return in_colab() and Path(drive_dir).is_dir()


def autosync(cmd, drive_dir=DRIVE_DIR):
    """Après une commande de `AUTOSYNC_TOOLS` sur Colab, Drive monté : `summary.json` des
    dossiers de sortie (`runs.write_summary`), puis copie dans `<drive_dir>/runs/`. Rend
    les dossiers copiés. Hors Colab, sans Drive, ou `EMBEDDEDCV_AUTOSYNC=0` : rien. Une
    erreur est affichée, jamais levée : la collecte ne casse pas une cellule.

    Appelée par `commandes.run` : chaque run d'un balayage et chaque évaluation (`_sweep`,
    `_infer`, `_train`) sont sur le Drive dès leur fin, sans attendre la copie périodique
    de la cellule d'entraînement (10 min), qui ne couvre ni les évaluations ni `_infer`."""
    if os.environ.get("EMBEDDEDCV_AUTOSYNC", "1") == "0" or not _drive_ready(drive_dir):
        return []
    if len(cmd) < 2 or cmd[1] not in AUTOSYNC_TOOLS:
        return []
    dirs = []
    for opt in ("--out", "--markdown"):
        if opt in cmd[:-1]:
            p = Path(cmd[cmd.index(opt) + 1])
            d = p.parent if p.suffix else p
            if d.parts[:1] == ("build",) and d not in dirs:
                dirs.append(d)
    done = []
    for d in dirs:
        try:
            from tools.notebooks import runs

            runs.write_summary(d)
            if (ROOT / d).is_dir():
                sync_outputs(d.as_posix(), drive_dir)
                done.append(d)
        except Exception as e:  # noqa: BLE001  (collecte au mieux)
            print(f"autosync {d} : {e.__class__.__name__}: {e}")
    return done

