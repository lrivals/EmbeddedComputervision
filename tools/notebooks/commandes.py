"""Commandes CLI des notebooks (M14) : construction et exécution en sous-processus.

Chaque fonction `cmd_*` rend la liste d'arguments d'un outil de `tools/`, dans l'ordre des
profils `tools/m11.sh` et `tools/m12.sh` ; `python/tests/test_notebooks.py` les compare aux
scripts. `run(cmd)` affiche la commande équivalente, puis relaie la sortie ligne à ligne :
une interruption du noyau arrête le sous-processus, et le dernier checkpoint reste.

Pas de dépendance hors bibliothèque standard : ce module est importé par les notebooks.
"""

import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Hyperparamètres d'affinage des profils `<jeu>-train` de tools/m11.sh (T11.4, T11.5, T11.7).
M11_TRAIN = {"iters": 4000, "batch": 16, "lr": 1e-3, "burn_in": 500, "multiscale": True,
             "workers": 4, "save_every": None}
M11_CFG_DIR = "build/m11/cfg"

HISTORY = []  # commandes lancées par `run`, pour le résumé du notebook


def _opt(cmd, name, value):
    """Ajoute `name value` si la valeur est donnée (None, "" et False : rien ; True : drapeau)."""
    if value is None or value is False or value == "":
        return cmd
    cmd.append(name)
    if value is not True:
        cmd.append(str(value))
    return cmd


def _size(size):
    """`SIZE` d'un notebook (None, 416, "640x192") → texte de l'option `--size`."""
    return None if size in (None, "") else str(size)


def cfg_variant(size=None, channels=None):
    """Suffixe des cfg et dossiers d'affinage de tools/m11.sh (`VAR`) : -640x192, -c1."""
    return (f"-{size}" if _size(size) else "") + (f"-c{channels}" if channels else "")


def cfg_path(net, dataset, size=None, channels=None):
    """Cfg d'affinage de `tools/m11.sh <jeu>-prep` : build/m11/cfg/<net>-<jeu><VAR>.cfg."""
    return f"{M11_CFG_DIR}/{net}-{dataset}{cfg_variant(size, channels)}.cfg"


# --------------------------------------------------------------------------- outils

def cmd_anchors(dataset, size, out, data_root=None):
    """`tools/kmeans_anchors.py` (profil `<jeu>-prep`)."""
    cmd = ["python", "tools/kmeans_anchors.py", "--dataset", dataset,
           "--size", _size(size) or "416", "--out", str(out)]
    return _opt(cmd, "--data-root", data_root)


def cmd_make_cfg(base, dataset, anchors, out, size=None, channels=None):
    """`tools/make_cfg.py` (profil `<jeu>-prep`) ; `anchors` au format « w,h  w,h … »."""
    cmd = ["python", "tools/make_cfg.py", "--base", base, "--dataset", dataset]
    _opt(cmd, "--size", _size(size))
    _opt(cmd, "--channels", channels)
    return cmd + ["--anchors", anchors, "--out", str(out)]


def init_args(init, init_net=None):
    """`INIT` d'un notebook → options `--init` (et `--init-net`) de train.py."""
    cmd = _opt([], "--init", init)
    return _opt(cmd, "--init-net", init_net)


def cmd_train(net, out, dataset="voc", init="coco", init_net=None, resume=False, qat="",
              qat_steps=None, admm=None, admm_rho=None, admm_growth=None, admm_every=None,
              iters=4000, batch=16, lr=1e-3, burn_in=500, multiscale=True, size=None,
              subset=0, save_every=None, workers=4, device="cpu", data_root=None, seed=0):
    """`tools/train.py`. Ordre des options : celui de m11.sh (`train`) ; QAT et ADMM à la
    place de `--dataset`, comme m12.sh (`qat`, `admm`). `--dataset voc` (défaut) est omis."""
    cmd = ["python", "tools/train.py", "--net", str(net)]
    if dataset != "voc":
        cmd += ["--dataset", dataset]
    cmd += init_args(init, init_net)
    _opt(cmd, "--resume", bool(resume))
    _opt(cmd, "--qat", qat)
    if qat:
        _opt(cmd, "--qat-steps", qat_steps)
    _opt(cmd, "--admm", admm)
    if admm:
        _opt(cmd, "--admm-rho", admm_rho)
        _opt(cmd, "--admm-growth", admm_growth)
        _opt(cmd, "--admm-every", admm_every)
    cmd += ["--iters", str(iters), "--batch", str(batch), "--lr", f"{lr:g}",
            "--burn-in", str(burn_in)]
    _opt(cmd, "--multiscale", bool(multiscale))
    _opt(cmd, "--size", _size(size))
    _opt(cmd, "--subset", subset or None)
    _opt(cmd, "--save-every", save_every)
    _opt(cmd, "--seed", seed or None)  # 0 : défaut de train.py, omis (T15.2)
    cmd += ["--workers", str(workers), "--device", device]
    _opt(cmd, "--data-root", data_root)
    return cmd + ["--out", str(out)]


# Mémoire d'un processus d'évaluation (réseau, lot de 8 images en float32, im2col), large.
JOB_MEMORY = 1.5e9


def auto_jobs(jobs=None):
    """`JOBS` des notebooks : le nombre donné, ou None → un processus par cœur CPU
    disponible, plafonné par la mémoire (`JOB_MEMORY` chacun). Colab : 2 sur le runtime
    CPU, des dizaines sur un runtime TPU (le TPU ne sert pas, l'évaluation est en NumPy)."""
    if jobs:
        return int(jobs)
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    try:
        mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return n
    return max(1, min(n, int(mem // JOB_MEMORY)))


def cmd_eval_voc(net, weights, dataset="voc", resize="stretch", subset=0, metric=None,
                 split=None, size=None, data_root=None, out=None, markdown=None, jobs=1):
    """`tools/eval_voc.py` (fin du profil `<jeu>-train`) ; sorties du notebook à la fin.
    `jobs` None : `auto_jobs` ; plus de 1 : `--jobs` (inférence répartie sur des processus)."""
    jobs = auto_jobs(jobs)
    cmd = ["python", "tools/eval_voc.py", "--net", str(net), "--weights", str(weights),
           "--dataset", dataset, "--resize", resize, "--subset", str(subset)]
    _opt(cmd, "--metric", metric)
    _opt(cmd, "--split", split)
    _opt(cmd, "--size", _size(size))
    _opt(cmd, "--data-root", data_root)
    _opt(cmd, "--out", out)
    _opt(cmd, "--markdown", markdown)
    return _opt(cmd, "--jobs", jobs if jobs > 1 else None)


def cmd_calibrate(net, weights, dataset, out, markdown, images=500, split=None,
                  data_root=None):
    """`tools/calibrate.py` ; `--out` et `--markdown` toujours donnés : rien dans results/."""
    cmd = ["python", "tools/calibrate.py", "--net", str(net), "--weights", str(weights),
           "--dataset", dataset, "--images", str(images)]
    _opt(cmd, "--split", split)
    _opt(cmd, "--data-root", data_root)
    return cmd + ["--out", str(out), "--markdown", str(markdown)]


def cmd_eval_quant(net, out, variants="float,int", subset=0, jobs=16, dataset="voc",
                   weights=None, calib=None, model_dir=None, metric=None, split=None,
                   size=None, data_root=None):
    """`tools/eval_quant.py` comme `evalq` de m11.sh et m12.sh (stretch, 1 fil BLAS) ;
    `jobs` None : `auto_jobs`."""
    cmd = ["python", "tools/eval_quant.py", "--net", str(net), "--resize", "stretch",
           "--jobs", str(auto_jobs(jobs)), "--blas-threads", "1", "--subset", str(subset),
           "--out", str(out), "--variants", variants]
    if dataset != "voc":
        cmd += ["--dataset", dataset]
    _opt(cmd, "--weights", weights)
    _opt(cmd, "--calib", calib)
    _opt(cmd, "--model-dir", model_dir)
    _opt(cmd, "--metric", metric)
    _opt(cmd, "--split", split)
    _opt(cmd, "--size", _size(size))
    return _opt(cmd, "--data-root", data_root)


def cmd_quant_lowbit(net, checkpoint, out, qat="", admm=False):
    """`tools/quant_lowbit.py` après QAT (`--scheme`) ou ADMM (`--weights pow2`), m12.sh."""
    cmd = ["python", "tools/quant_lowbit.py", "--net", str(net)]
    cmd += ["--weights", "pow2"] if admm else ["--scheme", qat]
    return cmd + ["--checkpoint", str(checkpoint), "--out", str(out)]


def cmd_act_hist(net, weights, datasets, calib, out, images=200):
    """`tools/act_hist.py` : histogrammes d'activations L00-L04 (T11.3)."""
    return ["python", "tools/act_hist.py", "--net", str(net), "--weights", str(weights),
            "--datasets", datasets, "--images", str(images), "--calib", str(calib),
            "--out", str(out)]


def cmd_data_stats(dataset, out, splits=None, size=None, resize="letterbox", net=None,
                   sample=0, gallery=0, seed=0, only=None, data_root=None, figures=True):
    """`tools/data_stats.py` (notebooks `stats`, M16), option par option."""
    cmd = ["python", "tools/data_stats.py", "--dataset", dataset]
    _opt(cmd, "--split", ",".join(splits) if isinstance(splits, (list, tuple)) else splits)
    _opt(cmd, "--size", _size(size))
    cmd += ["--resize", resize]
    _opt(cmd, "--net", net)
    cmd += ["--sample", str(sample), "--gallery", str(gallery), "--seed", str(seed)]
    _opt(cmd, "--only", ",".join(only) if isinstance(only, (list, tuple)) else only)
    _opt(cmd, "--data-root", data_root)
    _opt(cmd, "--figures", figures)
    return cmd + ["--out", str(out)]


def cmd_figures(kind, path):
    """Mode automatique de M13 : figures de `path` dans `<path>/figures/`."""
    return ["python", "-m", "tools.figures", "--run", kind, "--dir", str(path)]


# ----------------------------------------------------------------------- exécution

def anchors_k(path, k):
    """Ancres à `k` de la table de kmeans_anchors.py, comme `anchors_k` de m11.sh."""
    import re

    for line in Path(path).read_text().splitlines():
        if line.startswith(f"| {k} |"):
            return re.findall(r"`([^`]*)`", line)[0]
    raise ValueError(f"{path} : pas de ligne k = {k}")


def show(cmd):
    return "$ " + shlex.join(cmd)


def _autosync(cmd):
    """Sorties de `cmd` sur le Drive dès sa fin (Colab seulement, `colab.autosync`)."""
    from tools.notebooks import colab

    colab.autosync(cmd)


def run(cmd, log=None, check=True):
    """Lance `cmd` depuis la racine du dépôt (le `python` du noyau), relaie sa sortie ligne à
    ligne et rend son code de retour. Ctrl-C / interruption du noyau : SIGINT au processus,
    qui s'arrête proprement ; l'exception est relancée."""
    HISTORY.append(shlex.join(cmd))
    print(show(cmd), flush=True)
    argv = [sys.executable if a == "python" and i == 0 else a for i, a in enumerate(cmd)]
    t0 = time.time()
    proc = subprocess.Popen(argv, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    f = open(log, "a") if log else None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if f:
                f.write(line)
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
        print(f"interrompu après {time.time() - t0:.0f} s ; reprise : relancer la cellule")
        _autosync(cmd)  # checkpoint de l'interruption
        raise
    finally:
        if f:
            f.close()
    print(f"  ({time.time() - t0:.0f} s)")
    if not proc.returncode:
        _autosync(cmd)
    if check and proc.returncode:
        raise subprocess.CalledProcessError(proc.returncode, shlex.join(cmd))
    return proc.returncode
