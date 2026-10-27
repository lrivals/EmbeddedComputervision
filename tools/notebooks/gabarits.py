"""Gabarits des notebooks (T14.1, T14.3 à T14.8) : cellules communes, `infer` et `train`.

Structure fixe de chaque notebook : titre, cellule `parameters` (compatible papermill),
cellule d'environnement, corps du gabarit, résumé. Les cellules de code n'appellent que le
code du dépôt (`tools.notebooks.commandes` pour les outils CLI, `yolo.*` pour l'affichage).
Le JSON est écrit sans sorties, cellules en listes de lignes, trié et indenté comme
l'enregistre Jupyter : ouvrir puis enregistrer un notebook non exécuté ne change rien.
"""

import json
import textwrap

from tools.notebooks import OUT, ROOT
from tools.notebooks.colab import DRIVE_DIR
from tools.notebooks.commandes import M11_TRAIN
from tools.notebooks.matrice import (SPLIT_IMAGES, V4_ARCH, V4_INIT_NET, V5_ARCH, how_to_get,
                                     palier_of, prerequis)

REPO_URL = "https://github.com/lrivals/EmbeddedComputervision.git"
REV = "main"
COLAB = "https://colab.research.google.com/github/lrivals/EmbeddedComputervision/blob/main"
ROLE_TITLES = {"infer": "inference", "train": "training",
               "sweep": "batch × subset sweep", "stats": "dataset statistics"}
SUBSET_FIRST = 50  # premier passage, palier R (règles de M12)
# Notebooks de référence, réglés pour le split complet (palier N) : leurs mAP se comparent
# à results/ (T14.3 : flottant 56,30 ; T14.4 : entier 55,66, results/map_int8.md).
REFERENCE = {"voc/tiny-yolov2-voc_infer.ipynb": {"SUBSET": 0, "INT8": True}}
# YOLOv4-tiny runs to do first (M24): full split by default. T24.5: C-sim pass over the
# 5,000 COCO images (on the PC, ≈ 1.5 h on 18 cores); T24.6: VOC, then VisDrone.
FULL_RUN = {"coco/tiny-yolov4-coco_infer.ipynb": {"SUBSET": 0, "INT8": True, "HW": True},
            "voc/tiny-yolov4-voc_train.ipynb": {"SUBSET": 0},
            "voc/tiny-yolov4-voc_infer.ipynb": {"SUBSET": 0, "INT8": True},
            "visdrone/tiny-yolov4-visdrone_train.ipynb": {"SUBSET": 0},
            "visdrone/tiny-yolov4-visdrone_infer.ipynb": {"SUBSET": 0, "INT8": True},
            # T37.2: VOC2007 test through MAPPINGS, the baseline of the VOC fine-tunings.
            "voc/yolov5n-coco_infer.ipynb": {"SUBSET": 0},
            "voc/tiny-yolov7-coco_infer.ipynb": {"SUBSET": 0},
            "voc/yolov8n-coco_infer.ipynb": {"SUBSET": 0}}
# Run long VisDrone (M15, T15.8) : géométrie de l'éval (stretch) à l'entraînement, 6 000
# itérations (≈ 15 époques à lot 16), LR ÷ 10 à 80 % et 90 %.
VISDRONE_TRAIN = {"ITERS": 6000, "STEPS": "4800,5400", "SCALES": "0.1,0.1",
                  "RESIZE_TRAIN": "stretch"}
# Grille par défaut du notebook _sweep (T14.10) : 3 lots × (500 images, tout le jeu),
# graine 0 (autres graines : bruit d'un run, T15.2).
SWEEP = {"batches": [8, 16, 32], "subsets": [500, 0], "seeds": [0], "iters": 600}

# Paramètres des profils M12 pour le notebook voc/tiny-yolov3-voc_train (T14.7).
M12_QAT = {"NET": "tiny-yolov2-voc", "INIT": "weights/yolov2-tiny-voc.weights",
           "QAT": "w4a4", "QAT_STEPS": "build/m9/models/tiny-yolov2-voc-w4a4-ptq/steps.json",
           "LR": 1e-4, "BURN_IN": 50, "BATCH": 8, "ITERS": 600, "SAVE_EVERY": 100,
           "WORKERS": 2, "MULTISCALE": False}
M12_ADMM = {"NET": "tiny-yolov2-voc", "INIT": "weights/yolov2-tiny-voc.weights",
            "ADMM": "build/notebooks/voc/tiny-yolov3-voc/plan_mixed6.json", "ADMM_RHO": 1e-3,
            "ADMM_GROWTH": 1.3, "ADMM_EVERY": 50, "LR": 1e-4, "BURN_IN": 50, "BATCH": 8,
            "ITERS": 600, "SAVE_EVERY": 100, "WORKERS": 2, "MULTISCALE": False}


# ------------------------------------------------------------------------ cellules

def _lines(text):
    lines = text.strip("\n").split("\n")
    return [line + "\n" for line in lines[:-1]] + [lines[-1]]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(textwrap.dedent(text))}


def code(text, tags=()):
    meta = {"tags": list(tags)} if tags else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": _lines(textwrap.dedent(text))}


def notebook(cells, gpu=False):
    for i, c in enumerate(cells):
        c["id"] = f"c{i:02d}"
    meta = {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}}
    if gpu:
        meta["accelerator"] = "GPU"
        meta["colab"] = {"gpuType": "T4", "provenance": []}
    return {"cells": cells, "metadata": meta, "nbformat": 4, "nbformat_minor": 5}


def dumps(nb):
    """Format d'enregistrement de Jupyter : JSON trié, indentation 1, fin de ligne finale."""
    return json.dumps(nb, indent=1, sort_keys=True, ensure_ascii=False) + "\n"


# ------------------------------------------------------------------------- en-tête

def _doc_of(task):
    """T11.4 → docs/tasks/M11-….md ; T9.2.3 → M9.2-….md (jalon : tout sauf le dernier)."""
    milestone = task[1:].rsplit(".", 1)[0]
    hits = sorted((ROOT / "docs" / "tasks").glob(f"M{milestone}-*.md"))
    return f"../../docs/tasks/{hits[0].name}" if hits else None


def _task_links(tasks):
    out = []
    for t in tasks:
        doc = _doc_of(t)
        out.append(f"[{t}]({doc})" if doc else t)
    return ", ".join(out)


def header(nb):
    first = palier_of(nb, SUBSET_FIRST, 0)
    if nb.role == "train" and nb.arch == V5_ARCH:
        cost = ("tier **N** (`EPOCHS` epochs of `tools/train_v5.py`, days on CPU: "
                "[long runs](../../docs/tasks/long-runs.md)); final evaluation on `SUBSET` "
                "images")
    elif nb.role == "train":
        full = palier_of(nb, 0, M11_TRAIN["iters"])
        cost = (f"tier **{full}** ({M11_TRAIN['iters']} iterations by default, "
                "several hours on CPU); final evaluation on `SUBSET` images")
    elif nb.role == "sweep":
        n = len(SWEEP["batches"]) * len(SWEEP["subsets"])
        cost = (f"tier **{palier_of(nb, 0, SWEEP['iters'])}** ({n} runs of "
                f"{SWEEP['iters']} iterations by default); each run is evaluated on "
                "`SUBSET` images, then compared with the others")
    else:
        n = SPLIT_IMAGES.get(nb.dataset)
        full = palier_of(nb)
        cost = (f"tier **{first}** on the first pass (`SUBSET = {SUBSET_FIRST}`), "
                f"**{full}** on the full split ({n or 'size not recorded'} images, "
                "`SUBSET = 0`)")
        if nb.path.as_posix() in REFERENCE:
            cost += (". **Reference notebook**, set to the full split (`SUBSET = 0`, "
                     "`INT8 = True`): expected mAP 56.30 in float (T14.3) and 55.66 in "
                     "integer (T14.4), `results/map_int8.md`; set `SUBSET = "
                     f"{SUBSET_FIRST}` for a quick try")
    full_run = FULL_RUN.get(nb.path.as_posix())
    if full_run:
        settings = ", ".join(f"`{k} = {v!r}`" for k, v in full_run.items())
        cost += (f". **Full run by default** ({settings}); set `SUBSET = {SUBSET_FIRST}` for "
                 "a quick try")
        if full_run.get("HW"):
            cost += (". The `HW` stage compiles the golden model and the C-sim with cmake and "
                     "g++: run it on the PC, not on Colab")
    req = prerequis(nb)
    reqs = "\n".join(f"- {k}: `{v}` — {how_to_get(k, nb)}" for k, v in req.items()
                     if not (nb.trains and k == "cfg"))
    domain = (f" Out-of-domain {nb.family.upper()} weights: only the shared classes are "
              "evaluated." if nb.out_of_domain and nb.role == "infer" else "")
    colab = f"{COLAB}/notebooks/{nb.path.as_posix()}"
    return md(textwrap.dedent("""
        # {model} on {dataset} — {role}

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})

        Role: {role}; {cost}. Tasks: {tasks};
        notebooks: [M14](../../docs/tasks/M14-notebooks.md).{domain}

        Prerequisites:
        @REQS@

        Each step prints the `python tools/…` command it runs: a result can be reproduced
        from the command line. Outputs in `{out}/`, never in `results/` (an mAP on `SUBSET`
        images is not publishable, M12 rules). Notebook generated by
        `python -m tools.notebooks` (`make notebooks`): edit the template
        `tools/notebooks/gabarits.py`, not this file.
        """).format(model=nb.model, dataset=nb.dataset, role=ROLE_TITLES[nb.role], colab=colab,
                    cost=cost, tasks=_task_links(nb.tasks), domain=domain,
                    out=nb.out_dir).replace("@REQS@", reqs))


# ---------------------------------------------------------------------- paramètres

VISDRONE_NOTE = textwrap.dedent("""

    ### VisDrone long runs (M15)

    One run per folder: set `OUT` to `RUNS_DIR/runs/<name>` so that the `_infer` notebook
    finds it (`RUN = '<name>'`, `COMPARE`), for example:

    | run | settings |
    |---|---|
    | `long-416` | defaults: `RESIZE_TRAIN = 'stretch'`, `CROP = 0` |
    | `long-416-crop640` | `CROP = 640`; evaluate with `TILES = 640` in `_infer` |
    | `long-608` | `SIZE = '608'` and `NET` = cfg at 608 (`tools/m11.sh visdrone-prep` with `SIZE=608`) |

    `RESIZE_TRAIN = 'stretch'` aligns the training on the evaluation (`RESIZE`): with
    letterbox, objects are 1.3 to 1.8 times shorter than at evaluation. The anchors of the
    cfg stay those of the letterbox: every run is evaluated with the same cfg.
    """)


def _tiled(nb):
    """Notebooks à options de géométrie et de tuiles (M15) : VisDrone affiné seulement ; les
    autres gardent leurs cellules telles quelles."""
    return nb.dataset == "visdrone" and (nb.role == "train"
                                         or (nb.role == "infer" and nb.finetuned))


def _tiles_args(nb, src):
    """Arguments `tiles` et `overlap` des évaluations (`@TILES@`) des notebooks `_tiled`."""
    return src.replace("@TILES@", ",\n" + " " * 38 + "tiles=TILES, overlap=OVERLAP"
                       if _tiled(nb) else "")


# Retouches des cellules du notebook _train VisDrone (M15) : géométrie d'entraînement,
# découpes, paliers du LR ; évaluation finale par tuiles si CROP. Textes après dedent.
TRAIN_RETOUCHES = (
    ("train=True, channels=channels)",
     "train=True, channels=channels,\n" + " " * 21 + "resize=RESIZE_TRAIN, crop=CROP)"),
    ('admm=bool(ADMM), rev=TRACE["rev"])',
     "admm=bool(ADMM), resize=RESIZE_TRAIN,\n" + " " * 16 + 'crop=CROP, rev=TRACE["rev"])'),
    ("WORKERS, DEVICE, DATA_ROOT),",
     "WORKERS, DEVICE, DATA_ROOT,\n" + " " * 22
     + "resize=RESIZE_TRAIN, crop=CROP, steps=STEPS, scales=SCALES),"),
    ("markdown=result,\n" + " " * 25 + "jobs=JOBS))",
     "markdown=result,\n" + " " * 25 + "jobs=JOBS, tiles=CROP))"),
)


def _retouch(cells, nb):
    """Cellules de `train_body` avec `TRAIN_RETOUCHES` pour les notebooks `_tiled`."""
    if not _tiled(nb):
        return cells
    done = set()
    for c in cells:
        if c["cell_type"] != "code":
            continue
        src = "".join(c["source"])
        for old, new in TRAIN_RETOUCHES:
            if old in src:
                src = src.replace(old, new)
                done.add(old)
        c["source"] = _lines(src)
    missing = [old for old, _ in TRAIN_RETOUCHES if old not in done]
    if missing:
        raise ValueError(f"retouches with no target in {nb.path}: {missing}")
    return cells


def _assign(pairs):
    width = max(len(k) for k, _, _ in pairs)
    return "\n".join(f"{k:{width}s} = {v!r}" + (f"  # {c}" if c else "") for k, v, c in pairs)


def _v5(nb):
    """YOLOv5-style decoding (T33.4): the `v5` notebooks and the Darknet YOLOv7-tiny."""
    return nb.arch == V5_ARCH or nb.net == "tiny-yolov7-coco"


def _v8(nb):
    """YOLOv8n (M35): anchor-free DFL head, `--preset v8` (NMS 0.7, T35.3), v8 loss."""
    return nb.arch == V5_ARCH and nb.model.startswith("yolov8")


def _preset(nb, cells):
    """`preset=PRESET` in the `tools/eval_voc.py` calls of the YOLOv8n notebooks."""
    if _v8(nb):
        for c in cells:
            if c["cell_type"] == "code":
                c["source"] = _lines("".join(c["source"]).replace(
                    "jobs=JOBS)", "jobs=JOBS, preset=PRESET)"))
    return cells


def _final(nb):
    """Weights written by `tools/train_v5.py` (Ultralytics state dict or Darknet)."""
    return "final.weights" if nb.model.startswith("tiny-") else "final.npz"


def _weights_param(nb):
    if nb.role == "train" and nb.arch == V5_ARCH:
        return ("WEIGHTS", None, f"None: EMA weights of OUT ({_final(nb)}), evaluated at "
                "the end; or a path")
    if nb.role == "infer" and nb.finetuned:
        return ("WEIGHTS", None, "None: final.weights of run RUN; or a path")
    if nb.role == "infer":
        return ("WEIGHTS", nb.weights, "weights evaluated")
    if nb.role == "sweep":
        return ("WEIGHTS", None, "per run: OUT/runs/<run>/final.weights")
    return ("WEIGHTS", f"{nb.out_dir}/final.weights", "weights produced, evaluated at the end")


def params(nb):
    p = [("NET", nb.net, "name in CFG_FILES or cfg (relative to the root)"),
         ("DATASET", nb.dataset, "key of DATASETS"),
         ("SPLIT", None, "None: evaluation split of the dataset"),
         _weights_param(nb),
         ("DEVICE", "gpu" if nb.trains else "cpu",
          "cpu | gpu (training only; the rest on CPU)"),
         ("SUBSET", SUBSET_FIRST, "images evaluated; 0: full split (tier N)"),
         ("SIZE", nb.size, "input WxH (e.g. '640x192'); None: the one of the cfg"),
         ("RESIZE", "v5" if _v5(nb) else "stretch",
          "v5 (letterbox and thresholds of YOLOv5, T33.4)" if _v5(nb)
          else "stretch (M12 reference) | letterbox"),
         *([("PRESET", "v8", "eval_voc preset with RESIZE v5: NMS 0.7 of YOLOv8 (T35.3)")]
           if _v8(nb) else []),
         ("METRIC", None, "None: the one of the dataset (coco for coco and flir)"),
         ("DATA_ROOT", None, "dataset root; None: data/<dataset> (Colab: Drive folder)")]
    if nb.role == "infer" and nb.finetuned:
        p += [("RUNS_DIR", nb.train_dir, "runs of the _train and _sweep notebooks"),
              ("RUN", None, "None: the most recent; e.g. 'train', 'b16-sall', 'b8-s500'"),
              ("COMPARE", False, "evaluates every run and compares them (T14.11)")]
        if _tiled(nb):
            p += [("TILES", 0, "tiles of N px of the original image (runs with CROP = N); "
                   "0: whole image"),
                  ("OVERLAP", 0.2, "tile overlap")]
    if nb.role == "train" and nb.arch == V5_ARCH:
        p += [("INIT", "coco", "coco (heads reinitialized) | he"),
              ("EPOCHS", 30, "epochs of the recipe (warmup 3, linear lr)"),
              ("ITERS", 0, "stop after n batches (0: all epochs; a short try: 20)"),
              ("BATCH", 16, "batch (gradients accumulated to 64)"),
              ("LR", 0.01, "lr0 (final lr: 0.01·lr0)"),
              ("SIZE_TRAIN", 640, "square training input"),
              ("TRAIN_SUBSET", 0, "first n training images (0: all)"),
              ("WORKERS", 4, None)]
    elif nb.role == "infer" and nb.arch == V5_ARCH:
        pass  # float only: the integer model of YOLOv5n is T33.8
    elif nb.role == "infer":
        p += [("INT8", False, "integer part: calibration and INT8 mAP (T14.4)"),
              ("CALIB_IMAGES", 500, "calibration images (calib split of the dataset)"),
              ("ACT_HIST", False, "activation histograms L00-L04 (T11.3)"),
              ("HW", False, "hardware part (needs INT8): export, C++ golden, C-sim on "
               "SUBSET images, mAP at the three stages (T8.2, T24.5)")]
    else:
        v4 = nb.arch == V4_ARCH  # YOLOv4-tiny (T24.6) : base et poids COCO de yolov4-tiny
        base = V4_INIT_NET if v4 else "tiny-yolov3-voc"
        p += [("BASE", base, "base of make_cfg.py"),
              ("CHANNELS", nb.channels, "channels of the cfg (1: thermal)"),
              ("ANCHORS_K", 6, "anchors of the cfg (6: YOLOv4-tiny)" if v4
               else "anchors of the cfg (6: Tiny-YOLOv3)"),
              ("INIT", nb.weights if v4 else "coco", "coco | he | path of a .weights"),
              ("INIT_NET", V4_INIT_NET if v4 else None,
               "network of the INIT weights if it differs from NET")]
        if nb.role == "sweep":
            p += [("BATCHES", SWEEP["batches"], "batch sizes tried"),
                  ("TRAIN_SUBSETS", SWEEP["subsets"],
                   "training images (first n); 0: the whole split"),
                  ("SEEDS", SWEEP["seeds"], "seeds (train.py --seed); T15.2: [1, 2]"),
                  ("ITERS", SWEEP["iters"], "iterations per run (tier N from 600)"),
                  ("SKIP_DONE", True, "skips the runs that already have final.weights")]
        else:
            p += [("ITERS", M11_TRAIN["iters"], "iterations (tier N above 600)"),
                  ("BATCH", M11_TRAIN["batch"], None)]
        if nb.role == "train" and _tiled(nb):
            p += [("RESIZE_TRAIN", "letterbox", "letterbox | stretch: geometry at "
                   "training (stretch: the one of the evaluation, RESIZE)"),
                  ("CROP", 0, "N×N px crops of the original image (tiles); 0: whole "
                   "image"),
                  ("STEPS", "", "iterations where the LR drops (e.g. '4800,5400')"),
                  ("SCALES", "", "LR factors at STEPS (e.g. '0.1,0.1')")]
        p += [("LR", M11_TRAIN["lr"], None),
              ("BURN_IN", M11_TRAIN["burn_in"], None),
              ("MULTISCALE", M11_TRAIN["multiscale"], "320-608 every 10 batches"),
              ("SAVE_EVERY", M11_TRAIN["save_every"], "None: default of train.py (200)"),
              ("WORKERS", M11_TRAIN["workers"], None)]
        if nb.role == "train":
            p += [("QAT", "", "wXaY scheme (e.g. 'w4a4'): QAT (T9.3, T12.5)"),
              ("QAT_STEPS", None, "steps.json of quant_lowbit.py (PTQ)"),
              ("ADMM", None, "JSON weight plan ({}: mixed6 everywhere, T9.2, T12.6)"),
              ("ADMM_RHO", 1e-3, None),
              ("ADMM_GROWTH", 1.3, None),
              ("ADMM_EVERY", 100, None)]
    p += [("DRIVE_DIR", DRIVE_DIR, "Colab: archives data/<dataset>.tar"
           + (", outputs and resume in runs/" if nb.trains else "")
           + "; None: no Drive"),
          ("JOBS", None, "evaluation processes; None: one per CPU core, capped "
           "by the memory (Colab TPU runtime: dozens)"),
          ("SHOW", 6, "images shown"),
          ("OUT", nb.out_dir, "notebook outputs"),
          ("REPO_URL", REPO_URL, "Colab: cloned repository"),
          ("REV", REV, "Colab: revision")]
    ref = {**REFERENCE.get(nb.path.as_posix(), {}), **FULL_RUN.get(nb.path.as_posix(), {})}
    if nb.role == "train" and _tiled(nb):
        ref = {**VISDRONE_TRAIN, **ref}
    p = [(k, ref.get(k, v), c) for k, v, c in p]
    return [md("## Parameters"), code(_assign(p), tags=("parameters",))]


# --------------------------------------------------------------------- environnement

def environment(nb):
    pre = ""
    if nb.role == "infer":
        needs = ("WEIGHTS", "NET", repr(how_to_get("weights", nb)))
        if nb.finetuned:  # T14.11 : poids d'un run de _train ou _sweep
            pre = f"""
        from tools.notebooks import colab, runs

        if COLAB and DRIVE_DIR:  # runs trained on Colab: copied to <DRIVE_DIR>/runs/
            env.mount_drive(DRIVE_DIR)
            colab.restore_outputs(RUNS_DIR, DRIVE_DIR)
        RUNS = runs.find_runs(RUNS_DIR)
        print(f"runs of {{RUNS_DIR}}:\\n{{runs.listing(RUNS)}}")
        CHOSEN = None
        if WEIGHTS is None:
            CHOSEN = runs.pick(RUNS_DIR, RUN, hint={needs[2]})
            WEIGHTS, OUT = CHOSEN.weights, f"{{OUT}}/{{CHOSEN.name}}"
            print(f"chosen run: {{CHOSEN.name}} ({{WEIGHTS}})")"""
            if nb.arch == V5_ARCH and nb.net.endswith(".cfg"):  # Darknet cfg of train_v5.py
                pre += """
            NET = f"{CHOSEN.dir}/net.cfg\""""
    elif nb.role == "stats":  # ni poids ni cfg exigés : la cfg manquante est remplacée
        needs = ("()", "None", "''")
    elif nb.arch == V5_ARCH:  # COCO weights of the fine-tuning
        needs = (repr(nb.weights) + " if INIT == 'coco' else ()", "None", "''")
    else:
        needs = ("INIT if str(INIT).endswith('.weights') else "
                 "'weights/yolov3-tiny.weights' if INIT == 'coco' else ()", "None", "''")
    post = ""
    if nb.role == "stats":
        post = """
        if str(NET).endswith(".cfg") and not env.restore_cfg(NET):
            print(f"cfg {NET} missing (tools/m11.sh {DATASET}-prep): anchors and grids of "
                  "tiny-yolov3-coco for the collisions")
            NET = "tiny-yolov3-coco\"""".rstrip()
    return [md("""
        ## Environment

        Local: nothing is installed or downloaded, the cell checks the prerequisites and
        stops with the command to run. Colab: clone of the repository, installation, CuPy if
        `DEVICE = "gpu"`, weights, then data by `colab.prepare_data` (archive
        `<DRIVE_DIR>/data/<dataset>.tar` or direct download; ExDark and FLIR: archive only,
        made on the PC by `tools/get_datasets.sh pack <dataset>`).
        """), code(f"""
        import os
        import subprocess
        import sys
        from pathlib import Path

        COLAB = "google.colab" in sys.modules


        def repo_root():
            for d in (Path.cwd(), *Path.cwd().parents):
                if (d / "tools" / "notebooks" / "commandes.py").exists():
                    return d
            return None


        ROOT = repo_root()
        if COLAB:
            if ROOT is None:
                ROOT = Path("/content/EmbeddedComputervision")
                if not ROOT.exists():
                    subprocess.run(["git", "clone", REPO_URL, str(ROOT)], check=True)
            # Clone of an earlier session: reset to REV (notebook and code of the same commit).
            subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin", REV], check=True)
            subprocess.run(["git", "-C", str(ROOT), "checkout", "-q", "FETCH_HEAD"], check=True)
            for m in [m for m in sys.modules if m.split(".")[0] in ("tools", "yolo")]:
                del sys.modules[m]  # modules imported before the update
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                            f"{{ROOT}}/python[data,plots]"], check=True)
            if DEVICE == "gpu":
                # CPU runtime: no driver, CuPy would say "cudaErrorInsufficientDriver".
                try:
                    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                except FileNotFoundError:
                    smi = None
                if smi is None or smi.returncode != 0:
                    raise RuntimeError("Colab runtime without a GPU: create a new Colab server "
                                       "of GPU type (T4, L4 or A100), or DEVICE = 'cpu'")
                cuda = smi.stdout.split("CUDA Version:")[-1].split()[0]  # driver version
                cupy_pkg = "cupy-cuda13x" if int(cuda.split(".")[0]) >= 13 else "cupy-cuda12x"
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", cupy_pkg],
                               check=True)
        elif ROOT is None:
            raise RuntimeError("repository root not found: open the notebook from notebooks/")
        os.chdir(ROOT)
        for p in (ROOT, ROOT / "python"):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        from tools.notebooks import commandes as C
        from tools.notebooks import env

        TRACE = env.trace(DEVICE)
        env.check_device(DEVICE){pre}
        env.prepare(DATASET, {needs[0]}, {needs[1]}, DATA_ROOT, COLAB, hint={needs[2]},
                    drive_dir=DRIVE_DIR){post}
        Path(OUT).mkdir(parents=True, exist_ok=True)
        """)]


# ---------------------------------------------------------------------------- infer

def infer_body(nb):
    cells = _infer_cells(nb)
    if nb.arch == V5_ARCH:  # `tools.eval_voc.load_net`: Ultralytics checkpoint, no torch
        for c in cells:
            if c["cell_type"] == "code":
                c["source"] = _lines("".join(c["source"]).replace(
                    "from yolo.io.darknet_weights import load_darknet_weights\n"
                    "from yolo.models.tiny_yolo import build\n",
                    "from tools.eval_voc import load_net\n").replace(
                    "@NET@", "net = load_net(NET, WEIGHTS)").replace(
                    # cache named with the resize mode (`runs.eval_paths`: `_v5`)
                    'map_float = EVAL_DIR / f"map_float_s{SUBSET or \'all\'}.md"\n'
                    '    dets = EVAL_DIR / f"dets_s{SUBSET or \'all\'}"',
                    "map_float, dets = runs.eval_paths(EVAL_DIR, SUBSET, RESIZE)"))
        _preset(nb, cells)
    else:
        for c in cells:
            if c["cell_type"] == "code":
                c["source"] = _lines("".join(c["source"]).replace(
                    "@NET@", "net = build(NET)\nload_darknet_weights(net, WEIGHTS)"))
    return cells


def _infer_cells(nb):
    cells = [md("""
        ## Network and weights

        `yolo.models.tiny_yolo.build` and `yolo.io.darknet_weights.load_darknet_weights`.
        """), code("""
        import json

        import numpy as np
        from IPython.display import Markdown, display
        from PIL import Image

        from tools.detect import draw
        from tools.notebooks.matrice import SPLIT_IMAGES, palier
        from yolo.data import datasets
        from yolo.data.letterbox import input_size, parse_size, size_label
        from yolo.infer.pipeline import detect, preprocess
        from yolo.io.darknet_weights import load_darknet_weights
        from yolo.models.tiny_yolo import build

        @NET@
        size = input_size(net.net, parse_size(SIZE) if SIZE else None)
        channels = net.net["input"][0]
        family, names = datasets.model_family(DATASET, net.net["classes"])
        print(f"{NET}: {net.net['classes']} classes ({family}), input {size_label(size)}, "
              f"{channels} channel(s); weights {WEIGHTS}")
        """), md("""
        ## Detections on a few images

        `yolo.infer.pipeline.preprocess` and `detect` (demo threshold), boxes by
        `tools/detect.py:draw`.
        """), code("""
        for s in datasets.load(DATASET, DATA_ROOT, SPLIT)[:SHOW]:
            with Image.open(s["image"]) as img:
                x, wh = preprocess(img, size, RESIZE, "pil", channels)
                boxes, scores, labels = detect(net, x[None], [wh], RESIZE)[0]
                shown = draw(img, boxes, scores, labels, names)
            shown.thumbnail((640, 640))
            print(f"{s['id']}: {len(labels)} detections")
            display(shown)
        """)]
    if nb.out_of_domain:
        cells += [md("""
            ## Class mapping (out of domain, T11.2)

            `MAPPINGS` of `yolo.data.datasets`, as it is; the classes of the dataset with no
            equivalent are removed from the ground truth, the detections of the other classes
            of the model are ignored (`eval_view`).
            """), code("""
            view = datasets.eval_view(DATASET, net.net["classes"])
            mapping = datasets.MAPPINGS[(DATASET, family)]
            rows = ["| dataset class | model class |", "|---|---|"]
            rows += [f"| {c} | {mapping.get(c, '— (no equivalent, ignored)')} |"
                     for c in datasets.DATASETS[DATASET].classes]
            display(Markdown("\\n".join(rows)))
            print(f"classes evaluated ({len(view.names)}): {', '.join(view.names)}")
            print(f"model classes ignored: {len(names) - len(view.names)}")
            """)]
    evaluate = """
        map_float = Path(OUT) / "map_float.md"
        map_float.unlink(missing_ok=True)
        C.run(C.cmd_eval_voc(NET, WEIGHTS, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                             DATA_ROOT, out=f"{OUT}/dets", markdown=map_float, jobs=JOBS))"""
    pr = 'Path(OUT, "dets", "figures")'
    if nb.finetuned and _tiled(nb):  # VisDrone (M15) : tuiles, caches par géométrie
        evaluate = """
        if CHOSEN is not None:
            EVAL_DIR = Path(RUNS_DIR, "eval", CHOSEN.name)
            runs.evaluate(CHOSEN, NET, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                          DATA_ROOT, out_dir=EVAL_DIR, jobs=JOBS, tiles=TILES, overlap=OVERLAP)
            map_float, dets = runs.eval_paths(EVAL_DIR, SUBSET, RESIZE, TILES)
        else:
            map_float = Path(OUT) / "map_float.md"
            map_float.unlink(missing_ok=True)
            C.run(C.cmd_eval_voc(NET, WEIGHTS, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                                 DATA_ROOT, out=f"{OUT}/dets", markdown=map_float, jobs=JOBS,
                                 tiles=TILES, overlap=OVERLAP))
            dets = Path(OUT, "dets")"""
        pr = 'Path(dets, "figures")'
    elif nb.finetuned:  # run choisi : même cache que la comparaison, évalué une fois (T15.1)
        evaluate = """
        if CHOSEN is not None:
            EVAL_DIR = Path(RUNS_DIR, "eval", CHOSEN.name)
            runs.evaluate(CHOSEN, NET, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                          DATA_ROOT, out_dir=EVAL_DIR, jobs=JOBS)
            map_float = EVAL_DIR / f"map_float_s{SUBSET or 'all'}.md"
            dets = EVAL_DIR / f"dets_s{SUBSET or 'all'}"
        else:""" + textwrap.indent(textwrap.dedent(evaluate), " " * 12) + """
            dets = Path(OUT, "dets")"""
        pr = 'Path(dets, "figures")'
    cells += [md("""
        ## Float mAP (T14.3)

        `tools/eval_voc.py`, metric of the dataset (`Dataset.metric`; COCO and FLIR: AP@[.5:.95]
        and AP50). PR curves of the automatic mode of M13 (VOC metric).
        """ + ("""
        The chosen run is evaluated in `RUNS_DIR/eval/<run>/`, as by the comparison
        below, which reuses this result: an mAP already computed for this `SUBSET` is not
        computed again.
        """ if nb.finetuned else "")), code("""
        print(f"{SUBSET or 'all'} images: tier "
              f"{palier(SUBSET or SPLIT_IMAGES.get(DATASET))} (M12 rules)")"""
        + evaluate + """
        display(Markdown(map_float.read_text()))
        for png in sorted(""" + pr + """.glob("pr*.png")):
            display(Image.open(png))
        """)]
    if nb.finetuned:
        cells += [md("""
            ## Comparison of the runs (T14.11)

            With `COMPARE = True`, each run found in `RUNS_DIR` (`_train` notebook and
            `_sweep` sweep) is evaluated on the same `SUBSET` images; an mAP already computed
            for this `SUBSET` is reused. The run evaluated above is chosen by `RUN`.
            """), code(_tiles_args(nb, """
            if COMPARE:
                rows = [runs.evaluate(r, NET, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                                      DATA_ROOT, out_dir=f"{RUNS_DIR}/eval/{r.name}", jobs=JOBS@TILES@)
                        for r in RUNS]
                display(Markdown(runs.table(rows)))
            else:
                print("COMPARE = False: comparison skipped; runs found:")
                print(runs.listing(RUNS))
            """))]
    if nb.arch == V5_ARCH and nb.finetuned:  # T37.4: against the COCO weights (T37.2)
        ood = nb.model.removesuffix(f"-{nb.dataset}") + "-coco"
        cells += [md(f"""
            ## Fine-tuned against out-of-domain

            The run against the COCO weights evaluated through `MAPPINGS` by
            `notebooks/{nb.dataset}/{ood}_infer.ipynb` (T37.2, its last execution): the
            baseline the fine-tuning must beat. Compare at the same `SUBSET` (images in the
            evaluation column).
            """), code(f"""
            rows = ["| weights | mAP | evaluation |", "|---|---|---|"]
            for label, table in (("fine-tuned", map_float),
                                 ("COCO, out of domain", Path("{OUT}/{nb.dataset}/{ood}/map_float.md"))):
                m = runs.read_map(table)
                head = (Path(table).read_text().splitlines()[0].lstrip("# ") if m is not None
                        else "— (run notebooks/{nb.dataset}/{ood}_infer.ipynb)")
                rows.append(f"| {{label}} | {{'—' if m is None else f'{{m:.2f}}'}} | {{head}} |")
            display(Markdown("\\n".join(rows)))
            """)]
    if nb.arch == V5_ARCH:  # float only (integer sections: T37.5)
        return cells + [md("## Summary"), code("""
            print(map_float.read_text())
            print(f"outputs: {OUT}/; revision {TRACE['rev']}")
            print("equivalent commands:")
            for c in C.HISTORY:
                print("  " + c)
            """)]
    cells += [md("""
        ## Integer part (T14.4)

        Enabled by `INT8 = True`. Calibration by `tools/calibrate.py` on the `calib` split
        of the dataset (scales and report in `OUT`, not in `results/`), then
        `tools/eval_quant.py --variants float,int`. On the full split (`SUBSET = 0`),
        tier **N**: count 12 to 25 min for VOC2007 test on top of the calibration.
        """), code("""
        map_int = None
        if INT8:
            if not SUBSET:
                print("SUBSET = 0: full split, tier N (several tens of minutes)")
            calib = Path(OUT) / "calib.json"
            C.run(C.cmd_calibrate(NET, WEIGHTS, DATASET, calib, Path(OUT) / "calibration.md",
                                  CALIB_IMAGES, data_root=DATA_ROOT))
            map_int = Path(OUT) / f"map_{RESIZE}_{SUBSET or 'complet'}.json"
            HW_DIR = Path(OUT) / "hw"  # integer detections per image, for the hardware part
            HW_DIR.mkdir(parents=True, exist_ok=True)
            C.run(C.cmd_eval_quant(NET, map_int, "float,int", SUBSET, JOBS, DATASET,
                                   weights=WEIGHTS, calib=calib, metric=METRIC, split=SPLIT,
                                   size=SIZE, data_root=DATA_ROOT,
                                   save_dets=HW_DIR / "int.jsonl" if HW else None))
            d = json.loads(map_int.read_text())
            fl, it = d["results"]["float"], d["results"]["int"]
            rows = ["| class | float | integer | gap |", "|---|---|---|---|"]
            rows += [f"| {c} | {100 * a:.2f} | {100 * b:.2f} | {100 * (b - a):+.2f} |"
                     for c, a, b in zip(d["classes"], fl["aps"], it["aps"])]
            rows.append(f"| **mAP** | **{100 * fl['map']:.2f}** | **{100 * it['map']:.2f}** | "
                        f"{100 * (it['map'] - fl['map']):+.2f} |")
            display(Markdown(f"{d['images']} images, {d['metric']}\\n\\n" + "\\n".join(rows)))
            for png in sorted((map_int.parent / "figures").glob(f"*{map_int.stem.removeprefix('map_')}*.png")):
                display(Image.open(png))
            if ACT_HIST:
                others = "voc" if DATASET == "voc" else f"voc,{DATASET}"
                C.run(C.cmd_act_hist(NET, WEIGHTS, others, calib, Path(OUT) / "act_hist"))
                display(Image.open(Path(OUT) / "act_hist" / "act_hist.png"))
        else:
            print("INT8 = False: integer part skipped")
        """), md("""
        ## Hardware part (T8.2, T24.5)

        Enabled by `HW = True` (with `INT8 = True`: same calibration, integer detections of
        the previous cell). Chain of the `coco-v4` profile of `tools/m11.sh`:

        1. `tools/export_model.py`: integer model (manifest, blobs) and dumps of 3 images of
           the dataset in `OUT/model`;
        2. C++ golden (`golden_run`) == Python dumps to the byte, layer by layer and
           detections (`tools/compare_dumps.py`, as `make golden-check`);
        3. FPGA stage in C-sim: the HLS kernel compiled without `ap_int` behind the ARM
           driver (`yolo_bench`, `tools/bench_sim.sh`), on the int8 inputs of the integer
           model (`tools/make_inputs.py`), for the first `SUBSET` images;
        4. `tools/map_stades.py`: mAP at the three stages and integer == C-sim equality image
           by image; `tools/perf_model.py`: cycles and ms of the current kernel (projection).

        Count ≈ 1 s of C-sim per image and per core (C-sim interrupted: run the cell again,
        the finished batches are kept as long as the exported model does not change).
        """), code("""
        import hashlib
        import shutil

        map_hw = None
        if HW:
            if not INT8:
                raise ValueError("HW = True needs INT8 = True (calibration and integer "
                                 "detections)")
            model = Path(OUT) / "model"
            C.run(C.cmd_export(NET, WEIGHTS, calib, model, DATASET, DATA_ROOT))
            for c in C.cmd_cmake("cpp/golden", "build/golden", "golden_run"):
                C.run(c)
            for dump in sorted(p for p in (model / "dumps").iterdir() if p.is_dir()):
                got = HW_DIR / "golden" / dump.name
                C.run(C.cmd_golden_run(model, dump / "input.npy", got))
                C.run(C.cmd_compare_dumps(dump, got))
            # C-sim detections of another exported model: erased (otherwise resumed).
            blobs = hashlib.sha256(b"".join((model / f).read_bytes() for f in
                                            ("weights.bin", "bias.bin", "requant.bin")))
            stamp = HW_DIR / "model.sha256"
            if not stamp.exists() or stamp.read_text() != blobs.hexdigest():
                shutil.rmtree(HW_DIR / "sim", ignore_errors=True)
                stamp.write_text(blobs.hexdigest())
            C.run(C.cmd_make_inputs(NET, DATASET, SUBSET, HW_DIR, DATA_ROOT))
            shutil.copy(map_int, HW_DIR / "eval_float_int.json")
            for c in C.cmd_cmake("sw", "build/sw-fast", "yolo_bench",
                                 "-DACC_NO_APINT -march=native", ["-DSW_BACKEND=sim"]):
                C.run(c)
            C.run(C.cmd_bench_sim(HW_DIR, model, JOBS), env={"BENCH": C.BENCH_FAST})
            map_hw = HW_DIR / "map_stades.md"
            C.run(C.cmd_map_stades(NET, DATASET, HW_DIR, map_hw, SUBSET, DATA_ROOT))
            display(Markdown(map_hw.read_text()))
            C.run(C.cmd_perf_model(model / "manifest.json"))
        else:
            print("HW = False: hardware part skipped")
        """), md("## Summary"), code("""
        print(map_float.read_text())
        if map_int:
            d = json.loads(map_int.read_text())
            for v, r in d["results"].items():
                print(f"{v:6s} mAP {100 * r['map']:.2f} ({d['images']} images)")
        if map_hw:
            print(map_hw.read_text().split("## AP")[0])
        print(f"outputs: {OUT}/; revision {TRACE['rev']}")
        print("equivalent commands:")
        for c in C.HISTORY:
            print("  " + c)
        """)]
    return cells


# ---------------------------------------------------------------------------- train

def _prep_preview_cells():
    """Préparation (ancres, cfg) et aperçu des cibles, communs à `train` et `sweep`."""
    return [md("""
        ## Preparation (datasets other than VOC)

        k-means anchors on the training split (`tools/kmeans_anchors.py`, table in `OUT`),
        then a cfg with N classes (`tools/make_cfg.py`), as `tools/m11.sh <dataset>-prep`.
        Skipped if the cfg already exists.
        """), code("""
        import csv
        import json

        import numpy as np
        from IPython.display import Markdown, display
        from PIL import Image

        from tools.detect import draw
        from tools.notebooks.matrice import estimate
        from yolo.data import datasets
        from yolo.data.letterbox import input_size, parse_size, size_label
        from yolo.models.tiny_yolo import load_cfg

        if not str(NET).endswith(".cfg"):
            print(f"{NET}: cfg of the repository, no preparation")
        elif Path(NET).exists():
            print(f"{NET} exists: preparation skipped")
        else:
            anchors = Path(OUT) / f"anchors_{DATASET}{'_' + str(SIZE) if SIZE else ''}.md"
            C.run(C.cmd_anchors(DATASET, SIZE, anchors, DATA_ROOT))
            C.run(C.cmd_make_cfg(BASE, DATASET, C.anchors_k(anchors, ANCHORS_K), NET, SIZE,
                                 CHANNELS))
        """), md("""
        ## Preview of the targets

        A few augmented images (`yolo.data.loader.VOCDataset`, hence `yolo.data.augment`)
        with their boxes: a check of the annotations before hours of computation.
        """), code("""
        cfg = load_cfg(NET)
        size = input_size(cfg, parse_size(SIZE) if SIZE else None)
        channels = cfg["input"][0]
        names = datasets.DATASETS[DATASET].classes
        samples = datasets.load(DATASET, DATA_ROOT, role="train")
        print(f"{len(samples)} training images, input {size_label(size)}, {channels} channel(s)")
        from yolo.data.loader import VOCDataset

        preview = VOCDataset(samples[:SHOW], train=True, channels=channels)
        for k in range(len(preview)):
            chw, boxes, labels = preview.load(k, size, np.random.default_rng(k))
            hwc = (np.clip(chw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(hwc[..., 0] if channels == 1 else hwc)
            display(draw(img, boxes, np.ones(len(labels)), labels, names))
        """)]


def train_v5_body(nb):
    """Fine-tuning with the YOLOv5 recipe (`tools/train_v5.py`, T33.7, T34.4) and its
    YOLOv8 settings (T35.6)."""
    net_eval = ("RUN / 'net.cfg'" if nb.model.startswith("tiny-") else "NET")
    v8 = ("""
        YOLOv8n (T35.6): v8 loss (task-aligned assigner, BCE + CIoU + DFL), `optimizer=auto`
        (AdamW or SGD by the rule of the reference, written to `optimizer.txt`), mosaic off
        for the last 10 epochs.
        """ if _v8(nb) else "")
    return _preset(nb, [md("""
        ## Fine-tuning

        `tools/train_v5.py` in a subprocess: SGD Nesterov with three parameter groups,
        warmup then linear lr, gradients accumulated to a nominal batch of 64, weight EMA,
        mosaic / random perspective / HSV / flip, heads reinitialized from the COCO weights
        with the bias prior of the reference (YOLOv7-tiny: the OTA loss of
        `hyp.scratch.tiny.yaml`). A `checkpoint.npz` is written every 200 batches; running
        the cell again resumes (`--resume`).
        """ + v8), code("""
        import csv

        from IPython.display import Markdown, display

        RUN = Path(OUT)
        RUN.mkdir(parents=True, exist_ok=True)
        resume = (RUN / "checkpoint.npz").exists()
        C.run(C.cmd_train_v5(NET, RUN, DATASET, INIT, resume, EPOCHS, ITERS, BATCH, SIZE_TRAIN,
                             WORKERS, DEVICE, DATA_ROOT, TRAIN_SUBSET, LR), log=RUN / "log.txt")
        """), md("""
        ## Losses

        `loss.csv`, per batch and per image: box, objectness and class terms (YOLOv5n,
        YOLOv7-tiny), box, class and DFL terms (YOLOv8n). Last rows, then one curve per term.
        """), code("""
        rows = list(csv.DictReader((RUN / "loss.csv").open())) if (RUN / "loss.csv").exists() else []
        parts = [k for k in ("box", "obj", "cls", "dfl") if rows and k in rows[0]]
        for r in rows[-10:]:
            print(", ".join(f"{k} {r[k]}" for k in ("it", "lr", "loss", *parts)))
        if rows:
            import matplotlib.pyplot as plt

            it = [int(r["it"]) for r in rows]
            fig, axes = plt.subplots(1, len(parts) + 1, figsize=(4 * (len(parts) + 1), 3))
            for ax, k in zip(axes, ("loss", *parts)):
                ax.plot(it, [float(r[k]) for r in rows], lw=0.8)
                ax.set_title(k)
                ax.set_xlabel("batch")
            fig.tight_layout()
            plt.show()
        """), md("""
        ## Evaluation

        The EMA weights by `tools/eval_voc.py --preset v5` (T33.4) on `SUBSET` images.
        """), code("""
        result = RUN / "map_float.md"
        result.unlink(missing_ok=True)
        C.run(C.cmd_eval_voc(@NET_EVAL@, WEIGHTS or RUN / "@FINAL@", DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                             DATA_ROOT, out=RUN / "dets", markdown=result, jobs=JOBS))
        display(Markdown(result.read_text()))
        """.replace("@NET_EVAL@", net_eval).replace("@FINAL@", _final(nb))), md("## Summary"), code("""
        if (RUN / "optimizer.txt").exists():
            print((RUN / "optimizer.txt").read_text().strip())
        print(f"outputs: {RUN}/ (checkpoint.npz, loss.csv, final weights); revision {TRACE['rev']}")
        print("equivalent commands:")
        for c in C.HISTORY:
            print("  " + c)
        """)])


def train_body(nb):
    if nb.arch == V5_ARCH:
        return train_v5_body(nb)
    qat = _assign([(k, v, None) for k, v in M12_QAT.items()])
    admm = _assign([(k, v, None) for k, v in M12_ADMM.items()])
    variants = ""
    if nb.dataset == "voc" and nb.arch != V4_ARCH:  # profils M12 : Tiny-YOLOv2 / v3
        variants = textwrap.dedent("""

        ### QAT and ADMM variants (T14.7)

        Parameters of the `tools/m12.sh qat ref` profile (T12.5); `QAT_STEPS` comes from the
        PTQ `tools/quant_lowbit.py --scheme w4a4` (T12.4):

        ```python
        @QAT@
        ```

        Parameters of the `tools/m12.sh admm ref` profile (T12.6); the plan `{}` (mixed6
        everywhere) is written if it is missing:

        ```python
        @ADMM@
        ```

        The evaluation then goes through `tools/quant_lowbit.py` and
        `tools/eval_quant.py --variants int --model-dir`, not through the float mAP.
        """).replace("@QAT@", qat).replace("@ADMM@", admm)
    affinage = textwrap.dedent("""
        ## Fine-tuning

        `tools/train.py` in a subprocess whose output is relayed line by line; an
        interruption of the kernel leaves the last `checkpoint.npz` (every `SAVE_EVERY`), and
        running the cell again resumes (`--resume`). On Colab with `DRIVE_DIR`, `OUT` is
        copied to `<DRIVE_DIR>/runs/` every 10 min and at the end of the cell
        (`colab.sync_outputs`), then restored at the start of a session
        (`colab.restore_outputs`): a cut session resumes from the last copied checkpoint, on
        the same backend (T12.11, T14.8).
        """) + variants + (VISDRONE_NOTE if _tiled(nb) else "")
    return _retouch([*_prep_preview_cells(), md(affinage), code("""
        import threading

        from tools.notebooks import colab, runs

        RUN = Path(OUT)
        SYNC = COLAB and bool(DRIVE_DIR)
        if SYNC:  # resume after a session cut: outputs of the previous session
            env.mount_drive(DRIVE_DIR)
            colab.restore_outputs(OUT, DRIVE_DIR)
        RUN.mkdir(parents=True, exist_ok=True)
        if ADMM and not Path(ADMM).exists():
            Path(ADMM).parent.mkdir(parents=True, exist_ok=True)
            Path(ADMM).write_text("{}\\n")  # mixed6 everywhere, as tools/m12.sh admm
        seconds = estimate(ITERS, BATCH, DEVICE, RUN / "loss.csv")
        print(f"{ITERS} iterations × {BATCH} images: "
              + (f"≈ {seconds / 3600:.1f} h on {DEVICE}" if seconds else
                 f"duration not measured on {DEVICE} (T12.11-e)"))
        resume = (RUN / "checkpoint.npz").exists()
        runs.write_meta(RUN, batch=BATCH, subset=0, iters=ITERS, lr=LR, device=DEVICE,
                        net=NET, dataset=DATASET, qat=QAT, admm=bool(ADMM), rev=TRACE["rev"])
        stop = threading.Event()


        def sync_loop(every=600):  # copy to Drive every 10 min, checkpoint included
            while not stop.wait(every):
                colab.sync_outputs(OUT, DRIVE_DIR)


        if SYNC:
            threading.Thread(target=sync_loop, daemon=True).start()
        try:
            C.run(C.cmd_train(NET, RUN, DATASET, INIT, INIT_NET, resume, QAT, QAT_STEPS, ADMM,
                              ADMM_RHO, ADMM_GROWTH, ADMM_EVERY, ITERS, BATCH, LR, BURN_IN,
                              MULTISCALE, None, 0, SAVE_EVERY, WORKERS, DEVICE, DATA_ROOT),
                  log=RUN / "log.txt")
        finally:
            stop.set()
            if SYNC:
                colab.sync_outputs(OUT, DRIVE_DIR)
        """), md("""
        ## Curves

        Loss and learning rate of `loss.csv` (automatic mode of M13); running the cell again
        during a long training redraws up to the last point written.
        """), code("""
        C.run(C.cmd_figures("train", RUN))
        for png in sorted((RUN / "figures").glob("*.png")):
            display(Image.open(png))
        """), md("""
        ## Evaluation

        `final.weights` by `tools/eval_voc.py` (`infer` template, T14.3); with QAT or ADMM,
        the low-precision model by `tools/quant_lowbit.py`, then `tools/eval_quant.py`.
        """), code("""
        if QAT or ADMM:
            C.run(C.cmd_quant_lowbit(NET, RUN / "checkpoint.npz", RUN / "model", QAT, bool(ADMM)))
            result = RUN / "map.json"
            C.run(C.cmd_eval_quant(NET, result, "int", SUBSET, JOBS, DATASET,
                                   model_dir=RUN / "model", metric=METRIC, split=SPLIT,
                                   size=SIZE, data_root=DATA_ROOT))
            d = json.loads(result.read_text())
            print(f"integer mAP {100 * d['results']['int']['map']:.2f} ({d['images']} images)")
        else:
            result = RUN / "map_float.md"
            result.unlink(missing_ok=True)
            C.run(C.cmd_eval_voc(NET, RUN / "final.weights", DATASET, RESIZE, SUBSET, METRIC,
                                 SPLIT, SIZE, DATA_ROOT, out=RUN / "dets", markdown=result,
                                 jobs=JOBS))
            display(Markdown(result.read_text()))
        """), md("""
        ## Backend speed (T12.11-e)

        Seconds per image of the last iterations (`loss.csv`); the largest batch that fits
        in GPU memory is found by running again with a growing `BATCH`. A number that comes
        from weights trained on GPU names the backend.
        """), code("""
        rows = list(csv.DictReader((RUN / "loss.csv").open())) if (RUN / "loss.csv").exists() else []
        if rows:
            last = rows[-50:]
            s = sum(float(r["seconds"]) for r in last) / len(last) / BATCH
            print(f"{s:.3f} s/image on {DEVICE}, batch {BATCH}, {len(rows)} iterations logged")
        """), md("## Summary"), code("""
        print(f"outputs: {RUN}/ (checkpoint.npz, loss.csv, final.weights, figures/); "
              f"backend {DEVICE}; revision {TRACE['rev']}")
        print(f"evaluation: {result}")
        print("equivalent commands:")
        for c in C.HISTORY:
            print("  " + c)
        """)], nb)


def sweep_body(nb):
    return [*_prep_preview_cells(), md("""
        ## Sweep plan (T14.10)

        One run per cell of the `BATCHES × TRAIN_SUBSETS × SEEDS` grid, every other
        parameter equal (those of `tools/m11.sh <dataset>-train`, except `ITERS`).
        `TRAIN_SUBSETS` takes the first n images of the training split
        (`train.py --subset`), 0 the whole split. A non-zero seed (`train.py --seed`:
        heads, batch draw, multiscale) suffixes the run (`b32-sall-g1`); seed 0 keeps the
        short name.
        At equal iterations, a larger batch sees more images: the "epochs" column shows it.
        The learning rate is not scaled with the batch.
        """), code("""
        import threading

        from tools.notebooks import colab, runs

        PLAN = [(b, s, g, runs.run_name(b, s, g)) for b in BATCHES for s in TRAIN_SUBSETS
                for g in SEEDS]
        rows = ["| run | batch | images | epochs | estimated duration |", "|---|---|---|---|---|"]
        total = 0
        for b, s, g, name in PLAN:
            n = min(s, len(samples)) if s else len(samples)
            sec = estimate(ITERS, b, DEVICE)
            total += sec or 0
            done = (Path(OUT) / "runs" / name / "final.weights").exists()
            rows.append(f"| {name}{' (done)' if done else ''} | {b} | {n} | "
                        f"{ITERS * b / n:.2f} | "
                        + (f"{sec / 3600:.1f} h |" if sec else "not measured |"))
        display(Markdown("\\n".join(rows)))
        print(f"{len(PLAN)} runs of {ITERS} iterations"
              + (f", ≈ {total / 3600:.1f} h on {DEVICE} in total" if total else ""))
        """), md("""
        ## Trainings

        Each run goes to `OUT/runs/<run>/` (`b16-sall`, `b8-s500`…), with its `run.json`.
        Running the cell again resumes the interrupted run (`--resume`) and skips the
        finished ones (`SKIP_DONE`). On Colab with `DRIVE_DIR`, `OUT` is copied to
        `<DRIVE_DIR>/runs/` every 10 min and at the end of the cell (T14.8).
        """), code("""
        SYNC = COLAB and bool(DRIVE_DIR)
        if SYNC:
            env.mount_drive(DRIVE_DIR)
            colab.restore_outputs(OUT, DRIVE_DIR)
        stop = threading.Event()


        def sync_loop(every=600):
            while not stop.wait(every):
                colab.sync_outputs(OUT, DRIVE_DIR)


        if SYNC:
            threading.Thread(target=sync_loop, daemon=True).start()
        try:
            for b, s, g, name in PLAN:
                run_dir = Path(OUT) / "runs" / name
                if SKIP_DONE and (run_dir / "final.weights").exists():
                    print(f"{name}: already trained, skipped")
                    continue
                resume = (run_dir / "checkpoint.npz").exists()
                runs.write_meta(run_dir, batch=b, subset=s, seed=g, iters=ITERS, lr=LR,
                                device=DEVICE, net=NET, dataset=DATASET, rev=TRACE["rev"])
                C.run(C.cmd_train(NET, run_dir, DATASET, INIT, INIT_NET, resume, iters=ITERS,
                                  batch=b, lr=LR, burn_in=BURN_IN, multiscale=MULTISCALE,
                                  subset=s, save_every=SAVE_EVERY, workers=WORKERS,
                                  device=DEVICE, data_root=DATA_ROOT, seed=g),
                      log=run_dir / "log.txt")
        finally:
            stop.set()
            if SYNC:
                colab.sync_outputs(OUT, DRIVE_DIR)
        """), md("""
        ## Comparison

        Float mAP of each run on the same `SUBSET` images (`tools/eval_voc.py`, in
        `OUT/eval/<run>/`, a cache shared with the comparison of the `_infer` notebooks:
        reused if already computed), final loss and speed; loss curves overlaid
        (`tools.figures.resultats.plot_training`, M13). The inference notebooks
        `<model>_infer` read these runs (`RUN`, `COMPARE`).
        """), code("""
        from tools.figures import MissingSource
        from tools.figures import resultats as FR

        RESULTS = [runs.evaluate(r, NET, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE, DATA_ROOT,
                                 out_dir=f"{OUT}/eval/{r.name}", jobs=JOBS)
                   for r in runs.find_runs(OUT) if r.name != runs.TRAIN]
        display(Markdown(runs.table(RESULTS)))
        curves = []
        for b, s, g, name in PLAN:
            try:
                curves.append(FR.load_training(Path(OUT) / "runs" / name))
            except MissingSource:
                pass
        if curves:
            for png in FR.plot_training(curves, Path(OUT) / "figures", "balayage"):
                if str(png).endswith(".png"):
                    display(Image.open(png))
        """), md("## Summary"), code("""
        print(f"runs: {OUT}/runs/; backend {DEVICE}; revision {TRACE['rev']}")
        best = max((r for r in RESULTS if r["mAP"] is not None), key=lambda r: r["mAP"],
                   default=None)
        if best:
            print(f"best run on {SUBSET or 'all'} images: {best['run']} "
                  f"(mAP {best['mAP']:.2f}); inference: RUN = {best['run']!r}")
        print("equivalent commands:")
        for c in C.HISTORY:
            print("  " + c)
        """)]


# ---------------------------------------------------------------------------- stats (M16)

STATS_SAMPLE = 200   # images lues pour les pixels (palier selon SAMPLE)
STATS_GALLERY = 8    # images de la galerie par split

# Question propre à chaque jeu (docs/tasks/M16-presentation-jeux.md#questions-par-jeu) :
# (question, renvoi, cellule de code). Les cellules n'appellent que tools/data_stats.py.
QUESTIONS = {
    "voc": ("Are the low-AP classes (bottle, pottedplant) rare or small?",
            "T13.7, T13.13", """
        ap = DS.voc_ap()
        if ap:
            print(f"float AP per class: {ap['source']} (Tiny-YOLOv2 VOC, VOC2007 test)")
        display(Markdown(DS.md_par_classe(STATS, ap.get("ap") if ap else None)))
        """),
    "coco": ("What share of the objects is small (< 32²) once resized to 416?", "T11.1", """
        g = DS.whole(STATS)["geometrie"]
        for mode in ("orig", "letterbox", "stretch"):
            c = g[mode]["coco"]
            print(f"{mode:9s}: small {100 * c[0] / max(sum(c), 1):.1f} %, medium "
                  f"{100 * c[1] / max(sum(c), 1):.1f} %, large {100 * c[2] / max(sum(c), 1):.1f} %")
        if DS.group(STATS, "test").get("comptes"):
            print(f"iscrowd (test): {DS.group(STATS, 'test')['comptes']['crowd']}")
        """),
    "kitti": ("`letterbox` or `stretch`: how many objects fall under 8 px of height?",
              "T11.4, T15", """
        g = DS.whole(STATS)["geometrie"]
        for mode in ("letterbox", "stretch"):
            print(f"{mode:9s} {g['size'][1]}×{g['size'][0]}: height < 8 px "
                  f"{100 * g[mode]['h_under']['8']:.1f} %, < 16 px {100 * g[mode]['h_under']['16']:.1f} %")
        # Non-square input of T11.4: geometry only, no pixels.
        out_nc = f"{OUT}/640x192"
        C.run(C.cmd_data_stats(DATASET, out_nc, SPLITS, "640x192", "letterbox", NET,
                               only=["geometrie"], data_root=DATA_ROOT, figures=False))
        display(Markdown(DS.md_geometrie(json.loads(Path(out_nc, "stats.json").read_text()))))
        """),
    "visdrone": ("Share of the objects smaller than a 26×26 cell; target collisions.",
                 "T11.5, T15", """
        g = DS.whole(STATS)["geometrie"][STATS["params"]["resize"]]
        print("smaller than a cell: " + ", ".join(f"{k} {100 * v:.1f} %"
                                                           for k, v in g["fit_cell"].items()))
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    "flir": ("Distribution of the thermal levels, and what the INT8 of L00 keeps of it.", "T11.7", """
        px = DS.whole(STATS).get("images", {}).get("pixels")
        if px:
            lv = px["levels"]
            print(f"{px['images']} images, {px['channels']} channel(s); 8-bit levels used "
                  f"{lv['8bit']} (99 % of the pixels on {lv['8bit_99']}) → {lv['int8']} INT8 "
                  f"levels at the input scale {lv['input_scale']:.5f} (1/127, yolo.quant)")
        else:
            print("SAMPLE = 0: no pixel read")
        """),
    "exdark": ("Brightness per lighting type, and gap to the VOC calibration images.",
               "T11.3", """
        light = STATS.get("exdark_light")
        if light:
            rows = ["| lighting | images | mean luminance |", "|---|---|---|"]
            rows += [f"| {k} | {v['images']} | {v['lum_mean']:.1f} |" for k, v in light.items()]
            display(Markdown("\\n".join(rows)))
        ref = STATS.get("voc_ref")
        if ref:
            print(f"VOC2007 test ({ref['images']} images): mean luminance {ref['lum_mean']:.1f}")
        show("eclairage")
        """),
    "crowdhuman": ("Objects per image against the 256 slots; share of occluded objects.",
                   "T11.6", """
        for name, a in DS._parts(STATS):
            d, c = a.get("densite"), a.get("comptes")
            if d and c:
                print(f"{name:12s}: {c['images']} images, > 256 objects {d['over']['256']} "
                      f"({100 * d['over']['256'] / max(c['images'], 1):.2f} %), > 1,024 "
                      f"{d['over']['1024']}; difficult (mask, ignore) "
                      f"{100 * c['difficult'] / max(c['objects'], 1):.1f} % of the boxes")
        """),
    # Drone and thermal datasets of M18 (docs/tasks/M18-jeux-drone.md#questions-par-jeu).
    "auair": ("What is left of the objects of a 1920×1080 frame resized to 416, class by "
              "class?", "T18.1", """
        g = DS.whole(STATS)["geometrie"][STATS["params"]["resize"]]
        print("smaller than a cell: " + ", ".join(f"{k} {100 * v:.1f} %"
                                                           for k, v in g["fit_cell"].items()))
        print(f"height < 8 px: {100 * g['h_under']['8']:.1f} %")
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    "dronevehicle": ("Vehicles per image against the 256 slots, and their size at 416.",
                     "T18.2", """
        g = DS.whole(STATS)["geometrie"][STATS["params"]["resize"]]
        print("smaller than a cell: " + ", ".join(f"{k} {100 * v:.1f} %"
                                                           for k, v in g["fit_cell"].items()))
        for name, a in DS._parts(STATS):
            d, c = a.get("densite"), a.get("comptes")
            if d and c:
                print(f"{name:12s}: {c['images']} images, > 256 objects {d['over']['256']}")
        """),
    "hituav": ("Distribution of the thermal levels, and what the INT8 of L00 keeps of it.",
               "T18.3", """
        px = DS.whole(STATS).get("images", {}).get("pixels")
        if px:
            lv = px["levels"]
            print(f"{px['images']} images, {px['channels']} channel(s); 8-bit levels used "
                  f"{lv['8bit']} (99 % of the pixels on {lv['8bit_99']}) → {lv['int8']} INT8 "
                  f"levels at the input scale {lv['input_scale']:.5f} (1/127, yolo.quant)")
        else:
            print("SAMPLE = 0: no pixel read")
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    "uavdt": ("`letterbox` or `stretch` for 1024×540: how many objects fall under 8 px of "
              "height?", "T18.4", """
        g = DS.whole(STATS)["geometrie"]
        for mode in ("letterbox", "stretch"):
            print(f"{mode:9s} {g['size'][1]}×{g['size'][0]}: height < 8 px "
                  f"{100 * g[mode]['h_under']['8']:.1f} %, < 16 px {100 * g[mode]['h_under']['16']:.1f} %")
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    # ASL Alphabet (M21, docs/tasks/M21-asl-alphabet.md#stats-question).
    "asl": ("How much of the input does the hand cover, and which classes lose the most images "
            "to the pseudo-box step?", "T21.1", """
        import json
        from collections import Counter
        from pathlib import Path
        from yolo.data.datasets import DATA_DIR
        g = DS.whole(STATS)["geometrie"]
        mode = STATS["params"]["resize"]
        print(f"median hand side: {g['orig']['median_side']:.0f} px of 200, "
              f"{g[mode]['median_side']:.0f} px at the {g['size'][1]}×{g['size'][0]} input")
        rej = Path(DATA_ROOT or DATA_DIR / "asl") / "rejected.json"
        if rej.exists():
            lost = Counter(r["class"] for r in json.loads(rej.read_text()))
            print(f"letter images left out (no hand found): {sum(lost.values())}; most: "
                  + ", ".join(f"{c} {n}" for c, n in lost.most_common(8)))
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    # TrashNet and Oxford-IIIT Pet (M32, docs/tasks/M32-trashnet-pets.md#stats-questions).
    "trashnet": ("How much of the image does one piece of waste cover, and which classes lose "
                 "the most images to the labeller?", "T32.8", """
        import json
        from collections import Counter
        from pathlib import Path
        import numpy as np
        from yolo.data.datasets import DATA_DIR
        root = Path(DATA_ROOT or DATA_DIR / "trashnet")
        kept = [e for s in ("train", "val", "test")
                for e in json.loads((root / f"annotations_{s}.json").read_text())]
        cover = np.array([(b[2] - b[0]) * (b[3] - b[1]) / (e["width"] * e["height"])
                          for e in kept for b in e["boxes"]])
        print(f"box share of the image: median {np.median(cover):.2f}, "
              f"q10 {np.quantile(cover, 0.1):.2f}, q90 {np.quantile(cover, 0.9):.2f}")
        rej = json.loads((root / "rejected.json").read_text())
        n = Counter(e["class"] for e in kept)
        lost = Counter(r["class"] for r in rej)
        print("left out by the labeller: " + ", ".join(
            f"{c} {lost[c]} / {n[c] + lost[c]} ({100 * lost[c] / (n[c] + lost[c]):.1f} %)"
            for c in sorted(n, key=lambda c: -lost[c] / (n[c] + lost[c]))))
        print("reasons: " + ", ".join(f"{k} {v}" for k, v in
                                      Counter(r["reason"] for r in rej).most_common()))
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    "pets": ("How much larger is the body box than the head box, and how many images hold "
             "more than one animal?", "T32.8", """
        import json
        from pathlib import Path
        import numpy as np
        from yolo.data.datasets import DATA_DIR
        root = Path(DATA_ROOT or DATA_DIR / "pets")
        es = [e for s in ("train", "val", "test")
              for e in json.loads((root / f"annotations_{s}.json").read_text())]
        area = lambda b: (b[2] - b[0]) * (b[3] - b[1])
        r = np.array([area(e["boxes"][0]) / area(e["head"]) for e in es if e["head"]])
        print(f"body / head box area, {len(r)} trainval images: median {np.median(r):.1f}, "
              f"q10 {np.quantile(r, 0.1):.1f}, q90 {np.quantile(r, 0.9):.1f}")
        multi = sum(e["components"] > 1 for e in es)
        print(f"mask in several parts over 1 % of the image: {multi} of {len(es)} images "
              f"({100 * multi / len(es):.1f} %); mostly one animal cut by a coat, a hand or a "
              "leash, a few with two animals (one box over both)")
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    # PREWID (M38, docs/tasks/M38-prewid.md#stats-question).
    "prewid": ("How far are the workers from each camera (box height per camera), and what "
               "share of the frames is empty?", "T38.3", """
        import json
        from pathlib import Path
        import numpy as np
        from yolo.data.datasets import DATA_DIR
        root = Path(DATA_ROOT or DATA_DIR / "prewid")
        es = {s: json.loads((root / f"annotations_{s}.json").read_text())
              for s in ("train", "val", "test", "lab")}
        rows = ["| Camera | Image height | Frames | Empty (train / val / test / lab) | Boxes "
                "| Box height px: q10 / median / q90 | Under 32 px |", "|---|---|---|---|---|---|---|"]
        for cam in sorted({e["camera"] for v in es.values() for e in v}):
            fr = {s: [e for e in v if e["camera"] == cam] for s, v in es.items()}
            allf = [e for v in fr.values() for e in v]
            h = np.array([b[3] - b[1] for e in allf for b in e["boxes"]])
            empty = " / ".join(f"{100 * sum(not e['boxes'] for e in v) / len(v):.0f} %"
                               if v else "-" for v in fr.values())
            q = np.quantile(h, [0.1, 0.5, 0.9])
            rows.append(f"| {cam} | {allf[0]['height']} | {len(allf)} | {empty} | {len(h)} "
                        f"| {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} | {int((h < 32).sum())} |")
        display(Markdown("\\n".join(rows)))
        display(Markdown(DS.md_par_classe(STATS)))
        """),
}


def _fiche_md(dataset):
    from tools import data_stats

    f = data_stats.FICHES.get(dataset)
    if not f:
        return "Datasheet missing from `FICHES` (tools/data_stats.py)."
    rows = [("source", f["source"]), ("version", f["version"]), ("licence", f["license"]),
            ("sensor", f["sensor"]), ("typical resolution", f["resolution"])]
    rows += [(f"official `{sp}`", f"{i or '—'} images, {o or '—'} objects")
             for sp, (i, o) in f["official"].items()]
    rows += [("loader", n) for n in f.get("notes", [])]
    return "\n".join(["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows])


def stats_header(nb):
    from tools.notebooks.matrice import NOTEBOOKS, READS_HEADERS, palier

    others = [o for o in NOTEBOOKS.values() if o.dataset == nb.dataset and o.role != "stats"]
    links = ", ".join(f"[{o.name}]({o.path.name})" for o in others) or "none"
    colab = f"{COLAB}/notebooks/{nb.path.as_posix()}"
    ann = "M (image headers read)" if nb.dataset in READS_HEADERS else "R"
    q, ref, _ = QUESTIONS.get(nb.dataset, ("—", "—", ""))
    return md(textwrap.dedent("""
        # Statistics of {dataset}

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})

        Presentation and statistical analysis of the dataset, read from the point of view of
        Tiny-YOLO and the embedded target (input 416, 13×13 and 26×26 grids, 256 slots of
        `yolo_post`, INT8). Tasks: {tasks}; milestone:
        [M16](../../docs/tasks/M16-presentation-jeux.md); synthesis:
        [stats-jeux.md](../../docs/tasks/stats-jeux.md). Other notebooks of the dataset:
        {links}.

        Tier: annotations **{ann}** (whole split), pixels **{px}** (`SAMPLE = {sample}`
        images per part). Prerequisites: data `{marker}` — {how}. No weights, no GPU.

        Own question: {q} (see: {ref}).

        @FICHE@

        No computation here: `tools/data_stats.py` computes (`stats.json`, `stats.md`,
        gallery), `tools/figures/donnees.py` plots; each step prints the command. Outputs in
        `{out}/`, never in `results/`. Notebook generated by `python -m tools.notebooks`
        (`make notebooks`): edit `tools/notebooks/gabarits.py`, not this file.
        """).format(dataset=nb.dataset, colab=colab, tasks=_task_links(nb.tasks), links=links,
                    ann=ann, px=palier(STATS_SAMPLE), sample=STATS_SAMPLE,
                    marker=prerequis(nb)["data"], how=how_to_get("data", nb), q=q,
                    ref=ref, out=nb.out_dir).replace("@FICHE@", _fiche_md(nb.dataset)))


def stats_params(nb):
    p = [("DATASET", nb.dataset, "key of DATASETS"),
         ("SPLITS", None, "list of splits; None: train then test splits of the dataset"),
         ("SIZE", None, "S or 'WxH'; None: input of the cfg of the dataset (416 otherwise)"),
         ("RESIZE", "letterbox", "letterbox | stretch (quantities seen by the network)"),
         ("SAMPLE", STATS_SAMPLE, "images read for the pixels, per part; 0: none"),
         ("GALLERY", STATS_GALLERY, "gallery images per split"),
         ("SEED", 0, "draw of the images (pixels, gallery, scatter plots)"),
         ("NET", nb.net, "cfg whose anchors and grids are used for the collisions"),
         ("DEVICE", "cpu", "CPU kernel: nothing runs on GPU"),
         ("DATA_ROOT", None, "dataset root; None: data/<dataset> (Colab: Drive folder)"),
         ("DRIVE_DIR", DRIVE_DIR, "Colab: archives data/<dataset>.tar; None: no Drive"),
         ("OUT", nb.out_dir, "notebook outputs"),
         ("REPO_URL", REPO_URL, "Colab: cloned repository"),
         ("REV", REV, "Colab: revision")]
    return [md("## Parameters"), code(_assign(p), tags=("parameters",))]


def _section(title, text, body):
    return [md(f"## {title}\n\n{textwrap.dedent(text).strip()}"), code(body)]


def stats_body(nb):
    q, ref, qcode = QUESTIONS.get(nb.dataset, ("—", "—", "print('no own question')"))
    cells = _section("Computation (T16.0)", """
        `tools/data_stats.py` reads the annotations of each split (whole), reads the pixels
        of `SAMPLE` images per part (drawn with `SEED`), draws the gallery and plots the
        figures. The quantities "seen by the network" are computed after `RESIZE` to `SIZE`.
        """, """
        import json

        from IPython.display import Image as Img, Markdown, display

        from tools import data_stats as DS
        from tools.notebooks.matrice import SPLIT_IMAGES, palier_stats
        from yolo.models.tiny_yolo import load_cfg

        if SIZE is None:
            _, h, w = load_cfg(NET)["input"]
            SIZE = h if h == w else f"{w}x{h}"
        ann, px = palier_stats(DATASET, SAMPLE)
        print(f"tier: annotations {ann}, pixels {px} ({SAMPLE} images per part); "
              f"input {SIZE} ({RESIZE}); cfg {NET}")
        C.run(C.cmd_data_stats(DATASET, OUT, SPLITS, SIZE, RESIZE, NET, SAMPLE, GALLERY, SEED,
                               data_root=DATA_ROOT))
        STATS = json.loads(Path(OUT, "stats.json").read_text())
        FIG = Path(OUT, "figures")


        def show(name):
            png = FIG / f"{name}.png"
            if png.exists():
                display(Img(filename=str(png)))


        def table(fn):
            text = fn(STATS)
            if text:
                display(Markdown(text))
        """)
    cells += _section("Presentation and gallery (T16.4)", """
        Classes of the dataset and their VOC and COCO mapping (`MAPPINGS`), then `GALLERY`
        images per split, one image per class; ground truth drawn by `tools/detect.py:draw`.
        At a fixed `SEED`, the gallery does not change.
        """, """
        from yolo.data import datasets as D

        rows = ["| class | VOC | COCO |", "|---|---|---|"]
        for c in D.DATASETS[DATASET].classes:
            m = [c if DATASET == f else D.MAPPINGS.get((DATASET, f), {}).get(c, "—")
                 for f in ("voc", "coco")]
            rows.append(f"| {c} | {m[0]} | {m[1]} |")
        display(Markdown("\\n".join(rows)))
        for g, sheet in STATS.get("galerie", {}).get("planches", {}).items():
            if g != "suspectes":
                print(f"gallery: {g}")
                display(Img(filename=str(Path(OUT, sheet))))
        """)
    cells += _section("Counts and classes (T16.5)", """
        Counts per split (annotations, whole split), against the official counts of the
        datasheet; objects per class and co-occurrence.
        """, """
        table(DS.md_synthese)
        chk = DS.check_official(STATS)
        for sp, i, o, (a, b), ok in chk:
            print(f"{sp}: {i} images, {o} objects; official {a or '—'} / {b or '—'}: "
                  f"{'ok' if ok else 'gap (see the notes of the datasheet)'}")
        table(DS.md_classes)
        show("classes")
        show("cooccurrence")
        """)
    cells += _section("Box geometry (T16.6)", """
        Sizes in original and input pixels (letterbox and stretch at `SIZE`), COCO
        categories before and after resizing, share of the objects smaller than a cell of
        each head, box centres.
        """, """
        table(DS.md_geometrie)
        show("tailles")
        show("letterbox_stretch")
        show("centres")
        """)
    cells += _section("Density and collisions (T16.7)", """
        Objects per image against the 256 slots of `yolo_post`; targets lost when two
        objects fall on the same cell and the same anchor (`build_targets`, anchors of `NET`).
        """, """
        table(DS.md_densite)
        show("densite")
        show("collisions")
        """)
    cells += _section("Images (T16.8)", """
        Resolutions on the whole split (annotations); channels, intensities and luminance on
        `SAMPLE` images per part, against VOC2007 test (INT8 scales of M4).
        """, """
        table(DS.md_images)
        show("resolutions")
        show("intensites")
        """)
    cells += _section("Anchors (T16.9)", """
        Boxes of the train group in letterbox input pixels, anchors of the cfg, of VOC, of
        COCO and of the k-means (`tools/kmeans_anchors.py`, same `--size` and `--seed`).
        """, """
        table(DS.md_ancres)
        show("ancres")
        show("ancres_k")
        """)
    cells += _section("Annotation quality (T16.10)", """
        Boxes removed or clipped by the loader, degenerate, duplicates, regions removed;
        the analysis reports, it does not correct (a correction goes through M11).
        """, """
        table(DS.md_qualite)
        sus = DS.whole(STATS).get("qualite", {}).get("suspects", [])
        for d in sus:
            print(f"{d['id']}: score {d['score']} (removed {d['retirées']}, clipped "
                  f"{d['rognées']}, degenerate {d['dégénérées']}, duplicates {d['doublons']})")
        sheet = STATS.get("galerie", {}).get("planches", {}).get("suspectes")
        if sheet:
            display(Img(filename=str(Path(OUT, sheet))))
        """)
    cells += _section("Gap between splits and out-of-domain coverage (T16.11)", """
        Total variation distance between train and test, per quantity; share of the objects
        that have a VOC or COCO class (what the out-of-domain notebooks of T14.5 evaluate).
        """, """
        table(DS.md_ecart)
        show("ecart")
        """)
    cells += [md(f"## Own question of the dataset\n\n{q} (see: {ref})"), code(qcode)]
    cells += [md("## Summary"), code("""
        table(DS.md_synthese)
        print(f"outputs: {OUT}/ (stats.json, stats.md, figures/, galerie/); "
              f"revision {TRACE['rev']}")
        print("equivalent commands:")
        for c in C.HISTORY:
            print("  " + c)
        """)]
    return cells


def render(nb):
    if nb.role == "stats":
        return notebook([stats_header(nb), *stats_params(nb), *environment(nb),
                         *stats_body(nb)])
    body = {"infer": infer_body, "train": train_body, "sweep": sweep_body}[nb.role](nb)
    return notebook([header(nb), *params(nb), *environment(nb), *body], gpu=nb.trains)


# ------------------------------------------------------------- visionneuse de figures

VIEWER = "figures_live.ipynb"  # affichage des figures, hors registre modèle × jeu × rôle
# Figures versionnées, affichées sans exécution (cellules Markdown) ; celles de build/ (non
# versionnées) par la cellule de code, à lancer avec un noyau local.
VIEWER_STATIC = ["docs/tasks/figures", "results/figures"]
VIEWER_DIRS = ["build/figures", "build/notebooks"]


# Contexte d'une famille du registre tools/figures (titres : FAMILY_TITLES), en tête de section.
FAMILY_INTROS = {
    "resultats": "What the project gives, measured: the accuracy (mAP) at the float, "
                 "integer and C-sim stages, the effect of the weight format and of the "
                 "quantization, the cost in cycles and resources of the accelerator, and its "
                 "place among published work. The numbers come from the outputs of `build/` "
                 "and `results/*.md`; none is copied by hand.",
    "modeles": "The two networks of the project, Tiny-YOLOv2 VOC and Tiny-YOLOv3 COCO, seen "
               "from above: layer graph, cost per layer (MACs, parameters, activations), "
               "memory footprint against the budget of the board, anchors, reading of a YOLO "
               "output, and the VOC data used for training and evaluation.",
    "materiel": "The accelerator on the Zynq SoC: the ARM / programmable logic split, the "
                "single MAC engine and its tiling, the streaming alternative, and the chain "
                "that checks each hand-over bit for bit, from float NumPy to the board.",
    "maths": "The whole chain is rewritten by hand in pure NumPy, without PyTorch, OpenCV, "
             "scikit-learn or devkit (ADR 0001). Each figure takes one building block, from "
             "the convolution to the quantization, through its mathematical functions.",
    "reseaux": "The level below \"Models\": each layer and each tensor of Tiny-YOLOv2 VOC and "
               "Tiny-YOLOv3 COCO, taken from `python/yolo/models/specs.py`, up to following "
               "one layer from float to the HLS kernel.",
    "projet": "How the project was built: milestones and timeline (from the commits), "
              "progress of the tasks, code volume, tests and dependencies between "
              "milestones.",
}


# PNG dont le nom ne commence pas par celui de la figure qui les trace (préfixe → figure).
EXTRA_PREFIXES = {"echelles_poids": "calibration"}


def _figure_cells():
    """Cellules Markdown des PNG versionnés, rangés par famille puis par figure du registre
    tools/figures (titre, légende, source, commande), liens `![nom](../chemin)` ; les PNG
    que le registre ne réclame pas finissent dans « Autres figures »."""
    from tools.figures import FIGURES
    from tools.figures.__main__ import FAMILY_TITLES, images_of

    pngs = sorted(p for top in VIEWER_STATIC for p in (ROOT / top).rglob("*.png"))
    # Un PNG revient à la figure au nom le plus long qui le réclame (ancres / ancres_k).
    owner = {}
    for fig in FIGURES.values():
        for p in images_of(fig):
            if p in pngs and len(fig.name) > len(owner.get(p, "")):
                owner[p] = fig.name
    for p in pngs:
        for prefix, name in EXTRA_PREFIXES.items():
            if p not in owner and p.stem.startswith(prefix):
                owner[p] = name

    def link(p):
        return f"![{p.stem}](../{p.relative_to(ROOT).as_posix()})"

    cells = []
    for fam, title in FAMILY_TITLES.items():
        figs = [(f, [p for p in pngs if owner.get(p) == f.name])
                for f in FIGURES.values() if f.family == fam]
        figs = [(f, imgs) for f, imgs in figs if imgs]
        if not figs:
            continue
        toc = [f"- {f.task} — `{f.name}`" for f, _ in figs]
        cells.append(md("\n".join([f"## {title}", "", FAMILY_INTROS.get(fam, ""), "", *toc])))
        for f, imgs in figs:
            lines = [f"### {f.task} — `{f.name}`", "", f.caption, "",
                     f"- Source: `{f.source}`", f"- Regenerate: `{f.command}`", ""]
            for p in imgs:
                lines += ([f"**{p.stem}**", ""] if len(imgs) > 1 else []) + [link(p), ""]
            cells.append(md("\n".join(lines)))
    orphans = [p for p in pngs if p not in owner]
    if orphans:
        lines = ["## Other figures", "",
                 "Versioned PNG files that no figure of the `tools/figures/` registry produces.", ""]
        for p in orphans:
            lines += [f"**{p.relative_to(ROOT).as_posix()}**", "", link(p), ""]
        cells.append(md("\n".join(lines)))
    return cells


def viewer():
    """Notebook d'affichage seul : les PNG versionnés en cellules Markdown (visibles dès
    l'ouverture, dans VS Code comme sur GitHub, sans rien exécuter), puis une cellule
    facultative pour ceux de build/. Régénéré par `make notebooks` après `make figures`."""
    return notebook([
        md("""\
        # Figures

        The project explained in pictures (milestone M13): the architecture of the networks
        and of the accelerator, the measured results, the NumPy reimplementation and the
        history of the project. Each figure is plotted by `tools/figures/` from data of the
        repository; no number is typed by hand.

        **Reading.** One section per family, opened by a paragraph of context and the list
        of its figures. Each figure gives its task (`T13.x`, detailed in
        [M13-figures.md](../docs/tasks/M13-figures.md)), what it shows, the data it reads
        and the command that plots it again. The same gallery exists in Markdown:
        [results/figures.md](../results/figures.md).

        **Display only.** The versioned PNG files (`results/figures/`, `docs/tasks/figures/`)
        are in Markdown cells: nothing to run, in VS Code as on GitHub. Only the last
        section, for `build/`, needs a local kernel.

        **Update.** `make figures` plots the PNG files again, then `make notebooks`
        regenerates this notebook from the disk. It is generated by
        `python -m tools.notebooks`: edit `tools/notebooks/gabarits.py:viewer`, not this
        file.
        """),
        *_figure_cells(),
        md("""\
        ## Unversioned figures (`build/`, local kernel)

        Some figures are not published: those computed on a subset of images (M12 rule, for
        example the per-layer sensitivity) and the outputs of the training and inference
        notebooks. They stay in `build/`, on the PC that produced them; the cell below shows
        them, with a local kernel (not Colab).

        - `DIRS`: folders walked, relative to the repository root;
        - `FILTER`: substring of the path to keep only some images;
        - `WIDTH`: display width in pixels.

        The images are shown from the most recent to the oldest.
        """),
        code(f"""\
        DIRS   = {VIEWER_DIRS!r}  # relative to the repository root
        FILTER = ''  # substring of the path (e.g. 'balayage'); '': everything
        WIDTH  = 900  # display width (px)
        """, tags=("parameters",)),
        code("""\
        from pathlib import Path

        from IPython.display import Image, Markdown, display

        # Repository root: current folder or a parent; on Colab, the clone of the notebooks.
        CANDIDATES = (Path.cwd(), *Path.cwd().parents, Path("/content/EmbeddedComputervision"))
        ROOT = next((d for d in CANDIDATES if (d / "tools" / "figures").is_dir()), None)
        if ROOT is None:
            raise SystemExit(f"repository not found from {Path.cwd()}: open the notebook with "
                             "a local kernel (the PNG files are on the PC, not on Colab)")
        print(f"repository: {ROOT}")
        for d in DIRS:
            pngs = sorted((p for p in (ROOT / d).rglob("*.png") if FILTER in str(p)),
                          key=lambda p: -p.stat().st_mtime)
            print(f"{d}: {len(pngs)} image(s)")
            if pngs:
                display(Markdown(f"## `{d}` ({len(pngs)})"))
            for p in pngs:
                display(Markdown(f"`{p.relative_to(ROOT)}`"))
                display(Image(data=p.read_bytes(), format="png", width=WIDTH))
        """),
    ])


# ------------------------------------------------- analyse transversale des balayages

ANALYSIS = "analyse_balayages.ipynb"  # hors registre modèle × jeu × rôle, comme VIEWER

# (clé de tools.figures.balayages.PLOTS, titre, ce que montre la figure et comment la lire)
ANALYSIS_SECTIONS = [
    ("grilles", "1. Batch × subset grid, dataset by dataset",
     "Each cell is a run: its score, then that score as a percentage of the best run of the "
     "dataset (★). The colour follows that percentage, on the same scale for every dataset: "
     "the *shape* of the grids compares despite mAP values from 2 to 36. An \"all\" column "
     "darker than the \"500 im.\" column means that seeing the whole split pays; a darker "
     "\"batch 32\" row, that a large batch pays."),
    ("classements", "2. Do the datasets rank the runs the same way?",
     "Left, the rank of each run in each dataset (1 at the top): a flat line at the top is "
     "a run that is always good. Colour: the batch (light 8, dark 32); solid line: whole "
     "split, dashes: 500 images. Right, the Spearman rank correlation between the rankings "
     "of two datasets: 1, same order; 0, no link. With 6 runs, |ρ| ≥ 0.83 is needed to "
     "conclude at 5 %; the mAP on 50 images are noisy (M12 rule)."),
    ("effets", "3. Main effects: subset and batch",
     "Left, each arrow goes from the run on 500 images (hollow) to the run on the whole "
     "split (filled), at equal batch: an arrow to the right is a gain. Right, the relative "
     "score per batch, one point per dataset, and the mean over the datasets (diamonds). At "
     "600 iterations, the batch also sets the number of images seen: batch and data volume "
     "do not separate in this grid."),
    ("epoques", "4. Epochs seen: more images, or the same ones again?",
     "The horizontal axis counts the passes over the training data "
     "(batch × iterations / images, log scale). The runs on the whole split (filled) stay "
     "under a few epochs; those on 500 images (hollow, grey area) see the same images 10 to "
     "38 times. If the score rose with the epochs alone, the two clouds would continue each "
     "other; a drop in the grey area is the sign of overfitting."),
    ("perte", "5. Final loss against the score",
     "One panel per dataset (own scales). ρ is the rank correlation between final loss and "
     "score: negative, a low loss goes with a good score; close to zero or positive, the "
     "loss does not predict the mAP. The runs on 500 images (hollow) often have the lowest "
     "losses: they fit a small set, without generalizing."),
    ("courbes", "6. Loss curves",
     "Smoothed training loss, one line per dataset (rows) and per subset (columns), one "
     "colour per batch. The grey area is the burn-in: the learning rate rises there from 0 "
     "to its value, over 500 of the 600 iterations. A run trained in an earlier session has "
     "no log in the notebook: it is read from "
     "`build/notebooks/<dataset>/<model>/runs/<run>/loss.csv` after `make harvest` (harvest "
     "from the Drive), otherwise marked \"no log\"."),
    ("composantes", "7. What is the final loss made of?",
     "Share of each term of the YOLO loss over the last 50 logged iterations: coord "
     "(position and size of the boxes), obj (confidence on the objects), noobj (confidence "
     "on the background), cls (classes). The number on the right is the total loss. A "
     "dominant obj share says that the network hardly knows where the objects are; a cls "
     "share, that it finds them but names them badly."),
    ("classes", "8. AP per class",
     "AP@0.5 of each class for each run (runs sorted from best to worst). The grey cells "
     "are classes with no instance in the 50 evaluated images: their AP does not exist, it "
     "is not a zero. The colour scale is a square root, to tell apart the AP from 1 to 10. A "
     "pale line over its whole length is a class that no configuration learns in 600 "
     "iterations."),
    ("publies", "9. Fine-tuning against the published weights",
     "Best fine-tuned run, and the published Darknet weights (Tiny-YOLOv3 COCO, Tiny-YOLOv2 "
     "VOC) run out of domain on the same 50 images, each dataset at its scale. The published "
     "weights are only evaluated on the classes that have an equivalent in their classes "
     "(\"n cl.\", `MAPPINGS`): their score compares with the fine-tuned run as an "
     "indication only. A fine-tuned run below the published weights says that iterations "
     "are missing."),
    ("cout", "10. Cost of a run",
     "Duration of a 600-iteration run on the Colab GPU against its relative score. The "
     "duration follows the batch (images processed), not the subset. Duration: the one of "
     "the cell, or the sum of the `seconds` column of `loss.csv` for a run completed from "
     "`build/`."),
]


def analysis():
    """Notebook d'analyse des balayages : relit les `_sweep` et `_infer` exécutés (sorties
    versionnées), sans GPU ni données ; tout calcul est dans `tools.notebooks.balayages`,
    tout tracé dans `tools.figures.balayages`."""
    from tools.notebooks.matrice import NOTEBOOKS

    sweeps = [nb for nb in NOTEBOOKS.values() if nb.role == "sweep"]
    links = ", ".join(f"[{nb.dataset}]({nb.path.as_posix()})" for nb in sweeps)
    cells = [
        md(f"""\
        # Analysis of the batch × subset sweeps, all datasets

        The `<model>_sweep` notebooks train six runs per dataset (batches 8, 16 and 32 ×
        first 500 images or the whole split, 600 iterations, COCO init) and compare them
        dataset by dataset. This notebook compares them **across datasets**: which
        configuration wins, consistently or not, and why (images seen, overfitting, dynamics
        and components of the loss, classes learnt), then the gap to the published weights.

        **Source.** The versioned outputs of the sweeps ({links}) and of the inferences of
        the published weights (`tiny-yolov3-coco_infer`, `tiny-yolov2-voc_infer`), read by
        `tools/notebooks/balayages.py`; the figures come from `tools/figures/balayages.py`.
        Nothing is recomputed or typed by hand: a sweep run again and committed shows up
        here after `Run All`. No GPU, no data. The runs with no log in their notebook
        (earlier Colab session) are completed by their `loss.csv` under `build/notebooks/`,
        brought back from the Drive by `make harvest`.

        **Caution.** Tier R: scores on the **first 50 evaluation images**, not publishable
        (M12 rules); a class with one or two instances jumps there from 0 to 100.
        **Score** = mAP@0.5 of the dataset: 11-point mAP, except FLIR (COCO metric), whose
        AP50 is taken; **relative score** = score / best run of the dataset. Context and
        tables: [resultats-balayages.md](../docs/tasks/resultats-balayages.md).

        Generated by `python -m tools.notebooks`: edit
        `tools/notebooks/gabarits.py:analysis`, not this file.
        """),
        md("## Parameters"),
        code("""\
        DATASETS = None  # datasets analysed (e.g. ['voc', 'kitti']); None: every executed sweep
        OUT      = 'build/notebooks/balayages'  # PNG files and data (relative to the repository root)
        SMOOTH   = 25  # smoothing of the loss curves (iterations)
        """, tags=("parameters",)),
        md("## Environment"),
        code("""\
        import os
        import sys
        from pathlib import Path

        from IPython.display import Image, Markdown, display

        CANDIDATES = (Path.cwd(), *Path.cwd().parents, Path("/content/EmbeddedComputervision"))
        ROOT = next((d for d in CANDIDATES if (d / "tools" / "notebooks").is_dir()), None)
        if ROOT is None:
            raise SystemExit(f"repository not found from {Path.cwd()}")
        os.chdir(ROOT)
        sys.path[:0] = [str(ROOT), str(ROOT / "python")]

        from tools.figures import balayages as FB
        from tools.notebooks import balayages as B

        data, pending = B.load_all(datasets=DATASETS)
        if not data:
            raise SystemExit("no executed sweep in notebooks/*/*_sweep.ipynb")
        OBS = B.observations(data)


        def show(paths):
            for p in paths:
                display(Image(data=Path(p).read_bytes(), format="png"))


        print(f"repository: {ROOT}")
        for d in data.values():
            print(f"  {d['label']:10s} {len(d['runs'])} runs, {d['metric']}, revision {d['rev']}, "
                  f"{d['notebook']}")
        for ds, p in pending.items():
            print(f"  {ds:10s} not executed yet ({p})")
        """),
        md("""\
        ## Overview

        One dataset per row: best and worst run, ratio between the two, range of the final
        losses and best published weights. Then every run, from best to worst in each
        dataset.
        """),
        code("""\
        display(Markdown(B.summary_table(data)))
        display(Markdown(B.runs_table(data)))
        """),
    ]
    for key, title, text in ANALYSIS_SECTIONS:
        call = (f"FB.PLOTS[{key!r}](data, OUT, window=SMOOTH)" if key == "courbes"
                else f"FB.PLOTS[{key!r}](data, OUT)")
        cells += [md(f"## {title}\n\n{text}"),
                  code(f"""\
                  show({call})
                  display(Markdown("**Finding.** " + OBS[{key!r}]))
                  """)]
    cells += [
        md("""\
        ## Conclusions

        Computed on the loaded sweeps; they update with them.
        """),
        code("""\
        display(Markdown(B.conclusions(data, pending)))
        print("equivalent command:")
        print(f"  python -m tools.notebooks.balayages --runs --json {OUT}/balayages.json")
        print(f"  python -m tools.figures balayages  # PNG files in docs/tasks/figures/resultats/")
        """),
    ]
    return notebook(cells)


# ------------------------------------------- cross-dataset analysis of the statistics

STATS_ANALYSIS = "analyse_jeux.ipynb"  # outside the model × dataset × role registry

# (key of tools.figures.jeux.PLOTS, title, what the figure shows and how to read it)
STATS_SECTIONS = [
    ("taille", "1. Scale and density",
     "Images and objects per dataset (log scale), then objects per image: median (●) and "
     "maximum (○). The dashed line is the 256 slots of the hardware post-processing "
     "(`yolo_post`, M9.1): an image to its right loses detections on the board. Green: "
     "ground views; indigo: drone views (same colours in every figure)."),
    ("objets", "2. Object size seen by the network",
     "COCO size categories (small < 32², medium < 96², large) at the original resolution "
     "(pale bar) and after letterbox to 416 (dark bar). The shift between the two bars is "
     "what the resize costs: a dataset with wide or large images (KITTI, VisDrone, UAVDT, "
     "AU-AIR) turns most of its objects into small ones."),
    ("cellules", "3. Objects smaller than a grid cell",
     "Share of objects whose longest side is below the stride of each Tiny-YOLOv3 head "
     "(32 px for 13×13, 16 px for 26×26), with letterbox (●, aspect kept) and stretch "
     "(○, aspect lost); right, objects less than 8 px high. A long bar between ○ and ● "
     "says that stretching the image would recover many objects."),
    ("densite", "4. Lost training targets: anchors or crowding?",
     "Share of non-difficult objects that `build_targets` drops, split by cause. Grey: the "
     "best anchor of the box belongs to no head (with the Tiny-YOLOv3 COCO masks 3,4,5 / "
     "1,2,3, anchor 0 = 10×14 px is unused, so the smallest boxes are never trained). "
     "Colour: two objects on the same cell and anchor (only one is kept). The grey part is "
     "fixed by dataset anchors and masks (M11 cfg); the coloured part needs a finer grid."),
    ("classes", "5. Class balance",
     "Share of each class, sorted, one panel per dataset (log scale). Below each panel: the "
     "most frequent class, the most / least frequent ratio and the normalised entropy "
     "(1 = all classes equally frequent). A steep staircase means rare classes the "
     "fine-tuning will barely see."),
    ("images", "6. Pixels against the INT8 calibration images",
     "Mean luminance (bar) and per-channel means (R, G, B ticks), then contrast (mean "
     "channel standard deviation), against VOC (dashed), whose images calibrated the INT8 "
     "scales (M4). Right: the letterbox factor 416 / longest side of the main resolution; "
     "small factors shrink objects the most."),
    ("ancres", "7. Anchors",
     "Left: mean IoU between the training boxes and their best anchor, for the anchors in "
     "use (○), k-means anchors computed on the dataset (●) and the Tiny-YOLOv2 VOC anchors "
     "(×). The gain written on the right is what re-clustering the anchors brings. Right: "
     "the k-means anchors (k = 6) of every dataset against the COCO anchors (□); anchor 0 "
     "(×) is the one no COCO head uses."),
    ("ecart", "8. Train ↔ test gap",
     "Total variation distance between the train and test distributions of each quantity "
     "(0: same histogram, 1: disjoint). Above about 0.1, scores on the test split say less "
     "about the training split, and the reverse."),
    ("couverture", "9. What the published weights can name",
     "Share of objects whose class has an equivalent among the VOC (20) or COCO (80) "
     "classes (`MAPPINGS`): only these objects are scored in the out-of-domain notebooks "
     "(T14.5) of `tiny-yolov2-voc` and `tiny-yolov3-coco`."),
    ("qualite", "10. Annotation quality",
     "Boxes clipped to the image, removed by the loader, under 2 px at the input, "
     "duplicates (IoU > 0.95) and ignore regions (DontCare…) removed by the loader. "
     "Colour: share of the column maximum, to spot the outlier of each column."),
    ("difficulte", "11. Overview: which dataset is hardest for the embedded Tiny-YOLO?",
     "Seven indicators, each normalised from 0 (easiest dataset) to 1 (hardest), and their "
     "mean; datasets sorted by that mean. Raw values are written in the cells. The weights "
     "are equal by choice: the ranking is a reading aid, not a score to optimise."),
]


def stats_analysis():
    """Cross-dataset analysis notebook: reads the executed `_stats` notebooks (versioned
    outputs), without data or GPU; parsing is in `tools.notebooks.jeux`, plots in
    `tools.figures.jeux`."""
    from tools.notebooks.matrice import NOTEBOOKS

    stats = [nb for nb in NOTEBOOKS.values() if nb.role == "stats"]
    links = ", ".join(f"[{nb.dataset}]({nb.path.as_posix()})" for nb in stats)
    cells = [
        md(f"""\
        # Cross-dataset comparison of the dataset statistics

        Each `<ds>_stats` notebook describes one dataset from the point of view of
        Tiny-YOLO and the embedded target (input 416, 13×13 and 26×26 grids, 256 slots of
        `yolo_post`, INT8). This notebook puts them side by side: how many objects and how
        dense, how small once resized, how many training targets are lost and why, how
        balanced the classes are, how far the pixels are from the INT8 calibration images,
        how well the anchors fit, and which dataset is hardest overall.

        **Source.** The versioned outputs of the statistics notebooks ({links}), read by
        `tools/notebooks/jeux.py`; the figures come from `tools/figures/jeux.py`. Nothing
        is recomputed or typed by hand: a statistics notebook re-run and committed shows up
        here after `Run All`. No data, no GPU.

        **Caution.** Annotation counts cover whole splits; pixel statistics cover 200
        images per split. A notebook run without its M11 cfg (`build/m11/cfg/`, absent on
        a fresh Colab runtime) uses the Tiny-YOLOv3 COCO anchors and masks for the lost
        targets and the "anchors in use": the provenance table below says which. A run that
        stopped after its tables without error is used and marked `partial`.
        Synthesis of each dataset: [stats-jeux.md](../docs/tasks/stats-jeux.md).

        Generated by `python -m tools.notebooks`: edit
        `tools/notebooks/gabarits.py:stats_analysis`, not this file.
        """),
        md("## Parameters"),
        code("""\
        DATASETS = None  # datasets analysed (e.g. ['voc', 'kitti']); None: every executed _stats notebook
        OUT      = 'build/notebooks/jeux'  # PNG files (relative to the repository root)
        """, tags=("parameters",)),
        md("## Environment"),
        code("""\
        import os
        import sys
        from pathlib import Path

        from IPython.display import Image, Markdown, display

        CANDIDATES = (Path.cwd(), *Path.cwd().parents, Path("/content/EmbeddedComputervision"))
        ROOT = next((d for d in CANDIDATES if (d / "tools" / "notebooks").is_dir()), None)
        if ROOT is None:
            raise SystemExit(f"repository not found from {Path.cwd()}")
        os.chdir(ROOT)
        sys.path[:0] = [str(ROOT), str(ROOT / "python")]

        from tools.figures import jeux as FJ
        from tools.notebooks import jeux as J

        data, pending = J.load_all(datasets=DATASETS)
        if not data:
            raise SystemExit("no executed notebook in notebooks/*/*_stats.ipynb")
        OBS = J.observations(data)


        def show(paths):
            for p in paths:
                display(Image(data=Path(p).read_bytes(), format="png"))


        print(f"repository: {ROOT}")
        print(f"{len(data)} datasets loaded: {', '.join(d['label'] for d in data.values())}")
        for ds, why in pending.items():
            print(f"  not analysed: {ds} — {why}")
        display(Markdown(J.provenance_table(data)))
        """),
        md("""\
        ## Overview

        One row per dataset (whole dataset, all splits). Sizes are seen by the network
        (letterbox 416); "< fine cell" is the share of objects under the 16 px stride of the
        26×26 head; "of which collisions" is the part of the lost targets due to two objects
        on the same cell and anchor (the rest is anchors, section 4).
        """),
        code("""\
        display(Markdown(J.summary_table(data)))
        """),
    ]
    for key, title, text in STATS_SECTIONS:
        cells += [md(f"## {title}\n\n{text}"),
                  code(f"""\
                  show(FJ.PLOTS[{key!r}](data, OUT))
                  display(Markdown("**Finding.** " + OBS[{key!r}]))
                  """)]
    cells += [
        md("""\
        ## Conclusions

        Computed on the loaded datasets; they update with them.
        """),
        code("""\
        display(Markdown(J.conclusions(data, pending)))
        print("equivalent command:")
        print(f"  python -m tools.notebooks.jeux --json {OUT}/jeux.json --figures {OUT}")
        """),
    ]
    return notebook(cells)


# ------------------------------------------------------------- M36: decision explanation

EXPLAIN = "explain.ipynb"  # outside the model × dataset × role registry (T36.9)


def explain():
    """Explanation notebook (T36.9): runs `tools/explain.py` and shows its maps and tables;
    no computation of its own, like the other notebooks."""
    cells = [
        md("""\
        # Decision explanation (M36)

        Why the detector output a box, and why it missed one, at the float, integer and
        golden stages. Each cell runs a command of `tools/explain.py` in a subprocess and
        shows what it writes: PNG maps in `build/explain/maps/<net>/`, JSON tables in
        `build/explain/`. The full protocol and the results are in
        [results/explain.md](../results/explain.md); the interactive viewer is
        `make explain`.

        Cost (PC CPU, one process): Grad-CAM and saliency about 1 s per map, Integrated
        Gradients 64 steps about 1 min, D-RISE 300 masks 1.5 min (float) to 3 min (integer).

        Generated by `python -m tools.notebooks`: edit
        `tools/notebooks/gabarits.py:explain`, not this file.
        """),
        md("## Parameters"),
        code("""\
        NET     = 'tiny-yolov2-voc'          # tiny-yolov2-voc, tiny-yolov3-coco, tiny-yolov4-coco
        IMAGES  = '000001,000004'            # VOC2007 test ids
        METHODS = 'gradcam,saliency,lrp_int@int,lrp_float'  # method[@stage], see tools/explain.py
        JOBS    = 4                          # processes of the table commands
        N       = 10                         # images of the table commands (reduced runs)
        """, tags=("parameters",)),
        md("## Environment"),
        code("""\
        import json
        import os
        import subprocess
        import sys
        from pathlib import Path

        from IPython.display import Image, Markdown, display

        CANDIDATES = (Path.cwd(), *Path.cwd().parents, Path("/content/EmbeddedComputervision"))
        ROOT = next((d for d in CANDIDATES if (d / "tools" / "explain.py").is_file()), None)
        if ROOT is None:
            raise SystemExit(f"repository not found from {Path.cwd()}")
        os.chdir(ROOT)


        def run(*args):
            cmd = [sys.executable, "tools/explain.py", *args]
            print("$", " ".join(cmd))
            subprocess.run(cmd, check=True)
        """),
        md("""\
        ## Maps of the best matched detection

        One PNG per image: the image, then each method's map over it, the explained box in
        white.
        """),
        code("""\
        run("map", "--nets", NET, "--image", IMAGES, "--methods", METHODS)
        for i in IMAGES.split(","):
            p = ROOT / "build" / "explain" / "maps" / NET / f"{i}.png"
            if p.exists():
                display(Image(filename=str(p)))
        """),
        md("""\
        ## Integer attribution (T36.3)

        Exact split of the accumulators (Σ qW·qx == acc − qb) and the relevance kept per
        layer, then the tables of `results/explain.md` for the reduced run.
        """),
        code("""\
        run("intattr", "--nets", NET, "--n", str(N), "--jobs", str(JOBS))
        run("report")
        text = (ROOT / "results" / "explain.md").read_text()
        a = text.index("<!-- explain:intattr:begin -->")
        b = text.index("<!-- explain:intattr:end -->")
        display(Markdown(text[a:b]))
        """),
        md("""\
        ## Miss causes of the persons (T36.5)

        The cause of each missed person at the float and integer stages, by a fixed order of
        tests (`yolo.explain.misses`).
        """),
        code("""\
        run("misses", "--nets", NET, "--n", str(5 * N), "--jobs", str(JOBS))
        res = json.loads((ROOT / "build" / "explain" / "misses.json").read_text())[NET]
        from collections import Counter
        for st in ("float", "int"):
            print(st, dict(Counter(r[st] or "detected" for r in res)))
        """),
    ]
    return notebook(cells)
