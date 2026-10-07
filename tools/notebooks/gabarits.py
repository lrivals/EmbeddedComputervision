"""Gabarits des notebooks (T14.1, T14.3 à T14.8) : cellules communes, `infer` et `train`.

Structure fixe de chaque notebook : titre, cellule `parameters` (compatible papermill),
cellule d'environnement, corps du gabarit, résumé. Les cellules de code n'appellent que le
code du dépôt (`tools.notebooks.commandes` pour les outils CLI, `yolo.*` pour l'affichage).
Le JSON est écrit sans sorties, cellules en listes de lignes, trié et indenté comme
l'enregistre Jupyter : ouvrir puis enregistrer un notebook non exécuté ne change rien.
"""

import json
import textwrap

from tools.notebooks import ROOT
from tools.notebooks.colab import DRIVE_DIR
from tools.notebooks.commandes import M11_TRAIN
from tools.notebooks.matrice import SPLIT_IMAGES, how_to_get, palier_of, prerequis

REPO_URL = "https://github.com/lrivals/EmbeddedComputervision.git"
REV = "main"
COLAB = "https://colab.research.google.com/github/lrivals/EmbeddedComputervision/blob/main"
ROLE_TITLES = {"infer": "inférence", "train": "entraînement",
               "sweep": "balayage lot × sous-ensemble", "stats": "statistiques du jeu"}
SUBSET_FIRST = 50  # premier passage, palier R (règles de M12)
# Notebooks de référence, réglés pour le split complet (palier N) : leurs mAP se comparent
# à results/ (T14.3 : flottant 56,30 ; T14.4 : entier 55,66, results/map_int8.md).
REFERENCE = {"voc/tiny-yolov2-voc_infer.ipynb": {"SUBSET": 0, "INT8": True}}
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
    if nb.role == "train":
        full = palier_of(nb, 0, M11_TRAIN["iters"])
        cost = (f"palier **{full}** ({M11_TRAIN['iters']} itérations par défaut, "
                "plusieurs heures sur CPU) ; évaluation finale sur `SUBSET` images")
    elif nb.role == "sweep":
        n = len(SWEEP["batches"]) * len(SWEEP["subsets"])
        cost = (f"palier **{palier_of(nb, 0, SWEEP['iters'])}** ({n} runs de "
                f"{SWEEP['iters']} itérations par défaut) ; chaque run est évalué sur "
                "`SUBSET` images, puis comparé aux autres")
    else:
        n = SPLIT_IMAGES.get(nb.dataset)
        full = palier_of(nb)
        cost = (f"palier **{first}** au premier passage (`SUBSET = {SUBSET_FIRST}`), "
                f"**{full}** sur le split complet ({n or 'taille non relevée'} images, "
                "`SUBSET = 0`)")
        if nb.path.as_posix() in REFERENCE:
            cost += (". **Notebook de référence**, réglé sur le split complet (`SUBSET = 0`, "
                     "`INT8 = True`) : mAP attendues 56,30 en flottant (T14.3) et 55,66 en "
                     "entier (T14.4), `results/map_int8.md` ; mettre `SUBSET = "
                     f"{SUBSET_FIRST}` pour un essai rapide")
    req = prerequis(nb)
    reqs = "\n".join(f"- {k} : `{v}` — {how_to_get(k, nb)}" for k, v in req.items()
                     if not (nb.trains and k == "cfg"))
    domain = (f" Poids {nb.family.upper()} hors domaine : seules les classes communes sont "
              "évaluées." if nb.out_of_domain and nb.role == "infer" else "")
    colab = f"{COLAB}/notebooks/{nb.path.as_posix()}"
    return md(textwrap.dedent("""
        # {model} sur {dataset} — {role}

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})

        Rôle : {role} ; {cost}. Tâches : {tasks} ;
        notebooks : [M14](../../docs/tasks/M14-notebooks.md).{domain}

        Prérequis :
        @REQS@

        Chaque étape affiche la commande `python tools/…` qu'elle lance : un résultat se
        reproduit en ligne de commande. Sorties dans `{out}/`, jamais dans `results/`
        (une mAP sur `SUBSET` images n'est pas publiable, règles de M12). Notebook généré par
        `python -m tools.notebooks` (`make notebooks`) : modifier le gabarit
        `tools/notebooks/gabarits.py`, pas ce fichier.
        """).format(model=nb.model, dataset=nb.dataset, role=ROLE_TITLES[nb.role], colab=colab,
                    cost=cost, tasks=_task_links(nb.tasks), domain=domain,
                    out=nb.out_dir).replace("@REQS@", reqs))


# ---------------------------------------------------------------------- paramètres

VISDRONE_NOTE = textwrap.dedent("""

    ### Runs longs VisDrone (M15)

    Un run par dossier : régler `OUT` sur `RUNS_DIR/runs/<nom>` pour que le notebook
    `_infer` le trouve (`RUN = '<nom>'`, `COMPARE`), par exemple :

    | run | réglages |
    |---|---|
    | `long-416` | défauts : `RESIZE_TRAIN = 'stretch'`, `CROP = 0` |
    | `long-416-crop640` | `CROP = 640` ; à évaluer avec `TILES = 640` dans `_infer` |
    | `long-608` | `SIZE = '608'` et `NET` = cfg à 608 (`tools/m11.sh visdrone-prep` avec `SIZE=608`) |

    `RESIZE_TRAIN = 'stretch'` aligne l'entraînement sur l'éval (`RESIZE`) : en letterbox,
    les objets sont 1,3 à 1,8 fois moins hauts qu'à l'éval. Les ancres de la cfg restent
    celles du letterbox : tous les runs s'évaluent avec la même cfg.
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
        raise ValueError(f"retouches sans cible dans {nb.path} : {missing}")
    return cells


def _assign(pairs):
    width = max(len(k) for k, _, _ in pairs)
    return "\n".join(f"{k:{width}s} = {v!r}" + (f"  # {c}" if c else "") for k, v, c in pairs)


def _weights_param(nb):
    if nb.role == "infer" and nb.finetuned:
        return ("WEIGHTS", None, "None : final.weights du run RUN ; ou un chemin")
    if nb.role == "infer":
        return ("WEIGHTS", nb.weights, "poids évalués")
    if nb.role == "sweep":
        return ("WEIGHTS", None, "par run : OUT/runs/<run>/final.weights")
    return ("WEIGHTS", f"{nb.out_dir}/final.weights", "poids produits, évalués à la fin")


def params(nb):
    p = [("NET", nb.net, "nom de CFG_FILES ou cfg (relative à la racine)"),
         ("DATASET", nb.dataset, "clé de DATASETS"),
         ("SPLIT", None, "None : split d'évaluation du jeu"),
         _weights_param(nb),
         ("DEVICE", "gpu" if nb.trains else "cpu",
          "cpu | gpu (entraînement seulement ; le reste sur CPU)"),
         ("SUBSET", SUBSET_FIRST, "images évaluées ; 0 : split complet (palier N)"),
         ("SIZE", nb.size, "entrée LxH (ex. '640x192') ; None : celle de la cfg"),
         ("RESIZE", "stretch", "stretch (référence M12) | letterbox"),
         ("METRIC", None, "None : celle du jeu (coco pour coco et flir)"),
         ("DATA_ROOT", None, "racine du jeu ; None : data/<jeu> (Colab : dossier Drive)")]
    if nb.role == "infer" and nb.finetuned:
        p += [("RUNS_DIR", nb.train_dir, "runs des notebooks _train et _sweep"),
              ("RUN", None, "None : le plus récent ; ex. 'train', 'b16-sall', 'b8-s500'"),
              ("COMPARE", False, "évalue tous les runs et les compare (T14.11)")]
        if _tiled(nb):
            p += [("TILES", 0, "tuiles N px de l'image d'origine (runs à CROP = N) ; "
                   "0 : image entière"),
                  ("OVERLAP", 0.2, "recouvrement des tuiles")]
    if nb.role == "infer":
        p += [("INT8", False, "volet entier : calibration et mAP INT8 (T14.4)"),
              ("CALIB_IMAGES", 500, "images de calibration (split calib du jeu)"),
              ("ACT_HIST", False, "histogrammes d'activations L00-L04 (T11.3)")]
    else:
        base = "tiny-yolov3-voc"
        p += [("BASE", base, "base de make_cfg.py"),
              ("CHANNELS", nb.channels, "canaux de la cfg (1 : thermique)"),
              ("ANCHORS_K", 6, "ancres de la cfg (6 : Tiny-YOLOv3)"),
              ("INIT", "coco", "coco | he | chemin d'un .weights"),
              ("INIT_NET", None, "réseau des poids de INIT s'il diffère de NET")]
        if nb.role == "sweep":
            p += [("BATCHES", SWEEP["batches"], "tailles de lot essayées"),
                  ("TRAIN_SUBSETS", SWEEP["subsets"],
                   "images d'entraînement (n premières) ; 0 : tout le split"),
                  ("SEEDS", SWEEP["seeds"], "graines (train.py --seed) ; T15.2 : [1, 2]"),
                  ("ITERS", SWEEP["iters"], "itérations par run (palier N dès 600)"),
                  ("SKIP_DONE", True, "saute les runs qui ont déjà final.weights")]
        else:
            p += [("ITERS", M11_TRAIN["iters"], "itérations (palier N au-delà de 600)"),
                  ("BATCH", M11_TRAIN["batch"], None)]
        if nb.role == "train" and _tiled(nb):
            p += [("RESIZE_TRAIN", "letterbox", "letterbox | stretch : géométrie à "
                   "l'entraînement (stretch : celle de l'éval, RESIZE)"),
                  ("CROP", 0, "découpes N×N px de l'image d'origine (tuiles) ; 0 : image "
                   "entière"),
                  ("STEPS", "", "itérations où le LR baisse (ex. '4800,5400')"),
                  ("SCALES", "", "facteurs du LR à STEPS (ex. '0.1,0.1')")]
        p += [("LR", M11_TRAIN["lr"], None),
              ("BURN_IN", M11_TRAIN["burn_in"], None),
              ("MULTISCALE", M11_TRAIN["multiscale"], "320-608 tous les 10 lots"),
              ("SAVE_EVERY", M11_TRAIN["save_every"], "None : défaut de train.py (200)"),
              ("WORKERS", M11_TRAIN["workers"], None)]
        if nb.role == "train":
            p += [("QAT", "", "schéma wXaY (ex. 'w4a4') : QAT (T9.3, T12.5)"),
              ("QAT_STEPS", None, "steps.json de quant_lowbit.py (PTQ)"),
              ("ADMM", None, "plan JSON de poids ({} : mixed6 partout, T9.2, T12.6)"),
              ("ADMM_RHO", 1e-3, None),
              ("ADMM_GROWTH", 1.3, None),
              ("ADMM_EVERY", 100, None)]
    p += [("DRIVE_DIR", DRIVE_DIR, "Colab : archives data/<jeu>.tar"
           + (", sorties et reprise dans runs/" if nb.trains else "")
           + " ; None : pas de Drive"),
          ("JOBS", None, "processus des évaluations ; None : un par cœur CPU, plafonné "
           "par la mémoire (runtime TPU de Colab : des dizaines)"),
          ("SHOW", 6, "images affichées"),
          ("OUT", nb.out_dir, "sorties du notebook"),
          ("REPO_URL", REPO_URL, "Colab : dépôt cloné"),
          ("REV", REV, "Colab : révision")]
    ref = REFERENCE.get(nb.path.as_posix(), {})
    if nb.role == "train" and _tiled(nb):
        ref = {**VISDRONE_TRAIN, **ref}
    p = [(k, ref.get(k, v), c) for k, v, c in p]
    return [md("## Paramètres"), code(_assign(p), tags=("parameters",))]


# --------------------------------------------------------------------- environnement

def environment(nb):
    pre = ""
    if nb.role == "infer":
        needs = ("WEIGHTS", "NET", repr(how_to_get("poids", nb)))
        if nb.finetuned:  # T14.11 : poids d'un run de _train ou _sweep
            pre = f"""
        from tools.notebooks import colab, runs

        if COLAB and DRIVE_DIR:  # runs entraînés sur Colab : copiés dans <DRIVE_DIR>/runs/
            env.mount_drive(DRIVE_DIR)
            colab.restore_outputs(RUNS_DIR, DRIVE_DIR)
        RUNS = runs.find_runs(RUNS_DIR)
        print(f"runs de {{RUNS_DIR}} :\\n{{runs.listing(RUNS)}}")
        CHOSEN = None
        if WEIGHTS is None:
            CHOSEN = runs.pick(RUNS_DIR, RUN, hint={needs[2]})
            WEIGHTS, OUT = CHOSEN.weights, f"{{OUT}}/{{CHOSEN.name}}"
            print(f"run choisi : {{CHOSEN.name}} ({{WEIGHTS}})")"""
    elif nb.role == "stats":  # ni poids ni cfg exigés : la cfg manquante est remplacée
        needs = ("()", "None", "''")
    else:
        needs = ("INIT if str(INIT).endswith('.weights') else "
                 "'weights/yolov3-tiny.weights' if INIT == 'coco' else ()", "None", "''")
    post = ""
    if nb.role == "stats":
        post = """
        if str(NET).endswith(".cfg") and not env.restore_cfg(NET):
            print(f"cfg {NET} absente (tools/m11.sh {DATASET}-prep) : ancres et grilles de "
                  "tiny-yolov3-coco pour les collisions")
            NET = "tiny-yolov3-coco\"""".rstrip()
    return [md("""
        ## Environnement

        Local : rien n'est installé ni téléchargé, la cellule vérifie les prérequis et
        s'arrête avec la commande à lancer. Colab : clone du dépôt, installation, CuPy si
        `DEVICE = "gpu"`, poids, puis données par `colab.prepare_data` (archive
        `<DRIVE_DIR>/data/<jeu>.tar` ou téléchargement direct ; ExDark et FLIR : archive
        seulement, faite sur le PC par `tools/get_datasets.sh pack <jeu>`).
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
            # Clone d'une session précédente : remis à REV (notebook et code du même commit).
            subprocess.run(["git", "-C", str(ROOT), "fetch", "-q", "origin", REV], check=True)
            subprocess.run(["git", "-C", str(ROOT), "checkout", "-q", "FETCH_HEAD"], check=True)
            for m in [m for m in sys.modules if m.split(".")[0] in ("tools", "yolo")]:
                del sys.modules[m]  # modules importés avant la mise à jour
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e",
                            f"{{ROOT}}/python[data,plots]"], check=True)
            if DEVICE == "gpu":
                # Runtime CPU : pas de pilote, CuPy dirait « cudaErrorInsufficientDriver ».
                try:
                    smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
                except FileNotFoundError:
                    smi = None
                if smi is None or smi.returncode != 0:
                    raise RuntimeError("runtime Colab sans GPU : créer un nouveau serveur Colab "
                                       "de type GPU (T4, L4 ou A100), ou DEVICE = 'cpu'")
                cuda = smi.stdout.split("CUDA Version:")[-1].split()[0]  # version du pilote
                cupy_pkg = "cupy-cuda13x" if int(cuda.split(".")[0]) >= 13 else "cupy-cuda12x"
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", cupy_pkg],
                               check=True)
        elif ROOT is None:
            raise RuntimeError("racine du dépôt introuvable : ouvrir le notebook depuis notebooks/")
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
    cells = [md("""
        ## Réseau et poids

        `yolo.models.tiny_yolo.build` et `yolo.io.darknet_weights.load_darknet_weights`.
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

        net = build(NET)
        load_darknet_weights(net, WEIGHTS)
        size = input_size(net.net, parse_size(SIZE) if SIZE else None)
        channels = net.net["input"][0]
        family, names = datasets.model_family(DATASET, net.net["classes"])
        print(f"{NET} : {net.net['classes']} classes ({family}), entrée {size_label(size)}, "
              f"{channels} canal(aux) ; poids {WEIGHTS}")
        """), md("""
        ## Détections sur quelques images

        `yolo.infer.pipeline.preprocess` et `detect` (seuil de la démo), boîtes de
        `tools/detect.py:draw`.
        """), code("""
        for s in datasets.load(DATASET, DATA_ROOT, SPLIT)[:SHOW]:
            with Image.open(s["image"]) as img:
                x, wh = preprocess(img, size, RESIZE, "pil", channels)
                boxes, scores, labels = detect(net, x[None], [wh], RESIZE)[0]
                shown = draw(img, boxes, scores, labels, names)
            shown.thumbnail((640, 640))
            print(f"{s['id']} : {len(labels)} détections")
            display(shown)
        """)]
    if nb.out_of_domain:
        cells += [md("""
            ## Correspondance des classes (hors domaine, T11.2)

            `MAPPINGS` de `yolo.data.datasets`, telle quelle ; les classes du jeu sans
            équivalent sont retirées des vérités, les détections des autres classes du
            modèle sont ignorées (`eval_view`).
            """), code("""
            view = datasets.eval_view(DATASET, net.net["classes"])
            mapping = datasets.MAPPINGS[(DATASET, family)]
            rows = ["| classe du jeu | classe du modèle |", "|---|---|"]
            rows += [f"| {c} | {mapping.get(c, '— (sans équivalent, ignorée)')} |"
                     for c in datasets.DATASETS[DATASET].classes]
            display(Markdown("\\n".join(rows)))
            print(f"classes évaluées ({len(view.names)}) : {', '.join(view.names)}")
            print(f"classes du modèle ignorées : {len(names) - len(view.names)}")
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
        ## mAP flottante (T14.3)

        `tools/eval_voc.py`, métrique du jeu (`Dataset.metric` ; COCO et FLIR : AP@[.5:.95]
        et AP50). Courbes PR du mode automatique de M13 (métrique VOC).
        """ + ("""
        Le run choisi est évalué dans `RUNS_DIR/eval/<run>/`, comme par la comparaison
        plus bas, qui réutilise ce résultat : une mAP déjà calculée pour ce `SUBSET` n'est
        pas recalculée.
        """ if nb.finetuned else "")), code("""
        print(f"{SUBSET or 'toutes les'} images : palier "
              f"{palier(SUBSET or SPLIT_IMAGES.get(DATASET))} (règles de M12)")"""
        + evaluate + """
        display(Markdown(map_float.read_text()))
        for png in sorted(""" + pr + """.glob("pr*.png")):
            display(Image.open(png))
        """)]
    if nb.finetuned:
        cells += [md("""
            ## Comparaison des runs (T14.11)

            Avec `COMPARE = True`, chaque run trouvé dans `RUNS_DIR` (notebook `_train` et
            balayage `_sweep`) est évalué sur les mêmes `SUBSET` images ; une mAP déjà
            calculée pour ce `SUBSET` est réutilisée. Le run évalué plus haut se choisit par
            `RUN`.
            """), code(_tiles_args(nb, """
            if COMPARE:
                rows = [runs.evaluate(r, NET, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                                      DATA_ROOT, out_dir=f"{RUNS_DIR}/eval/{r.name}", jobs=JOBS@TILES@)
                        for r in RUNS]
                display(Markdown(runs.table(rows)))
            else:
                print("COMPARE = False : comparaison sautée ; runs trouvés :")
                print(runs.listing(RUNS))
            """))]
    cells += [md("""
        ## Volet entier (T14.4)

        Activé par `INT8 = True`. Calibration par `tools/calibrate.py` sur le split `calib`
        du jeu (échelles et rapport dans `OUT`, pas dans `results/`), puis
        `tools/eval_quant.py --variants float,int`. Sur le split complet (`SUBSET = 0`),
        palier **N** : compter 12 à 25 min pour VOC2007 test en plus de la calibration.
        """), code("""
        map_int = None
        if INT8:
            if not SUBSET:
                print("SUBSET = 0 : split complet, palier N (plusieurs dizaines de minutes)")
            calib = Path(OUT) / "calib.json"
            C.run(C.cmd_calibrate(NET, WEIGHTS, DATASET, calib, Path(OUT) / "calibration.md",
                                  CALIB_IMAGES, data_root=DATA_ROOT))
            map_int = Path(OUT) / f"map_{RESIZE}_{SUBSET or 'complet'}.json"
            C.run(C.cmd_eval_quant(NET, map_int, "float,int", SUBSET, JOBS, DATASET,
                                   weights=WEIGHTS, calib=calib, metric=METRIC, split=SPLIT,
                                   size=SIZE, data_root=DATA_ROOT))
            d = json.loads(map_int.read_text())
            fl, it = d["results"]["float"], d["results"]["int"]
            rows = ["| classe | flottant | entier | écart |", "|---|---|---|---|"]
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
            print("INT8 = False : volet entier sauté")
        """), md("## Résumé"), code("""
        print(map_float.read_text())
        if map_int:
            d = json.loads(map_int.read_text())
            for v, r in d["results"].items():
                print(f"{v:6s} mAP {100 * r['map']:.2f} ({d['images']} images)")
        print(f"sorties : {OUT}/ ; révision {TRACE['rev']}")
        print("commandes équivalentes :")
        for c in C.HISTORY:
            print("  " + c)
        """)]
    return cells


# ---------------------------------------------------------------------------- train

def _prep_preview_cells():
    """Préparation (ancres, cfg) et aperçu des cibles, communs à `train` et `sweep`."""
    return [md("""
        ## Préparation (jeux hors VOC)

        Ancres k-means sur le split d'entraînement (`tools/kmeans_anchors.py`, table dans
        `OUT`), puis cfg à N classes (`tools/make_cfg.py`), comme `tools/m11.sh <jeu>-prep`.
        Sautée si la cfg existe déjà.
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
            print(f"{NET} : cfg du dépôt, pas de préparation")
        elif Path(NET).exists():
            print(f"{NET} existe : préparation sautée")
        else:
            anchors = Path(OUT) / f"anchors_{DATASET}{'_' + str(SIZE) if SIZE else ''}.md"
            C.run(C.cmd_anchors(DATASET, SIZE, anchors, DATA_ROOT))
            C.run(C.cmd_make_cfg(BASE, DATASET, C.anchors_k(anchors, ANCHORS_K), NET, SIZE,
                                 CHANNELS))
        """), md("""
        ## Aperçu des cibles

        Quelques images augmentées (`yolo.data.loader.VOCDataset`, donc `yolo.data.augment`)
        avec leurs boîtes : contrôle des annotations avant des heures de calcul.
        """), code("""
        cfg = load_cfg(NET)
        size = input_size(cfg, parse_size(SIZE) if SIZE else None)
        channels = cfg["input"][0]
        names = datasets.DATASETS[DATASET].classes
        samples = datasets.load(DATASET, DATA_ROOT, role="train")
        print(f"{len(samples)} images d'entraînement, entrée {size_label(size)}, {channels} canal(aux)")
        from yolo.data.loader import VOCDataset

        preview = VOCDataset(samples[:SHOW], train=True, channels=channels)
        for k in range(len(preview)):
            chw, boxes, labels = preview.load(k, size, np.random.default_rng(k))
            hwc = (np.clip(chw.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
            img = Image.fromarray(hwc[..., 0] if channels == 1 else hwc)
            display(draw(img, boxes, np.ones(len(labels)), labels, names))
        """)]


def train_body(nb):
    qat = _assign([(k, v, None) for k, v in M12_QAT.items()])
    admm = _assign([(k, v, None) for k, v in M12_ADMM.items()])
    variants = ""
    if nb.dataset == "voc":
        variants = textwrap.dedent("""

        ### Variantes QAT et ADMM (T14.7)

        Paramètres du profil `tools/m12.sh qat ref` (T12.5) ; `QAT_STEPS` vient de la PTQ
        `tools/quant_lowbit.py --scheme w4a4` (T12.4) :

        ```python
        @QAT@
        ```

        Paramètres du profil `tools/m12.sh admm ref` (T12.6) ; le plan `{}` (mixed6
        partout) est écrit s'il manque :

        ```python
        @ADMM@
        ```

        L'évaluation passe alors par `tools/quant_lowbit.py` puis
        `tools/eval_quant.py --variants int --model-dir`, pas par la mAP flottante.
        """).replace("@QAT@", qat).replace("@ADMM@", admm)
    affinage = textwrap.dedent("""
        ## Affinage

        `tools/train.py` dans un sous-processus dont la sortie est relayée ligne à ligne ; une
        interruption du noyau laisse le dernier `checkpoint.npz` (tous les `SAVE_EVERY`), et
        relancer la cellule reprend (`--resume`). Sur Colab avec `DRIVE_DIR`, `OUT` est copié
        dans `<DRIVE_DIR>/runs/` toutes les 10 min et en fin de cellule
        (`colab.sync_outputs`), puis restauré en début de session (`colab.restore_outputs`) :
        une session coupée reprend au dernier checkpoint copié, sur le même backend
        (T12.11, T14.8).
        """) + variants + (VISDRONE_NOTE if _tiled(nb) else "")
    return _retouch([*_prep_preview_cells(), md(affinage), code("""
        import threading

        from tools.notebooks import colab, runs

        RUN = Path(OUT)
        SYNC = COLAB and bool(DRIVE_DIR)
        if SYNC:  # reprise après coupure de session : sorties de la session précédente
            env.mount_drive(DRIVE_DIR)
            colab.restore_outputs(OUT, DRIVE_DIR)
        RUN.mkdir(parents=True, exist_ok=True)
        if ADMM and not Path(ADMM).exists():
            Path(ADMM).parent.mkdir(parents=True, exist_ok=True)
            Path(ADMM).write_text("{}\\n")  # mixed6 partout, comme tools/m12.sh admm
        seconds = estimate(ITERS, BATCH, DEVICE, RUN / "loss.csv")
        print(f"{ITERS} itérations × {BATCH} images : "
              + (f"≈ {seconds / 3600:.1f} h sur {DEVICE}" if seconds else
                 f"durée non mesurée sur {DEVICE} (T12.11-e)"))
        resume = (RUN / "checkpoint.npz").exists()
        runs.write_meta(RUN, batch=BATCH, subset=0, iters=ITERS, lr=LR, device=DEVICE,
                        net=NET, dataset=DATASET, qat=QAT, admm=bool(ADMM), rev=TRACE["rev"])
        stop = threading.Event()


        def sync_loop(every=600):  # copie sur Drive toutes les 10 min, checkpoint compris
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
        ## Courbes

        Perte et taux d'apprentissage de `loss.csv` (mode automatique de M13) ; relancer la
        cellule pendant un entraînement long redessine jusqu'au dernier point écrit.
        """), code("""
        C.run(C.cmd_figures("train", RUN))
        for png in sorted((RUN / "figures").glob("*.png")):
            display(Image.open(png))
        """), md("""
        ## Évaluation

        `final.weights` par `tools/eval_voc.py` (gabarit `infer`, T14.3) ; en QAT ou ADMM,
        modèle basse précision par `tools/quant_lowbit.py` puis `tools/eval_quant.py`.
        """), code("""
        if QAT or ADMM:
            C.run(C.cmd_quant_lowbit(NET, RUN / "checkpoint.npz", RUN / "model", QAT, bool(ADMM)))
            result = RUN / "map.json"
            C.run(C.cmd_eval_quant(NET, result, "int", SUBSET, JOBS, DATASET,
                                   model_dir=RUN / "model", metric=METRIC, split=SPLIT,
                                   size=SIZE, data_root=DATA_ROOT))
            d = json.loads(result.read_text())
            print(f"mAP entière {100 * d['results']['int']['map']:.2f} ({d['images']} images)")
        else:
            result = RUN / "map_float.md"
            result.unlink(missing_ok=True)
            C.run(C.cmd_eval_voc(NET, RUN / "final.weights", DATASET, RESIZE, SUBSET, METRIC,
                                 SPLIT, SIZE, DATA_ROOT, out=RUN / "dets", markdown=result,
                                 jobs=JOBS))
            display(Markdown(result.read_text()))
        """), md("""
        ## Vitesse du backend (T12.11-e)

        Secondes par image des dernières itérations (`loss.csv`) ; le lot maximal qui tient
        en mémoire GPU se trouve en relançant avec `BATCH` croissant. Un chiffre issu de
        poids entraînés sur GPU indique le backend.
        """), code("""
        rows = list(csv.DictReader((RUN / "loss.csv").open())) if (RUN / "loss.csv").exists() else []
        if rows:
            last = rows[-50:]
            s = sum(float(r["seconds"]) for r in last) / len(last) / BATCH
            print(f"{s:.3f} s/image sur {DEVICE}, lot {BATCH}, {len(rows)} itérations journalisées")
        """), md("## Résumé"), code("""
        print(f"sorties : {RUN}/ (checkpoint.npz, loss.csv, final.weights, figures/) ; "
              f"backend {DEVICE} ; révision {TRACE['rev']}")
        print(f"évaluation : {result}")
        print("commandes équivalentes :")
        for c in C.HISTORY:
            print("  " + c)
        """)], nb)


def sweep_body(nb):
    return [*_prep_preview_cells(), md("""
        ## Plan du balayage (T14.10)

        Un run par case de la grille `BATCHES × TRAIN_SUBSETS × SEEDS`, tous les autres
        paramètres égaux (ceux de `tools/m11.sh <jeu>-train`, `ITERS` à part).
        `TRAIN_SUBSETS` prend les n premières images du split d'entraînement
        (`train.py --subset`), 0 tout le split. Une graine non nulle (`train.py --seed` :
        têtes, tirage des lots, multiscale) suffixe le run (`b32-sall-g1`) ; la graine 0
        garde le nom court.
        À itérations égales, un lot plus grand voit plus d'images : la colonne « époques »
        le montre. Le taux d'apprentissage n'est pas ajusté au lot.
        """), code("""
        import threading

        from tools.notebooks import colab, runs

        PLAN = [(b, s, g, runs.run_name(b, s, g)) for b in BATCHES for s in TRAIN_SUBSETS
                for g in SEEDS]
        rows = ["| run | lot | images | époques | durée estimée |", "|---|---|---|---|---|"]
        total = 0
        for b, s, g, name in PLAN:
            n = min(s, len(samples)) if s else len(samples)
            sec = estimate(ITERS, b, DEVICE)
            total += sec or 0
            done = (Path(OUT) / "runs" / name / "final.weights").exists()
            rows.append(f"| {name}{' (fait)' if done else ''} | {b} | {n} | "
                        f"{ITERS * b / n:.2f} | "
                        + (f"{sec / 3600:.1f} h |" if sec else "non mesurée |"))
        display(Markdown("\\n".join(rows)))
        print(f"{len(PLAN)} runs de {ITERS} itérations"
              + (f", ≈ {total / 3600:.1f} h sur {DEVICE} au total" if total else ""))
        """), md("""
        ## Entraînements

        Chaque run va dans `OUT/runs/<run>/` (`b16-sall`, `b8-s500`…), avec son `run.json`.
        Relancer la cellule reprend le run interrompu (`--resume`) et saute ceux qui sont
        finis (`SKIP_DONE`). Sur Colab avec `DRIVE_DIR`, `OUT` est copié dans
        `<DRIVE_DIR>/runs/` toutes les 10 min et en fin de cellule (T14.8).
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
                    print(f"{name} : déjà entraîné, sauté")
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
        ## Comparaison

        mAP flottante de chaque run sur les mêmes `SUBSET` images (`tools/eval_voc.py`,
        dans `OUT/eval/<run>/`, cache partagé avec la comparaison des notebooks `_infer` :
        réutilisée si déjà calculée), perte finale et vitesse ; courbes de perte superposées
        (`tools.figures.resultats.plot_training`, M13). Les notebooks d'inférence
        `<modèle>_infer` lisent ces runs (`RUN`, `COMPARE`).
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
        """), md("## Résumé"), code("""
        print(f"runs : {OUT}/runs/ ; backend {DEVICE} ; révision {TRACE['rev']}")
        best = max((r for r in RESULTS if r["mAP"] is not None), key=lambda r: r["mAP"],
                   default=None)
        if best:
            print(f"meilleur run sur {SUBSET or 'toutes les'} images : {best['run']} "
                  f"(mAP {best['mAP']:.2f}) ; inférence : RUN = {best['run']!r}")
        print("commandes équivalentes :")
        for c in C.HISTORY:
            print("  " + c)
        """)]


# ---------------------------------------------------------------------------- stats (M16)

STATS_SAMPLE = 200   # images lues pour les pixels (palier selon SAMPLE)
STATS_GALLERY = 8    # images de la galerie par split

# Question propre à chaque jeu (docs/tasks/M16-presentation-jeux.md#questions-par-jeu) :
# (question, renvoi, cellule de code). Les cellules n'appellent que tools/data_stats.py.
QUESTIONS = {
    "voc": ("Les classes à faible AP (bottle, pottedplant) sont-elles rares ou petites ?",
            "T13.7, T13.13", """
        ap = DS.voc_ap()
        if ap:
            print(f"AP flottante par classe : {ap['source']} (Tiny-YOLOv2 VOC, VOC2007 test)")
        display(Markdown(DS.md_par_classe(STATS, ap.get("ap") if ap else None)))
        """),
    "coco": ("Quelle part des objets est petite (< 32²) une fois réduite à 416 ?", "T11.1", """
        g = DS.whole(STATS)["geometrie"]
        for mode in ("orig", "letterbox", "stretch"):
            c = g[mode]["coco"]
            print(f"{mode:9s} : petits {100 * c[0] / max(sum(c), 1):.1f} %, moyens "
                  f"{100 * c[1] / max(sum(c), 1):.1f} %, grands {100 * c[2] / max(sum(c), 1):.1f} %")
        if DS.group(STATS, "test").get("comptes"):
            print(f"iscrowd (test) : {DS.group(STATS, 'test')['comptes']['crowd']}")
        """),
    "kitti": ("`letterbox` ou `stretch` : combien d'objets passent sous 8 px de haut ?",
              "T11.4, T15", """
        g = DS.whole(STATS)["geometrie"]
        for mode in ("letterbox", "stretch"):
            print(f"{mode:9s} {g['size'][1]}×{g['size'][0]} : hauteur < 8 px "
                  f"{100 * g[mode]['h_under']['8']:.1f} %, < 16 px {100 * g[mode]['h_under']['16']:.1f} %")
        # Entrée non carrée de T11.4 : géométrie seule, sans pixels.
        out_nc = f"{OUT}/640x192"
        C.run(C.cmd_data_stats(DATASET, out_nc, SPLITS, "640x192", "letterbox", NET,
                               only=["geometrie"], data_root=DATA_ROOT, figures=False))
        display(Markdown(DS.md_geometrie(json.loads(Path(out_nc, "stats.json").read_text()))))
        """),
    "visdrone": ("Part des objets plus petits qu'une cellule 26×26 ; collisions de cibles.",
                 "T11.5, T15", """
        g = DS.whole(STATS)["geometrie"][STATS["params"]["resize"]]
        print("plus petits qu'une cellule : " + ", ".join(f"{k} {100 * v:.1f} %"
                                                           for k, v in g["fit_cell"].items()))
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    "flir": ("Distribution des niveaux thermiques, et ce qu'en garde l'INT8 de L00.", "T11.7", """
        px = DS.whole(STATS).get("images", {}).get("pixels")
        if px:
            lv = px["levels"]
            print(f"{px['images']} images, {px['channels']} canal(aux) ; niveaux 8 bits occupés "
                  f"{lv['8bit']} (99 % des pixels sur {lv['8bit_99']}) → {lv['int8']} niveaux "
                  f"INT8 à l'échelle d'entrée {lv['input_scale']:.5f} (1/127, yolo.quant)")
        else:
            print("SAMPLE = 0 : pas de lecture de pixels")
        """),
    "exdark": ("Luminosité par type d'éclairage, et écart aux images de calibration VOC.",
               "T11.3", """
        light = STATS.get("exdark_light")
        if light:
            rows = ["| éclairage | images | luminance moy. |", "|---|---|---|"]
            rows += [f"| {k} | {v['images']} | {v['lum_mean']:.1f} |" for k, v in light.items()]
            display(Markdown("\\n".join(rows)))
        ref = STATS.get("voc_ref")
        if ref:
            print(f"VOC2007 test ({ref['images']} images) : luminance moyenne {ref['lum_mean']:.1f}")
        show("eclairage")
        """),
    "crowdhuman": ("Objets par image face aux 256 emplacements ; part d'objets occultés.",
                   "T11.6", """
        for name, a in DS._parts(STATS):
            d, c = a.get("densite"), a.get("comptes")
            if d and c:
                print(f"{name:12s} : {c['images']} images, > 256 objets {d['over']['256']} "
                      f"({100 * d['over']['256'] / max(c['images'], 1):.2f} %), > 1 024 "
                      f"{d['over']['1024']} ; difficult (mask, ignore) "
                      f"{100 * c['difficult'] / max(c['objects'], 1):.1f} % des boîtes")
        """),
    # Jeux drone et thermiques de M18 (docs/tasks/M18-jeux-drone.md#questions-par-jeu).
    "auair": ("Que reste-t-il des objets d'une trame 1920×1080 réduite à 416, classe par "
              "classe ?", "T18.1", """
        g = DS.whole(STATS)["geometrie"][STATS["params"]["resize"]]
        print("plus petits qu'une cellule : " + ", ".join(f"{k} {100 * v:.1f} %"
                                                           for k, v in g["fit_cell"].items()))
        print(f"hauteur < 8 px : {100 * g['h_under']['8']:.1f} %")
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    "dronevehicle": ("Véhicules par image face aux 256 emplacements, et leur taille à 416.",
                     "T18.2", """
        g = DS.whole(STATS)["geometrie"][STATS["params"]["resize"]]
        print("plus petits qu'une cellule : " + ", ".join(f"{k} {100 * v:.1f} %"
                                                           for k, v in g["fit_cell"].items()))
        for name, a in DS._parts(STATS):
            d, c = a.get("densite"), a.get("comptes")
            if d and c:
                print(f"{name:12s} : {c['images']} images, > 256 objets {d['over']['256']}")
        """),
    "hituav": ("Distribution des niveaux thermiques, et ce qu'en garde l'INT8 de L00.",
               "T18.3", """
        px = DS.whole(STATS).get("images", {}).get("pixels")
        if px:
            lv = px["levels"]
            print(f"{px['images']} images, {px['channels']} canal(aux) ; niveaux 8 bits occupés "
                  f"{lv['8bit']} (99 % des pixels sur {lv['8bit_99']}) → {lv['int8']} niveaux "
                  f"INT8 à l'échelle d'entrée {lv['input_scale']:.5f} (1/127, yolo.quant)")
        else:
            print("SAMPLE = 0 : pas de lecture de pixels")
        display(Markdown(DS.md_par_classe(STATS)))
        """),
    "uavdt": ("`letterbox` ou `stretch` pour du 1024×540 : combien d'objets passent sous 8 px "
              "de haut ?", "T18.4", """
        g = DS.whole(STATS)["geometrie"]
        for mode in ("letterbox", "stretch"):
            print(f"{mode:9s} {g['size'][1]}×{g['size'][0]} : hauteur < 8 px "
                  f"{100 * g[mode]['h_under']['8']:.1f} %, < 16 px {100 * g[mode]['h_under']['16']:.1f} %")
        display(Markdown(DS.md_par_classe(STATS)))
        """),
}


def _fiche_md(dataset):
    from tools import data_stats

    f = data_stats.FICHES.get(dataset)
    if not f:
        return "Fiche absente de `FICHES` (tools/data_stats.py)."
    rows = [("source", f["source"]), ("version", f["version"]), ("licence", f["license"]),
            ("capteur", f["sensor"]), ("résolution typique", f["resolution"])]
    rows += [(f"officiel `{sp}`", f"{i or '—'} images, {o or '—'} objets")
             for sp, (i, o) in f["official"].items()]
    rows += [("chargeur", n) for n in f.get("notes", [])]
    return "\n".join(["| | |", "|---|---|"] + [f"| {k} | {v} |" for k, v in rows])


def stats_header(nb):
    from tools.notebooks.matrice import NOTEBOOKS, READS_HEADERS, palier

    others = [o for o in NOTEBOOKS.values() if o.dataset == nb.dataset and o.role != "stats"]
    links = ", ".join(f"[{o.name}]({o.path.name})" for o in others) or "aucun"
    colab = f"{COLAB}/notebooks/{nb.path.as_posix()}"
    ann = "M (en-têtes d'image lus)" if nb.dataset in READS_HEADERS else "R"
    q, ref, _ = QUESTIONS.get(nb.dataset, ("—", "—", ""))
    return md(textwrap.dedent("""
        # Statistiques de {dataset}

        [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab})

        Présentation et analyse statistique du jeu, lues du point de vue de Tiny-YOLO et de la
        cible embarquée (entrée 416, grilles 13×13 et 26×26, 256 emplacements de `yolo_post`,
        INT8). Tâches : {tasks} ; jalon : [M16](../../docs/tasks/M16-presentation-jeux.md) ;
        synthèse : [stats-jeux.md](../../docs/tasks/stats-jeux.md). Autres notebooks du jeu :
        {links}.

        Palier : annotations **{ann}** (split entier), pixels **{px}** (`SAMPLE = {sample}`
        images par partie). Prérequis : données `{marker}` — {how}. Ni poids ni GPU.

        Question propre : {q} (renvoi : {ref}).

        @FICHE@

        Aucun calcul ici : `tools/data_stats.py` calcule (`stats.json`, `stats.md`, galerie),
        `tools/figures/donnees.py` trace ; chaque étape affiche la commande. Sorties dans
        `{out}/`, jamais dans `results/`. Notebook généré par `python -m tools.notebooks`
        (`make notebooks`) : modifier `tools/notebooks/gabarits.py`, pas ce fichier.
        """).format(dataset=nb.dataset, colab=colab, tasks=_task_links(nb.tasks), links=links,
                    ann=ann, px=palier(STATS_SAMPLE), sample=STATS_SAMPLE,
                    marker=prerequis(nb)["données"], how=how_to_get("données", nb), q=q,
                    ref=ref, out=nb.out_dir).replace("@FICHE@", _fiche_md(nb.dataset)))


def stats_params(nb):
    p = [("DATASET", nb.dataset, "clé de DATASETS"),
         ("SPLITS", None, "liste de splits ; None : splits train puis test du jeu"),
         ("SIZE", None, "S ou 'LxH' ; None : entrée de la cfg du jeu (416 sinon)"),
         ("RESIZE", "letterbox", "letterbox | stretch (grandeurs vues par le réseau)"),
         ("SAMPLE", STATS_SAMPLE, "images lues pour les pixels, par partie ; 0 : aucune"),
         ("GALLERY", STATS_GALLERY, "images de la galerie par split"),
         ("SEED", 0, "tirage des images (pixels, galerie, nuages)"),
         ("NET", nb.net, "cfg dont les ancres et grilles servent aux collisions"),
         ("DEVICE", "cpu", "noyau CPU : rien ne tourne sur GPU"),
         ("DATA_ROOT", None, "racine du jeu ; None : data/<jeu> (Colab : dossier Drive)"),
         ("DRIVE_DIR", DRIVE_DIR, "Colab : archives data/<jeu>.tar ; None : pas de Drive"),
         ("OUT", nb.out_dir, "sorties du notebook"),
         ("REPO_URL", REPO_URL, "Colab : dépôt cloné"),
         ("REV", REV, "Colab : révision")]
    return [md("## Paramètres"), code(_assign(p), tags=("parameters",))]


def _section(title, text, body):
    return [md(f"## {title}\n\n{textwrap.dedent(text).strip()}"), code(body)]


def stats_body(nb):
    q, ref, qcode = QUESTIONS.get(nb.dataset, ("—", "—", "print('pas de question propre')"))
    cells = _section("Calcul (T16.0)", """
        `tools/data_stats.py` lit les annotations de chaque split (en entier), lit les pixels
        de `SAMPLE` images par partie (tirées avec `SEED`), dessine la galerie et trace les
        figures. Les grandeurs « vues par le réseau » sont calculées après `RESIZE` à `SIZE`.
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
        print(f"palier : annotations {ann}, pixels {px} ({SAMPLE} images par partie) ; "
              f"entrée {SIZE} ({RESIZE}) ; cfg {NET}")
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
    cells += _section("Présentation et galerie (T16.4)", """
        Classes du jeu et leur correspondance VOC et COCO (`MAPPINGS`), puis `GALLERY` images
        par split, une image par classe ; vérités terrain dessinées par `tools/detect.py:draw`.
        À `SEED` fixé, la galerie ne change pas.
        """, """
        from yolo.data import datasets as D

        rows = ["| classe | VOC | COCO |", "|---|---|---|"]
        for c in D.DATASETS[DATASET].classes:
            m = [c if DATASET == f else D.MAPPINGS.get((DATASET, f), {}).get(c, "—")
                 for f in ("voc", "coco")]
            rows.append(f"| {c} | {m[0]} | {m[1]} |")
        display(Markdown("\\n".join(rows)))
        for g, sheet in STATS.get("galerie", {}).get("planches", {}).items():
            if g != "suspectes":
                print(f"galerie : {g}")
                display(Img(filename=str(Path(OUT, sheet))))
        """)
    cells += _section("Comptes et classes (T16.5)", """
        Comptes par split (annotations, split entier), face aux effectifs officiels de la
        fiche ; objets par classe et co-occurrence.
        """, """
        table(DS.md_synthese)
        chk = DS.check_official(STATS)
        for sp, i, o, (a, b), ok in chk:
            print(f"{sp} : {i} images, {o} objets ; officiel {a or '—'} / {b or '—'} : "
                  f"{'ok' if ok else 'écart (voir les notes de la fiche)'}")
        table(DS.md_classes)
        show("classes")
        show("cooccurrence")
        """)
    cells += _section("Géométrie des boîtes (T16.6)", """
        Tailles en pixels d'origine et d'entrée (letterbox et stretch à `SIZE`), catégories
        COCO avant et après redimensionnement, part des objets plus petits qu'une cellule de
        chaque tête, centres des boîtes.
        """, """
        table(DS.md_geometrie)
        show("tailles")
        show("letterbox_stretch")
        show("centres")
        """)
    cells += _section("Densité et collisions (T16.7)", """
        Objets par image face aux 256 emplacements de `yolo_post` ; cibles perdues quand deux
        objets tombent sur la même cellule et la même ancre (`build_targets`, ancres de `NET`).
        """, """
        table(DS.md_densite)
        show("densite")
        show("collisions")
        """)
    cells += _section("Images (T16.8)", """
        Résolutions sur le split entier (annotations) ; canaux, intensités et luminance sur
        `SAMPLE` images par partie, face à VOC2007 test (échelles INT8 de M4).
        """, """
        table(DS.md_images)
        show("resolutions")
        show("intensites")
        """)
    cells += _section("Ancres (T16.9)", """
        Boîtes du groupe train en pixels d'entrée letterbox, ancres de la cfg, de VOC, de
        COCO et du k-means (`tools/kmeans_anchors.py`, mêmes `--size` et `--seed`).
        """, """
        table(DS.md_ancres)
        show("ancres")
        show("ancres_k")
        """)
    cells += _section("Qualité des annotations (T16.10)", """
        Boîtes retirées ou rognées par le chargeur, dégénérées, doublons, régions retirées ;
        l'analyse signale, elle ne corrige pas (une correction passe par M11).
        """, """
        table(DS.md_qualite)
        sus = DS.whole(STATS).get("qualite", {}).get("suspects", [])
        for d in sus:
            print(f"{d['id']} : score {d['score']} (retirées {d['retirées']}, rognées "
                  f"{d['rognées']}, dégénérées {d['dégénérées']}, doublons {d['doublons']})")
        sheet = STATS.get("galerie", {}).get("planches", {}).get("suspectes")
        if sheet:
            display(Img(filename=str(Path(OUT, sheet))))
        """)
    cells += _section("Écart entre splits et couverture hors domaine (T16.11)", """
        Distance en variation totale entre train et test, par grandeur ; part des objets qui
        ont une classe VOC ou COCO (ce qu'évaluent les notebooks hors domaine de T14.5).
        """, """
        table(DS.md_ecart)
        show("ecart")
        """)
    cells += [md(f"## Question propre au jeu\n\n{q} (renvoi : {ref})"), code(qcode)]
    cells += [md("## Résumé"), code("""
        table(DS.md_synthese)
        print(f"sorties : {OUT}/ (stats.json, stats.md, figures/, galerie/) ; "
              f"révision {TRACE['rev']}")
        print("commandes équivalentes :")
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
    "resultats": "Ce que donne le projet, mesuré : la précision (mAP) aux stades flottant, "
                 "entier et C-sim, l'effet du format des poids et de la quantification, le "
                 "coût en cycles et en ressources de l'accélérateur, et sa place face aux "
                 "travaux publiés. Les chiffres viennent des sorties de `build/` et de "
                 "`results/*.md` ; aucun n'est recopié à la main.",
    "modeles": "Les deux réseaux du projet, Tiny-YOLOv2 VOC et Tiny-YOLOv3 COCO, vus d'en "
               "haut : graphe des couches, coût par couche (MACs, paramètres, activations), "
               "empreinte mémoire face au budget de la carte, ancres, lecture d'une sortie "
               "YOLO, et les données VOC qui servent à l'entraînement et à l'évaluation.",
    "materiel": "L'accélérateur sur le SoC Zynq : la répartition ARM / logique programmable, "
                "le moteur unique de MACs et son tuilage, l'alternative streaming, et la "
                "chaîne qui vérifie bit à bit chaque passage, du NumPy flottant à la carte.",
    "maths": "Toute la chaîne est réécrite à la main en NumPy pur, sans PyTorch, OpenCV, "
             "scikit-learn ni devkit (ADR 0001). Chaque figure reprend une brique, de la "
             "convolution à la quantification, par ses fonctions mathématiques.",
    "reseaux": "Le niveau en dessous de « Modèles » : chaque couche et chaque tenseur de "
               "Tiny-YOLOv2 VOC et Tiny-YOLOv3 COCO, tirés de `python/yolo/models/specs.py`, "
               "jusqu'au suivi d'une couche du flottant au noyau HLS.",
    "projet": "Comment le projet s'est construit : jalons et chronologie (depuis les "
              "commits), avancement des tâches, volume de code, tests et dépendances "
              "entre jalons.",
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
                     f"- Source : `{f.source}`", f"- Régénérer : `{f.command}`", ""]
            for p in imgs:
                lines += ([f"**{p.stem}**", ""] if len(imgs) > 1 else []) + [link(p), ""]
            cells.append(md("\n".join(lines)))
    orphans = [p for p in pngs if p not in owner]
    if orphans:
        lines = ["## Autres figures", "",
                 "PNG versionnés qu'aucune figure du registre `tools/figures/` ne produit.", ""]
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

        Le projet expliqué en images (jalon M13) : l'architecture des réseaux et de
        l'accélérateur, les résultats mesurés, la réimplémentation NumPy et l'histoire du
        projet. Chaque figure est tracée par `tools/figures/` à partir de données du dépôt ;
        aucun chiffre n'est saisi à la main.

        **Lecture.** Une section par famille, ouverte par un paragraphe de contexte et la
        liste de ses figures. Chaque figure donne sa tâche (`T13.x`, détaillée dans
        [M13-figures.md](../docs/tasks/M13-figures.md)), ce qu'elle montre, la donnée
        qu'elle lit et la commande qui la retrace. La même galerie existe en Markdown :
        [results/figures.md](../results/figures.md).

        **Affichage seul.** Les PNG versionnés (`results/figures/`, `docs/tasks/figures/`)
        sont dans des cellules Markdown : rien à exécuter, dans VS Code comme sur GitHub.
        Seule la dernière section, pour `build/`, demande un noyau local.

        **Mise à jour.** `make figures` retrace les PNG, puis `make notebooks` régénère ce
        notebook depuis le disque. Il est généré par `python -m tools.notebooks` : modifier
        `tools/notebooks/gabarits.py:viewer`, pas ce fichier.
        """),
        *_figure_cells(),
        md("""\
        ## Figures non versionnées (`build/`, noyau local)

        Certaines figures ne sont pas publiées : celles calculées sur un sous-ensemble
        d'images (règle M12, par exemple la sensibilité par couche) et les sorties des
        notebooks d'entraînement et d'inférence. Elles restent dans `build/`, sur le PC qui
        les a produites ; la cellule ci-dessous les affiche, avec un noyau local (pas Colab).

        - `DIRS` : dossiers parcourus, relatifs à la racine du dépôt ;
        - `FILTER` : sous-chaîne du chemin pour ne garder que certaines images ;
        - `WIDTH` : largeur d'affichage en pixels.

        Les images s'affichent de la plus récente à la plus ancienne.
        """),
        code(f"""\
        DIRS   = {VIEWER_DIRS!r}  # relatifs à la racine du dépôt
        FILTER = ''  # sous-chaîne du chemin (ex. 'balayage') ; '' : tout
        WIDTH  = 900  # largeur d'affichage (px)
        """, tags=("parameters",)),
        code("""\
        from pathlib import Path

        from IPython.display import Image, Markdown, display

        # Racine du dépôt : dossier courant ou un parent ; sur Colab, le clone des notebooks.
        CANDIDATES = (Path.cwd(), *Path.cwd().parents, Path("/content/EmbeddedComputervision"))
        ROOT = next((d for d in CANDIDATES if (d / "tools" / "figures").is_dir()), None)
        if ROOT is None:
            raise SystemExit(f"dépôt introuvable depuis {Path.cwd()} : ouvrir le notebook avec "
                             "un noyau local (les PNG sont sur le PC, pas sur Colab)")
        print(f"dépôt : {ROOT}")
        for d in DIRS:
            pngs = sorted((p for p in (ROOT / d).rglob("*.png") if FILTER in str(p)),
                          key=lambda p: -p.stat().st_mtime)
            print(f"{d} : {len(pngs)} image(s)")
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
    ("grilles", "1. Grille lot × sous-ensemble, jeu par jeu",
     "Chaque case est un run : son score, puis ce score en pourcentage du meilleur run du "
     "jeu (★). La couleur suit ce pourcentage, sur la même échelle pour tous les jeux : on "
     "compare la *forme* des grilles malgré des mAP qui vont de 2 à 36. Une colonne « tout » "
     "plus foncée que la colonne « 500 im. » veut dire que voir tout le split paie ; une "
     "ligne « lot 32 » plus foncée, qu'un grand lot paie."),
    ("classements", "2. Les jeux classent-ils les runs de la même façon ?",
     "À gauche, le rang de chaque run dans chaque jeu (1 en haut) : une ligne plate en haut "
     "est un run toujours bon. Couleur : le lot (clair 8, foncé 32) ; trait plein : tout le "
     "split, tirets : 500 images. À droite, la corrélation de rangs de Spearman entre les "
     "classements de deux jeux : 1, même ordre ; 0, aucun lien. Avec 6 runs, il faut "
     "|ρ| ≥ 0,83 pour conclure à 5 % ; les mAP sur 50 images sont bruitées (règle M12)."),
    ("effets", "3. Effets principaux : sous-ensemble et lot",
     "À gauche, chaque flèche va du run à 500 images (creux) au run sur tout le split "
     "(plein), à lot égal : une flèche vers la droite est un gain. À droite, le score "
     "relatif par lot, un point par jeu, et la moyenne sur les jeux (losanges). À 600 "
     "itérations, le lot fixe aussi le nombre d'images vues : lot et volume de données ne "
     "se séparent pas dans cette grille."),
    ("epoques", "4. Époques vues : voir plus d'images ou revoir les mêmes ?",
     "L'axe horizontal compte les passages sur les données d'entraînement "
     "(lot × itérations / images, échelle log). Les runs sur tout le split (pleins) restent "
     "sous quelques époques ; ceux à 500 images (creux, zone grisée) revoient 10 à 38 fois "
     "les mêmes images. Si le score montait avec les époques seules, les deux nuages "
     "se prolongeraient ; un décrochage dans la zone grisée signe le surapprentissage."),
    ("perte", "5. Perte finale face au score",
     "Un panneau par jeu (échelles propres). ρ est la corrélation de rangs entre perte "
     "finale et score : négative, la perte basse va avec le bon score ; proche de zéro ou "
     "positive, la perte ne prédit pas la mAP. Les runs à 500 images (creux) ont souvent "
     "les pertes les plus basses : ils ajustent un petit jeu, sans généraliser."),
    ("courbes", "6. Courbes de perte",
     "Perte d'entraînement lissée, une ligne par jeu (rangées) et par sous-ensemble "
     "(colonnes), une couleur par lot. La zone grisée est le burn-in : le taux "
     "d'apprentissage y monte de 0 à sa valeur, sur 500 des 600 itérations. Un run "
     "entraîné lors d'une session précédente n'a pas de journal dans le notebook : il est "
     "relu dans `build/notebooks/<jeu>/<modèle>/runs/<run>/loss.csv` après `make harvest` "
     "(récolte du Drive), sinon marqué « sans journal »."),
    ("composantes", "7. De quoi est faite la perte finale ?",
     "Part de chaque terme de la perte YOLO dans les 50 dernières itérations journalisées : "
     "coord (position et taille des boîtes), obj (confiance sur les objets), noobj "
     "(confiance sur le fond), cls (classes). Le chiffre à droite est la perte totale. Une "
     "part obj dominante dit que le réseau sait mal où sont les objets ; une part cls, "
     "qu'il les trouve mais les nomme mal."),
    ("classes", "8. AP par classe",
     "AP@0,5 de chaque classe pour chaque run (runs triés du meilleur au moins bon). Les "
     "cases grises sont des classes sans instance dans les 50 images évaluées : leur AP "
     "n'existe pas, ce n'est pas un zéro. L'échelle de couleur est en racine carrée, pour "
     "distinguer les AP de 1 à 10. Une ligne pâle sur toute sa longueur est une classe "
     "qu'aucune configuration n'apprend en 600 itérations."),
    ("publies", "9. Affinage face aux poids publiés",
     "Meilleur run affiné, et les poids Darknet publiés (Tiny-YOLOv3 COCO, Tiny-YOLOv2 VOC) "
     "lancés hors domaine sur les mêmes 50 images, chaque jeu à son échelle. Les poids "
     "publiés ne sont évalués que sur les classes qui ont un équivalent dans leurs classes "
     "(« n cl. », `MAPPINGS`) : leur score ne se compare au run affiné qu'à titre "
     "indicatif. Un run affiné sous les poids publiés dit qu'il manque des itérations."),
    ("cout", "10. Coût d'un run",
     "Durée d'un run de 600 itérations sur le GPU Colab face à son score relatif. La durée "
     "suit le lot (images traitées), pas le sous-ensemble. Durée : celle de la cellule, ou "
     "la somme de la colonne `seconds` de `loss.csv` pour un run complété depuis `build/`."),
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
        # Analyse des balayages lot × sous-ensemble, tous jeux

        Les notebooks `<modèle>_sweep` entraînent six runs par jeu (lots 8, 16 et 32 ×
        500 premières images ou tout le split, 600 itérations, init COCO) et les comparent
        jeu par jeu. Ce notebook les compare **entre jeux** : quelle configuration gagne,
        de façon constante ou non, et pourquoi (images vues, surapprentissage, dynamique et
        composantes de la perte, classes apprises), puis l'écart aux poids publiés.

        **Source.** Les sorties versionnées des balayages ({links}) et des inférences des
        poids publiés (`tiny-yolov3-coco_infer`, `tiny-yolov2-voc_infer`), relues par
        `tools/notebooks/balayages.py` ; les figures viennent de
        `tools/figures/balayages.py`. Rien n'est recalculé ni saisi à la main : un
        balayage relancé et versionné se retrouve ici après `Run All`. Ni GPU ni données.
        Les runs sans journal dans leur notebook (session Colab précédente) sont complétés
        par leur `loss.csv` sous `build/notebooks/`, rapatrié du Drive par `make harvest`.

        **Prudence.** Palier R : scores sur les **50 premières images** d'évaluation, non
        publiables (règles de M12) ; une classe à une ou deux instances y saute de 0 à
        100. **Score** = mAP@0,5 du jeu : mAP 11 points, sauf FLIR (métrique COCO) dont on
        prend l'AP50 ; **score relatif** = score / meilleur run du jeu. Contexte et tables :
        [resultats-balayages.md](../docs/tasks/resultats-balayages.md).

        Généré par `python -m tools.notebooks` : modifier
        `tools/notebooks/gabarits.py:analysis`, pas ce fichier.
        """),
        md("## Paramètres"),
        code("""\
        DATASETS = None  # jeux analysés (ex. ['voc', 'kitti']) ; None : tous les balayages exécutés
        OUT      = 'build/notebooks/balayages'  # PNG et données (relatif à la racine du dépôt)
        SMOOTH   = 25  # lissage des courbes de perte (itérations)
        """, tags=("parameters",)),
        md("## Environnement"),
        code("""\
        import os
        import sys
        from pathlib import Path

        from IPython.display import Image, Markdown, display

        CANDIDATES = (Path.cwd(), *Path.cwd().parents, Path("/content/EmbeddedComputervision"))
        ROOT = next((d for d in CANDIDATES if (d / "tools" / "notebooks").is_dir()), None)
        if ROOT is None:
            raise SystemExit(f"dépôt introuvable depuis {Path.cwd()}")
        os.chdir(ROOT)
        sys.path[:0] = [str(ROOT), str(ROOT / "python")]

        from tools.figures import balayages as FB
        from tools.notebooks import balayages as B

        data, pending = B.load_all(datasets=DATASETS)
        if not data:
            raise SystemExit("aucun balayage exécuté dans notebooks/*/*_sweep.ipynb")
        OBS = B.observations(data)


        def show(paths):
            for p in paths:
                display(Image(data=Path(p).read_bytes(), format="png"))


        print(f"dépôt : {ROOT}")
        for d in data.values():
            print(f"  {d['label']:10s} {len(d['runs'])} runs, {d['metric']}, révision {d['rev']}, "
                  f"{d['notebook']}")
        for ds, p in pending.items():
            print(f"  {ds:10s} pas encore exécuté ({p})")
        """),
        md("""\
        ## Vue d'ensemble

        Un jeu par ligne : meilleur et pire run, rapport entre les deux, plage des pertes
        finales et meilleur poids publié. Puis tous les runs, du meilleur au moins bon
        dans chaque jeu.
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
                  display(Markdown("**Constat.** " + OBS[{key!r}]))
                  """)]
    cells += [
        md("""\
        ## Conclusions

        Calculées sur les balayages chargés ; elles se mettent à jour avec eux.
        """),
        code("""\
        display(Markdown(B.conclusions(data, pending)))
        print("commande équivalente :")
        print(f"  python -m tools.notebooks.balayages --runs --json {OUT}/balayages.json")
        print(f"  python -m tools.figures balayages  # PNG dans docs/tasks/figures/resultats/")
        """),
    ]
    return notebook(cells)
