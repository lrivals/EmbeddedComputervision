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
               "sweep": "balayage lot × sous-ensemble"}
SUBSET_FIRST = 50  # premier passage, palier R (règles de M12)
# Grille par défaut du notebook _sweep (T14.10) : 3 lots × (500 images, tout le jeu).
SWEEP = {"batches": [8, 16, 32], "subsets": [500, 0], "iters": 600}

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
                  ("ITERS", SWEEP["iters"], "itérations par run (palier N dès 600)"),
                  ("SKIP_DONE", True, "saute les runs qui ont déjà final.weights")]
        else:
            p += [("ITERS", M11_TRAIN["iters"], "itérations (palier N au-delà de 600)"),
                  ("BATCH", M11_TRAIN["batch"], None)]
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
          ("JOBS", 4, "processus des évaluations"),
          ("SHOW", 6, "images affichées"),
          ("OUT", nb.out_dir, "sorties du notebook"),
          ("REPO_URL", REPO_URL, "Colab : dépôt cloné"),
          ("REV", REV, "Colab : révision")]
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
        if WEIGHTS is None:
            CHOSEN = runs.pick(RUNS_DIR, RUN, hint={needs[2]})
            WEIGHTS, OUT = CHOSEN.weights, f"{{OUT}}/{{CHOSEN.name}}"
            print(f"run choisi : {{CHOSEN.name}} ({{WEIGHTS}})")"""
    else:
        needs = ("INIT if str(INIT).endswith('.weights') else "
                 "'weights/yolov3-tiny.weights' if INIT == 'coco' else ()", "None", "''")
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
                subprocess.run([sys.executable, "-m", "pip", "install", "-q", "cupy-cuda12x"],
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
                    drive_dir=DRIVE_DIR)
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
    cells += [md("""
        ## mAP flottante (T14.3)

        `tools/eval_voc.py`, métrique du jeu (`Dataset.metric` ; COCO et FLIR : AP@[.5:.95]
        et AP50). Courbes PR du mode automatique de M13 (métrique VOC).
        """), code("""
        print(f"{SUBSET or 'toutes les'} images : palier "
              f"{palier(SUBSET or SPLIT_IMAGES.get(DATASET))} (règles de M12)")
        map_float = Path(OUT) / "map_float.md"
        map_float.unlink(missing_ok=True)
        C.run(C.cmd_eval_voc(NET, WEIGHTS, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                             DATA_ROOT, out=f"{OUT}/dets", markdown=map_float))
        display(Markdown(map_float.read_text()))
        for png in sorted(Path(OUT, "dets", "figures").glob("pr*.png")):
            display(Image.open(png))
        """)]
    if nb.finetuned:
        cells += [md("""
            ## Comparaison des runs (T14.11)

            Avec `COMPARE = True`, chaque run trouvé dans `RUNS_DIR` (notebook `_train` et
            balayage `_sweep`) est évalué sur les mêmes `SUBSET` images ; une mAP déjà
            calculée pour ce `SUBSET` est réutilisée. Le run évalué plus haut se choisit par
            `RUN`.
            """), code("""
            if COMPARE:
                rows = [runs.evaluate(r, NET, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE,
                                      DATA_ROOT, out_dir=f"{RUNS_DIR}/eval/{r.name}")
                        for r in RUNS]
                display(Markdown(runs.table(rows)))
            else:
                print("COMPARE = False : comparaison sautée ; runs trouvés :")
                print(runs.listing(RUNS))
            """)]
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
        """) + variants
    return [*_prep_preview_cells(), md(affinage), code("""
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
                                 SPLIT, SIZE, DATA_ROOT, out=RUN / "dets", markdown=result))
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
        """)]


def sweep_body(nb):
    return [*_prep_preview_cells(), md("""
        ## Plan du balayage (T14.10)

        Un run par case de la grille `BATCHES × TRAIN_SUBSETS`, tous les autres paramètres
        égaux (ceux de `tools/m11.sh <jeu>-train`, `ITERS` à part). `TRAIN_SUBSETS` prend les
        n premières images du split d'entraînement (`train.py --subset`), 0 tout le split.
        À itérations égales, un lot plus grand voit plus d'images : la colonne « époques »
        le montre. Le taux d'apprentissage n'est pas ajusté au lot.
        """), code("""
        import threading

        from tools.notebooks import colab, runs

        PLAN = [(b, s, runs.run_name(b, s)) for b in BATCHES for s in TRAIN_SUBSETS]
        rows = ["| run | lot | images | époques | durée estimée |", "|---|---|---|---|---|"]
        total = 0
        for b, s, name in PLAN:
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
            for b, s, name in PLAN:
                run_dir = Path(OUT) / "runs" / name
                if SKIP_DONE and (run_dir / "final.weights").exists():
                    print(f"{name} : déjà entraîné, sauté")
                    continue
                resume = (run_dir / "checkpoint.npz").exists()
                runs.write_meta(run_dir, batch=b, subset=s, iters=ITERS, lr=LR, device=DEVICE,
                                net=NET, dataset=DATASET, rev=TRACE["rev"])
                C.run(C.cmd_train(NET, run_dir, DATASET, INIT, INIT_NET, resume, iters=ITERS,
                                  batch=b, lr=LR, burn_in=BURN_IN, multiscale=MULTISCALE,
                                  subset=s, save_every=SAVE_EVERY, workers=WORKERS,
                                  device=DEVICE, data_root=DATA_ROOT),
                      log=run_dir / "log.txt")
        finally:
            stop.set()
            if SYNC:
                colab.sync_outputs(OUT, DRIVE_DIR)
        """), md("""
        ## Comparaison

        mAP flottante de chaque run sur les mêmes `SUBSET` images (`tools/eval_voc.py`,
        réutilisée si déjà calculée), perte finale et vitesse ; courbes de perte superposées
        (`tools.figures.resultats.plot_training`, M13). Les notebooks d'inférence
        `<modèle>_infer` lisent ces runs (`RUN`, `COMPARE`).
        """), code("""
        from tools.figures import MissingSource
        from tools.figures import resultats as FR

        RESULTS = [runs.evaluate(r, NET, DATASET, RESIZE, SUBSET, METRIC, SPLIT, SIZE, DATA_ROOT)
                   for r in runs.find_runs(OUT) if r.name != runs.TRAIN]
        display(Markdown(runs.table(RESULTS)))
        curves = []
        for b, s, name in PLAN:
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


def render(nb):
    body = {"infer": infer_body, "train": train_body, "sweep": sweep_body}[nb.role](nb)
    return notebook([header(nb), *params(nb), *environment(nb), *body], gpu=nb.trains)
