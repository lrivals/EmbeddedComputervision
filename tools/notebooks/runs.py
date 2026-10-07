"""Runs d'entraînement des notebooks (T14.10, T14.11) : nommage, recherche, comparaison.

Disposition sous `build/notebooks/<jeu>/<modèle>/` (le `RUNS_DIR` des notebooks) :

    final.weights, loss.csv, …         run « train » (notebook <modèle>_train)
    runs/b16-sall/final.weights, …     runs du balayage (notebook <modèle>_sweep)
    runs/b8-s500/run.json              lot, sous-ensemble, itérations, backend, révision
    eval/<run>/                        évaluations des notebooks d'inférence

Un run compte dès qu'il a un `final.weights`. Sur Colab, `colab.sync_outputs(OUT)` copie
tout le dossier, `runs/` compris. Pas de dépendance hors bibliothèque standard.
"""

import csv
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from tools.notebooks import ROOT

TRAIN = "train"  # nom du run du notebook _train, à la racine de RUNS_DIR


@dataclass
class Run:
    name: str
    dir: str             # relatif à la racine du dépôt
    weights: str
    meta: dict = field(default_factory=dict)
    mtime: float = 0.0


def run_name(batch, subset):
    """Nom d'une case du balayage : b16-sall (tout le jeu), b8-s500 (500 images)."""
    return f"b{batch}-s{subset or 'all'}"


def _rel(p):
    p = Path(p)
    p = p if p.is_absolute() else ROOT / p
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def write_meta(run_dir, **params):
    """`run.json` du run : paramètres et date, pour la table comparative."""
    d = ROOT / run_dir
    d.mkdir(parents=True, exist_ok=True)
    meta = {**params, "date": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "run.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def _meta(d):
    try:
        return json.loads((d / "run.json").read_text())
    except (OSError, ValueError):
        return {}


def find_runs(model_dir):
    """Runs de `model_dir` qui ont un `final.weights` : `train` d'abord, puis `runs/*`."""
    root = ROOT / model_dir
    dirs = [(TRAIN, root)] + [(d.name, d) for d in sorted((root / "runs").glob("*"))
                              if d.is_dir()]
    out = []
    for name, d in dirs:
        w = d / "final.weights"
        if w.is_file():
            out.append(Run(name, _rel(d), _rel(w), _meta(d), w.stat().st_mtime))
    return out


def listing(runs):
    if not runs:
        return "aucun run entraîné"
    def when(r):
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(r.mtime))
    return "\n".join(f"  {r.name:12s} {r.weights}  ({when(r)})" for r in runs)


def pick(model_dir, name=None, hint=""):
    """Run `name`, ou le plus récent ; lève `env.Prerequis` avec la liste sinon."""
    from tools.notebooks.env import Prerequis

    runs = find_runs(model_dir)
    if name is None and runs:
        return max(runs, key=lambda r: r.mtime)
    hit = [r for r in runs if r.name == name]
    if hit:
        return hit[0]
    want = f"run {name!r}" if name else "un run entraîné"
    raise Prerequis(f"{want} introuvable dans {model_dir} ; runs disponibles :\n"
                    f"{listing(runs)}\n  à lancer : {hint or 'le notebook _train ou _sweep'}")


# ----------------------------------------------------------------------- comparaison

def read_map(md_path):
    """mAP (en %) d'une table de `tools/eval_voc.py` : ligne `**mAP**` (VOC), sinon première
    valeur du résumé COCO (AP@[.5:.95]). None si le fichier ou la valeur manque."""
    p = ROOT / md_path
    if not p.exists():
        return None
    text = p.read_text()
    m = re.findall(r"\|\s*\*\*mAP\*\*\s*\|\s*\*\*([\d.]+)\*\*", text)
    if m:
        return float(m[-1])
    lines = text.splitlines()
    for i, line in enumerate(lines[:-2]):
        if line.startswith("|---") and lines[i - 1].lstrip("| ").startswith("AP"):
            return float(lines[i + 1].strip("| ").split("|")[0])
    return None


def _loss_rows(run_dir):
    p = ROOT / run_dir / "loss.csv"
    if not p.exists():
        return []
    with p.open() as f:
        return list(csv.DictReader(f))


def last_loss(run_dir, n=50):
    """Perte moyenne des `n` dernières itérations journalisées."""
    rows = _loss_rows(run_dir)[-n:]
    return sum(float(r["loss"]) for r in rows) / len(rows) if rows else None


def s_per_image(run_dir, batch, n=50):
    rows = _loss_rows(run_dir)[-n:]
    return sum(float(r["seconds"]) for r in rows) / len(rows) / batch if rows and batch else None


def evaluate(run, net, dataset, resize, subset, metric=None, split=None, size=None,
             data_root=None, out_dir=None):
    """mAP flottante du run par `tools/eval_voc.py`, réutilisée si déjà calculée pour ce
    `subset` ; rend la ligne de la table comparative."""
    from tools.notebooks import commandes as C

    out = Path(out_dir or run.dir)
    md = out / f"map_float_s{subset or 'all'}.md"
    if read_map(md) is None:
        (ROOT / md).unlink(missing_ok=True)
        C.run(C.cmd_eval_voc(net, run.weights, dataset, resize, subset, metric, split, size,
                             data_root, out=out / f"dets_s{subset or 'all'}", markdown=md))
    return row(run, read_map(md), subset)


def row(run, map_, subset=None):
    m = run.meta
    batch = m.get("batch")
    return {"run": run.name, "lot": batch, "images": m.get("subset", "") or "tout",
            "itérations": m.get("iters", ""), "backend": m.get("device", ""),
            "perte finale": last_loss(run.dir), "mAP": map_,
            "s/image": s_per_image(run.dir, batch), "éval.": subset or "tout"}


def table(rows):
    """Table markdown des runs, triée par mAP décroissante."""
    if not rows:
        return "aucun run"
    cols = list(rows[0])

    def fmt(v):
        if v is None or v == "":
            return "—"
        return f"{v:.2f}" if isinstance(v, float) else str(v)

    rows = sorted(rows, key=lambda r: -(r["mAP"] if r["mAP"] is not None else -1))
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


# --------------------------------------------------------------------- résumés (T14.8)

def parse_eval(text):
    """Bloc de `tools/eval_voc.py` → {metric, map, ap, ap50, coco, classes, classes50,
    ms_image} ; AP de classe None si la classe n'a aucune instance (-100 de COCO)."""
    out = {"metric": "coco" if "métrique COCO" in text else "voc", "map": None, "ap": None,
           "ap50": None, "coco": {}, "classes": {}, "classes50": {}, "ms_image": None}
    m = re.search(r"\((\d+) ms/image\)", text)
    if m:
        out["ms_image"] = float(m.group(1))
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if not line.startswith("| ") or i + 2 >= len(lines) or not lines[i + 1].startswith("|---"):
            continue
        head = [c.strip() for c in line.strip().strip("|").split("|")]
        if head[:2] == ["AP", "AP50"] and not out["coco"]:  # résumé COCO
            vals = [float(c) for c in lines[i + 2].strip().strip("|").split("|")]
            out["coco"] = dict(zip(head, vals))
        elif head[0] == "Classe" and not out["classes"]:
            for row in lines[i + 2:]:
                if not row.startswith("|"):
                    break
                cells = [c.strip() for c in row.strip().strip("|").split("|")]
                if cells[0] == "**mAP**":
                    out["map"] = float(cells[1].strip("*"))
                    continue
                vals = [float(v) for v in cells[1:]]
                vals = [None if v < 0 else v for v in vals]
                out["classes"][cells[0]] = vals[0]
                if len(vals) > 1:
                    out["classes50"][cells[0]] = vals[1]
    if out["metric"] == "coco" and out["coco"]:
        out["ap"], out["ap50"] = out["coco"]["AP"], out["coco"]["AP50"]
        out["map"] = out["ap50"]
    if out["metric"] == "voc":
        out["ap50"] = out["map"]
    return out


LOSS_PARTS = ("coord", "obj", "noobj", "cls")


def summarize(run_dir, n=50):
    """Résumé d'un dossier de run ou d'évaluation, lisible sans le notebook : `run.json`,
    journal `loss.csv` (itérations, perte et composantes moyennes des `n` dernières, durée,
    s/image) et chaque `map_float_*.md` (mAP, AP par classe). Clés absentes si la source
    manque."""
    d = ROOT / run_dir
    out = {"dir": _rel(d)}
    meta = _meta(d)
    if meta:
        out["meta"] = meta
    rows = _loss_rows(d)
    if rows:
        tail = rows[-n:]
        mean = lambda k: sum(float(r[k]) for r in tail) / len(tail)  # noqa: E731
        seconds = sum(float(r["seconds"]) for r in rows)
        out["train"] = {"iters": int(float(rows[-1]["it"])), "final_loss": mean("loss"),
                        "final_parts": {k: mean(k) for k in LOSS_PARTS if k in rows[0]},
                        "seconds": seconds,
                        "s_per_image": s_per_image(d, meta.get("batch"), n)}
    evals = {}
    for md in sorted(d.glob("map_float*.md")):
        ev = parse_eval(md.read_text())
        if ev["map"] is not None:
            evals[md.stem] = ev
    if evals:
        out["eval"] = evals
    return out


def write_summary(run_dir):
    """`summary.json` du dossier (voir `summarize`) ; rend son chemin, ou None s'il n'y a
    rien à résumer."""
    s = summarize(run_dir)
    if len(s) == 1:  # seulement "dir"
        return None
    p = ROOT / run_dir / "summary.json"
    p.write_text(json.dumps(s, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return p


def write_summaries(model_dir):
    """`summary.json` de chaque run (`runs/*`, run `train` à la racine) et évaluation
    (`eval/*`) de `model_dir`, et `<model_dir>/summary.json` qui les rassemble."""
    root = ROOT / model_dir
    dirs = [root] + sorted(p for sub in ("runs", "eval") for p in (root / sub).glob("*")
                           if p.is_dir())
    items = {}
    for d in dirs:
        s = summarize(d)
        if len(s) > 1:
            items[_rel(d)] = s
            if d != root:
                (d / "summary.json").write_text(
                    json.dumps(s, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    p = root / "summary.json"
    if items:
        p.write_text(json.dumps(items, indent=1, ensure_ascii=False, sort_keys=True) + "\n")
    return p if items else None
