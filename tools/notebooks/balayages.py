"""Analyse transversale des balayages lot × sous-ensemble (T14.10) : relit les notebooks
`notebooks/<jeu>/<modèle>_sweep.ipynb` exécutés et les `_infer` des poids publiés.

    python -m tools.notebooks.balayages                      # table de synthèse
    python -m tools.notebooks.balayages --json build/notebooks/balayages/balayages.json

Les runs vivent sur le runtime qui les a entraînés (Colab, Drive) ; les sorties
versionnées des notebooks (règle M14) suffisent : journal de `tools/train.py` (perte et
composantes par itération, durée), tables de `tools/eval_voc.py` (AP par classe, mAP ou
résumé COCO). Rien n'est recalculé, rien n'est saisi à la main. Stdlib + NumPy (ADR 0001).

Score commun : la mAP@0,5 du jeu, c'est-à-dire la mAP 11 points (VOC, VisDrone, KITTI,
ExDark…) ou l'AP50 COCO pour les jeux à métrique COCO (FLIR) ; l'AP@[.5:.95] de FLIR est
gardée à part (`ap`). Score relatif : score / meilleur run du jeu, pour comparer des jeux
dont les mAP diffèrent d'un ordre de grandeur.
"""

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from tools.notebooks import NB_DIR, ROOT
from tools.notebooks.runs import parse_eval, summarize  # noqa: F401  (parse_eval : API)

ORDER = ("voc", "visdrone", "kitti", "flir", "exdark", "crowdhuman")  # ordre des figures
LABELS = {"voc": "VOC", "visdrone": "VisDrone", "kitti": "KITTI", "flir": "FLIR",
          "exdark": "ExDark", "crowdhuman": "CrowdHuman", "coco": "COCO"}
BASELINES = ("tiny-yolov3-coco", "tiny-yolov2-voc")  # poids publiés, hors domaine
CURVE = ("it", "lr", "loss", "coord", "obj", "noobj", "cls")
PARTS = ("coord", "obj", "noobj", "cls")
LAST = 50  # perte finale : moyenne des LAST dernières itérations (comme runs.last_loss)

_IT = re.compile(r"^it\s+(\d+)\s+lr\s+(\S+)\s+perte\s+(\S+)\s+coord\s+(\S+)\s+obj\s+(\S+)"
                 r"\s+noobj\s+(\S+)\s+cls\s+(\S+)", re.M)


# ----------------------------------------------------------------------- notebooks

def _cells(path):
    """[(id, texte de sortie)] des cellules de code : flux et Markdown affiché, concaténés."""
    nb = json.loads(Path(path).read_text())
    out = []
    for c in nb["cells"]:
        if c["cell_type"] != "code":
            continue
        parts = []
        for o in c.get("outputs", []):
            if "text" in o:
                parts.append("".join(o["text"]))
            elif "text/markdown" in o.get("data", {}):
                parts.append("".join(o["data"]["text/markdown"]) + "\n")
        out.append((c.get("id", ""), "".join(parts), "".join(c["source"])))
    return out


def _param(cells, name):
    """Valeur littérale d'un paramètre de la cellule `parameters` (`NAME = …`)."""
    for _, _, src in cells:
        m = re.search(rf"^{name}\s*=\s*(.+?)(?:\s+#.*)?$", src, re.M)
        if m:
            try:
                return eval(m.group(1), {"__builtins__": {}})  # littéral du gabarit
            except Exception:  # noqa: BLE001
                return m.group(1)
    return None


def _rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def _executed(path):
    from tools.notebooks.__main__ import executed_ok

    text = Path(path).read_text()
    code = [c for c in json.loads(text)["cells"] if c["cell_type"] == "code"]
    return any(c.get("outputs") for c in code) and executed_ok(text) is None


def find_sweeps(nb_dir=NB_DIR):
    """{jeu: chemin} des `*_sweep.ipynb`, exécutés ou non, dans l'ordre ORDER."""
    found = {p.parent.name: p for p in Path(nb_dir).glob("*/*_sweep.ipynb")}
    return {k: found[k] for k in sorted(found, key=lambda k: (ORDER + (k,)).index(k))}


# ---------------------------------------------------------------------- évaluations

def _option(cmd, name):
    m = re.search(rf"--{name}\s+(\S+)", cmd)
    return m.group(1) if m else None


def _segments(text, tool):
    """[(commande, sortie)] des appels `$ python tools/<tool>` ; la sortie s'arrête à la
    commande suivante, quel que soit l'outil."""
    marks = [(m.start(), m.group(1)) for m in re.finditer(r"^\$ python tools/(\S+)", text, re.M)]
    segs = []
    for (a, t), (b, _) in zip(marks, marks[1:] + [(len(text), None)]):
        if t == tool:
            cmd, _, body = text[a:b].partition("\n")
            segs.append((cmd, body))
    return segs


# --------------------------------------------------------------------------- balayage

def _name(name):
    """« b16-s500 » → (16, 500) ; « b32-sall » → (32, 0)."""
    m = re.fullmatch(r"b(\d+)-s(\d+|all)", name)
    return (int(m.group(1)), 0 if m.group(2) == "all" else int(m.group(2))) if m else None


def _parse_train(cmd, body):
    rows = np.array([[float(v) for v in m] for m in _IT.findall(body)])
    m = re.search(r"^(\d+) images, (\d+) lots par époque", body, re.M)
    dur = re.findall(r"^\s+\((\d+) s\)", body, re.M)
    resumed = "--resume" in cmd.split()
    return {"iters": int(_option(cmd, "iters")), "lr": float(_option(cmd, "lr")),
            "burn_in": int(_option(cmd, "burn-in") or 0),
            "images": int(m.group(1)) if m else None,
            # Run repris (--resume) : journal et durée partiels.
            "duration_s": float(dur[-1]) if dur and not resumed else None,
            "curve": {k: rows[:, j] for j, k in enumerate(CURVE)} if len(rows) else None,
            "resumed": resumed}


def _comparison(text):
    """Lignes de la table de `runs.table` (cellule de comparaison) : {run: {colonne: texte}}."""
    out, head = {}, None
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if line.startswith("| run |"):
            head = cells
        elif head and line.startswith("| ") and _name(cells[0]) and len(cells) == len(head):
            out[cells[0]] = dict(zip(head, cells))
        elif not line.startswith("|"):
            head = None
    return out


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_sweep(path):
    """Notebook de balayage exécuté → {dataset, net, metric, n_train, rev, runs: [...]}, runs
    dans l'ordre du plan (lot, puis 500 images avant tout le split). Un run entraîné dans
    une session précédente (« déjà entraîné, sauté ») n'a ni courbe ni durée : sa perte
    finale et sa mAP viennent de la table comparative."""
    path = Path(path)
    cells = _cells(path)
    text = "\n".join(t for _, t, _ in cells)
    iters = _param(cells, "ITERS")
    runs = {}

    def get(name):
        b, s = _name(name)
        return runs.setdefault(name, {"run": name, "batch": b, "subset": s, "iters": iters,
                                      "images": None, "duration_s": None, "curve": None,
                                      "resumed": False, "map": None})

    for cmd, body in _segments(text, "train.py"):
        get(Path(_option(cmd, "out")).name).update(_parse_train(cmd, body))
    for cmd, body in _segments(text, "eval_voc.py"):
        r = get(Path(_option(cmd, "weights")).parent.name)
        r.update(parse_eval(body))
    for name, row in _comparison(text).items():
        r = get(name)
        r["table_loss"] = _num(row.get("perte finale"))
        if r["map"] is None:  # évaluation réutilisée : seule la table la donne
            r["map"] = _num(row.get("mAP"))
    sall = [r["images"] for r in runs.values() if r["subset"] == 0 and r["images"]]
    n_train = max(sall) if sall else None
    for r in runs.values():
        curve = r["curve"]
        full = curve is not None and not r["resumed"]
        tail = {k: float(curve[k][-LAST:].mean()) for k in ("loss",) + PARTS} if curve is not None else {}
        # Perte finale : celle de la table (loss.csv complet), sinon le journal affiché.
        r["final_loss"] = r.pop("table_loss", None) or tail.get("loss")
        r["final_parts"] = {k: tail[k] for k in PARTS} if tail else {}
        r["full_log"] = full
        n = r["subset"] or n_train
        r["images"] = r["images"] or (min(r["subset"], n_train or r["subset"]) if r["subset"] else n_train)
        r["epochs"] = r["iters"] * r["batch"] / n if n and r["iters"] else None
    order = sorted(runs.values(), key=lambda r: (r["batch"], r["subset"] == 0, r["subset"]))
    metric = next((r["metric"] for r in order if "metric" in r), None)
    m = re.search(r"révision (\w+)", text)
    return {"dataset": path.parent.name, "label": LABELS.get(path.parent.name, path.parent.name),
            "net": _param(cells, "NET"), "notebook": _rel(path),
            "metric": metric, "n_train": n_train, "iters": iters,
            "burn_in": _param(cells, "BURN_IN"), "rev": m.group(1) if m else None,
            "runs": order}


def parse_baselines(dataset, nb_dir=NB_DIR):
    """{modèle: évaluation} des notebooks `_infer` des poids publiés du jeu, s'ils ont tourné.
    `n_classes` : classes évaluées (avec équivalent dans le modèle), `present` : celles qui
    ont des instances dans les images évaluées."""
    out = {}
    for model in BASELINES:
        p = Path(nb_dir) / dataset / f"{model}_infer.ipynb"
        if not p.exists():
            continue
        text = "\n".join(t for _, t, _ in _cells(p))
        segs = _segments(text, "eval_voc.py")
        if not segs:
            continue
        ev = parse_eval(segs[0][1])
        if ev["map"] is None:
            continue
        ev["n_classes"] = len(ev["classes"])
        ev["present"] = sum(v is not None for v in ev["classes"].values())
        out[model] = ev
    return out


def _finish(d):
    """Scores relatifs, rangs et meilleur run d'un jeu."""
    scored = [r for r in d["runs"] if r.get("map") is not None]
    top = max((r["map"] for r in scored), default=None)
    for r in d["runs"]:
        r["rel"] = r["map"] / top if top and r.get("map") is not None else None
    ranks = _ranks([-r["map"] for r in scored])
    for r, k in zip(scored, ranks):
        r["rank"] = float(k)
    d["best"] = max(scored, key=lambda r: r["map"])["run"] if scored else None
    d["top"] = top
    return d


def complete_from_build(d, build=None):
    """Runs sans journal dans le notebook (entraînés lors d'une session précédente, ou
    repris) : courbe, durée et composantes depuis `loss.csv` du run sous `build`
    (`build/notebooks/<jeu>/<modèle>/runs/<run>/`, rapatrié par `make harvest`). Rend les
    runs complétés."""
    model = Path(d["notebook"]).stem.removesuffix("_sweep")
    base = Path(build) if build else ROOT / "build" / "notebooks"
    done = []
    for r in d["runs"]:
        csv_path = base / d["dataset"] / model / "runs" / r["run"] / "loss.csv"
        if r["full_log"] or not csv_path.is_file():
            continue
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        r["curve"] = {k: np.array([float(x[src]) for x in rows])
                      for k, src in zip(CURVE, ("it", "lr", "loss") + PARTS)}
        tail = {k: float(r["curve"][k][-LAST:].mean()) for k in ("loss",) + PARTS}
        r["final_parts"] = {k: tail[k] for k in PARTS}
        r["final_loss"] = r["final_loss"] or tail["loss"]
        r["duration_s"] = sum(float(x["seconds"]) for x in rows)
        r["resumed"], r["full_log"], r["from_build"] = False, True, True
        done.append(r["run"])
    return done


def load_all(nb_dir=NB_DIR, datasets=None, build=None):
    """{jeu: balayage} des notebooks exécutés (`datasets` : sous-liste), complétés par les
    journaux de `build` s'il y en a ; et {jeu: chemin} des balayages pas encore lancés."""
    data, pending = {}, {}
    for ds, p in find_sweeps(nb_dir).items():
        if datasets and ds not in datasets:
            continue
        if not _executed(p):
            pending[ds] = _rel(p)
            continue
        d = parse_sweep(p)
        d["completed"] = complete_from_build(d, build)
        d["baselines"] = parse_baselines(ds, nb_dir)
        data[ds] = _finish(d)
    return data, pending


# ----------------------------------------------------------------------- statistiques

def _ranks(values):
    """Rangs 1..n (1 : plus petite valeur), ex aequo au rang moyen."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    ranks = np.empty(len(v))
    ranks[order] = np.arange(1, len(v) + 1)
    for x in np.unique(v):
        tie = v == x
        ranks[tie] = ranks[tie].mean()
    return ranks


def spearman(a, b):
    """Corrélation de rangs de Spearman (Pearson sur les rangs moyens)."""
    ra, rb = _ranks(a), _ranks(b)
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else float("nan")


def run_names(data):
    """Runs présents dans tous les jeux, dans l'ordre du plan."""
    names = None
    for d in data.values():
        cur = [r["run"] for r in d["runs"] if r.get("map") is not None]
        names = cur if names is None else [n for n in names if n in cur]
    return names or []


def score_matrix(data, key="map"):
    """(jeux, runs, matrice jeux × runs de `key`)."""
    names = run_names(data)
    sets = list(data)
    m = np.array([[next(r[key] for r in data[ds]["runs"] if r["run"] == n) for n in names]
                  for ds in sets], dtype=float)
    return sets, names, m


def rank_agreement(data):
    """(jeux, matrice de Spearman jeux × jeux) sur les scores des runs communs."""
    sets, _, m = score_matrix(data)
    rho = np.array([[spearman(a, b) for b in m] for a in m])
    return sets, rho


def effects(data):
    """Effets principaux, en score relatif : {sall_minus_s500: {jeu: {lot: Δ}},
    by_batch: {sous-ensemble: {lot: moyenne sur les jeux}}}."""
    out = {"sall_minus_s500": {}, "by_batch": {}}
    for ds, d in data.items():
        rel = {(r["batch"], r["subset"]): r["rel"] for r in d["runs"] if r["rel"] is not None}
        out["sall_minus_s500"][ds] = {b: rel[(b, 0)] - rel[(b, s)] for (b, s) in rel
                                      if s and (b, 0) in rel}
    keys = {(r["batch"], r["subset"]) for d in data.values() for r in d["runs"]}
    for b, s in sorted(keys):
        vals = [r["rel"] for d in data.values() for r in d["runs"]
                if (r["batch"], r["subset"]) == (b, s) and r["rel"] is not None]
        out["by_batch"].setdefault(s, {})[b] = float(np.mean(vals)) if vals else None
    return out


# ------------------------------------------------------------------------- texte

def _f(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")


def metric_label(d):
    return "AP50 (COCO)" if d["metric"] == "coco" else "mAP 11 pts"


def summary_table(data):
    """Table Markdown : un jeu par ligne, meilleur et pire run, écart, poids publiés."""
    lines = ["| jeu | images d'entr. | métrique | meilleur run | score | pire run | score | "
             "écart max/min | perte finale (min–max) | meilleur poids publié |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for ds, d in data.items():
        scored = [r for r in d["runs"] if r.get("map") is not None]
        best = max(scored, key=lambda r: r["map"])
        worst = min(scored, key=lambda r: r["map"])
        losses = [r["final_loss"] for r in d["runs"] if r["final_loss"] is not None]
        base = max(d["baselines"].items(), key=lambda kv: kv[1]["map"], default=None)
        base_txt = f"{base[0]} {_f(base[1]['map'])}" if base else "—"
        ratio = best["map"] / worst["map"] if worst["map"] else float("inf")
        lines.append(f"| {d['label']} | {d['n_train'] or '—'} | {metric_label(d)} | "
                     f"**{best['run']}** | **{_f(best['map'])}** | {worst['run']} | "
                     f"{_f(worst['map'])} | ×{_f(ratio, 1)} | "
                     f"{_f(min(losses), 1)}–{_f(max(losses), 1)} | {base_txt} |")
    return "\n".join(lines)


def runs_table(data):
    """Table Markdown longue : tous les runs de tous les jeux."""
    lines = ["| jeu | run | lot | images | époques | perte finale | score | relatif | rang | "
             "durée (min) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for d in data.values():
        for r in sorted(d["runs"], key=lambda r: r.get("rank") or 99):
            lines.append(f"| {d['label']} | {r['run']} | {r['batch']} | "
                         f"{r['subset'] or 'tout'} | {_f(r['epochs'])} | {_f(r['final_loss'])} | "
                         f"{_f(r.get('map'))} | {_f(r['rel'])} | {_f(r.get('rank'), 1)} | "
                         f"{_f(r['duration_s'] / 60 if r['duration_s'] else None, 1)} |")
    return "\n".join(lines)


def observations(data):
    """Constats chiffrés, un par figure du notebook d'analyse : {clé: Markdown}."""
    obs = {}
    sets, names, m = score_matrix(data)
    rel = m / m.max(axis=1, keepdims=True)
    mean_rel = rel.mean(axis=0)
    order = np.argsort(-mean_rel)
    wins = {n: sum(data[ds]["best"] == n for ds in sets) for n in names}
    obs["grilles"] = (
        f"Sur {len(sets)} jeux, le score relatif moyen classe les runs ainsi : "
        + ", ".join(f"`{names[i]}` {_f(mean_rel[i])}" for i in order)
        + ". Meilleur run par jeu : "
        + ", ".join(f"{data[ds]['label']} `{data[ds]['best']}`" for ds in sets)
        + f" (`{max(wins, key=wins.get)}` gagne {max(wins.values())} fois sur {len(sets)}).")
    s, rho = rank_agreement(data)
    iu = np.triu_indices(len(s), 1)
    pairs = sorted(((rho[i, j], s[i], s[j]) for i, j in zip(*iu)), reverse=True)
    if pairs:
        obs["classements"] = (
            f"Spearman moyen entre jeux : {_f(float(np.mean(rho[iu])))}. Classements les plus "
            f"proches : {LABELS[pairs[0][1]]}–{LABELS[pairs[0][2]]} ({_f(pairs[0][0])}) ; les "
            f"plus éloignés : {LABELS[pairs[-1][1]]}–{LABELS[pairs[-1][2]]} "
            f"({_f(pairs[-1][0])}). Avec 6 runs, |ρ| < 0,83 n'est pas significatif à 5 %.")
    eff = effects(data)
    deltas = [v for per in eff["sall_minus_s500"].values() for v in per.values()]
    pos = sum(v > 0 for v in deltas)
    bb = eff["by_batch"]
    obs["effets"] = (
        f"Tout le split bat 500 images dans {pos} cas sur {len(deltas)} (jeu × lot), de "
        f"{_f(float(np.mean(deltas)))} en score relatif moyen. Score relatif moyen par lot : "
        + " ; ".join(f"{'tout' if s_ == 0 else str(s_) + ' images'} : "
                     + ", ".join(f"lot {b} {_f(v)}" for b, v in sorted(per.items()))
                     for s_, per in sorted(bb.items(), key=lambda kv: kv[0] != 0)) + ".")
    ep = [(r["epochs"], r["rel"], r["subset"]) for d in data.values() for r in d["runs"]
          if r["epochs"] and r["rel"] is not None]
    x, y = np.log10([e for e, _, _ in ep]), np.array([v for _, v, _ in ep])
    e500 = [e for e, _, s in ep if s]
    eall = [e for e, _, s in ep if not s]
    obs["epoques"] = (
        f"Corrélation de rangs entre époques vues et score relatif, tous jeux : "
        f"{_f(spearman(x, y))}. Les runs à 500 images font {_f(min(e500), 1)} à "
        f"{_f(max(e500), 1)} époques ; ceux sur tout le split, {_f(min(eall), 2)} à "
        f"{_f(max(eall), 2)}. Score relatif moyen : {_f(float(np.mean([v for _, v, s in ep if not s])))} "
        f"sur tout le split, {_f(float(np.mean([v for _, v, s in ep if s])))} à 500 images.")
    lines = []
    for ds, d in data.items():
        pts = [(r["final_loss"], r["map"]) for r in d["runs"]
               if r["final_loss"] is not None and r.get("map") is not None]
        rho_ = spearman([p[0] for p in pts], [p[1] for p in pts])
        low = min(d["runs"], key=lambda r: r["final_loss"])
        lines.append(f"{d['label']} ρ = {_f(rho_)} (perte la plus basse : `{low['run']}`, "
                     f"rang {_f(low.get('rank'), 0)})")
    obs["perte"] = ("Corrélation de rangs perte finale / score, par jeu : " + " ; ".join(lines)
                    + ". Une corrélation positive veut dire : plus la perte est basse, plus "
                    "la mAP est basse (surapprentissage des runs à 500 images).")
    lines = []
    for ds, d in data.items():
        by_b = {}
        for r in d["runs"]:
            if r["curve"] is not None and not r["resumed"]:
                by_b.setdefault(r["batch"], []).append(r["final_loss"])
        if len(by_b) > 1:
            lo, hi = min(by_b), max(by_b)
            gain = 1 - np.mean(by_b[hi]) / np.mean(by_b[lo])
            lines.append(f"{d['label']} {_f(100 * gain, 0)} %")
    obs["courbes"] = (
        "Les courbes des trois lots se superposent pendant l'essentiel du burn-in (LR "
        "montant en (it/burn-in)⁴ : moins de 20 % de sa valeur avant 66 % du burn-in) et ne "
        "s'écartent qu'à la fin. Perte finale du plus grand lot, "
        "en moins par rapport au plus petit (runs avec journal) : "
        + (", ".join(lines) if lines else "—") + ".")
    parts = []
    for ds, d in data.items():
        r = next(r for r in d["runs"] if r["run"] == d["best"])
        tot = sum(r["final_parts"].values()) or 1
        dom = max(r["final_parts"], key=r["final_parts"].get)
        parts.append(f"{d['label']} {dom} {_f(100 * r['final_parts'][dom] / tot, 0)} %")
    obs["composantes"] = ("Terme dominant de la perte finale du meilleur run : "
                          + ", ".join(parts) + ".")
    cls = []
    for ds, d in data.items():
        r = next(r for r in d["runs"] if r["run"] == d["best"])
        ap = r.get("classes50") or r.get("classes") or {}
        present = {k: v for k, v in ap.items() if v is not None}
        nz = [k for k, v in present.items() if v >= 5]
        top = max(present, key=present.get) if present else "—"
        cls.append(f"{d['label']} {len(nz)}/{len(present)} (meilleure : {top} "
                   f"{_f(present.get(top), 1)})")
    obs["classes"] = ("Classes présentes au-dessus de 5 points d'AP@0,5, meilleur run : "
                      + " ; ".join(cls) + ".")
    gaps = []
    for ds, d in data.items():
        if not d["baselines"]:
            continue
        name, b = max(d["baselines"].items(), key=lambda kv: kv[1]["map"])
        gaps.append(f"{d['label']} {_f(d['top'])} contre {_f(b['map'])} ({name}, "
                    f"{b['n_classes']} classes évaluées)")
    obs["publies"] = ("Meilleur run affiné face au meilleur poids publié : " + " ; ".join(gaps)
                      + "." if gaps else "Aucune inférence des poids publiés exécutée.")
    dur = [(r["batch"], r["duration_s"]) for d in data.values() for r in d["runs"]
           if r["duration_s"]]
    obs["cout"] = ("Durée moyenne d'un run de 600 itérations : "
                   + ", ".join(f"lot {b} {_f(np.mean([s for bb_, s in dur if bb_ == b]) / 60, 1)} min"
                               for b in sorted({b for b, _ in dur})) + ".")
    return obs


def conclusions(data, pending=None):
    """Synthèse Markdown du notebook d'analyse, calculée."""
    sets, names, m = score_matrix(data)
    rel = m / m.max(axis=1, keepdims=True)
    mean_rel = rel.mean(axis=0)
    best = names[int(np.argmax(mean_rel))]
    eff = effects(data)
    deltas = [v for per in eff["sall_minus_s500"].values() for v in per.values()]
    out = [f"- **Configuration la plus robuste : `{best}`**, score relatif moyen "
           f"{_f(float(mean_rel.max()))} sur {len(sets)} jeux "
           f"(meilleur run de {sum(data[ds]['best'] == best for ds in sets)} jeux).",
           f"- **Tout le split plutôt que 500 images** : gain moyen de "
           f"{_f(float(np.mean(deltas)))} en score relatif ({sum(v > 0 for v in deltas)} cas "
           f"sur {len(deltas)}).",
           "- **Affinage trop court** : le meilleur run reste sous les poids publiés hors "
           "domaine sur "
           + (", ".join(f"{d['label']} ({_f(d['top'])} contre "
                        f"{_f(max(b['map'] for b in d['baselines'].values()))})"
                        for d in data.values() if d["baselines"] and d["top"]
                        < max(b["map"] for b in d["baselines"].values())) or "aucun jeu")
           + " ; 600 itérations (moins de 7 époques) ne suffisent pas à réapprendre les têtes."]
    if pending:
        out.append("- Balayages pas encore exécutés : "
                   + ", ".join(f"`{p}`" for p in pending.values()) + ".")
    return "\n".join(out)


# ---------------------------------------------------------------------------- CLI

def to_json(data):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items()}
        if isinstance(v, list):
            return [conv(x) for x in v]
        return v
    return json.dumps(conv(data), indent=1, ensure_ascii=False)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.notebooks.balayages",
                                 description=__doc__.split("\n\n")[0])
    ap.add_argument("--datasets", default=None, help="jeux séparés par des virgules")
    ap.add_argument("--json", type=Path, default=None, help="écrit toutes les données")
    ap.add_argument("--runs", action="store_true", help="table de tous les runs")
    args = ap.parse_args(argv)
    data, pending = load_all(datasets=args.datasets.split(",") if args.datasets else None)
    if not data:
        raise SystemExit("aucun balayage exécuté dans notebooks/*/*_sweep.ipynb")
    print(summary_table(data))
    if args.runs:
        print()
        print(runs_table(data))
    print()
    print(conclusions(data, pending))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(to_json(data))
        print(f"\ndonnées : {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
