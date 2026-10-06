"""Figures automatiques en fin de run (M13) : entraînement, évaluation, modèle de cycles.

    from tools.figures.auto import after_run
    after_run("train", args.out)              # build/train/<run>/figures/
    python -m tools.figures --run m12 --dir build/m12/qat   # depuis un script shell

Actif par défaut ; `YOLO_FIGURES=0` (ou `--no-figures` des outils) le coupe. Les figures
vont dans `<sorties du run>/figures/`, jamais dans `results/` : seul `make figures` publie.
Ce mode n'échoue jamais : matplotlib absent ou erreur de tracé donnent un avertissement,
et le code de retour du run reste inchangé.
"""

import json
import os
import traceback
from pathlib import Path

from tools.figures import ROOT

KINDS = ("train", "eval-quant", "eval-voc", "map-stades", "perf-model", "m12", "scan")


def enabled():
    return os.environ.get("YOLO_FIGURES", "1").strip().lower() not in ("0", "false", "non", "")


def _train(path, out):
    from tools.figures import resultats as r

    return r.plot_training([r.load_training(path)], out, "entrainement")


def _eval_json(path, out):
    from tools.figures import resultats as r

    d = json.loads(Path(path).read_text())
    stem = Path(path).stem.removeprefix("map_")
    paths = r.plot_eval_json(d, out, f"map_{stem}")
    if any(k.startswith("fq:") for k in d["results"]):
        f, fall, n, per = r.load_fq_sensitivity(path)
        paths += r.plot_sensitivity([(f"INT8 seule couche face au flottant ({n} images)",
                                      {"INT8 (fake-quant)": per})], out, f"sensibilite_{stem}")
    return paths


def _eval_voc(path, out, dets=None, samples=None, names=None, title=""):
    """Courbes PR d'une évaluation : détections et vérités passées par l'outil, sinon les
    fichiers comp4 de `path` face à VOC2007 test."""
    from tools.figures import resultats as r

    if dets is None:
        samples = samples or r.test_samples()
        curves = {Path(path).name: r.load_pr(path, samples, names)}
    else:
        curves = {Path(path).name: r.pr_curves(dets, samples, list(names))}
    if len(next(iter(curves.values()))) > 20:
        print("figures : courbes PR limitées aux jeux de 20 classes au plus")
        return []
    return r.plot_pr(curves, out, "pr", title or f"{Path(path).name}, {len(samples)} images")


def _map_stades(path, out):
    from tools.figures import resultats as r

    return r.plot_map_stades(json.loads(Path(path).read_text()), out)


def _perf_model(path, out):
    from tools.figures import resultats as r

    paths = r.plot_cascade(*r.load_cascade(), out)
    csv = ROOT / "build" / "hls" / "cycles_conv.csv"
    if csv.exists():
        paths += r.plot_cycles(r.load_cycles(csv), out)
    try:
        b, pts = r.load_roofline_layers()
        paths += r.plot_roofline_layers(b, pts, r.NET, out, f"roofline_couches_{b['file']}")
    except Exception as e:  # pyyaml absent : la cascade suffit
        print(f"figures : roofline par couche sautée ({e})")
    return paths


def _scan(path, out):
    """Tout ce qu'on reconnaît sous `path` : runs d'entraînement et tables JSON de mAP."""
    path = Path(path)
    paths = []
    for loss in sorted(path.rglob("loss.csv")):
        if "figures" not in loss.parts:
            paths += _train(loss.parent, out / loss.parent.relative_to(path)
                            if loss.parent != path else out)
    summary = []
    for js in sorted(path.rglob("*.json")):
        if "figures" in js.parts:
            continue
        try:
            d = json.loads(js.read_text())
        except (ValueError, UnicodeDecodeError):
            continue
        if isinstance(d, dict) and isinstance(d.get("results"), dict):
            summary += [(f"{js.relative_to(path).with_suffix('')} · {k}", v["map"],
                         d.get("images", 0)) for k, v in d["results"].items()
                        if isinstance(v, dict) and "map" in v]
            try:
                paths += _eval_json(js, out)
            except Exception as e:
                print(f"figures : {js.name} ignoré ({e})")
    if len(summary) > 1:
        from tools.figures import resultats as r

        paths += r.plot_map_summary(summary, out, title=f"mAP de {path.name}/ "
                                    "(hachuré : sous-ensemble d'images)")
    return paths


HANDLERS = {"train": _train, "eval-quant": _eval_json, "eval-voc": _eval_voc,
            "map-stades": _map_stades, "perf-model": _perf_model, "m12": _scan, "scan": _scan}


def after_run(kind, path, out=None, **kw):
    """Figures du run `kind` dont les sorties sont dans `path` (fichier ou dossier).

    `out` : défaut `<path>/figures/` (ou `<dossier du fichier>/figures/`). Rend la liste des
    fichiers écrits ; [] si désactivé ou en cas d'échec (avertissement imprimé).
    """
    if not enabled():
        return []
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("figures : matplotlib absent, figures sautées (pip install -e 'python[plots]')")
        return []
    if kind not in HANDLERS:
        print(f"figures : type de run inconnu {kind!r} ({', '.join(KINDS)})")
        return []
    path = Path(path)
    if out is None:
        out = (path if path.is_dir() or not path.suffix else path.parent) / "figures"
    try:
        paths = HANDLERS[kind](path, Path(out), **kw)
    except Exception as e:
        print(f"figures : échec du tracé ({type(e).__name__}: {e}) — le run n'est pas affecté")
        if os.environ.get("YOLO_FIGURES_DEBUG"):
            traceback.print_exc()
        return []
    pngs = [p for p in paths if str(p).endswith(".png")]
    if pngs:
        print("figures : " + ", ".join(_short(p) for p in pngs))
    return paths


def _short(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
