"""python -m tools.notebooks <nom|jeu|rôle|all> … ; --list ; --check (make ci)."""

import argparse
import sys
from pathlib import Path

from tools.notebooks import NB_DIR, ROLES, ROOT
from tools.notebooks.gabarits import COLAB, ROLE_TITLES, SUBSET_FIRST, VIEWER, dumps, render, viewer
from tools.notebooks.matrice import NOTEBOOKS, how_to_get, palier_of, prerequis
from tools.notebooks.commandes import M11_TRAIN


def select(targets):
    """Noms (`kitti/tiny-yolov3-kitti_train` ou `tiny-yolov3-kitti_train`), jeux, rôles ou
    `all` → notebooks, dans l'ordre du registre."""
    out = []
    for t in targets:
        if t == "all":
            hits = list(NOTEBOOKS.values())
        elif t in ROLES:
            hits = [nb for nb in NOTEBOOKS.values() if nb.role == t]
        else:
            hits = [nb for key, nb in NOTEBOOKS.items()
                    if t in (key, nb.dataset, nb.name, str(nb.path))]
        if not hits:
            raise SystemExit(f"notebook, jeu ou rôle inconnu : {t} (voir --list)")
        out += hits
    seen = set()
    return [nb for nb in out if not (nb.path in seen or seen.add(nb.path))]


def _palier(nb):
    """« premier passage / complet » pour l'inférence ; ITERS par défaut pour l'entraînement."""
    if nb.trains:
        return palier_of(nb, 0, M11_TRAIN["iters"])
    return f"{palier_of(nb, SUBSET_FIRST)} / {palier_of(nb)}"


def index():
    """notebooks/README.md, réécrit depuis le registre (comme results/figures.md)."""
    lines = ["# Notebooks (M14)", "",
             "Index généré par `python -m tools.notebooks all` (`make notebooks`) depuis le "
             "registre de `tools/notebooks/matrice.py` ; ne pas éditer à la main. Les "
             "notebooks sont versionnés vides, ou exécutés en entier sans erreur (sorties visibles "
             "ici) ; leurs résultats vont dans "
             "`build/notebooks/<jeu>/<modèle>/`. Voir "
             "[docs/tasks/M14-notebooks.md](../docs/tasks/M14-notebooks.md).", "",
             f"Palier (règles de M12) : premier passage (`SUBSET = {SUBSET_FIRST}`) / passage "
             f"complet (`SUBSET = 0`, ou `ITERS = {M11_TRAIN['iters']}` à l'entraînement).", "",
             f"Affichage seul : [{VIEWER}]({VIEWER}) montre les figures (M13, runs) et les "
             "réaffiche dès qu'une image est produite.", "",
             "| Jeu | Modèle | Rôle | Palier | Prérequis | Colab |", "|---|---|---|---|---|---|"]
    for nb in NOTEBOOKS.values():
        pal = _palier(nb)
        req = "<br>".join(f"{k} : `{v}` ({how_to_get(k, nb)})" for k, v in prerequis(nb).items()
                          if not (nb.trains and k == "cfg"))
        badge = (f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]"
                 f"({COLAB}/notebooks/{nb.path.as_posix()})")
        lines.append(f"| {nb.dataset} | [{nb.model}]({nb.path.as_posix()}) | "
                     f"{ROLE_TITLES[nb.role]} | {pal} | {req} | {badge} |")
    return "\n".join(lines) + "\n"


def generate(nbs, out_dir=NB_DIR):
    """{chemin: texte} des notebooks `nbs` et de l'index."""
    files = {Path(out_dir) / nb.path: dumps(render(nb)) for nb in nbs}
    files[Path(out_dir) / VIEWER] = dumps(viewer())
    files[Path(out_dir) / "README.md"] = index()
    return files


# Métadonnées propres au noyau qui a exécuté le notebook (VS Code, Colab) : ignorées.
RUN_META = ("kernelspec", "language_info", "widgets")


def strip_outputs(text):
    """Texte du notebook sans sorties ni métadonnées d'exécution (format du générateur)."""
    import json

    nb = json.loads(text)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            c["outputs"], c["execution_count"] = [], None
        c["metadata"].pop("execution", None)
    for k in RUN_META:
        nb["metadata"].pop(k, None)
    return nb


def executed_ok(text):
    """None si chaque cellule de code a tourné sans erreur ; sinon la raison."""
    import json

    code = [c for c in json.loads(text)["cells"] if c["cell_type"] == "code"]
    if any(c.get("execution_count") is None for c in code):
        return "exécution partielle"
    for c in code:
        for o in c.get("outputs", []):
            if o.get("output_type") == "error":
                return f"erreur {o.get('ename', '')} en cellule {c.get('id', '?')}"
    return None


def state(path, text):
    """État d'un fichier face au texte généré : à jour | exécuté | <raison de l'écart>.

    Règle M14 : un notebook versionné est soit sans sorties et identique au générateur,
    soit **entièrement exécuté sans erreur** avec les cellules du générateur (sorties
    visibles sur GitHub). Une exécution partielle ou en erreur ne se versionne pas."""
    if not path.exists():
        return "absent"
    cur = path.read_text()
    if cur == text:
        return "à jour"
    if path.suffix != ".ipynb":
        return "diffère du générateur"
    gen = strip_outputs(text)
    try:
        mine = strip_outputs(cur)
    except (ValueError, KeyError):
        return "JSON illisible"
    if mine != gen:
        return "diffère du générateur"
    why = executed_ok(cur)
    return f"exécuté, refusé : {why}" if why else "exécuté"


def _rel(p):
    p = Path(p).resolve()
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.notebooks", description=__doc__)
    ap.add_argument("targets", nargs="*", default=[], help="nom, jeu, rôle ou all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--check", action="store_true",
                    help="échoue si un notebook versionné diffère du générateur (sorties "
                    "admises si l'exécution est complète et sans erreur)")
    ap.add_argument("--out", type=Path, default=NB_DIR, help="dossier des notebooks")
    args = ap.parse_args(argv)

    if args.list:
        for key, nb in NOTEBOOKS.items():
            print(f"{key:40s} palier {_palier(nb):5s}  {nb.net}")
        return 0
    if args.check:
        files = generate(NOTEBOOKS.values(), args.out)
        states = {p: state(p, text) for p, text in files.items()}
        bad = [p for p, st in states.items() if st not in ("à jour", "exécuté")]
        run = [p for p, st in states.items() if st == "exécuté"]
        known = set(files)
        extra = [p for p in Path(args.out).rglob("*.ipynb") if p not in known
                 and ".ipynb_checkpoints" not in p.parts]
        for p in bad:
            print(f"{states[p]} : {_rel(p)}")
        for p in extra:
            print(f"absent du registre : {_rel(p)}")
        if bad or extra:
            print("régénérer : make notebooks")
            return 1
        print(f"{len(files)} fichiers à jour, dont {len(run)} notebooks exécutés sans erreur")
        return 0
    if not args.targets:
        ap.error("préciser un notebook, un jeu, un rôle ou all (voir --list)")
    files = generate(select(args.targets), args.out)
    for p, text in files.items():
        p.parent.mkdir(parents=True, exist_ok=True)
        st = state(p, text)
        if st == "exécuté":  # exécution complète gardée (règle M14)
            print(f"  gardé  {_rel(p)} (exécuté sans erreur)")
        elif st != "à jour":
            p.write_text(text)
            print(f"  écrit  {_rel(p)}" + (f" ({st})" if st.startswith("exécuté") else ""))
    print(f"{len(files) - 2} notebooks + {VIEWER}, index {_rel(args.out / 'README.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
