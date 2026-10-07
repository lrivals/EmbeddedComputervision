"""Figures du projet (M13, docs/tasks/M13-figures.md) : registre nom → générateur.

    python -m tools.figures all            # tout ce dont les données sont présentes (make figures)
    python -m tools.figures resultats      # une famille
    python -m tools.figures map_stades     # une figure
    python -m tools.figures --list

Un générateur a la forme `fn(out_dir) -> list[Path]` ; il lève `MissingSource` quand sa
donnée manque, et la figure est alors sautée avec un message (ce n'est pas une erreur).
Les figures publiées vont dans `results/figures/<famille>/`. Celles qui portent sur un
sous-ensemble d'images (`subset=True`, règle M12) vont dans `build/figures/<famille>/`, ou
dans `dest/<famille>/` si la figure en donne un (illustrations de `docs/tasks/`).
matplotlib n'est jamais importé depuis `python/yolo/` (ADR 0001) : il est chargé en
paresseux par `style.plt()`.
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
for _p in (ROOT, ROOT / "python"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

RESULTS = ROOT / "results" / "figures"
BUILD = ROOT / "build" / "figures"
FAMILIES = ("projet", "modeles", "resultats", "materiel", "maths", "reseaux")


class MissingSource(Exception):
    """Donnée d'entrée absente : la figure est sautée."""


@dataclass
class Figure:
    name: str
    family: str
    task: str
    caption: str
    source: str
    fn: Callable = field(repr=False)
    subset: bool = False
    dest: Path | None = None

    @property
    def command(self):
        return f"python -m tools.figures {self.name}"


FIGURES: dict[str, Figure] = {}


def figure(name, family, task, caption, source, subset=False, dest=None):
    """Décorateur : enregistre `fn(out_dir) -> list[Path]` sous `name`."""
    assert family in FAMILIES, family

    def deco(fn):
        assert name not in FIGURES, name
        FIGURES[name] = Figure(name, family, task, caption, source, fn, subset, dest)
        return fn
    return deco


def need(*paths):
    """Lève `MissingSource` si l'un des chemins manque ; rend les chemins."""
    paths = [Path(p) for p in paths]
    for p in paths:
        if not p.exists():
            raise MissingSource(f"{p.relative_to(ROOT) if p.is_relative_to(ROOT) else p} absent")
    return paths[0] if len(paths) == 1 else paths


def out_dir_of(fig, results=RESULTS, build=BUILD):
    if fig.dest is not None:
        return Path(fig.dest) / fig.family
    return (build if fig.subset else results) / fig.family


def select(targets):
    """Noms, familles ou `all` → liste de figures (ordre du registre)."""
    out = []
    for t in targets:
        if t == "all":
            out += FIGURES.values()
        elif t in FAMILIES:
            out += [f for f in FIGURES.values() if f.family == t]
        elif t in FIGURES:
            out.append(FIGURES[t])
        else:
            raise SystemExit(f"figure ou famille inconnue : {t} (voir --list)")
    seen = set()
    return [f for f in out if not (f.name in seen or seen.add(f.name))]


def run(figs, results=RESULTS, build=BUILD, verbose=True):
    """Génère `figs` ; rend {nom: (état, chemins ou message)} avec état ok | sautée | erreur."""
    status = {}
    for fig in figs:
        out = out_dir_of(fig, results, build)
        try:
            paths = fig.fn(out)
            status[fig.name] = ("ok", paths)
            if verbose:
                print(f"  ok      {fig.name:18s} → {', '.join(_rel(p) for p in paths if p.suffix == '.png')}")
        except MissingSource as e:
            status[fig.name] = ("sautée", str(e))
            if verbose:
                print(f"  sautée  {fig.name:24s} ({e})")
    return status


def _rel(p):
    p = Path(p)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name


from tools.figures import maths, materiel, modeles, projet, resultats, reseaux  # noqa: E402,F401  (enregistrement)
from tools.figures import balayages  # noqa: E402,F401  (après resultats : ordre du registre)
