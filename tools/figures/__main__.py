"""python -m tools.figures <nom|famille|all> … ; --list ; --run KIND --dir PATH (mode auto)."""

import argparse
import os
import sys
from pathlib import Path

from tools.figures import BUILD, FIGURES, RESULTS, out_dir_of, run, select

FAMILY_TITLES = {"resultats": "Résultats (D)", "modeles": "Modèles (B)",
                 "projet": "Développement du projet (E)", "materiel": "Matériel (C)",
                 "maths": "Réimplémentation de zéro (F)", "reseaux": "Réseaux en détail (G)"}


def images_of(fig, results=RESULTS, build=BUILD):
    d = out_dir_of(fig, results, build)
    return sorted(set(d.glob(f"{fig.name}.png")) | set(d.glob(f"{fig.name}_*.png")))


def gallery(status=None, results=RESULTS, build=BUILD):
    """Réécrit `results/figures.md` (à côté de `results/figures/`) depuis le registre."""
    status = status or {}
    path = Path(results).parent / "figures.md"
    lines = ["# Figures (M13)", "",
             "Galerie générée par `python -m tools.figures all` (`make figures`) à partir du "
             "registre de `tools/figures/` ; ne pas éditer à la main. Chaque figure se "
             "régénère par sa commande et lit ses données dans la source indiquée. Les "
             "figures absentes attendent leur donnée (voir la commande de la tâche).", ""]
    for fam, title in FAMILY_TITLES.items():
        figs = [f for f in FIGURES.values() if f.family == fam]
        if not figs:
            continue
        lines += [f"## {title}", ""]
        for f in figs:
            lines += [f"### {f.task} — `{f.name}`", "", f.caption, "",
                      f"- Commande : `{f.command}`", f"- Source : `{f.source}`"]
            imgs = images_of(f, results, build)
            if f.subset:
                lines.append("- Sous-ensemble d'images : sortie dans "
                             f"`build/figures/{f.family}/`, non publiée (règle M12).")
            elif imgs:
                lines.append("")
                for p in imgs:
                    rel = os.path.relpath(p, path.parent)
                    lines.append(f"![{p.stem}]({rel})")
            else:
                why = status.get(f.name, ("", "pas encore générée"))[1]
                lines.append(f"- Non générée : {why}.")
            lines.append("")
    path.write_text("\n".join(lines))
    return path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m tools.figures", description=__doc__)
    ap.add_argument("targets", nargs="*", default=[], help="nom, famille ou all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--results", type=Path, default=RESULTS)
    ap.add_argument("--build", type=Path, default=BUILD)
    ap.add_argument("--no-gallery", action="store_true")
    ap.add_argument("--run", default=None, help="mode auto : type de run (train, eval-quant…)")
    ap.add_argument("--dir", type=Path, default=None, help="mode auto : sorties du run")
    args = ap.parse_args(argv)

    if args.list:
        for f in FIGURES.values():
            print(f"{f.name:20s} {f.family:10s} {f.task:7s} {f.source}")
        return 0
    if args.run:
        from tools.figures.auto import after_run

        after_run(args.run, args.dir or Path.cwd())
        return 0
    if not args.targets:
        ap.error("préciser une figure, une famille ou all (voir --list)")
    status = run(select(args.targets), args.results, args.build)
    if "all" in args.targets and not args.no_gallery:
        print(f"→ {gallery(status, args.results, args.build)}")
    n_ok = sum(s == "ok" for s, _ in status.values())
    print(f"{n_ok} générées, {len(status) - n_ok} sautées")
    return 0


if __name__ == "__main__":
    sys.exit(main())
