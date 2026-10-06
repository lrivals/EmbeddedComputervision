"""Activations des premières couches sur deux jeux, face aux échelles INT8 (T11.3, §9.2).

    # VOC face à ExDark, échelles calibrées sur VOC
    python tools/act_hist.py --net tiny-yolov2-voc --datasets voc,exdark
    # mêmes histogrammes, échelles calibrées sur ExDark
    python tools/act_hist.py --net tiny-yolov2-voc --datasets voc,exdark \\
        --calib build/quant/tiny-yolov2-voc/calib-exdark.json

Pour chaque convolution d'indice ≤ `--max-layer` (L00-L04 par défaut) et chaque jeu :
percentiles de |x|, part écrêtée hors de ±127·s et **niveaux int8 utilisés** (|round(x/s)|
distincts, sur 128) avec l'échelle s de `--calib`. Des images sombres resserrent les
activations : peu de niveaux utilisés signale une échelle mal adaptée au jeu.
Sorties : `<out>/act_hist.md` et `<out>/act_hist.png` (si matplotlib est installé).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from yolo.data import datasets  # noqa: E402

# Palette catégorielle de référence (bleu, orange), un jeu par couleur.
COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def level_stats(values, scale):
    """(part écrêtée, niveaux |q| distincts utilisés sur 128) de `values` à l'échelle `scale`."""
    q = np.abs(np.round(np.asarray(values, np.float64) / scale))
    return float((q > 127).mean()), int(np.unique(np.minimum(q, 127)).size)


def table(rows):
    lines = ["| Couche | Jeu | p50 \\|x\\| | p99 | p99,99 | max | s | écrêté | niveaux utilisés |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| L{r['layer']:02d} | {r['dataset']} | {r['p50']:.3g} | {r['p99']:.3g} | "
                     f"{r['p9999']:.3g} | {r['max']:.3g} | {r['scale']:.3g} | "
                     f"{r['clip'] * 100:.3f} % | {r['levels']} / 128 |")
    return "\n".join(lines) + "\n"


def plot(values, scales, names, path):
    """Histogrammes de |x|/s (une colonne par couche), un contour par jeu, axe y log."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib absent : pas de figure")
        return False
    layers = sorted(scales)
    fig, axes = plt.subplots(1, len(layers), figsize=(3.2 * len(layers), 2.8), sharey=True)
    axes = np.atleast_1d(axes)
    bins = np.linspace(0, 140, 71)
    for ax, i in zip(axes, layers):
        for k, name in enumerate(names):
            ax.hist(np.abs(values[name][i]) / scales[i], bins=bins, histtype="step",
                    linewidth=2, color=COLORS[k % len(COLORS)], label=name, density=True)
        ax.axvline(127, color="#888780", linewidth=1, linestyle="--")
        ax.set_yscale("log")
        ax.set_title(f"L{i:02d}", fontsize=10)
        ax.set_xlabel("|x| / s (niveaux int8)", fontsize=9)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("densité")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def main():
    from calibrate import collect, load_fused, net_tag

    from yolo.quant.calibrate import load_scales

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--datasets", default="voc,exdark", help="jeux comparés, séparés par des virgules")
    ap.add_argument("--role", choices=("calib", "test"), default="test",
                    help="split de chaque jeu (défaut : celui de l'évaluation)")
    ap.add_argument("--images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-layer", type=int, default=4)
    ap.add_argument("--calib", type=Path, default=None,
                    help="échelles (défaut build/quant/<net>/calib.json, calibrées sur VOC)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None, help="défaut build/m11/act_hist/<net>")
    args = ap.parse_args()

    calib = args.calib or ROOT / "build" / "quant" / net_tag(args.net) / "calib.json"
    _, act_scales = load_scales(calib)
    fused = load_fused(args.net, args.weights)
    names = args.datasets.split(",")
    values, rows = {}, []
    for name in names:
        samples = datasets.load(name, role=args.role)
        idx = np.random.default_rng(args.seed).choice(
            len(samples), size=min(args.images, len(samples)), replace=False)
        stats = collect(fused, [samples[i]["image"] for i in sorted(idx)], workers=args.workers)
        values[name] = {i: stats.values(i) for i in stats.convs if i <= args.max_layer}
    scales = {i: act_scales[i] for i in values[names[0]]}
    for i in sorted(scales):
        for name in names:
            v = np.abs(values[name][i])
            clip, levels = level_stats(v, scales[i])
            rows.append({"layer": i, "dataset": name, "p50": np.percentile(v, 50),
                         "p99": np.percentile(v, 99), "p9999": np.percentile(v, 99.99),
                         "max": v.max(), "scale": scales[i], "clip": clip, "levels": levels})

    out = args.out or ROOT / "build" / "m11" / "act_hist" / net_tag(args.net)
    out.mkdir(parents=True, exist_ok=True)
    text = (f"### Activations de {args.net} : {', '.join(names)} ({args.images} images "
            f"{args.role} par jeu), échelles {calib}\n\n" + table(rows))
    (out / "act_hist.md").write_text(text)
    print(text)
    if plot(values, scales, names, out / "act_hist.png"):
        print(f"figure : {out / 'act_hist.png'}")


if __name__ == "__main__":
    main()
