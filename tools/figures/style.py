"""Style commun des figures M13 : palette, couleurs fixes, tailles, enregistrement.

Palette catégorielle validée pour les daltoniens (ordre fixe, jamais cyclé) ; chaque stade
et chaque réseau a sa couleur. Projection et mesure se distinguent aussi sans la couleur
(hachures, marqueurs creux, traits pointillés), pour rester lisibles en niveaux de gris.
"""

from pathlib import Path

PALETTE = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7",
           "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
GREY = "#b5b4ae"

STAGE_COLORS = {"flottant": PALETTE[0], "entier": PALETTE[1], "csim": PALETTE[2],
                "carte": PALETTE[6]}
STAGE_HATCH = {"flottant": "", "entier": "//", "csim": "..", "carte": "xx"}
NET_COLORS = {"tiny-yolov2-voc": PALETTE[0], "tiny-yolov3-coco": PALETTE[1]}
NET_LABELS = {"tiny-yolov2-voc": "Tiny-YOLOv2 VOC", "tiny-yolov3-coco": "Tiny-YOLOv3 COCO"}
NETS = tuple(NET_COLORS)

COL = 6.4   # largeur une colonne (pouces)
FULL = 10.0  # pleine page
DPI = 150

# Mesure : marque pleine ; projection / estimation : marque creuse, hachée, pointillée.
MEASURED = dict(fill=True, hatch="", linestyle="-")
PROJECTED = dict(fill=False, hatch="///", linestyle="--")

_plt = None


def plt():
    """matplotlib.pyplot chargé en paresseux, backend Agg, avec le style du projet."""
    global _plt
    if _plt is None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot

        pyplot.rcParams.update({
            "figure.facecolor": "white", "axes.facecolor": "white",
            "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "axes.titlesize": 10,
            "axes.titleweight": "bold", "axes.labelsize": 9, "axes.grid": True,
            "axes.spines.top": False, "axes.spines.right": False,
            "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
            "grid.color": GRID, "grid.linewidth": 0.6, "xtick.color": INK2,
            "ytick.color": INK2, "xtick.labelsize": 8, "ytick.labelsize": 8,
            "legend.fontsize": 8, "legend.frameon": False, "font.size": 9,
            "lines.linewidth": 1.6, "hatch.linewidth": 0.6, "svg.fonttype": "none",
        })
        _plt = pyplot
    return _plt


def save(fig, out_dir, name):
    """PNG 150 dpi + SVG dans `out_dir` ; ferme la figure ; rend les deux chemins."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png", out_dir / f"{name}.svg"]
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    plt().close(fig)
    return paths


def note(ax, text, loc="upper right"):
    """Petite mention en gris dans un coin (source, « projection C-sim »…)."""
    x, ha = (0.99, "right") if "right" in loc else (0.01, "left")
    y, va = (0.98, "top") if "upper" in loc else (0.02, "bottom")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=7, color=MUTED)


def smooth(values, window):
    """Moyenne glissante centrée sur `window` points (bords : fenêtre tronquée)."""
    import numpy as np

    v = np.asarray(values, dtype=np.float64)
    if window <= 1 or len(v) < 2:
        return v
    k = np.ones(min(window, len(v)))
    return np.convolve(v, k, "same") / np.convolve(np.ones_like(v), k, "same")
