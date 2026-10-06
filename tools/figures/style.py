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
            "lines.linewidth": 1.6, "hatch.linewidth": 0.6,
        })
        _plt = pyplot
    return _plt


def save(fig, out_dir, name):
    """PNG 150 dpi dans `out_dir` ; ferme la figure ; rend [chemin]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / f"{name}.png"]
    fig.savefig(paths[0], dpi=DPI, bbox_inches="tight")
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


def draw_boxes(ax, xyxy, texts=(), color=PALETTE[0], lw=1.6, ls="-", fontsize=6.5):
    """Rectangles `xyxy` (pixels) et étiquettes sur une image affichée par `imshow`."""
    from matplotlib.patches import Rectangle

    texts = list(texts) + [""] * (len(xyxy) - len(texts))
    for (x1, y1, x2, y2), t in zip(xyxy, texts):
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=lw, ls=ls))
        if t:
            ax.text(x1 + 1, y1 + 1, t, fontsize=fontsize, color="white", va="top",
                    bbox=dict(fc=color, ec="none", pad=0.6, alpha=0.9))


def image_axes(ax, title=""):
    """Axes d'image : sans grille ni graduations."""
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if title:
        ax.set_title(title)


def block(ax, x, y, w, h, text="", color=PALETTE[0], fill=0.14, hatch="", fontsize=8,
          weight="normal", ls="-", align="center"):
    """Bloc de schéma (coordonnées de données) : cadre coloré, fond léger, texte centré."""
    from matplotlib.colors import to_rgba
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.08",
                                fc=to_rgba(color, fill), ec=color, lw=1.2, hatch=hatch, ls=ls))
    if text:
        tx = x + w / 2 if align == "center" else x + 0.08
        ax.text(tx, y + h / 2, text, ha=align, va="center", fontsize=fontsize, color=INK,
                weight=weight, linespacing=1.25)


def arrow(ax, p, q, text="", color=INK2, ls="-", lw=1.2, fontsize=7, offset=(0, 0.12),
          both=False):
    """Flèche de `p` à `q` avec une étiquette au milieu."""
    ax.annotate("", q, p, arrowprops=dict(arrowstyle="<->" if both else "->", color=color, lw=lw,
                                          ls=ls, shrinkA=2, shrinkB=2))
    if text:
        ax.text((p[0] + q[0]) / 2 + offset[0], (p[1] + q[1]) / 2 + offset[1], text,
                ha="center", va="bottom", fontsize=fontsize, color=color)


def schema_axes(ax, xlim, ylim):
    """Axes de schéma : repère de données fixe, sans axes."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal")
    ax.axis("off")
