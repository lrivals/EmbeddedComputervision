"""Analyse comparative des balayages lot × sous-ensemble, tous jeux (T13.53, T14.10).

Données : `tools.notebooks.balayages.load_all()`, relues dans les sorties versionnées des
notebooks `notebooks/<jeu>/<modèle>_sweep.ipynb` et `_infer.ipynb` ; une fonction
`plot_*` par figure, qui prend ces données et rend la liste des PNG. Le notebook
`notebooks/analyse_balayages.ipynb` les enchaîne et les commente.

Encodages fixes dans toutes les figures : une couleur par jeu (`DS_COLORS`, ordre de
`balayages.ORDER`, jamais selon le rang), une teinte bleue de clair à foncé pour le lot
(8 → 32, grandeur ordonnée), marque pleine pour « tout le split » et creuse pour « 500
images » (comme `balayage_perte`, T13.53).
"""

import numpy as np

from tools.figures import ROOT, MissingSource, figure
from tools.figures import style as st

DS_COLORS = {"voc": st.PALETTE[0], "visdrone": st.PALETTE[1], "kitti": st.PALETTE[2],
             "flir": st.PALETTE[3], "exdark": st.PALETTE[4], "crowdhuman": st.PALETTE[6]}
BATCH_COLORS = {8: "#8fbbea", 16: "#2a78d6", 32: "#123f7a"}  # rampe bleue, clair → foncé
PART_COLORS = {"coord": st.PALETTE[0], "obj": st.PALETTE[1], "noobj": st.GREY,
               "cls": st.PALETTE[2]}
PART_LABELS = {"coord": "coord (boîtes)", "obj": "obj (objets)",
               "noobj": "noobj (fond)", "cls": "cls (classes)"}
SUBSET_LABEL = {0: "tout le split", 500: "500 images"}
FOOT = "50 images d'évaluation, palier R (non publiable) ; score = mAP@0,5 (AP50 pour FLIR)"


def _B():
    from tools.notebooks import balayages

    return balayages


def _color(ds):
    return DS_COLORS.get(ds, st.PALETTE[7])


def _batch_color(b):
    return BATCH_COLORS.get(b, st.MUTED)


def _subset_label(s):
    return SUBSET_LABEL.get(s, f"{s} images")


def _fmt(v, nd=2):
    return "—" if v is None else f"{v:.{nd}f}".replace(".", ",")


def _foot(fig, text=FOOT):
    """Mention en pied de figure, sous tout ce que tracent les axes (étiquettes comprises) :
    `st.save` enregistre en `bbox_inches="tight"`, la marge suit."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(r).y0 for ax in fig.axes) / fig.bbox.height
    fig.text(0.99, y0 - 0.01, text, ha="right", va="top", fontsize=7, color=st.MUTED)


def _scored(d):
    return [r for r in d["runs"] if r.get("map") is not None]


def _point(ax, x, y, ds, subset, size=46, z=3, label=None):
    """Marque d'un run : couleur du jeu, pleine (tout le split) ou creuse (500 images)."""
    c = _color(ds)
    ax.scatter([x], [y], s=size, zorder=z, facecolor=c if subset == 0 else "white",
               edgecolor=c, linewidth=1.6, label=label)


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", ["#f2f1ee", "#a9c9ef", "#2a78d6",
                                                     "#0f3366"])


def _div_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("div", [st.PALETTE[1], "#f2f1ee", st.PALETTE[0]])


def _ink_on(value, vmax=1.0):
    """Encre lisible sur une case de la rampe séquentielle."""
    return "white" if value is not None and value > 0.6 * vmax else st.INK


# ---------------------------------------------------------------- 1. grilles

def plot_grilles(data, out_dir, name="balayages_grilles"):
    """Une grille lot × sous-ensemble par jeu ; couleur = score / meilleur run du jeu."""
    plt = st.plt()
    sets = list(data)
    fig, axes = plt.subplots(1, len(sets), figsize=(st.FULL, 2.9), squeeze=False)
    cmap = _seq_cmap()
    for ax, ds in zip(axes[0], sets):
        d = data[ds]
        batches = sorted({r["batch"] for r in d["runs"]})
        subsets = sorted({r["subset"] for r in d["runs"]}, key=lambda s: (s == 0, s))
        grid = np.full((len(batches), len(subsets)), np.nan)
        for r in _scored(d):
            grid[batches.index(r["batch"]), subsets.index(r["subset"])] = r["rel"]
        ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
        for r in _scored(d):
            i, j = batches.index(r["batch"]), subsets.index(r["subset"])
            best = r["run"] == d["best"]
            ax.text(j, i, f"{_fmt(r['map'])}{' ★' if best else ''}\n{_fmt(r['rel'] * 100, 0)} %",
                    ha="center", va="center", fontsize=7.5, color=_ink_on(r["rel"]),
                    fontweight="bold" if best else "normal", linespacing=1.3)
        ax.set_xticks(range(len(subsets)), [_subset_label(s).replace(" le split", "")
                                             .replace(" images", " im.") for s in subsets])
        ax.set_yticks(range(len(batches)), [f"lot {b}" for b in batches]
                      if ax is axes[0][0] else [])
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{d['label']} ({_B().metric_label(d)})", fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 100))
    cb = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.025, pad=0.02)
    cb.set_label("% du meilleur run du jeu", fontsize=8)
    cb.outline.set_visible(False)
    fig.subplots_adjust(left=0.06, right=0.87, wspace=0.08)
    _foot(fig, "case : score, puis % du meilleur run ; ★ meilleur run. " + FOOT)
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 2. classements

def plot_classements(data, out_dir, name="balayages_classements"):
    """Rang de chaque run par jeu (gauche) ; accord des classements, Spearman (droite)."""
    B = _B()
    plt = st.plt()
    sets, names, m = B.score_matrix(data)
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(st.FULL, 3.8),
                                  gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(sets))
    ranks = np.array([B._ranks(-row) for row in m])  # 1 : meilleur
    for k, n in enumerate(names):
        b, s = B._name(n)
        c = _batch_color(b)
        ys = ranks[:, k]
        ax.plot(x, ys, color=c, lw=2, ls="-" if s == 0 else "--", zorder=2)
        ax.scatter(x, ys, s=52, zorder=3, facecolor=c if s == 0 else "white", edgecolor=c,
                   linewidth=1.6)
        ax.annotate(n, (x[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=7.5, color=st.INK2)
    ax.set_xticks(x, [data[ds]["label"] for ds in sets])
    ax.set_yticks(range(1, len(names) + 1))
    ax.set_ylim(len(names) + 0.5, 0.5)
    ax.set_xlim(-0.3, len(sets) - 0.3)
    ax.set_ylabel("rang du run dans le jeu (1 : meilleur)")
    ax.set_title("Classement des runs, jeu par jeu")
    ax.grid(axis="x", visible=False)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=_batch_color(b), lw=2, label=f"lot {b}")
               for b in sorted({B._name(n)[0] for n in names})]
    handles += [Line2D([], [], color=st.INK2, lw=1.4, ls="-", marker="o", label="tout le split"),
                Line2D([], [], color=st.INK2, lw=1.4, ls="--", marker="o", mfc="white",
                       label="500 images")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=5,
              fontsize=7.5)

    s, rho = B.rank_agreement(data)
    axr.imshow(rho, cmap=_div_cmap(), vmin=-1, vmax=1)
    for i in range(len(s)):
        for j in range(len(s)):
            axr.text(j, i, _fmt(rho[i, j]), ha="center", va="center", fontsize=7.5,
                     color="white" if abs(rho[i, j]) > 0.7 else st.INK)
    labels = [data[d]["label"] for d in s]
    axr.set_xticks(range(len(s)), labels, rotation=35, ha="right")
    axr.set_yticks(range(len(s)), labels)
    axr.grid(False)
    for sp in axr.spines.values():
        sp.set_visible(False)
    axr.set_title("Accord des classements (Spearman ρ)")
    fig.tight_layout()
    _foot(fig, "ρ = 1 : même ordre des 6 runs ; |ρ| < 0,83 non significatif à 5 % (n = 6). " + FOOT)
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 3. effets

def plot_effets(data, out_dir, name="balayages_effets"):
    """Effet du sous-ensemble (gauche, 500 images → tout le split) et du lot (droite)."""
    plt = st.plt()
    sets = list(data)
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(st.FULL, 4.0),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    rows, ylabels = [], []
    for ds in sets:
        d = data[ds]
        rel = {(r["batch"], r["subset"]): r["rel"] for r in _scored(d)}
        for b in sorted({r["batch"] for r in d["runs"]}):
            if (b, 0) in rel and (b, 500) in rel:
                rows.append((ds, b, rel[(b, 500)], rel[(b, 0)]))
                ylabels.append(f"{d['label']} · lot {b}")
    y = np.arange(len(rows))[::-1]
    for yi, (ds, b, a, z) in zip(y, rows):
        c = _color(ds)
        axl.annotate("", xy=(z, yi), xytext=(a, yi),
                     arrowprops=dict(arrowstyle="-|>", color=c, lw=1.6, shrinkA=4, shrinkB=4))
        _point(axl, a, yi, ds, 500)
        _point(axl, z, yi, ds, 0)
        axl.text(max(a, z) + 0.03, yi, f"{'+' if z >= a else '−'}{_fmt(abs(z - a) * 100, 0)} pts",
                 va="center", fontsize=7, color=st.INK2)
    axl.set_yticks(y, ylabels, fontsize=7.5)
    axl.set_xlim(0, 1.18)
    axl.set_xlabel("score relatif (1 : meilleur run du jeu)")
    axl.set_title("500 images (creux) → tout le split (plein)")
    axl.grid(axis="y", visible=False)

    batches = sorted({r["batch"] for d in data.values() for r in d["runs"]})
    rng = np.random.default_rng(0)
    for s, off in ((500, -0.12), (0, 0.12)):
        means = []
        for b in batches:
            vals = [(ds, r["rel"]) for ds in sets for r in _scored(data[ds])
                    if r["batch"] == b and r["subset"] == s]
            for ds, v in vals:
                _point(axr, batches.index(b) + off + rng.uniform(-0.05, 0.05), v, ds, s,
                       size=26, z=2)
            means.append(np.mean([v for _, v in vals]) if vals else np.nan)
        xs = np.arange(len(batches)) + off
        axr.plot(xs, means, color=st.INK, lw=2, ls="-" if s == 0 else "--", zorder=4,
                 marker="D", ms=6, mfc=st.INK if s == 0 else "white", mec=st.INK)
        axr.annotate(f"moyenne, {_subset_label(s)}", (xs[-1], means[-1]), xytext=(6, 0),
                     textcoords="offset points", va="center", fontsize=7, color=st.INK)
    axr.set_xticks(range(len(batches)), [f"lot {b}" for b in batches])
    axr.set_xlim(-0.5, len(batches) - 0.1)
    axr.set_ylim(0, 1.08)
    axr.set_ylabel("score relatif")
    axr.set_title("Effet du lot (points : jeux)")
    axr.grid(axis="x", visible=False)
    _legend_datasets(axr, sets, data, loc="lower right")
    fig.tight_layout()
    _foot(fig)
    return st.save(fig, out_dir, name)


def _legend_datasets(ax, sets, data, loc="best", **kw):
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], ls="", marker="o", ms=6, mfc=_color(ds), mec=_color(ds),
                      label=data[ds]["label"]) for ds in sets]
    ax.legend(handles=handles, loc=loc, fontsize=7, **kw)


# ---------------------------------------------------------------- 4. époques

def plot_epoques(data, out_dir, name="balayages_epoques"):
    """Époques vues (log) face au score relatif : voir plus d'images distinctes paie,
    revoir 500 images 10 à 38 fois ne paie pas."""
    plt = st.plt()
    sets = list(data)
    fig, ax = plt.subplots(figsize=(st.FULL, 3.8))
    for ds in sets:
        d = data[ds]
        pts = sorted((r for r in _scored(d) if r["epochs"]), key=lambda r: r["epochs"])
        for s in (0, 500):
            ser = [r for r in pts if r["subset"] == s]
            ax.plot([r["epochs"] for r in ser], [r["rel"] for r in ser], color=_color(ds),
                    lw=1, alpha=0.5, ls="-" if s == 0 else "--", zorder=1)
        for r in pts:
            _point(ax, r["epochs"], r["rel"], ds, r["subset"], size=18 + 1.6 * r["batch"])
    ax.set_xscale("log")
    ax.axvspan(8, 60, color=st.GRID, alpha=0.5, zorder=0, lw=0)
    eps = {s: [r["epochs"] for d in data.values() for r in d["runs"]
               if r["epochs"] and r["subset"] == s] for s in (0, 500)}
    if eps[500]:
        ax.text(42, 0.04, f"500 images :\n{min(eps[500]):.0f} à {max(eps[500]):.0f} passages",
                fontsize=7.5, color=st.INK2, ha="right")
    if eps[0]:
        ax.text(0.31, 0.04, f"tout le split : {max(eps[0]):.1f} époques au plus".replace(".", ","),
                fontsize=7.5, color=st.INK2)
    ax.set_xlim(0.22, 60)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("époques vues en 600 itérations (lot × 600 / images), échelle log")
    ax.set_ylabel("score relatif (1 : meilleur run du jeu)")
    ax.set_title("Nombre de passages sur les données face au score")
    _legend_datasets(ax, sets, data, loc="upper center", ncol=len(sets),
                     bbox_to_anchor=(0.5, -0.16))
    ax.text(0.99, 0.98, "taille de la marque : lot\npleine : tout le split, creuse : 500 images",
            transform=ax.transAxes, ha="right", va="top", fontsize=7, color=st.MUTED)
    _foot(fig)
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 5. perte / score

def _offset(r, pts):
    """Étiquette au-dessus à droite, en dessous si un voisin proche la masquerait."""
    sx = (max(q["final_loss"] for q in pts) - min(q["final_loss"] for q in pts)) or 1
    sy = (max(q["map"] for q in pts) - min(q["map"] for q in pts)) or 1
    for q in pts:
        if (q is not r and abs(q["final_loss"] - r["final_loss"]) < 0.15 * sx
                and 0 <= q["map"] - r["map"] < 0.1 * sy):
            return (4, -10)
    return (4, 3)


def plot_perte(data, out_dir, name="balayages_perte_score"):
    """Perte finale face au score, un panneau par jeu : une perte basse n'est pas une
    bonne mAP (les runs à 500 images surapprennent)."""
    B = _B()
    plt = st.plt()
    sets = list(data)
    fig, axes = plt.subplots(1, len(sets), figsize=(st.FULL, 3.0), squeeze=False)
    for ax, ds in zip(axes[0], sets):
        d = data[ds]
        pts = [r for r in _scored(d) if r["final_loss"] is not None]
        for r in pts:
            _point(ax, r["final_loss"], r["map"], ds, r["subset"], size=40)
            ax.annotate(r["run"], (r["final_loss"], r["map"]), xytext=_offset(r, pts),
                        textcoords="offset points", fontsize=6.5, color=st.INK2)
        rho = B.spearman([r["final_loss"] for r in pts], [r["map"] for r in pts])
        ax.set_title(f"{d['label']} — ρ = {_fmt(rho)}", fontsize=9)
        ax.set_xlabel("perte finale")
        ax.margins(x=0.25, y=0.15)
    axes[0][0].set_ylabel("score (%)")
    fig.tight_layout()
    _foot(fig, "ρ : Spearman perte / score ; > 0 : la perte la plus basse n'a pas le meilleur "
               "score. " + FOOT)
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 6. courbes

def plot_courbes(data, out_dir, name="balayages_courbes", window=25):
    """Pertes d'entraînement lissées : une ligne par jeu, tout le split / 500 images en
    colonnes, une couleur par lot ; burn-in grisé."""
    plt = st.plt()
    sets = [ds for ds in data if any(r["curve"] is not None for r in data[ds]["runs"])]
    if not sets:
        raise MissingSource("aucun journal d'entraînement dans les notebooks de balayage")
    fig, axes = plt.subplots(len(sets), 2, figsize=(st.FULL, 1.9 * len(sets) + 0.6),
                             sharex=True, squeeze=False)
    for i, ds in enumerate(sets):
        d = data[ds]
        for j, s in enumerate((0, 500)):
            ax = axes[i][j]
            burn = d.get("burn_in") or 0
            ax.axvspan(0, burn, color=st.GRID, alpha=0.6, lw=0, zorder=0)
            for r in d["runs"]:
                if r["subset"] != s or r["curve"] is None:
                    continue
                c = r["curve"]
                ax.plot(c["it"], st.smooth(c["loss"], window), color=_batch_color(r["batch"]),
                        lw=1.6, label=f"lot {r['batch']}" + (" (repris)" if r["resumed"] else ""))
            missing = [r["run"] for r in d["runs"] if r["subset"] == s and r["curve"] is None]
            if missing:
                st.note(ax, "sans journal : " + ", ".join(missing), loc="upper right")
            if i == 0:
                ax.set_title(_subset_label(s))
            if j == 0:
                ax.set_ylabel(f"{d['label']}\nperte", fontsize=8)
            # Le premier pas (têtes neuves) écrase l'échelle : on part de la fin du warm-up.
            tails = [r["curve"]["loss"][len(r["curve"]["loss"]) // 3:] for r in d["runs"]
                     if r["curve"] is not None]
            if tails:
                lo = min(float(np.min(t)) for t in tails)
                hi = max(float(np.percentile(t, 99)) for t in tails)
                ax.set_ylim(lo * 0.85, hi * 1.25)
            ax.legend(loc="upper right" if not missing else "lower left", fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel("itération")
    fig.tight_layout()
    _foot(fig, f"moyenne glissante sur {window} itérations ; zone grisée : burn-in (LR montant) ; "
               "axe coupé en haut : les premières itérations (têtes neuves) sortent du cadre")
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 7. composantes

def plot_composantes(data, out_dir, name="balayages_composantes"):
    """Part de chaque terme dans la perte finale (50 dernières itérations), par run."""
    plt = st.plt()
    sets = [ds for ds in data if any(r["final_parts"] for r in data[ds]["runs"])]
    if not sets:
        raise MissingSource("aucun journal d'entraînement dans les notebooks de balayage")
    fig, axes = plt.subplots(1, len(sets), figsize=(st.FULL, 3.0), sharey=False, squeeze=False)
    for k, (ax, ds) in enumerate(zip(axes[0], sets)):
        d = data[ds]
        runs = [r for r in d["runs"] if r["final_parts"]]
        y = np.arange(len(runs))[::-1]
        for yi, r in zip(y, runs):
            tot = sum(r["final_parts"].values())
            left = 0.0
            for p, v in r["final_parts"].items():
                ax.barh(yi, v / tot, 0.7, left=left, color=PART_COLORS[p],
                        edgecolor="white", linewidth=1, label=PART_LABELS[p] if yi == y[0] else None)
                left += v / tot
            ax.text(1.02, yi, _fmt(tot, 0), va="center", fontsize=7, color=st.INK2)
        ax.set_yticks(y, [r["run"] for r in runs], fontsize=7)
        ax.set_xlim(0, 1.18)
        ax.set_xticks([0, 0.5, 1], ["0", "50 %", "100 %"])
        ax.set_title(d["label"], fontsize=9)
        ax.grid(False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _foot(fig, "chiffre à droite : perte totale (journal du notebook, 50 dernières itérations)")
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 8. classes

def plot_classes(data, out_dir, name="balayages_classes"):
    """AP@0,5 par classe et par run, un panneau par jeu ; gris : classe absente des images
    évaluées."""
    plt = st.plt()
    sets = [ds for ds in data if any(r.get("classes") for r in data[ds]["runs"])]
    if not sets:
        raise MissingSource("aucune table d'AP par classe")
    sizes = [max(len(r.get("classes") or {}) for r in data[ds]["runs"]) for ds in sets]
    fig, axes = plt.subplots(1, len(sets), figsize=(st.FULL + 1.5, 0.2 * max(sizes) + 1.8),
                             gridspec_kw={"width_ratios": [1] * len(sets), "wspace": 0.75},
                             squeeze=False)
    cmap = _seq_cmap()
    cmap.set_bad("#bdbcb6")
    from matplotlib.colors import PowerNorm
    norm = PowerNorm(0.5, vmin=0, vmax=60)  # racine : les AP de 1 à 10 restent visibles
    for ax, ds in zip(axes[0], sets):
        d = data[ds]
        runs = sorted(_scored(d), key=lambda r: r["rank"])
        classes = list(next(r for r in runs if r.get("classes"))["classes"])
        grid = np.full((len(classes), len(runs)), np.nan)
        for j, r in enumerate(runs):
            ap = r.get("classes50") or r.get("classes") or {}
            for i, c in enumerate(classes):
                v = ap.get(c)
                if v is not None:
                    grid[i, j] = v
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto")
        for i in range(len(classes)):
            for j in range(len(runs)):
                v = grid[i, j]
                if not np.isnan(v) and v >= 1:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6,
                            color="white" if v > 25 else st.INK)
        ax.set_xticks(range(len(runs)), [r["run"] for r in runs], rotation=60, ha="right",
                      fontsize=6.5)
        ax.set_yticks(range(len(classes)), classes, fontsize=6.5)
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(d["label"], fontsize=9)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    cb = fig.colorbar(sm, ax=axes[0].tolist(), fraction=0.02, pad=0.01, extend="max")
    cb.set_label("AP@0,5 (%)", fontsize=8)
    cb.outline.set_visible(False)
    _foot(fig, "runs triés du meilleur au moins bon ; gris : classe sans instance dans les "
               "50 images. " + FOOT)
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 9. poids publiés

def plot_publies(data, out_dir, name="balayages_publies"):
    """Meilleur run affiné face aux poids publiés hors domaine : un panneau par jeu, chacun
    à son échelle (les scores vont de 2 à 68)."""
    plt = st.plt()
    sets = list(data)
    series = [("affiné", "meilleur run affiné", st.PALETTE[2]),
              ("tiny-yolov3-coco", st.NET_LABELS["tiny-yolov3-coco"] + " (publié)",
               st.NET_COLORS["tiny-yolov3-coco"]),
              ("tiny-yolov2-voc", st.NET_LABELS["tiny-yolov2-voc"] + " (publié)",
               st.NET_COLORS["tiny-yolov2-voc"])]
    fig, axes = plt.subplots(1, len(sets), figsize=(st.FULL, 3.0), squeeze=False)
    for ax, ds in zip(axes[0], sets):
        d = data[ds]
        bars = []
        for key, label, color in series:
            if key == "affiné":
                bars.append((d["top"], d["best"], color, label))
            elif key in d["baselines"]:
                b = d["baselines"][key]
                bars.append((b["map"], f"{b['n_classes']} cl.", color, label))
        top = max(v for v, *_ in bars)
        for k, (v, txt, color, label) in enumerate(bars):
            ax.bar(k, v, 0.72, color=color, label=label)
            ax.text(k, v + top * 0.02, f"{_fmt(v, 1)}\n{txt}", ha="center", va="bottom",
                    fontsize=6.5, color=st.INK, linespacing=1.1)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(series) - 0.4)
        ax.set_ylim(0, top * 1.3)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{d['label']}\n{_B().metric_label(d)}", fontsize=9)
        if not d["baselines"]:
            st.note(ax, "inférences des poids\npubliés pas exécutées")
    axes[0][0].set_ylabel("score (%)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in series]
    fig.legend(handles, [lab for _, lab, _ in series], loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    _foot(fig, "poids publiés : classes du jeu ayant un équivalent (MAPPINGS) seulement. " + FOOT)
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------- 10. coût

def plot_cout(data, out_dir, name="balayages_cout"):
    """Durée d'entraînement (GPU Colab) face au score relatif."""
    plt = st.plt()
    sets = list(data)
    fig, ax = plt.subplots(figsize=(st.FULL, 3.4))
    for ds in sets:
        for r in _scored(data[ds]):
            if r["duration_s"]:
                _point(ax, r["duration_s"] / 60, r["rel"], ds, r["subset"],
                       size=18 + 1.6 * r["batch"])
    batches = sorted({r["batch"] for d in data.values() for r in d["runs"]})
    for b in batches:
        mins = [r["duration_s"] / 60 for d in data.values() for r in d["runs"]
                if r["batch"] == b and r["duration_s"]]
        if mins:
            ax.axvline(np.median(mins), color=st.MUTED, lw=0.8, ls=":", zorder=0)
            ax.text(np.median(mins), 1.06, f"lot {b}", ha="center", fontsize=7, color=st.INK2)
    ax.set_ylim(0, 1.12)
    ax.set_xlabel("durée d'un run de 600 itérations (min, GPU Colab)")
    ax.set_ylabel("score relatif")
    ax.set_title("Coût d'un run face à son score", pad=14)
    _legend_datasets(ax, sets, data, loc="lower right", ncol=len(sets))
    _foot(fig, "durée : sortie du notebook, ou somme de `seconds` de loss.csv (make harvest). "
               "Pointillés : durée médiane par lot.")
    return st.save(fig, out_dir, name)


PLOTS = {"grilles": plot_grilles, "classements": plot_classements, "effets": plot_effets,
         "epoques": plot_epoques, "perte": plot_perte, "courbes": plot_courbes,
         "composantes": plot_composantes, "classes": plot_classes, "publies": plot_publies,
         "cout": plot_cout}


@figure("balayages", "resultats", "T13.53",
        "Analyse transversale des balayages lot × sous-ensemble : grilles, classements, effets "
        "du lot et du sous-ensemble, époques, perte, courbes, composantes, AP par classe, "
        "poids publiés et coût. 50 images, palier R.",
        "sorties des notebooks notebooks/*/*_sweep.ipynb et *_infer.ipynb", subset=True,
        dest=ROOT / "docs" / "tasks" / "figures")
def balayages(out_dir):
    data, _ = _B().load_all(ROOT / "notebooks")
    if not data:
        raise MissingSource("aucun notebook de balayage exécuté")
    out = []
    for fn in PLOTS.values():
        out += fn(data, out_dir)
    return out
