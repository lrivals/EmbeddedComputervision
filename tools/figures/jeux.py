"""Cross-dataset comparison of the dataset statistics (M16).

Data: `tools.notebooks.jeux.load_all()`, read from the versioned outputs of the
`notebooks/<ds>/<ds>_stats.ipynb` notebooks; one `plot_*` per question, which takes those
data and returns the list of PNG files. The notebook `notebooks/analyse_jeux.ipynb` chains
and comments them. Not in the `tools.figures` registry (M13 figures): run them with
`python -m tools.notebooks.jeux --figures <dir>`.

Fixed encodings in every figure: datasets on the y axis in `jeux.ORDER` (ground views, then
drone views, split by a thin rule), bar colour = view (ground / aerial, two validated
hues), letterbox = filled mark and stretch = hollow mark, published-weights classes in the
project colours of `style.NET_COLORS`. Sequential quantities use one blue ramp.
"""

import numpy as np

from tools.figures import style as st

FAMILY_COLORS = {"ground": st.PALETTE[2], "aerial": st.PALETTE[6]}
FAMILY_LABELS = {"ground": "ground view", "aerial": "drone view"}
SIZE_COLORS = ("#0f3366", "#2a78d6", "#a9c9ef")  # small → large: dark → light
FOOT = ("Source: executed notebooks/<ds>/<ds>_stats.ipynb (annotations: whole splits; "
        "pixels: 200 images per split).")


def _J():
    from tools.notebooks import jeux

    return jeux


def _seq_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("seq", ["#f2f1ee", "#a9c9ef", "#2a78d6",
                                                     "#0f3366"])


def _ink_on(value, vmax=1.0):
    return "white" if value is not None and value > 0.6 * vmax else st.INK


def _foot(fig, text=FOOT):
    """Footnote under everything the axes draw (labels included)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    y0 = min(ax.get_tightbbox(r).y0 for ax in fig.axes if ax.get_visible()) / fig.bbox.height
    fig.text(0.99, y0 - 0.01, text, ha="right", va="top", fontsize=7, color=st.MUTED)


def _sets(data):
    return list(data)


def _color(d):
    return FAMILY_COLORS[d["family"]]


def _ylabels(ax, data, sets):
    """Datasets top to bottom, a rule between ground and drone views."""
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([data[k]["label"] for k in sets])
    ax.set_ylim(len(sets) - 0.5, -0.5)
    fam = [data[k]["family"] for k in sets]
    for i in range(1, len(sets)):
        if fam[i] != fam[i - 1]:
            ax.axhline(i - 0.5, color=st.MUTED, lw=0.6, ls="-", zorder=0)
    ax.grid(axis="y", visible=False)


def _family_legend(fig, data, **kw):
    plt = st.plt()
    fams = [f for f in FAMILY_COLORS if any(d["family"] == f for d in data.values())]
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLORS[f]) for f in fams]
    fig.legend(handles, [FAMILY_LABELS[f] for f in fams], ncol=len(fams), **kw)


def _get(d, *keys):
    return _J()._get(d, *keys)


def _hbar(ax, data, sets, values, fmt="{:.1f}", log=False):
    """Horizontal bars, one per dataset, value printed at the end of the bar."""
    y = np.arange(len(sets))
    v = np.array([np.nan if x is None else x for x in values], dtype=np.float64)
    ax.barh(y, np.nan_to_num(v), height=0.62, color=[_color(data[k]) for k in sets],
            edgecolor="white", linewidth=1)
    if log:
        ax.set_xscale("log")
    hi = np.nanmax(v) if np.isfinite(v).any() else 1
    for yi, x in zip(y, v):
        if np.isfinite(x):
            ax.text(x * (1.08 if log else 1) + (0 if log else 0.01 * hi), yi, fmt.format(x),
                    va="center", fontsize=7, color=st.INK2)
    ax.grid(axis="x", visible=True)
    _ylabels(ax, data, sets)


def _heatmap(ax, data, sets, cols, values, norm=None, fmt="{:.2f}", vmax=None):
    """Datasets × columns grid; colour from `norm` (0–1) or the values, raw value written."""
    v = np.array([[np.nan if x is None else x for x in row] for row in values], np.float64)
    c = v if norm is None else np.array([[np.nan if x is None else x for x in row]
                                         for row in norm], np.float64)
    top = vmax if vmax is not None else (np.nanmax(c) if np.isfinite(c).any() else 1)
    cm = _seq_cmap()
    cm.set_bad("#e9e8e4")
    ax.imshow(np.ma.masked_invalid(c), cmap=cm, vmin=0, vmax=top, aspect="auto")
    for i in range(v.shape[0]):
        for j in range(v.shape[1]):
            txt = "—" if not np.isfinite(v[i, j]) else fmt.format(v[i, j])
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color=_ink_on(c[i, j] if np.isfinite(c[i, j]) else None, top))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_xticks(np.arange(len(cols) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(sets) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    _ylabels(ax, data, sets)


def _height(sets, per=0.34, base=1.4):
    return base + per * len(sets)


# ------------------------------------------------------------------- 1. scale

def plot_taille(data, out_dir, name="jeux_taille"):
    """Images and objects per dataset (log), objects per image (median, max)."""
    plt = st.plt()
    sets = _sets(data)
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, _height(sets)), sharey=True)
    _hbar(axes[0], data, sets, [data[k]["synthese"]["images"] for k in sets], "{:,.0f}", log=True)
    axes[0].set_title("Images")
    _hbar(axes[1], data, sets, [data[k]["synthese"]["objects"] for k in sets], "{:,.0f}",
          log=True)
    axes[1].set_title("Objects")
    ax = axes[2]
    for i, k in enumerate(sets):
        s = data[k]["synthese"]
        ax.plot([s["median"], s["max"]], [i, i], color=st.GREY, lw=1.6, zorder=1)
        ax.scatter([s["median"]], [i], s=40, color=_color(data[k]), zorder=3)
        ax.scatter([s["max"]], [i], s=40, facecolor="white", edgecolor=_color(data[k]),
                   linewidth=1.6, zorder=3)
        ax.text(s["max"] * 1.4, i, f"{s['median']:.0f} / {s['max']:.0f}", va="center",
                fontsize=7, color=st.INK2)
    ax.axvline(_J().SLOTS, color=st.INK2, lw=0.9, ls="--")
    ax.text(_J().SLOTS * 0.9, len(sets) - 0.55, f"{_J().SLOTS} yolo_post slots", fontsize=7,
            color=st.INK2, ha="right", va="bottom")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 6000)
    ax.set_title("Objects per image (● median, ○ max)")
    ax.grid(axis="x", visible=True)
    _ylabels(ax, data, sets)
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    _family_legend(fig, data, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _foot(fig)
    return st.save(fig, out_dir, name)


# ------------------------------------------------------------- 2. object size

def plot_objets(data, out_dir, name="jeux_objets"):
    """Small / medium / large (COCO areas) at the original resolution and after letterbox."""
    plt = st.plt()
    sets = _sets(data)
    fig, ax = plt.subplots(figsize=(st.FULL, _height(sets, 0.5)))
    h = 0.36
    for i, k in enumerate(sets):
        g = data[k].get("geometrie", {})
        for off, key, alpha in ((-h / 2 - 0.02, "orig", 0.55), (h / 2 + 0.02, "letterbox", 1.0)):
            r = g.get(key)
            if not r:
                continue
            left = 0
            for part, c in zip(("small", "medium", "large"), SIZE_COLORS):
                w = r[part] or 0
                ax.barh(i + off, w, left=left, height=h, color=c, alpha=alpha,
                        edgecolor="white", linewidth=1.5)
                if w >= 9:
                    ax.text(left + w / 2, i + off, f"{w:.0f}", ha="center", va="center",
                            fontsize=6.5, color="white" if c != SIZE_COLORS[2] else st.INK)
                left += w
            ax.text(101, i + off, f"{'original' if key == 'orig' else 'letterbox 416'} · "
                    f"median side {r['side']:.0f} px", va="center", fontsize=6.5, color=st.INK2)
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of objects (%)")
    ax.set_title("Object size: original resolution (pale) vs network input (letterbox 416)")
    _ylabels(ax, data, sets)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in SIZE_COLORS]
    fig.legend(handles, ["small < 32²", "medium < 96²", "large"], ncol=3, loc="upper center",
               bbox_to_anchor=(0.45, 1.04))
    fig.tight_layout()
    _foot(fig, "COCO size categories on box area in pixels. " + FOOT)
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------- 3. grid cells

def plot_cellules(data, out_dir, name="jeux_cellules"):
    """Share of objects smaller than a cell of each head, letterbox (●) vs stretch (○)."""
    plt = st.plt()
    sets = _sets(data)
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, _height(sets)), sharey=True)
    panels = [("13x13", "< 32 px: smaller than a 13×13 cell"),
              ("26x26", "< 16 px: smaller than a 26×26 cell")]
    for ax, (head, title) in zip(axes, panels):
        for i, k in enumerate(sets):
            g = data[k].get("geometrie", {})
            lb = _get(g, "letterbox", "cell", head)
            sr = _get(g, "stretch", "cell", head)
            c = _color(data[k])
            if lb is not None and sr is not None:
                ax.plot([sr, lb], [i, i], color=st.GREY, lw=1.6, zorder=1)
            if sr is not None:
                ax.scatter([sr], [i], s=40, facecolor="white", edgecolor=c, linewidth=1.6, zorder=3)
            if lb is not None:
                ax.scatter([lb], [i], s=40, color=c, zorder=3)
                ax.text(max(lb, sr or 0) + 3, i, f"{lb:.0f}", va="center", fontsize=7,
                        color=st.INK2)
        ax.set_xlim(-2, 105)
        ax.set_title(title)
        ax.set_xlabel("share of objects (%)")
        ax.grid(axis="x", visible=True)
        _ylabels(ax, data, sets)
    ax = axes[2]
    _hbar(ax, data, sets, [_get(data[k], "geometrie", "letterbox", "h8") for k in sets])
    ax.set_title("height < 8 px (letterbox)")
    ax.set_xlabel("share of objects (%)")
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    _family_legend(fig, data, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _foot(fig, "● letterbox 416 (aspect kept), ○ stretch 416 (aspect lost). " + FOOT)
    return st.save(fig, out_dir, name)


# ------------------------------------------------------------- 4. density

def plot_densite(data, out_dir, name="jeux_densite"):
    """Lost training targets split by cause: best anchor in no head vs cell collision."""
    plt = st.plt()
    sets = _sets(data)
    fig, ax = plt.subplots(figsize=(st.FULL, _height(sets)))
    for i, k in enumerate(sets):
        dn = data[k].get("densite") or {}
        o, c = dn.get("orphan_pct") or 0, dn.get("collide_pct") or 0
        ax.barh(i, o, height=0.62, color=st.GREY, edgecolor="white", linewidth=1.5)
        ax.barh(i, c, left=o, height=0.62, color=_color(data[k]), edgecolor="white",
                linewidth=1.5)
        ax.text(o + c + 1, i, f"{o + c:.1f} %  (no head {o:.1f} · collision {c:.2f})",
                va="center", fontsize=7, color=st.INK2)
    ax.set_xlim(0, 115)
    ax.set_xlabel("share of training targets dropped by build_targets (%)")
    ax.set_title("Lost targets: best anchor in no head (grey) vs same cell and anchor (colour)")
    ax.grid(axis="x", visible=True)
    _ylabels(ax, data, sets)
    _family_legend(fig, data, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    nets = sorted({data[k]["net"] for k in sets if data[k]["net"]})
    _foot(fig, f"Anchors and masks of {', '.join(nets)}; non-difficult objects. " + FOOT)
    return st.save(fig, out_dir, name)


# -------------------------------------------------------------- 5. classes

def plot_classes(data, out_dir, name="jeux_classes"):
    """Class balance: share of each class, sorted, one small panel per dataset."""
    plt = st.plt()
    sets = [k for k in _sets(data) if data[k].get("classes")]
    cols = 5
    rows = int(np.ceil(len(sets) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(st.FULL, 2.3 * rows), sharey=True,
                             squeeze=False)
    for ax, k in zip(axes.flat, sets):
        c = data[k]["classes"]
        cnt = np.array(c["counts"], dtype=np.float64)
        order = np.argsort(cnt)[::-1]
        share = 100 * cnt[order] / cnt.sum()
        x = np.arange(len(share))
        ax.bar(x, np.where(share > 0, share, np.nan), width=0.8, color=_color(data[k]),
               edgecolor="white", linewidth=0.8)
        ax.set_yscale("log")
        ax.set_ylim(0.001, 100)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        names = [c["names"][j] for j in order]
        ratio = c.get("ratio")
        ax.set_title(f"{data[k]['label']} · {len(cnt)} cl.", fontsize=8.5)
        ax.set_xlabel(f"top: {names[0]} {share[0]:.0f} %\n"
                      + (f"ratio {ratio:,.0f}× · " if ratio and len(cnt) > 1 else "")
                      + f"entropy {c.get('entropy', 1):.2f}", fontsize=6.5, color=st.INK2)
    for ax in axes.flat[len(sets):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("share of objects (%)")
    fig.suptitle("Class balance: classes sorted by frequency (log scale)", fontsize=10,
                 fontweight="bold")
    fig.tight_layout()
    _foot(fig, "ratio = most / least frequent class; entropy normalised (1 = uniform). " + FOOT)
    return st.save(fig, out_dir, name)


# --------------------------------------------------------------- 6. images

def plot_images(data, out_dir, name="jeux_images"):
    """Pixels against the VOC calibration images: luminance, channel means, contrast,
    letterbox downscale factor."""
    plt = st.plt()
    sets = _sets(data)
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL, _height(sets)), sharey=True)
    ref = _get(data.get(_J().VOC), "images", "luminance")
    ax = axes[0]
    ch = ("#d0453f", "#1f8a3c", "#2a78d6")  # R, G, B: channel identity, semantic colours
    for i, k in enumerate(sets):
        im = data[k].get("images") or {}
        lum = im.get("luminance")
        if lum is None:
            continue
        ax.barh(i, lum, height=0.62, color=_color(data[k]), edgecolor="white", linewidth=1)
        if len(im.get("mean") or []) == 3:
            for m, c in zip(im["mean"], ch):
                ax.plot([m, m], [i - 0.33, i + 0.33], color=c, lw=1.6)
        right = max([lum] + [m for m in (im.get("mean") or []) if m is not None])
        ax.text(right + 4, i, f"{lum:.0f}" + (" (1 ch.)" if im.get("channels") == 1 else ""),
                va="center", fontsize=7, color=st.INK2)
    if ref:
        ax.axvline(ref, color=st.INK2, lw=0.9, ls="--")
    ax.set_xlim(0, 255)
    ax.set_title("Mean luminance (ticks: R, G, B means)")
    ax.grid(axis="x", visible=True)
    _ylabels(ax, data, sets)
    _hbar(axes[1], data, sets, [np.mean(_get(data[k], "images", "std") or [np.nan])
                                for k in sets], "{:.0f}")
    axes[1].set_title("Contrast (mean channel std)")
    if ref:
        axes[1].axvline(np.mean(data[_J().VOC]["images"]["std"]), color=st.INK2, lw=0.9, ls="--")
    _hbar(axes[2], data, sets, [_get(data[k], "images", "scale") for k in sets], "×{:.2f}")
    for t, k in zip(axes[2].texts, [k for k in sets if _get(data[k], "images", "scale")]):
        m = data[k]["images"]["main"]
        t.set_text(f"{t.get_text()}  ({m[0]}×{m[1]})")
    axes[2].set_title("Letterbox factor to 416 (main resolution)")
    axes[2].set_xlim(0, 1.45)
    for a in axes[1:]:
        a.tick_params(labelleft=False)
    _family_legend(fig, data, loc="upper center", bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    _foot(fig, "Pixels on 200 images per split; dashed: VOC (INT8 calibration images, M4). "
               + FOOT)
    return st.save(fig, out_dir, name)


# -------------------------------------------------------------- 7. anchors

def plot_ancres(data, out_dir, name="jeux_ancres"):
    """Mean IoU of the boxes with the anchors in use (○) and with k-means anchors (●);
    k-means anchors of each dataset against the COCO anchors."""
    plt = st.plt()
    sets = [k for k in _sets(data) if data[k].get("ancres")]
    fig, axes = plt.subplots(1, 2, figsize=(st.FULL, _height(sets, 0.36, 1.8)),
                             gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    for i, k in enumerate(sets):
        a = data[k]["ancres"]["iou"]
        c = _color(data[k])
        cfg, best = a.get("cfg"), max(a.get("k5") or 0, a.get("k6") or 0)
        ax.plot([cfg, best], [i, i], color=st.GREY, lw=1.6, zorder=1)
        ax.scatter([cfg], [i], s=40, facecolor="white", edgecolor=c, linewidth=1.6, zorder=3)
        ax.scatter([best], [i], s=40, color=c, zorder=3)
        if a.get("voc2") is not None:
            ax.scatter([a["voc2"]], [i], marker="x", s=22, color=st.MUTED, zorder=2)
        ax.text(best + 0.012, i, f"+{best - cfg:.2f}", va="center", fontsize=7, color=st.INK2)
    ax.set_xlim(0.15, 0.85)
    ax.set_xlabel("mean IoU, box ↔ best anchor (train boxes, letterbox 416)")
    ax.set_title("○ anchors in use  ● k-means  × Tiny-YOLOv2 VOC")
    ax.grid(axis="x", visible=True)
    _ylabels(ax, data, sets)
    ax = axes[1]
    coco = data[sets[0]]["ancres"]["wh"].get("coco", [])
    for k in sets:
        wh = data[k]["ancres"]["wh"].get("k6") or []
        if not wh:
            continue
        wh = sorted(wh, key=lambda p: p[0] * p[1])
        w, h = np.array(wh).T
        ax.scatter(w, h, s=16, color=_color(data[k]), alpha=0.8, edgecolor="white",
                   linewidth=0.5, zorder=3)
        ax.text(w[-1] * 1.08, h[-1], data[k]["label"], fontsize=6.5, color=st.INK2,
                va="center")
    if coco:
        w, h = np.array(coco).T
        ax.scatter(w, h, marker="s", s=30, facecolor="none", edgecolor=st.INK, zorder=4,
                   label="COCO anchors")
        ax.scatter([w[0]], [h[0]], marker="x", s=40, color=st.INK, zorder=5,
                   label="anchor 0 (no head in COCO masks)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("anchor width (px at 416)")
    ax.set_ylabel("anchor height (px at 416)")
    ax.set_title("k-means anchors (k = 6) vs COCO anchors")
    ax.legend(loc="lower right")
    _family_legend(fig, data, loc="upper center", bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout()
    _foot(fig, "k-means: tools/kmeans_anchors.py, seed 0, best of k = 5 and 6. " + FOOT)
    return st.save(fig, out_dir, name)


# ----------------------------------------------------------------- 8. shift

def plot_ecart(data, out_dir, name="jeux_ecart"):
    """Train ↔ test total variation distance per quantity."""
    plt = st.plt()
    sets = [k for k in _sets(data) if data[k].get("ecart")]
    cols = list(_J().SHIFT)
    labels = ["classes", "box area", "box width", "box height", "box aspect", "objects/image"]
    vals = [[data[k]["ecart"].get(c) for c in cols] for k in sets]
    fig, ax = plt.subplots(figsize=(st.COL, _height(sets)))
    _heatmap(ax, data, sets, labels, vals, fmt="{:.3f}", vmax=0.35)
    ax.set_title("Train ↔ test gap (total variation, 0 = same distribution)")
    fig.tight_layout()
    _foot(fig, "Box sizes in letterbox 416 pixels. " + FOOT)
    return st.save(fig, out_dir, name)


# -------------------------------------------------------------- 9. coverage

def plot_couverture(data, out_dir, name="jeux_couverture"):
    """Share of objects whose class exists in VOC / COCO (published weights)."""
    plt = st.plt()
    sets = [k for k in _sets(data) if data[k].get("couverture")]
    fig, ax = plt.subplots(figsize=(st.COL, _height(sets)))
    h = 0.36
    series = (("VOC", "tiny-yolov2-voc", -h / 2 - 0.02), ("COCO", "tiny-yolov3-coco", h / 2 + 0.02))
    for fam, net, off in series:
        for i, k in enumerate(sets):
            v = data[k]["couverture"].get(fam)
            if v is None:
                continue
            ax.barh(i + off, v, height=h, color=st.NET_COLORS[net], edgecolor="white",
                    linewidth=1)
            ax.text(v + 1, i + off, f"{v:.1f}", va="center", fontsize=6.5, color=st.INK2)
    ax.set_xlim(0, 112)
    ax.set_xlabel("objects with an equivalent class (%)")
    ax.set_title("What the published weights can name")
    ax.grid(axis="x", visible=True)
    _ylabels(ax, data, sets)
    handles = [plt.Rectangle((0, 0), 1, 1, color=st.NET_COLORS[n]) for _, n, _ in series]
    ax.legend(handles, ["VOC classes (Tiny-YOLOv2 VOC)", "COCO classes (Tiny-YOLOv3 COCO)"],
              loc="lower left", bbox_to_anchor=(0, 1.06), ncol=1)
    fig.tight_layout()
    _foot(fig, "Class mapping: MAPPINGS of yolo.data.datasets. " + FOOT)
    return st.save(fig, out_dir, name)


# --------------------------------------------------------------- 10. quality

def plot_qualite(data, out_dir, name="jeux_qualite"):
    """Annotation issues as a share of objects (colour: per-column rank)."""
    plt = st.plt()
    sets = [k for k in _sets(data) if data[k].get("qualite")]
    cols = [("clipped_pct", "clipped (%)", "{:.2f}"), ("removed_pct", "removed (%)", "{:.2f}"),
            ("tiny_in_pct", "< 2 px at input (%)", "{:.2f}"),
            ("duplicates", "duplicates", "{:,.0f}"), ("regions_n", "ignore regions", "{:,.0f}")]
    vals = [[data[k]["qualite"].get(c) for c, _, _ in cols] for k in sets]
    arr = np.array([[np.nan if x is None else x for x in r] for r in vals], np.float64)
    norm = np.full_like(arr, np.nan)
    for j in range(arr.shape[1]):
        col = arr[:, j]
        hi = np.nanmax(col) if np.isfinite(col).any() else 0
        norm[:, j] = col / hi if hi > 0 else 0 * col
    fig, ax = plt.subplots(figsize=(st.COL, _height(sets)))
    v = arr.copy()
    _heatmap(ax, data, sets, [lab for _, lab, _ in cols], v.tolist(), norm.tolist(),
             fmt="{:,.2f}", vmax=1)
    for i in range(len(sets)):  # integer columns without decimals
        for j, (_, _, f) in enumerate(cols):
            if f == "{:,.0f}" and np.isfinite(arr[i, j]):
                for t in ax.texts:
                    if t.get_position() == (j, i):
                        t.set_text(f.format(arr[i, j]))
    ax.set_title("Annotation quality (colour: share of the column maximum)")
    fig.tight_layout()
    _foot(fig, "—: not checked by the loader. Ignore regions: DontCare / ignored areas. " + FOOT)
    return st.save(fig, out_dir, name)


# ------------------------------------------------------------- 11. overview

def plot_difficulte(data, out_dir, name="jeux_difficulte"):
    """Difficulty indicators, min–max normalised per column; datasets sorted by mean."""
    plt = st.plt()
    J = _J()
    sc = J.difficulty(data)
    sets = sorted(data, key=lambda k: sc["score"][k], reverse=True)
    keys = [k for k, _, _ in J.INDICATORS]
    labels = [lab for _, lab, _ in J.INDICATORS] + ["mean"]
    raw = [[sc["raw"][c][k] for c in keys] + [sc["score"][k]] for k in sets]
    norm = [[sc["norm"][c][k] for c in keys] + [sc["score"][k]] for k in sets]
    fig, ax = plt.subplots(figsize=(st.FULL * 0.8, _height(sets)))
    _heatmap(ax, data, sets, labels, raw, norm, fmt="{:.2f}", vmax=1)
    for t in ax.texts:  # integers for counts and percentages
        x, _ = t.get_position()
        if x in (0, 1, 3, 6):
            try:
                t.set_text(f"{float(t.get_text()):.0f}")
            except ValueError:
                pass
    ax.set_yticks(range(len(sets)))
    ax.set_yticklabels([data[k]["label"] for k in sets])
    for ln in list(ax.lines):
        ln.remove()  # sorted by score: the ground / drone rule no longer applies
    ax.axvline(len(keys) - 0.5, color=st.INK2, lw=1.2)
    ax.set_title("Embedded difficulty overview (colour: 0 easiest → 1 hardest per column)")
    fig.tight_layout()
    _foot(fig, "Raw values written; luminance gap = |luminance − VOC|; imbalance = log₁₀ of "
               "the most / least frequent class ratio. " + FOOT)
    return st.save(fig, out_dir, name)


PLOTS = {"taille": plot_taille, "objets": plot_objets, "cellules": plot_cellules,
         "densite": plot_densite, "classes": plot_classes, "images": plot_images,
         "ancres": plot_ancres, "ecart": plot_ecart, "couverture": plot_couverture,
         "qualite": plot_qualite, "difficulte": plot_difficulte}
