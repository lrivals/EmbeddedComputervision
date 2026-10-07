"""Figures des statistiques de jeux de données (M16, T16.1).

Une fonction par panneau de la section B de docs/tasks/M16-presentation-jeux.md, tracée
depuis `stats.json` de `tools/data_stats.py` sans relire le jeu. `plot_all(stats, out)`
trace tout ce que le fichier contient (`data_stats.py --figures`). `plot_voc_stats` est la
figure `voc_stats` de T13.7 (enregistrée dans `tools/figures/modeles.py`).
"""

from pathlib import Path

import numpy as np

from tools.figures import style as st

AREA_BINS = (32 ** 2, 96 ** 2)  # petits / moyens / grands (convention COCO, en pixels)
SEQ = "Blues"                   # cartes de chaleur : une teinte, du clair au foncé


def _edges(stats, key):
    return np.asarray(stats["bins"][key], dtype=np.float64)


def _share(h):
    h = np.asarray(h, dtype=np.float64)
    return h / h.sum() if h.sum() else h


def _span(edges, *hists):
    """Bornes des bacs occupés par l'un des histogrammes (axe log resserré sur les données)."""
    occ = np.flatnonzero(np.sum([np.asarray(h) > 0 for h in hists], axis=0))
    if not len(occ):
        return edges[0], edges[-1]
    return edges[max(occ[0] - 1, 0)], edges[min(occ[-1] + 2, len(edges) - 1)]


def _refs(a):
    """Ancres de référence de `ancres`, celles qui sont identiques réunies sous un nom."""
    out = {}
    for n, r in a["refs"].items():
        key = tuple(map(tuple, np.round(r["anchors"], 1)))
        if key in out:
            out[key]["names"].append(n)
        else:
            out[key] = {"names": [n], **r}
    return [(" = ".join(v["names"]), v) for v in out.values()]


def _parts(stats, groups=True):
    """[(nom, analyses)] : groupes train et test (ou splits), pour superposer les courbes."""
    out = []
    names = stats.get("groupes", {}) if groups else {}
    for g in ("train", "test"):
        v = names.get(g)
        if v is not None:
            out.append((f"{g} ({v})" if isinstance(v, str) else g,
                        stats["splits"][v] if isinstance(v, str) else v))
    return out or list(stats["splits"].items())


def _whole(stats):
    return stats.get("ensemble") or next(iter(stats["splits"].values()), {})


def _title(stats, text):
    return f"{stats['dataset']} — {text}"


# ---------------------------------------------------------------------------- T16.5

def plot_classes(stats, out_dir, name="classes"):
    """Objets par classe et par partie (échelle log si le déséquilibre dépasse 20)."""
    parts = [(n, a) for n, a in _parts(stats) if "comptes" in a]
    if not parts:
        return []
    plt = st.plt()
    classes = stats["classes"]
    fig, ax = plt.subplots(figsize=(st.FULL if len(classes) > 12 else st.COL, 3.8))
    x = np.arange(len(classes))
    width = 0.8 / len(parts)
    for k, (n, a) in enumerate(parts):
        c = np.asarray(a["comptes"]["per_class"])
        ax.bar(x + (k - (len(parts) - 1) / 2) * width, c, width * 0.95, color=st.PALETTE[k],
               hatch=("", "//", "..")[k % 3], edgecolor="white", linewidth=0.4,
               label=f"{n} : {c.sum():,} objets".replace(",", " "))
    allc = np.asarray(_whole(stats)["comptes"]["per_class"])
    nz = allc[allc > 0]
    if len(nz) and nz.max() / nz.min() > 20:
        ax.set_yscale("log")
    ax.set_xticks(x, classes, rotation=40 if len(classes) > 6 else 0, ha="right"
                  if len(classes) > 6 else "center")
    ax.set_ylabel("objets")
    ax.set_title(_title(stats, "objets par classe (split entier)"))
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_cooc(stats, out_dir, name="cooccurrence"):
    """Part des images d'une classe (ligne) qui contiennent aussi la classe en colonne."""
    c = _whole(stats).get("comptes")
    if not c or len(stats["classes"]) < 2:
        return []
    m = np.asarray(c["cooc"], dtype=np.float64)
    diag = np.maximum(np.diag(m), 1)
    p = m / diag[:, None]
    plt = st.plt()
    n = len(stats["classes"])
    fig, ax = plt.subplots(figsize=(min(st.FULL, 2 + 0.38 * n), min(st.FULL, 1.6 + 0.36 * n)))
    im = ax.imshow(p, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(n), stats["classes"], rotation=60, ha="right")
    ax.set_yticks(range(n), stats["classes"])
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.04, label="part des images de la ligne")
    ax.set_title(_title(stats, "co-occurrence des classes"))
    fig.tight_layout()
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------------------- T16.6

def plot_tailles(stats, out_dir, name="tailles"):
    """Aires des boîtes : image d'origine, letterbox et stretch à l'entrée ; seuils COCO et
    aire d'une cellule de chaque tête."""
    g = _whole(stats).get("geometrie")
    if not g:
        return []
    plt = st.plt()
    edges = _edges(stats, "area")
    fig, ax = plt.subplots(figsize=(st.COL, 3.6))
    lab = "×".join(str(v) for v in g["size"][::-1])
    for k, (mode, text, ls) in enumerate((("orig", "image d'origine", "-"),
                                          ("letterbox", f"letterbox {lab}", "-"),
                                          ("stretch", f"stretch {lab}", "--"))):
        ax.stairs(_share(g[mode]["area"]), edges, color=st.PALETTE[k], lw=1.6, ls=ls,
                  label=text)
    for b in AREA_BINS:
        ax.axvline(b, color=st.MUTED, lw=0.9, ls=":")
    for k, (grid, stride) in enumerate(g["strides"].items()):
        ax.axvline(stride ** 2, color=st.INK2, lw=0.9, ls="-.")
        ax.annotate(f"cellule {grid}", (stride ** 2, 0.02), xycoords=("data", "axes fraction"),
                    rotation=90, va="bottom", ha="right", fontsize=7, color=st.INK2)
    m = g["resize"]
    c = np.asarray(g[m]["coco"], dtype=np.float64)
    st.note(ax, f"{m} : petits {100 * c[0] / max(c.sum(), 1):.0f} %, moyens "
            f"{100 * c[1] / max(c.sum(), 1):.0f} %, grands {100 * c[2] / max(c.sum(), 1):.0f} %",
            "upper left")
    ax.set_xscale("log")
    ax.set_xlim(*_span(edges, *(g[m]["area"] for m in ("orig", "letterbox", "stretch"))))
    ax.set_xlabel("aire de la boîte (px²)")
    ax.set_ylabel("part des objets")
    ax.set_title(_title(stats, "aires des boîtes (seuils 32² et 96²)"))
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.92))
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_letterbox_stretch(stats, out_dir, name="letterbox_stretch"):
    """Hauteurs des boîtes à l'entrée en letterbox et en stretch (jeux non carrés)."""
    g = _whole(stats).get("geometrie")
    if not g:
        return []
    plt = st.plt()
    edges = _edges(stats, "side")
    fig, ax = plt.subplots(figsize=(st.COL, 3.4))
    for k, mode in enumerate(("letterbox", "stretch")):
        u = g[mode]["h_under"]["8"]
        ax.stairs(_share(g[mode]["h"]), edges, color=st.PALETTE[k + 1], lw=1.6,
                  ls=("-", "--")[k], label=f"{mode} : {100 * u:.1f} % sous 8 px")
    ax.axvline(8, color=st.MUTED, lw=0.9, ls=":")
    ax.set_xscale("log")
    ax.set_xlim(*_span(edges, g["letterbox"]["h"], g["stretch"]["h"]))
    ax.set_xlabel("hauteur de la boîte (px d'entrée)")
    ax.set_ylabel("part des objets")
    ax.set_title(_title(stats, "hauteurs à l'entrée : letterbox / stretch"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_centres(stats, out_dir, name="centres"):
    g = _whole(stats).get("geometrie")
    if not g:
        return []
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.COL * 0.7, st.COL * 0.6))
    im = ax.imshow(np.asarray(g["centers"]), cmap=SEQ, extent=(0, 1, 1, 0))
    st.image_axes(ax, _title(stats, "centres des boîtes"))
    ax.set_xlabel("x / largeur")
    ax.set_ylabel("y / hauteur")
    fig.colorbar(im, ax=ax, fraction=0.046, label="objets")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------------------- T16.7

def plot_densite(stats, out_dir, name="densite"):
    """Objets par image, avec les 256 emplacements de yolo_post (et 1 024)."""
    parts = [(n, a) for n, a in _parts(stats) if "densite" in a]
    if not parts:
        return []
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.COL, 3.4))
    top = 1
    for k, (n, a) in enumerate(parts):
        h = np.asarray(a["densite"]["per_image"], dtype=np.float64)
        top = max(top, len(h))
        ax.stairs(_share(h), np.arange(len(h) + 1) - 0.5, color=st.PALETTE[k], lw=1.4,
                  ls=("-", "--")[k % 2], label=f"{n} : méd. {a['densite']['median']:.0f}, "
                                               f"max {a['densite']['max']}")
    for s in (256, 1024):
        if top > s * 0.5:
            ax.axvline(s, color=st.INK2, lw=0.9, ls="-.")
            ax.annotate(f"{s}", (s, 0.98), xycoords=("data", "axes fraction"), rotation=90,
                        va="top", ha="right", fontsize=7, color=st.INK2)
    ax.set_yscale("log")
    if top > 60:
        ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("objets par image")
    ax.set_ylabel("part des images")
    ax.set_title(_title(stats, "objets par image"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_collisions(stats, out_dir, name="collisions"):
    """Part des cibles perdues par collision (même cellule, même ancre), par classe."""
    d = _whole(stats).get("densite")
    if not d:
        return []
    col = d["collisions"]
    obj = np.asarray(col["per_class"]["objects"], dtype=np.float64)
    lost = np.asarray(col["per_class"]["lost"], dtype=np.float64)
    keep = obj > 0
    if not keep.any():
        return []
    classes = np.asarray(stats["classes"])[keep]
    share = 100 * lost[keep] / obj[keep]
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.COL, max(2.2, 0.28 * len(classes) + 1)))
    y = np.arange(len(classes))
    ax.barh(y, share, 0.7, color=st.PALETTE[0])
    for yy, s, n in zip(y, share, lost[keep]):
        ax.text(s, yy, f" {s:.1f} % ({int(n)})", va="center", fontsize=7, color=st.INK2)
    ax.set_yticks(y, classes)
    ax.invert_yaxis()
    ax.set_xlabel("cibles perdues (% des objets de la classe)")
    ax.set_xlim(0, max(share.max() * 1.35, 1))
    ax.grid(axis="y", visible=False)
    ax.set_title(_title(stats, f"collisions de cibles ({100 * col['lost_share']:.2f} % au total)"))
    st.note(ax, f"ancres de {Path(col['net']).stem}", "lower right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------------------- T16.8

def plot_resolutions(stats, out_dir, name="resolutions"):
    im = _whole(stats).get("images")
    if not im or not im.get("aspect"):
        return []
    plt = st.plt()
    edges = _edges(stats, "image_aspect")
    fig, ax = plt.subplots(figsize=(st.COL, 3.2))
    ax.stairs(_share(im["aspect"]), edges, color=st.PALETTE[0], lw=1.6, fill=True, alpha=0.35)
    ax.stairs(_share(im["aspect"]), edges, color=st.PALETTE[0], lw=1.6)
    for r, text in ((1, "1:1"), (4 / 3, "4:3"), (16 / 9, "16:9"), (3.3, "KITTI")):
        if edges[0] < np.log2(r) < edges[-1]:
            ax.axvline(np.log2(r), color=st.MUTED, lw=0.8, ls=":")
            ax.annotate(text, (np.log2(r), 0.98), xycoords=("data", "axes fraction"),
                        rotation=90, va="top", ha="right", fontsize=7, color=st.INK2)
    top = ", ".join(f"{w}×{h}" for w, h, _ in im["resolutions"][:3])
    st.note(ax, f"{im['distinct']} résolutions ; {top}", "upper left")
    ax.set_xlabel("log2(largeur / hauteur) de l'image")
    ax.set_ylabel("part des images")
    ax.set_title(_title(stats, "rapport d'aspect des images (split entier)"))
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_intensites(stats, out_dir, name="intensites"):
    """Histogramme de luminance sur SAMPLE images, face à VOC2007 test (échelles INT8 de M4)."""
    px = _whole(stats).get("images", {}).get("pixels")
    if not px:
        return []
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.COL, 3.4))
    v = np.arange(257) - 0.5
    ax.stairs(_share(px["lum_hist"]), v, color=st.PALETTE[0], lw=1.6,
              label=f"{stats['dataset']} ({px['images']} images) : moy. {px['lum_mean']:.0f}")
    ref = stats.get("voc_ref")
    if ref:
        ax.stairs(_share(ref["lum_hist"]), v, color=st.PALETTE[1], lw=1.4, ls="--",
                  label=f"VOC2007 test ({ref['images']} images) : moy. {ref['lum_mean']:.0f}")
    lv = px["levels"]
    st.note(ax, f"niveaux occupés : {lv['8bit']} en 8 bits → {lv['int8']} en INT8 "
            f"(échelle 1/127)", "upper left")
    ax.set_xlim(-1, 256)
    ax.set_xlabel("luminance (8 bits)")
    ax.set_ylabel("part des pixels")
    ax.set_title(_title(stats, f"luminance sur {px['images']} images (graine {px['seed']})"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_eclairage(stats, out_dir, name="eclairage"):
    """ExDark : luminance moyenne par type d'éclairage, face à VOC."""
    light = stats.get("exdark_light")
    if not light:
        return []
    plt = st.plt()
    names = list(light)
    vals = [light[k]["lum_mean"] for k in names]
    fig, ax = plt.subplots(figsize=(st.COL, 3.2))
    y = np.arange(len(names))
    ax.barh(y, vals, 0.7, color=st.PALETTE[0])
    for yy, k, val in zip(y, names, vals):
        ax.text(val, yy, f" {val:.0f} ({light[k]['images']} im.)", va="center", fontsize=7,
                color=st.INK2)
    ref = stats.get("voc_ref")
    if ref:
        ax.axvline(ref["lum_mean"], color=st.PALETTE[1], lw=1.4, ls="--")
        ax.annotate(f"VOC {ref['lum_mean']:.0f}", (ref["lum_mean"], 0.02),
                    xycoords=("data", "axes fraction"), ha="right", rotation=90, fontsize=7,
                    color=st.INK2)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlim(0, 255)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("luminance moyenne (8 bits)")
    ax.set_title(_title(stats, "luminance par type d'éclairage"))
    fig.tight_layout()
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------------------- T16.9

def plot_ancres(stats, out_dir, name="ancres"):
    """Nuage (w, h) des boîtes à l'entrée, ancres de la cfg et du k-means (k = 6)."""
    a = stats.get("ancres")
    if not a:
        return []
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.COL, st.COL * 0.75))
    wh = np.asarray(a["scatter"], dtype=np.float64)
    ax.scatter(wh[:, 0], wh[:, 1], s=3, color=st.GREY, alpha=0.5, lw=0,
               label=f"{len(wh)} boîtes sur {a['boxes']}")
    cfg = next((k for k in a["refs"] if k.startswith("cfg")), None)
    if cfg:
        c = np.asarray(a["refs"][cfg]["anchors"])
        ax.scatter(c[:, 0], c[:, 1], s=60, marker="D", facecolor="white",
                   edgecolor=st.PALETTE[1], lw=1.8, label=f"{cfg} : IoU {a['refs'][cfg]['iou']:.3f}")
    km = np.asarray(a["kmeans"]["6"])
    ax.scatter(km[:, 0], km[:, 1], s=60, marker="o", color=st.PALETTE[0], edgecolor="white",
               lw=1.5, label=f"k-means k = 6 : IoU {a['kmeans_iou']['6']:.3f}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("largeur (px d'entrée, letterbox)")
    ax.set_ylabel("hauteur (px)")
    ax.set_title(_title(stats, "boîtes et ancres"))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_ancres_k(stats, out_dir, name="ancres_k"):
    a = stats.get("ancres")
    if not a:
        return []
    plt = st.plt()
    ks = sorted(int(k) for k in a["kmeans_iou"])
    fig, ax = plt.subplots(figsize=(st.COL, 3.4))
    ax.plot(ks, [a["kmeans_iou"][str(k)] for k in ks], "o-", color=st.PALETTE[0], ms=5,
            label="k-means (1 − IoU)")
    for j, (n, r) in enumerate(_refs(a)):
        k = len(r["anchors"])
        ax.plot([k], [r["iou"]], "D", ms=7, mfc="white", mec=st.PALETTE[1 + j % 7], mew=1.8,
                label=f"{n} : {r['iou']:.3f}")
    ax.set_xticks(ks)
    ax.set_xlabel("nombre d'ancres k")
    ax.set_ylabel("IoU moyenne de la meilleure ancre")
    ax.set_title(_title(stats, "IoU moyenne selon k"))
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


# ---------------------------------------------------------------------------- T16.11

def plot_ecart(stats, out_dir, name="ecart"):
    """Parts des classes en train et en test."""
    e = stats.get("ecart")
    if not e or "share_train" not in e:
        return []
    plt = st.plt()
    classes = stats["classes"]
    fig, ax = plt.subplots(figsize=(st.FULL if len(classes) > 12 else st.COL, 3.4))
    x = np.arange(len(classes))
    for k, (key, text) in enumerate((("share_train", "train"), ("share_test", "test"))):
        ax.bar(x + (k - 0.5) * 0.4, 100 * np.asarray(e[key]), 0.38, color=st.PALETTE[k],
               hatch=("", "//")[k], edgecolor="white", linewidth=0.4, label=text)
    ax.set_xticks(x, classes, rotation=40 if len(classes) > 6 else 0,
                  ha="right" if len(classes) > 6 else "center")
    ax.set_ylabel("part des objets (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title(_title(stats, f"classes en train et en test (variation totale "
                               f"{e.get('classes', 0):.3f})"))
    ax.legend(loc="upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


PANELS = (plot_classes, plot_cooc, plot_tailles, plot_letterbox_stretch, plot_centres,
          plot_densite, plot_collisions, plot_resolutions, plot_intensites, plot_eclairage,
          plot_ancres, plot_ancres_k, plot_ecart)


def plot_all(stats, out_dir):
    """Tous les panneaux que `stats` permet de tracer ; rend les chemins."""
    paths = []
    for fn in PANELS:
        paths += fn(stats, out_dir)
    return paths


# ---------------------------------------------------------------------------- T13.7 (VOC)

VOC_PARTS = {"trainval 07+12": "train", "test 2007": "test"}


def voc_stats_view(stats):
    """stats.json de VOC → {partie: per_class, area (histogramme), area_edges, per_image
    (comptes par nombre d'objets), images}, objets non-difficult (ceux de l'évaluation)."""
    out = {}
    for name, g in VOC_PARTS.items():
        v = stats["groupes"][g]
        a = stats["splits"][v] if isinstance(v, str) else v
        out[name] = {"per_class": np.asarray(a["comptes"]["per_class_easy"]),
                     "area": np.asarray(a["geometrie"]["orig"]["area"]),
                     "area_edges": _edges(stats, "area"),
                     "per_image": np.asarray(a["densite"]["per_image_easy"]),
                     "images": a["comptes"]["images"]}
    return out


def plot_voc_stats(view, out_dir, name="voc_stats"):
    """Objets par classe en trainval et en test, aires des boîtes, objets par image."""
    from yolo.data.voc import VOC_CLASSES

    plt = st.plt()
    fig = plt.figure(figsize=(st.FULL, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax = fig.add_subplot(gs[0, :])
    splits = list(view)
    x = np.arange(len(VOC_CLASSES))
    for k, sp in enumerate(splits):
        ax.bar(x + (k - 0.5) * 0.4, view[sp]["per_class"], 0.38, color=st.PALETTE[k],
               hatch=("", "//")[k], edgecolor="white", linewidth=0.4,
               label=f"{sp} : {view[sp]['per_class'].sum()} objets, {view[sp]['images']} images")
    ax.set_xticks(x, VOC_CLASSES, rotation=40, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("objets (non-difficult)")
    ax.set_title("Objets par classe")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)

    ax2 = fig.add_subplot(gs[1, 0])
    for k, sp in enumerate(splits):
        e = view[sp]["area_edges"]
        h = np.asarray(view[sp]["area"], dtype=np.float64)
        ax2.stairs(_share(h), e, color=st.PALETTE[k], lw=1.6, ls=("-", "--")[k], label=sp)
    for b in AREA_BINS:
        ax2.axvline(b, color=st.MUTED, lw=0.9, ls=":")
    last = view[splits[-1]]
    e, h = last["area_edges"], np.asarray(last["area"], dtype=np.float64)
    centers = np.sqrt(e[:-1] * e[1:])
    tot = max(h.sum(), 1)
    frac = [h[centers < AREA_BINS[0]].sum() / tot,
            h[(centers >= AREA_BINS[0]) & (centers < AREA_BINS[1])].sum() / tot,
            h[centers >= AREA_BINS[1]].sum() / tot]
    st.note(ax2, f"{splits[-1]} : petits {frac[0] * 100:.0f} %, moyens {frac[1] * 100:.0f} %, "
            f"grands {frac[2] * 100:.0f} %", "upper left")
    ax2.set_xscale("log")
    ax2.set_xlim(10, 500 * 500)
    ax2.set_xlabel("aire de la boîte (px², image d'origine)")
    ax2.set_ylabel("part des objets")
    ax2.set_title("Aires des boîtes (seuils 32² et 96²)")
    ax2.legend(loc="upper left", bbox_to_anchor=(0, 0.92))

    ax3 = fig.add_subplot(gs[1, 1])
    for k, sp in enumerate(splits):
        c = np.asarray(view[sp]["per_image"], dtype=np.float64)
        c = np.concatenate([c[:25], [c[25:].sum()]]) if len(c) > 25 else c
        ax3.stairs(_share(c), np.arange(len(c) + 1) - 0.5, color=st.PALETTE[k], lw=1.6,
                   ls=("-", "--")[k], label=sp)
    ax3.set_xlabel("objets par image (25 = 25 et plus)")
    ax3.set_ylabel("part des images")
    ax3.set_title("Objets par image")
    ax3.legend(loc="upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)
