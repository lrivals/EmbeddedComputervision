"""Figures des modèles (M13, section B) : graphe, profil par couche, mémoire, ancres, VOC."""

import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from tools.figures import ROOT, MissingSource, figure, need
from tools.figures import style as st

DEVKIT = ROOT / "data" / "VOCdevkit"
MANIFEST = "model/{net}/manifest.json"


def _manifest_net(net):
    from tools.count_macs import net_from_manifest

    return net_from_manifest(need(ROOT / MANIFEST.format(net=net)))


# --- T13.2 : profil par couche ------------------------------------------------------------

def load_profile(net):
    """[(id, type, MACs, paramètres, octets int8 de la sortie)] de `tools/count_macs.py`."""
    from yolo.models.specs import infer_shapes, layer_cost

    desc = _manifest_net(net)
    rows = []
    for i, (layer, (ins, outs)) in enumerate(zip(desc["layers"], infer_shapes(desc))):
        params, macs = layer_cost(layer, ins, outs)
        rows.append((i, layer["type"], int(macs), int(params), int(np.prod(outs))))
    return rows


def plot_profile(profiles, out_dir, name="profil_couches"):
    plt = st.plt()
    fig, axes = plt.subplots(4, 1, figsize=(st.FULL, 8.2), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 2]})
    nets = list(profiles)
    w = 0.8 / len(nets)
    specs = [(2, "MACs (G)", 1e9), (3, "paramètres (M)", 1e6), (4, "sortie int8 (Mo)", 1e6)]
    for k, net in enumerate(nets):
        rows = profiles[net]
        x = np.array([r[0] for r in rows]) + (k - (len(nets) - 1) / 2) * w
        color = st.NET_COLORS[net]
        for ax, (col, _, div) in zip(axes, specs):
            ax.bar(x, [r[col] / div for r in rows], w * 0.92, color=color,
                   hatch=("", "//")[k], edgecolor="white", linewidth=0.4,
                   label=f"{st.NET_LABELS[net]} : total {sum(r[col] for r in rows) / div:.3f}")
        macs = np.cumsum([r[2] for r in rows]) / sum(r[2] for r in rows) * 100
        prm = np.cumsum([r[3] for r in rows]) / max(1, sum(r[3] for r in rows)) * 100
        xi = [r[0] for r in rows]
        axes[3].plot(xi, macs, color=color, lw=1.8, label=f"MACs {st.NET_LABELS[net]}")
        axes[3].plot(xi, prm, color=color, lw=1.4, ls="--",
                     label=f"paramètres {st.NET_LABELS[net]}")
    for ax, (_, lab, _) in zip(axes, specs):
        ax.set_ylabel(lab)
        ax.legend(loc="upper right")
        ax.grid(axis="x", visible=False)
    axes[3].set_ylabel("cumul (%)")
    axes[3].set_ylim(0, 102)
    axes[3].legend(loc="lower right", ncol=2)
    axes[3].set_xlabel("indice de couche")
    n = max(len(r) for r in profiles.values())
    axes[3].set_xticks(range(n))
    axes[0].set_title("Profil par couche : les couches 13×13 portent les poids, les premières "
                      "les activations")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("profil_couches", "modeles", "T13.2",
        "MACs, paramètres et taille des sorties int8 par couche, Tiny-YOLOv2 et Tiny-YOLOv3 "
        "superposés, avec les cumuls.",
        "model/<net>/manifest.json via tools/count_macs.py")
def profil_couches(out_dir):
    profiles = {}
    for net in st.NETS:
        try:
            profiles[net] = load_profile(net)
        except MissingSource:
            pass
    if not profiles:
        raise MissingSource("aucun model/<net>/manifest.json (make export)")
    return plot_profile(profiles, out_dir)


# --- T13.1 : graphe couche par couche -----------------------------------------------------

LAYER_COLORS = {"conv": st.PALETTE[0], "maxpool": st.PALETTE[2], "route": st.PALETTE[3],
                "upsample": st.PALETTE[4], "yolo": st.PALETTE[1], "region": st.PALETTE[1]}


def graph_dot(net):
    """Source Graphviz : un nœud par couche (type, K/stride, C×H×W), bord ∝ MACs."""
    from yolo.models.specs import infer_shapes, layer_cost

    desc = _manifest_net(net)
    shapes = infer_shapes(desc)
    costs = [layer_cost(layer, i, o)[1] for layer, (i, o) in zip(desc["layers"], shapes)]
    top = max(costs) or 1
    lines = [f'digraph "{net}" {{', "  rankdir=TB; nodesep=0.25; ranksep=0.22;",
             '  node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=10, '
             'fillcolor="white"];', '  edge [color="#8a8984", arrowsize=0.6];',
             f'  input [label="entrée\\n{"×".join(map(str, desc["input"]))}", '
             'shape=box, style="rounded", color="#52514e"];']
    for i, (layer, (_, out)) in enumerate(zip(desc["layers"], shapes)):
        t = layer["type"]
        ks = f" {layer['k']}×{layer['k']}/{layer['s']}" if "k" in layer else ""
        if t == "upsample":
            ks = f" ×{layer['s']}"
        extra = f"\\n{costs[i] / 1e6:.0f} MMAC" if costs[i] else ""
        act = f" {layer['act']}" if t == "conv" else ""
        pen = 1 + 5 * costs[i] / top
        c = LAYER_COLORS.get(t, st.MUTED)
        lines.append(f'  L{i} [label="L{i:02d} {t}{ks}{act}\\n{out[0]}×{out[1]}×{out[2]}'
                     f'{extra}", color="{c}", penwidth={pen:.2f}];')
        srcs = layer["from"] if t == "route" else [i - 1]
        for j in srcs:
            j = j if j >= 0 else i + j
            lines.append(f"  {'input' if j < 0 else f'L{j}'} -> L{i};")
    lines.append("}")
    return "\n".join(lines) + "\n"


@figure("graphe", "modeles", "T13.1",
        "Graphe couche par couche (type, noyau et stride, forme de sortie C×H×W ; bord épais "
        "= beaucoup de MACs).",
        "model/<net>/manifest.json, yolo.models.specs (Graphviz dot)")
def graphe(out_dir):
    dot = shutil.which("dot")
    if not dot:
        raise MissingSource("Graphviz (dot) absent")
    paths = []
    for net in st.NETS:
        try:
            src = graph_dot(net)
        except MissingSource:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"graphe_{net}"
        stem.with_suffix(".dot").write_text(src)
        subprocess.run([dot, "-Tpng", "-Gdpi=150", "-o", str(stem.with_suffix(".png")),
                        str(stem.with_suffix(".dot"))], check=True)
        stem.with_suffix(".dot").unlink()
        paths.append(stem.with_suffix(".png"))
    if not paths:
        raise MissingSource("aucun manifest (make export)")
    return paths


# --- T13.3 : mémoire face au budget -------------------------------------------------------

def arena_size(manifest, header=ROOT / "sw" / "driver" / "program.hpp"):
    """Taille de l'arène d'activations DDR, comme `driver::build` (sw/driver/program.cpp)."""
    text = Path(header).read_text()
    align = int(re.search(r"ARENA_ALIGN = (\d+)", text).group(1))
    slack = int(re.search(r"ARENA_SLACK = (\d+)", text).group(1))
    return sum(-(-n // align) * align for n in manifest["buffers"].values()) + slack


def load_memory(net):
    """[(id, poids int8, biais int32, entrée, sortie)] des convs, arène et BRAM du noyau."""
    from yolo.models.specs import infer_shapes

    from tools import perf_model, roofline

    path = need(ROOT / MANIFEST.format(net=net))
    m = json.loads(path.read_text())
    desc = _manifest_net(net)
    rows = []
    for i, (layer, (ins, outs)) in enumerate(zip(desc["layers"], infer_shapes(desc))):
        if layer["type"] == "conv":
            w = ins[0] * outs[0] * layer["k"] ** 2
            rows.append((i, w, 4 * outs[0], int(np.prod(ins)), int(np.prod(outs))))
    t = perf_model.TILES
    bram = roofline.bram18(t["tm"], t["tn"], t["tr"], t["tc"])
    return rows, arena_size(m), bram, bram * 18 * 1024 // 8


def plot_memory(net, rows, arena, bram, bram_bytes, out_dir, name):
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.FULL, 4.2))
    x = np.arange(len(rows))
    parts = [("poids int8", 1), ("biais int32", 2), ("activations entrée", 3),
             ("activations sortie", 4)]
    bottom = np.zeros(len(rows))
    for k, (lab, col) in enumerate(parts):
        v = np.array([r[col] for r in rows], dtype=np.float64)
        ax.bar(x, v, 0.7, bottom=bottom, color=st.PALETTE[k], hatch=("", "", "//", "..")[k],
               edgecolor="white", linewidth=0.5, label=lab)
        bottom += v
    ax.set_yscale("log")
    ax.set_ylim(1e3, max(bottom.max(), arena) * 3)
    ax.axhline(bram_bytes, color=st.PALETTE[7], lw=1.4)
    ax.text(len(rows) - 0.5, bram_bytes * 1.12, f"tampons BRAM du noyau : {bram} BRAM18 = "
            f"{bram_bytes / 1024:.0f} Kio", ha="right", fontsize=7.5, color=st.INK2,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.axhline(arena, color=st.PALETTE[6], lw=1.4, ls="--")
    ax.text(len(rows) - 0.5, arena * 1.12, f"arène d'activations DDR du driver : "
            f"{arena / 1e6:.2f} Mo", ha="right", fontsize=7.5, color=st.INK2,
            bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_xticks(x, [f"L{r[0]:02d}" for r in rows])
    ax.set_ylabel("octets (log)")
    ax.set_title(f"{st.NET_LABELS.get(net, net)} — empreinte mémoire par conv : rien ne tient "
                 "en BRAM d'un seul coup, d'où le moteur couche par couche")
    ax.legend(loc="upper left", ncol=4)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("memoire", "modeles", "T13.3",
        "Empreinte mémoire par conv (poids, biais, activations) face aux tampons BRAM du "
        "noyau et à l'arène DDR du driver.",
        "model/<net>/manifest.json, sw/driver/program.hpp, tools/roofline.py")
def memoire(out_dir):
    paths = []
    for net in st.NETS:
        try:
            data = load_memory(net)
        except MissingSource:
            continue
        paths += plot_memory(net, *data, out_dir, f"memoire_{net}")
    if not paths:
        raise MissingSource("aucun manifest (make export)")
    return paths


# --- T13.4 : ancres -----------------------------------------------------------------------

def load_anchors_md(path):
    """results/anchors.md : {k: (ancres k-means, IoU, nom Darknet, ancres Darknet, IoU)}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        m = re.match(r"\| (\d+) \| `([^`]*)` \| ([\d.]+) \| ([^|]+) \| `([^`]*)` \| ([\d.]+) \|",
                     line)
        if m:
            def parse(s):
                return np.array([[float(v) for v in p.split(",")] for p in s.split()])
            out[int(m.group(1))] = (parse(m.group(2)), float(m.group(3)), m.group(4).strip(),
                                    parse(m.group(5)), float(m.group(6)))
    if not out:
        raise MissingSource(f"{path} sans tableau d'ancres")
    return out


def plot_anchors(wh, anchors, out_dir, name="ancres"):
    plt = st.plt()
    ks = sorted(anchors)
    fig, axes = plt.subplots(1, len(ks), figsize=(st.FULL, 4.6), sharey=True)
    for ax, k in zip(np.atleast_1d(axes), ks):
        ours, iou, dname, dark, diou = anchors[k]
        ax.hexbin(wh[:, 0], wh[:, 1], gridsize=60, bins="log", cmap="Greys", mincnt=1,
                  linewidths=0, rasterized=True)
        ax.plot(ours[:, 0], ours[:, 1], "o", ms=9, mfc=st.PALETTE[0], mec="white", mew=1.5,
                label=f"k-means k={k} : IoU {iou:.4f}")
        ax.plot(dark[:, 0], dark[:, 1], "D", ms=8, mfc="white", mec=st.PALETTE[1], mew=1.8,
                label=f"Darknet {dname} : IoU {diou:.4f}")
        lim = max(416, float(np.max(dark)) * 1.05)
        ax.set_xlim(0, lim)
        ax.set_ylim(0, lim)
        ax.set_aspect("equal")
        ax.set_xlabel("largeur (px, letterbox 416)")
        ax.set_title(f"k = {k}")
        ax.legend(loc="upper right")
    np.atleast_1d(axes)[0].set_ylabel("hauteur (px)")
    fig.suptitle(f"Boîtes VOC07+12 trainval non-difficult ({len(wh)}), densité log, et ancres",
                 fontsize=9)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("ancres", "modeles", "T13.4",
        "Nuage (w, h) des boîtes VOC avec les ancres k-means (distance 1 − IoU) et les "
        "ancres Darknet.",
        "data/VOCdevkit, results/anchors.md, tools/kmeans_anchors.py:dataset_wh")
def ancres(out_dir):
    anchors = load_anchors_md(need(ROOT / "results" / "anchors.md"))
    need(DEVKIT / "VOC2012" / "ImageSets" / "Main" / "trainval.txt")
    from yolo.data.voc import load_split

    from tools.kmeans_anchors import dataset_wh

    samples = [s for y in (2007, 2012) for s in load_split(DEVKIT, y, "trainval")]
    return plot_anchors(dataset_wh(samples), anchors, out_dir)


# --- T13.7 : statistiques VOC -------------------------------------------------------------

SPLITS = {"trainval 07+12": [(2007, "trainval"), (2012, "trainval")],
          "test 2007": [(2007, "test")]}
AREA_BINS = (32 ** 2, 96 ** 2)  # petits / moyens / grands (convention COCO, en pixels)


def load_voc_stats(devkit=None):
    """{split: {per_class (C,), areas (n,), per_image (images,)}}, objets non-difficult."""
    from yolo.data.voc import VOC_CLASSES, load_split

    devkit = devkit or DEVKIT
    out = {}
    for name, splits in SPLITS.items():
        for year, split in splits:
            need(Path(devkit) / f"VOC{year}" / "ImageSets" / "Main" / f"{split}.txt")
        samples = [s for y, sp in splits for s in load_split(devkit, y, sp)]
        per_class = np.zeros(len(VOC_CLASSES), dtype=np.int64)
        areas, per_image = [], []
        for s in samples:
            keep = ~s["difficult"]
            np.add.at(per_class, s["labels"][keep], 1)
            b = s["xyxy"][keep]
            areas.append((b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1))
            per_image.append(int(keep.sum()))
        out[name] = {"per_class": per_class, "areas": np.concatenate(areas),
                     "per_image": np.array(per_image), "images": len(samples)}
    return out


def plot_voc_stats(stats, out_dir, name="voc_stats"):
    from yolo.data.voc import VOC_CLASSES

    plt = st.plt()
    fig = plt.figure(figsize=(st.FULL, 6.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
    ax = fig.add_subplot(gs[0, :])
    splits = list(stats)
    x = np.arange(len(VOC_CLASSES))
    for k, sp in enumerate(splits):
        ax.bar(x + (k - 0.5) * 0.4, stats[sp]["per_class"], 0.38, color=st.PALETTE[k],
               hatch=("", "//")[k], edgecolor="white", linewidth=0.4,
               label=f"{sp} : {stats[sp]['per_class'].sum()} objets, {stats[sp]['images']} images")
    ax.set_xticks(x, VOC_CLASSES, rotation=40, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("objets (non-difficult)")
    ax.set_title("Objets par classe")
    ax.legend(loc="upper left")
    ax.grid(axis="x", visible=False)

    ax2 = fig.add_subplot(gs[1, 0])
    bins = np.logspace(1, np.log10(500 * 500), 60)
    for k, sp in enumerate(splits):
        a = stats[sp]["areas"]
        ax2.hist(a, bins=bins, histtype="step", lw=1.6, color=st.PALETTE[k], density=True,
                 ls=("-", "--")[k], label=sp)
    for b in AREA_BINS:
        ax2.axvline(b, color=st.MUTED, lw=0.9, ls=":")
    a = stats[splits[-1]]["areas"]
    frac = [np.mean(a < AREA_BINS[0]), np.mean((a >= AREA_BINS[0]) & (a < AREA_BINS[1])),
            np.mean(a >= AREA_BINS[1])]
    st.note(ax2, f"{splits[-1]} : petits {frac[0] * 100:.0f} %, moyens {frac[1] * 100:.0f} %, "
            f"grands {frac[2] * 100:.0f} %", "upper left")
    ax2.set_xscale("log")
    ax2.set_xlabel("aire de la boîte (px², image d'origine)")
    ax2.set_ylabel("densité")
    ax2.set_title("Aires des boîtes (seuils 32² et 96²)")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[1, 1])
    top = max(int(stats[sp]["per_image"].max()) for sp in splits)
    edges = np.arange(0, min(top, 25) + 2) - 0.5
    for k, sp in enumerate(splits):
        ax3.hist(np.minimum(stats[sp]["per_image"], 25), bins=edges, histtype="step", lw=1.6,
                 color=st.PALETTE[k], density=True, ls=("-", "--")[k], label=sp)
    ax3.set_xlabel("objets par image (25 = 25 et plus)")
    ax3.set_title("Objets par image")
    ax3.legend(loc="upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("voc_stats", "modeles", "T13.7",
        "Statistiques de VOC : objets par classe en trainval et en test, aires des boîtes, "
        "objets par image.",
        "data/VOCdevkit (comptes égaux à tools/voc_stats.py)")
def voc_stats(out_dir):
    return plot_voc_stats(load_voc_stats(), out_dir)
