"""Figures des réseaux en détail (M13, section G) : fiche, champ réceptif, flux, tête, une couche.

Les formes et coûts viennent de `yolo.models.specs` (`infer_shapes`, `layer_cost`) appliqué
aux descriptions des manifests (`tools/count_macs.net_from_manifest`), les ancres et masques
des `.cfg` (`yolo.models.tiny_yolo.load_cfg`).
"""

import csv
import math

import numpy as np

from tools.figures import ROOT, MissingSource, figure, need
from tools.figures import style as st

LAYER_COLORS = {"conv": st.PALETTE[0], "maxpool": st.PALETTE[2], "route": st.PALETTE[3],
                "upsample": st.PALETTE[4], "yolo": st.PALETTE[1], "region": st.PALETTE[1]}


def _desc(net):
    from tools.figures.modeles import _manifest_net

    return _manifest_net(net)


def _nets():
    """{net: description} des manifests présents."""
    out = {}
    for net in st.NETS:
        try:
            out[net] = _desc(net)
        except MissingSource:
            pass
    if not out:
        raise MissingSource("aucun model/<net>/manifest.json (make export)")
    return out


# --- T13.48 : fiche couche par couche -----------------------------------------------------

def receptive_fields(desc):
    """[(r, j)] par couche : champ réceptif r et pas cumulé j (pixels d'entrée) de la sortie.

    r_l = r_{l−1} + (k_l − 1)·j_{l−1}, j_l = j_{l−1}·s_l ; un upsample divise j par s, une
    route reprend ses sources (champ le plus large).
    """
    out = []
    r, j = 1, 1
    for layer in desc["layers"]:
        t = layer["type"]
        if t == "route":
            srcs = [out[i] for i in layer["from"]]
            r, j = max(s[0] for s in srcs), srcs[0][1]
        elif t in ("conv", "maxpool"):
            r, j = r + (layer["k"] - 1) * j, j * layer["s"]
        elif t == "upsample":
            j = j / layer["s"]
        out.append((r, j))
    return out


def layer_table(desc):
    """Lignes de la fiche : une par couche, formes de `infer_shapes`, coûts de `layer_cost`."""
    from yolo.models.specs import infer_shapes, layer_cost

    rows = []
    for i, (layer, (ins, outs), (r, j)) in enumerate(zip(desc["layers"], infer_shapes(desc),
                                                         receptive_fields(desc))):
        t = layer["type"]
        params, macs = layer_cost(layer, ins, outs)
        k = layer.get("k", "")
        rows.append({
            "id": i, "type": t, "k": k, "s": layer.get("s", ""),
            "pad": k // 2 if t == "conv" else "", "cin": ins[0], "cout": outs[0],
            "in_hw": f"{ins[1]}×{ins[2]}", "out_hw": f"{outs[1]}×{outs[2]}",
            "params": int(params), "macs": int(macs), "rf": int(r), "stride": j,
            "act": layer.get("act", ""), "bn": layer.get("bn", ""),
            "from": " ".join(map(str, layer.get("from", []))),
        })
    return rows


def plot_layer_table(net, rows, out_dir, name):
    plt = st.plt()
    tot_m = sum(r["macs"] for r in rows)
    tot_p = sum(r["params"] for r in rows)
    head = ["L", "type", "k/s/pad", "cin → cout", "entrée", "sortie", "paramètres", "MACs (M)",
            "part des MACs", "champ r", "pas j", "act.", "BN"]
    xs = [0, 0.4, 1.35, 2.25, 3.45, 4.35, 5.25, 6.3, 7.2, 8.85, 9.65, 10.25, 10.95]
    n = len(rows)
    fig, ax = plt.subplots(figsize=(st.FULL, 0.26 * (n + 3) + 0.6))
    ax.set_xlim(-0.1, 11.3)
    ax.set_ylim(n + 1.6, -0.9)
    ax.axis("off")
    for x, h in zip(xs, head):
        ax.text(x, -0.3, h, fontsize=7.5, weight="bold", va="center")
    ax.axhline(0.25, color=st.INK2, lw=0.8)
    for y, r in enumerate(rows, start=1):
        if y % 2:
            ax.axhspan(y - 0.5, y + 0.5, color=st.GRID, alpha=0.5, lw=0)
        t = r["type"]
        kspad = (f"{r['k']}/{r['s']}/{r['pad']}" if t == "conv" else
                 f"{r['k']}/{r['s']}" if t == "maxpool" else f"×{r['s']}" if t == "upsample"
                 else f"← {r['from']}" if t == "route" else "")
        cells = [f"{r['id']:02d}", t, kspad, f"{r['cin']} → {r['cout']}", r["in_hw"],
                 r["out_hw"], f"{r['params']:,}".replace(",", " ") if r["params"] else "",
                 f"{r['macs'] / 1e6:.1f}" if r["macs"] else "", "", str(r["rf"]),
                 f"{r['stride']:g}", r["act"] if t == "conv" else "",
                 ("oui" if r["bn"] else "non") if t == "conv" else ""]
        for x, c in zip(xs, cells):
            ax.text(x, y, c, fontsize=7.2, va="center",
                    color=LAYER_COLORS.get(t, st.INK) if x == xs[1] else st.INK)
        if r["macs"]:
            frac = r["macs"] / tot_m
            ax.barh(y, 1.5 * frac, left=xs[8], height=0.55, color=st.PALETTE[0])
            ax.text(xs[8] + 1.5 * frac + 0.05, y, f"{100 * frac:.1f} %", fontsize=6.5,
                    va="center", color=st.INK2)
    y = n + 1.1
    ax.axhline(n + 0.6, color=st.INK2, lw=0.8)
    ax.text(xs[0], y, "total", fontsize=7.5, weight="bold", va="center")
    ax.text(xs[6], y, f"{tot_p:,}".replace(",", " "), fontsize=7.5, weight="bold", va="center")
    ax.text(xs[7], y, f"{tot_m / 1e9:.3f} G", fontsize=7.5, weight="bold", va="center")
    ax.set_title(f"{st.NET_LABELS.get(net, net)} — fiche couche par couche "
                 "(yolo.models.specs : infer_shapes, layer_cost)")
    paths = st.save(fig, out_dir, name)
    with open(paths[0].with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return paths


@figure("fiche", "reseaux", "T13.48",
        "Fiche couche par couche : type, noyau, formes, paramètres, MACs et leur part, champ "
        "réceptif et pas cumulé (CSV à côté de l'image).",
        "model/<net>/manifest.json, yolo.models.specs")
def fiche(out_dir):
    paths = []
    for net, desc in _nets().items():
        paths += plot_layer_table(net, layer_table(desc), out_dir, f"fiche_{net}")
    return paths


# --- T13.49 : champ réceptif ---------------------------------------------------------------

def head_geometry(net):
    """[(id, S, pas en pixels, ancres en pixels)] des têtes, ancres et masques du .cfg."""
    from yolo.data.targets import ANCHOR_REF, heads
    from yolo.models.tiny_yolo import load_cfg

    spec = load_cfg(net)
    desc = _desc(net)
    rf = receptive_fields(desc)
    anchors = np.asarray(spec["anchors"], dtype=np.float64).reshape(-1, 2) * 416 / ANCHOR_REF
    return [(hid, int(416 / rf[hid][1]), rf[hid][1], rf[hid][0], anchors[mask])
            for hid, mask in heads(spec)]


def plot_receptive(nets, out_dir, name="champ_receptif"):
    from matplotlib.patches import Rectangle

    plt = st.plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(st.FULL, 4.6), gridspec_kw={"width_ratios":
                                                                              [1.3, 1]})
    for net, desc in nets.items():
        rf = receptive_fields(desc)
        c = st.NET_COLORS[net]
        lw = 3.2 if "v2" in net else 1.4  # tronc commun : v2 en trait épais sous v3
        ax.step(range(len(rf)), [r for r, _ in rf], where="post", color=c, lw=lw,
                alpha=0.55 if "v2" in net else 1, label=f"{st.NET_LABELS[net]} : champ r")
        ax.step(range(len(rf)), [j for _, j in rf], where="post", color=c, ls="--", lw=lw,
                alpha=0.55 if "v2" in net else 1, label=f"{st.NET_LABELS[net]} : pas j")
    ax.axhline(416, color=st.MUTED, lw=0.9, ls=":")
    ax.text(23, 430, "image 416", fontsize=7, color=st.MUTED, ha="right")
    ax.set_yscale("log", base=2)
    ax.set_xlabel("indice de couche")
    ax.set_ylabel("pixels d'entrée")
    ax.set_title("r_l = r_{l−1} + (k_l − 1)·j_{l−1},  j_l = j_{l−1}·s_l")
    ax.legend(loc="lower right", fontsize=7)
    # cellules sur une image
    img = ROOT / "data" / "VOCdevkit" / "VOC2007" / "JPEGImages" / "000009.jpg"
    if img.exists():
        from PIL import Image

        from yolo.data.letterbox import letterbox_image

        with Image.open(img) as im:
            ax2.imshow(letterbox_image(im.convert("RGB"), 416)[0], alpha=0.55)
    ax2.set_xlim(0, 416)
    ax2.set_ylim(416, 0)
    geoms = [(net, g) for net in nets for g in head_geometry(net)]
    centers = [(208 - 16, 208 - 16), (208 + 8, 208 + 8), (208 + 56, 208 + 72)]
    for k, ((net, (hid, S, j, r, anchors)), (cx, cy)) in enumerate(zip(geoms, centers)):
        c = st.PALETTE[k]
        cell = 416 / S
        x0, y0 = (cx // cell) * cell, (cy // cell) * cell
        ax2.add_patch(Rectangle((x0, y0), cell, cell, color=c, alpha=0.8))
        h = r / 2
        ax2.add_patch(Rectangle((x0 + cell / 2 - h, y0 + cell / 2 - h), r, r, fill=False, ec=c,
                                lw=1.2, ls="--"))
        for w_, h_ in anchors:
            ax2.add_patch(Rectangle((x0 + cell / 2 - w_ / 2, y0 + cell / 2 - h_ / 2), w_, h_,
                                    fill=False, ec=c, lw=0.7, alpha=0.8))
        ax2.plot([], [], color=c, label=f"{'v2' if 'v2' in net else 'v3'} L{hid} : {S}×{S}, "
                 f"pas {j:g} px, r = {r:g} px")
    st.image_axes(ax2, "cellule, champ réceptif (tirets) et ancres")
    ax2.legend(loc="lower center", fontsize=6.5, bbox_to_anchor=(0.5, -0.2), frameon=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("champ_receptif", "reseaux", "T13.49",
        "Champ réceptif et pas cumulé par couche, puis une cellule de chaque tête sur une image "
        "416 avec son champ réceptif et ses ancres.",
        "model/<net>/manifest.json, yolo.models.cfg (ancres, masques)")
def champ_receptif(out_dir):
    return plot_receptive(_nets(), out_dir)


# --- T13.50 : flux des tenseurs -------------------------------------------------------------

def tensor_layout(desc):
    """[(id, x, rangée, largeur, hauteur, forme, sources)] ; largeur ∝ log2 C, hauteur ∝
    log2 H ; une route qui ne suit pas sa source ouvre une nouvelle rangée."""
    from yolo.models.specs import infer_shapes

    shapes = infer_shapes(desc)
    pos, out = {}, []
    x, row = 0.0, 0
    for i, (layer, (_, o)) in enumerate(zip(desc["layers"], shapes)):
        t = layer["type"]
        srcs = layer.get("from", [i - 1]) if t == "route" else [i - 1]
        if t == "route" and srcs[0] != i - 1:
            row += 1
            px, _, pw = pos[srcs[0]]
            x = px + pw + 0.6
        w = 0.25 + 0.12 * math.log2(o[0])
        h = 0.4 + 0.32 * math.log2(o[1])
        out.append((i, x, row, w, h, o, srcs, t))
        pos[i] = (x, row, w)
        x += w + 0.5
    return out


def plot_flow(net, desc, out_dir, name):
    from matplotlib.patches import Polygon

    plt = st.plt()
    lay = tensor_layout(desc)
    rows = max(r for _, _, r, *_ in lay) + 1
    xmax = max(x + w for _, x, _, w, *_ in lay)
    fig, ax = plt.subplots(figsize=(st.FULL, 2.6 + 2.8 * rows))
    pitch = 6.2
    centers = {}
    for i, x, r, w, h, o, srcs, t in lay:
        yc = -r * pitch
        c = LAYER_COLORS.get(t, st.MUTED)
        dx = dy = 0.18
        y0 = yc - h / 2
        ax.add_patch(Polygon([(x, y0), (x + w, y0), (x + w, y0 + h), (x, y0 + h)], fc=c,
                             alpha=0.35, ec=c))
        ax.add_patch(Polygon([(x, y0 + h), (x + dx, y0 + h + dy), (x + w + dx, y0 + h + dy),
                              (x + w, y0 + h)], fc=c, alpha=0.55, ec=c))
        ax.add_patch(Polygon([(x + w, y0), (x + w + dx, y0 + dy), (x + w + dx, y0 + h + dy),
                              (x + w, y0 + h)], fc=c, alpha=0.2, ec=c))
        ax.text(x + w / 2, y0 - 0.12, f"L{i:02d} {t}  {o[0]}×{o[1]}×{o[2]}", ha="center",
                va="top", fontsize=5.8, rotation=90)
        centers[i] = (x, x + w, yc)
        for s in srcs:
            if s < 0 or s not in centers:
                continue
            sx0, sx1, sy = centers[s]
            if s == i - 1 and sy == yc:
                st.arrow(ax, (sx1 + 0.18, yc), (x, yc), lw=0.8)
            else:
                ax.annotate("", (x + w / 2, yc + h / 2 + 0.2), ((sx0 + sx1) / 2, sy - 3.4),
                            arrowprops=dict(arrowstyle="->", color=st.PALETTE[3], lw=1.2,
                                            connectionstyle="arc3,rad=0.15"))
    ax.set_xlim(-0.3, xmax + 0.5)
    ax.set_ylim(-(rows - 1) * pitch - 6.0, 3.0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{st.NET_LABELS.get(net, net)} — flux des tenseurs C×H×W (largeur ∝ log₂ C, "
                 "hauteur ∝ log₂ H ; formes de infer_shapes)")
    return st.save(fig, out_dir, name)


@figure("flux", "reseaux", "T13.50",
        "Flux des tenseurs couche par couche, un bloc par tenseur (largeur selon C, hauteur "
        "selon H) ; en v3, route, upsample et seconde tête.",
        "model/<net>/manifest.json, yolo.models.specs.infer_shapes")
def flux(out_dir):
    paths = []
    for net, desc in _nets().items():
        paths += plot_flow(net, desc, out_dir, f"flux_{net}")
    return paths


# --- T13.51 : la tête de sortie ------------------------------------------------------------

HEAD_FIELDS = ("t_x", "t_y", "t_w", "t_h", "t_o")


def head_vector(net="tiny-yolov2-voc", image="000001"):
    """Vecteur (5 + C) de la meilleure (ancre, cellule) d'une tête dumpée, entier et décodé.

    Sortie int8 de la tête (`model/<net>/dumps/<image>/L<id>.npy`), déquantifiée par l'échelle
    du manifest puis décodée par `decode_head` (mode v2 pour `region`, v3 sinon).
    """
    import json

    from yolo.data.targets import anchors_frac, heads
    from yolo.infer.decode import decode_head
    from yolo.models.tiny_yolo import load_cfg

    from tools.figures.resultats import layer_scales

    m = json.loads(need(ROOT / "model" / net / "manifest.json").read_text())
    spec = load_cfg(net)
    hid, mask = heads(spec)[0]
    q = np.load(need(ROOT / "model" / net / "dumps" / image / f"L{hid:02d}.npy"))
    q = q.reshape(-1, *q.shape[-3:])[0]
    scale = layer_scales(m)[hid]
    a, C = len(mask), spec["classes"]
    mode = "v2" if spec["layers"][hid]["type"] == "region" else "v3"
    boxes, obj, scores = decode_head(q[None] * scale, anchors_frac(spec["anchors"])[mask], C, mode)
    best = int(np.argmax(scores[0].max(axis=1)))
    S = q.shape[-1]
    k, i, j = best // (S * S), (best // S) % S, best % S
    qv = q.reshape(a, 5 + C, S, S)[k, :, i, j]
    return {"net": net, "mode": mode, "A": a, "C": C, "S": S, "anchor": k, "cell": (i, j),
            "q": qv, "t": qv * scale, "scale": scale, "box": boxes[0, best],
            "obj": obj[0, best], "scores": scores[0, best], "image": image}


def plot_head(v, out_dir, name="tete"):
    from matplotlib.patches import Rectangle

    from tools.figures.modeles import class_names

    plt = st.plt()
    names = class_names(v["net"])
    fig = plt.figure(figsize=(st.FULL, 6.4))
    ax = fig.add_axes([0.03, 0.62, 0.94, 0.3])
    a, C = v["A"], v["C"]
    for r in range(a):
        for c in range(5 + C):
            col = st.PALETTE[c] if c < 5 else st.PALETTE[6]
            ax.add_patch(Rectangle((c, -r), 0.92, 0.85, fc=col, alpha=0.25 if c >= 5 else 0.6,
                                   ec="white"))
            if r == 0:
                ax.text(c + 0.46, 1.05, HEAD_FIELDS[c] if c < 5 else names[c - 5][:5],
                        rotation=60, fontsize=6.5, ha="left", va="bottom")
        ax.text(-0.3, -r + 0.42, f"ancre {r}", ha="right", va="center", fontsize=7)
    ax.add_patch(Rectangle((0, -v["anchor"]), 5 + C - 0.08, 0.85, fill=False, ec=st.INK, lw=1.4))
    ax.set_xlim(-2.2, 5 + C + 0.4)
    ax.set_ylim(-a + 0.6, 3.4)
    ax.axis("off")
    ax.set_title(f"Sortie (A·(5 + C), S, S) = ({a * (5 + C)}, {v['S']}, {v['S']}) vue en "
                 f"(A, 5 + C, S, S) : {a} × (5 + {C}) — en v3 COCO, 3 × (5 + 80) = 255 par tête",
                 fontsize=9)
    # vecteur décodé
    ax2 = fig.add_axes([0.07, 0.08, 0.4, 0.42])
    x = np.arange(5)
    ax2.bar(x - 0.2, v["t"][:5], 0.4, color=st.PALETTE[0], label="t (int8 × échelle)")
    dec = [v["box"][0] * v["S"] - v["cell"][1], v["box"][1] * v["S"] - v["cell"][0],
           v["box"][2], v["box"][3], v["obj"]]
    ax2.bar(x + 0.2, dec, 0.4, color=st.PALETTE[1], label="décodé")
    for xi, qv in zip(x, v["q"][:5]):
        ax2.text(xi - 0.2, 0, f"q={qv}", rotation=90, fontsize=6.5, ha="center",
                 va="bottom" if v["t"][xi] < 0 else "top", color=st.INK2)
    ax2.set_xticks(x, [r"$\sigma(t_x)$", r"$\sigma(t_y)$", r"$p_w e^{t_w}$", r"$p_h e^{t_h}$",
                       r"$\sigma(t_o)$"])
    ax2.axhline(0, color=st.INK2, lw=0.8)
    ax2.legend(loc="upper left")
    ax2.set_title(f"cellule {v['cell']}, ancre {v['anchor']} ({v['image']}, échelle "
                  f"{v['scale']:.4f})", fontsize=8.5)
    ax3 = fig.add_axes([0.56, 0.08, 0.41, 0.42])
    top = np.argsort(-v["scores"])[:6]
    cls = v["scores"] / max(v["obj"], 1e-12)
    ax3.barh(range(len(top)), cls[top], color=st.PALETTE[6], alpha=0.7,
             label="softmax(t_c)" if v["mode"] == "v2" else "σ(t_c)")
    ax3.barh(range(len(top)), v["scores"][top], color=st.PALETTE[6], label="score σ(t_o)·p_c")
    ax3.set_yticks(range(len(top)), [names[c] for c in top])
    ax3.invert_yaxis()
    ax3.set_xlim(0, 1)
    ax3.legend(loc="lower right")
    ax3.set_title("classes : softmax (v2) ou sigmoïdes indépendantes (v3)", fontsize=8.5)
    return st.save(fig, out_dir, name)


@figure("tete", "reseaux", "T13.51",
        "La tête de sortie : disposition des canaux (ancres × (5 + C)), puis le vecteur d'une "
        "cellule en int8 et décodé (sigmoïdes, exponentielles, softmax).",
        "model/<net>/dumps/<image>/L<tête>.npy, manifest, yolo.infer.decode.decode_head")
def tete(out_dir):
    return plot_head(head_vector(), out_dir)


# --- T13.52 : une couche à travers toutes les représentations -------------------------------

ONE_LAYER = 2  # L02 de v2 : conv 3×3 16 → 32, BN, leaky, maxpool fusionné (L03)
STAGES = [  # (titre, fichier, fonctions vérifiées dans le fichier, lignes affichées)
    ("flottant", "python/yolo/models/graph.py", ["_conv_forward"],
     ["conv 3×3 (float32)", "BN : γ(x − μ)/σ + β", "leaky 0,1", "maxpool 2×2/2"]),
    ("BN fusionnée", "python/yolo/quant/fuse_bn.py", ["fuse_bn"],
     ["W' = s·W, s = γ/√(σ² + ε)", "b' = β + s(b − μ)", "float32"]),
    ("entier Python", "python/yolo/quant/int_layers.py",
     ["conv_acc", "requantize", "leaky_int", "clip_q", "maxpool_int"],
     ["conv_acc : int8·int8 → int32", "requantize : (acc·M0 + 2ⁿ⁻¹)≫n",
      "leaky_int : (13y + 64) ≫ 7", "clip_q → int8", "maxpool_int"]),
    ("golden C++", "cpp/golden/include/golden/conv.hpp", ["conv_layer", "requantize", "leaky_int"],
     ["golden::conv_layer", "golden::requantize", "golden::leaky_int", "int8 / int32"]),
    ("HLS (C-sim)", "hls/kernels/conv_pe.cpp", ["load_input", "load_weights", "compute"],
     ["load_input / load_weights", "compute : Tm × Tn MAC", "store_tile →",
      "golden::requantize"]),
]


def one_layer_facts(net="tiny-yolov2-voc", layer=ONE_LAYER):
    """Écarts mesurés entre stades pour `layer` : SNR flottant ↔ entier (dumps, T13.18),
    octets différents entier ↔ golden, état C-sim lu dans le suivi."""
    import re

    from tools.figures import materiel, resultats
    from tools.figures.projet import load_tracking

    for _, path, names, _ in STAGES:
        text = need(ROOT / path).read_text()
        missing = [n for n in names if not re.search(rf"\b{n}\b", text)]
        if missing:
            raise AssertionError(f"{path} : fonctions absentes {missing}")
    require = materiel.require_names
    require(materiel.OUTPUT_STAGE, ["store_tile", "golden::requantize"])
    errs = resultats.load_layer_errors(net)
    snr = [dict((i, s) for i, s, _ in rows).get(layer) for _, rows in errs]
    gold = resultats.load_golden_diffs(net)
    diffs = None if gold is None else [dict(rows).get(layer) for _, rows in gold]
    states = materiel.chain_states(load_tracking())
    return {"snr": [s for s in snr if s is not None], "golden": diffs,
            "csim": states[2], "images": len(errs)}


def plot_one_layer(facts, out_dir, name="une_couche"):
    plt = st.plt()
    n = len(STAGES)
    fig, ax = plt.subplots(figsize=(st.FULL, 4.0))
    st.schema_axes(ax, (0, 4.4 * n - 0.8), (0, 5.2))
    snr = facts["snr"]
    gaps = [
        "même réseau\n(fusion exacte)",
        f"SNR {min(snr):.1f} à {max(snr):.1f} dB\n({facts['images']} dumps)" if snr else "SNR ?",
        ("0 octet différent" if facts["golden"] and not any(facts["golden"]) else
         f"{facts['golden']} octets" if facts["golden"] else "golden non lancé") +
        f"\n({len(facts['golden'] or [])} dumps)",
        f"0 écart (tb_net)\n{facts['csim']}",
    ]
    widths = ["float32", "float32", "int8 → int32 → int8", "int8 → int32 → int8", "mots 64 bits"]
    for i, (title, path, _, lines) in enumerate(STAGES):
        x = 4.4 * i
        c = st.PALETTE[i]
        st.block(ax, x, 0.7, 3.6, 3.6, "", c, fill=0.1)
        ax.text(x + 1.8, 3.85, title, ha="center", fontsize=9, weight="bold")
        ax.text(x + 0.12, 3.4, "\n".join(lines), fontsize=6.4, va="top", linespacing=1.6)
        ax.text(x + 1.8, 0.9, path.rsplit("/", 1)[-1], ha="center", fontsize=6.3, color=st.MUTED)
        ax.text(x + 1.8, 4.55, widths[i], ha="center", fontsize=7, color=st.INK2)
        if i:
            st.arrow(ax, (x - 0.8, 2.5), (x, 2.5), lw=1.6)
            ax.text(x - 0.4, 0.45, gaps[i - 1], ha="center", va="top", fontsize=6.3,
                    color=st.PALETTE[5] if "0 " in gaps[i - 1] or "exacte" in gaps[i - 1]
                    else st.INK2)
    ax.set_title(f"Une couche (L{ONE_LAYER:02d} de Tiny-YOLOv2 : conv 3×3 16 → 32, BN, leaky, "
                 "maxpool) à travers toutes les représentations", fontsize=9)
    return st.save(fig, out_dir, name)


@figure("une_couche", "reseaux", "T13.52",
        "Une couche suivie du flottant au noyau HLS : formules, types et largeurs, écarts "
        "mesurés entre stades (SNR, octets).",
        "model/<net>/dumps, build/golden/out, suivi du README ; noms vérifiés dans le code")
def une_couche(out_dir):
    return plot_one_layer(one_layer_facts(), out_dir)
