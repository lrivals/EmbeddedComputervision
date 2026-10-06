"""Figures du matériel (M13, section C) : SoC, moteur, tuilage, streaming, vérification.

Les schémas lisent leurs noms et constantes dans le code (regmap.hpp, pragmas de
conv_pe.cpp, accel_config.hpp via `perf_model.kernel_config`, manifest) : un nom absent du
code fait échouer le générateur plutôt que de dessiner un schéma faux.
"""

import json
import re
import textwrap

import numpy as np

from tools.figures import ROOT, figure, need
from tools.figures import style as st

NET = "tiny-yolov2-voc"
REGMAP = "sw/driver/regmap.hpp"
KERNEL = "hls/kernels/conv_pe.cpp"
OUTPUT_STAGE = "hls/kernels/output_stage.hpp"


def _tools(name):
    import importlib

    return importlib.import_module(f"tools.{name}")


def require_names(path, names):
    """Vérifie que chaque nom apparaît dans `path` (sinon le schéma serait faux)."""
    text = need(ROOT / path).read_text()
    missing = [n for n in names if not re.search(rf"\b{re.escape(n)}\b", text)]
    if missing:
        raise AssertionError(f"{path} : noms absents du code {missing}")
    return text


# --- T13.8 : SoC --------------------------------------------------------------------------

def load_soc(net=NET):
    """{regs: [(nom, offset)], ports: [(port, bundle)], blobs, buffers} lus dans le code."""
    regs = re.findall(r"constexpr uint32_t (\w+) = (0x[0-9a-f]+);", need(ROOT / REGMAP).read_text())
    ports = re.findall(r"INTERFACE m_axi port=(\w+) offset=slave bundle=(\w+)",
                       need(ROOT / KERNEL).read_text())
    m = json.loads(need(ROOT / "model" / net / "manifest.json").read_text())
    return {"regs": [(n, int(o, 16)) for n, o in regs], "ports": ports,
            "blobs": m["blobs"], "buffers": m["buffers"]}


def plot_soc(soc, out_dir, name="soc"):
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.FULL, 5.6))
    st.schema_axes(ax, (0, 20), (0, 11))
    c_ps, c_pl, c_ddr = st.PALETTE[0], st.PALETTE[1], st.PALETTE[6]
    # PS
    st.block(ax, 0.2, 0.4, 6.2, 10.2, "", c_ps, fill=0.05)
    ax.text(0.4, 10.3, "PS : ARM Cortex-A53 (Linux)", fontsize=9, weight="bold", va="top")
    st.block(ax, 0.6, 8.0, 5.4, 1.4, "yolo_app / yolo_bench\nprétraitement, post-traitement",
             c_ps)
    st.block(ax, 0.6, 5.6, 5.4, 1.8, "driver sw/driver/\nprogram.cpp : arène, descripteurs\n"
             "accel_driver.cpp : une couche = un appel", c_ps)
    st.block(ax, 0.6, 3.3, 2.6, 1.7, "Device « uio »\n/dev/uioN\n/dev/udmabufN", c_ps)
    st.block(ax, 3.4, 3.3, 2.6, 1.7, "Device « sim »\nregistres émulés\n→ noyau C-sim", c_ps,
             ls="--")
    ax.text(3.3, 2.8, "device.hpp : deux réalisations", fontsize=7, ha="center", color=st.MUTED)
    st.arrow(ax, (3.3, 8.0), (3.3, 7.4))
    st.arrow(ax, (1.9, 5.6), (1.9, 5.0))
    st.arrow(ax, (4.7, 5.6), (4.7, 5.0))
    # AXI-Lite
    regs = "\n".join(f"0x{o:02x}  {n}" for n, o in soc["regs"]
                     if not n.startswith("AP_") and n not in ("END", "SPAN"))
    st.block(ax, 7.0, 3.0, 3.6, 7.6, "", st.PALETTE[3], fill=0.08)
    ax.text(8.8, 10.3, "AXI-Lite « control »", fontsize=8.5, weight="bold", ha="center",
            va="top")
    ax.text(7.25, 9.7, regs, fontsize=7, family="monospace", va="top", linespacing=1.35)
    ax.text(8.8, 3.3, "regmap.hpp", fontsize=7, ha="center", color=st.MUTED)
    st.arrow(ax, (6.0, 4.2), (7.0, 6.5), "write32 / read32", fontsize=6.5)
    # PL
    st.block(ax, 11.2, 3.0, 4.0, 7.6, "", c_pl, fill=0.05)
    ax.text(13.2, 10.3, "PL : noyau yolo_conv", fontsize=9, weight="bold", ha="center", va="top")
    st.block(ax, 11.6, 7.6, 3.2, 1.8, "moteur de conv\n(voir moteur.png)", c_pl)
    for k, (port, bundle) in enumerate(soc["ports"]):
        y = 6.6 - 0.75 * k
        st.block(ax, 11.6, y, 3.2, 0.55, f"m_axi {port} → {bundle}", c_pl, fill=0.22,
                 fontsize=7)
    st.arrow(ax, (10.6, 8.5), (11.6, 8.5), "ap_start", fontsize=6.5)
    # DDR
    st.block(ax, 15.8, 0.4, 4.0, 10.2, "", c_ddr, fill=0.05)
    ax.text(17.8, 10.3, "DDR (zone contiguë)", fontsize=9, weight="bold", ha="center", va="top")
    items = [(f"activations {k}", v) for k, v in soc["buffers"].items()]
    items += [(k.removesuffix(".bin"), v) for k, v in soc["blobs"].items()]
    total = sum(v for _, v in items)
    small = [v for _, v in items if 8.4 * v / total < 0.45]
    room = 8.4 - 0.45 * len(small) - 0.08 * len(items)
    big = total - sum(small)
    y = 9.7
    for lab, v in items:
        h = 0.45 if v in small else room * v / big
        y -= h + 0.08
        st.block(ax, 16.1, y, 3.4, h, f"{lab} : {v / 1e6:.2f} Mo" if v > 1e5 else
                 f"{lab} : {v / 1e3:.1f} Ko", c_ddr, fill=0.2, fontsize=7)
    st.arrow(ax, (14.8, 5.0), (16.1, 5.0), "AXI HP", fontsize=6.5, both=True)
    st.arrow(ax, (6.0, 1.2), (15.8, 1.2), "l'ARM écrit l'entrée et lit les têtes (virt) ; le "
             "noyau lit et écrit en adresses physiques (phys)", fontsize=7, both=True)
    ax.set_title(f"SoC KV260 : PS, registres, noyau et arène DDR ({NET}, manifest)")
    return st.save(fig, out_dir, name)


@figure("soc", "materiel", "T13.8",
        "Schéma du SoC : application et driver sur l'ARM, registres AXI-Lite, ports m_axi du "
        "noyau et découpage de l'arène DDR.",
        "sw/driver/regmap.hpp, pragmas de hls/kernels/conv_pe.cpp, model/<net>/manifest.json")
def soc(out_dir):
    return plot_soc(load_soc(), out_dir)


# --- T13.9 : moteur unique ----------------------------------------------------------------

ENGINE_NAMES = ("load_input", "load_weights", "compute", "conv_tile", "in_buf", "w_buf")
STAGE_NAMES = ("store_tile", "write_row")
DEMO_LAYER = 6  # L06 de v2 : conv 3×3, 64 → 128, 52×52 (maxpool fusionné)


def load_engine(net=NET, layer=DEMO_LAYER):
    """Constantes du noyau et cycles par tuile d'une couche (perf_model)."""
    pm = _tools("perf_model")
    require_names(KERNEL, ENGINE_NAMES)
    require_names(OUTPUT_STAGE, STAGE_NAMES)
    layers = pm.conv_layers(need(ROOT / "model" / net / "manifest.json"))
    d = next(x for x in layers if x["layer"] == layer)
    k = pm.KERNEL
    g = pm.tile_grid(d, fold=k["fold"], tile_pool=k["tile_pool"])
    cyc = pm.layer_cycles(d, **pm.TILES, **{x: k[x] for x in ("width", "trim", "requant",
                                                                "fold", "tile_pool")})
    n_tiles = g["n_r"] * g["n_c"] * g["n_m"]
    per = {x: cyc[x] / n_tiles for x in ("load_in", "load_w", "compute", "store")}
    return {"cfg": k, "d": d, "grid": g, "per_tile": per, "cycles": cyc, "tiles": n_tiles}


def plot_engine(e, out_dir, name="moteur"):
    plt = st.plt()
    k = e["cfg"]
    fig = plt.figure(figsize=(st.FULL, 6.6))
    ax = fig.add_axes([0, 0.32, 1, 0.62])
    st.schema_axes(ax, (0, 20), (0, 8))
    c_ld, c_mac, c_out = st.PALETTE[0], st.PALETTE[1], st.PALETTE[2]
    st.block(ax, 0.2, 2.0, 2.2, 4.0, "DDR\nact_in\nwts\nprm", st.PALETTE[6])
    w = k["width"] * 8
    st.block(ax, 3.3, 4.6, 3.6, 1.6, "load_input\nin_buf[Tn][IR][IC]\nping-pong in_a / in_b",
             c_ld)
    st.block(ax, 3.3, 1.8, 3.6, 1.6, "load_weights\nw_buf[Tm][Tn][3][3]\nping-pong w_a / w_b", c_ld)
    st.arrow(ax, (2.4, 5.0), (3.3, 5.4), f"mots {w} bits", fontsize=6.5)
    st.arrow(ax, (2.4, 3.0), (3.3, 2.6), f"mots {w} bits", fontsize=6.5)
    # réseau de MACs
    st.block(ax, 8.0, 1.4, 4.6, 5.2, "", c_mac, fill=0.06)
    tm, tn = k["tm"], k["tn"]
    xs, ys = np.meshgrid(np.linspace(8.3, 12.3, tn), np.linspace(1.9, 5.7, tm))
    ax.plot(xs.ravel(), ys.ravel(), "s", ms=1.6, color=c_mac)
    ax.text(10.3, 6.25, f"compute : Tm × Tn = {tm} × {tn} = {tm * tn} MAC int8 / cycle",
            ha="center", fontsize=7.5, weight="bold")
    st.arrow(ax, (6.9, 5.4), (8.0, 4.6), f"Tn = {tn} voies")
    st.arrow(ax, (6.9, 2.6), (8.0, 3.2), "poids")
    st.block(ax, 13.4, 3.0, 1.9, 2.0, "accumulateur\nint32\nout_a / out_b", c_mac)
    st.arrow(ax, (12.6, 4.0), (13.4, 4.0))
    st.block(ax, 16.0, 2.3, 3.8, 3.4, f"store_tile\n(output_stage.hpp)\nrequantification "
             f"×{k['requant']}\nleaky entière, clip\nmaxpool fusionné\nwrite_row → act_out", c_out)
    st.arrow(ax, (15.3, 4.0), (16.0, 4.0), f"Tm = {tm}")
    ax.text(10, 0.6, f"Tuile Tr × Tc = {k['tr']} × {k['tc']} (maxpool : {k['tile_pool']} × "
            f"{k['tile_pool']}) ; trim = {k['trim']}, pliage L00 = {k['fold']} — "
            "constantes de hls/kernels/accel_config.hpp", ha="center", fontsize=7.5, color=st.INK2)
    ax.set_title("Moteur unique : chargeurs ping-pong, réseau de MACs, étage de sortie")
    # chronogramme
    ax2 = fig.add_axes([0.08, 0.06, 0.88, 0.22])
    p = e["per_tile"]
    ld = max(p["load_in"], p["load_w"])
    comp, store = p["compute"], p["store"]
    rows = {"chargement": 2, "calcul": 1, "stockage": 0}
    colors = {"chargement": c_ld, "calcul": c_mac, "stockage": c_out}
    # Deux tampons par entrée : la tuile i se charge quand la tuile i − 2 a libéré le sien.
    ld_end, cp_end, sto_end = [], [], []
    for i in range(3):
        a = max(ld_end[-1] if ld_end else 0.0, cp_end[i - 2] if i >= 2 else 0.0)
        b = max(a + ld, cp_end[-1] if cp_end else 0.0)
        c = max(b + comp, sto_end[-1] if sto_end else 0.0)
        ld_end.append(a + ld)
        cp_end.append(b + comp)
        sto_end.append(c + store)
        for x, w_, r in ((a, ld, "chargement"), (b, comp, "calcul"), (c, store, "stockage")):
            ax2.broken_barh([(x, w_)], (rows[r] - 0.35, 0.7), fc=colors[r], ec="white", lw=1.5,
                            alpha=0.9, hatch="//" if r == "stockage" else "")
            ax2.text(x + w_ / 2, rows[r], f"t{i}", ha="center", va="center", fontsize=7,
                     color="white")
    ax2.set_yticks(list(rows.values()), list(rows))
    ax2.set_xlabel("cycles (moyenne par tuile, perf_model.layer_cycles)")
    d = e["d"]
    ax2.set_title(f"Recouvrement sur 3 tuiles, L{d['layer']:02d} ({d['cin']} → {d['cout']}, "
                  f"{d['h']}×{d['w']}) : chargement {ld:.0f}, calcul {comp:.0f}, stockage "
                  f"{store:.0f} cycles", fontsize=8.5)
    ax2.grid(axis="y", visible=False)
    return st.save(fig, out_dir, name)


@figure("moteur", "materiel", "T13.9",
        "Moteur unique : chargeurs ping-pong, réseau Tm × Tn de MACs, accumulateur, étage de "
        "sortie, et chronogramme du recouvrement sur 3 tuiles.",
        "hls/kernels/accel_config.hpp, conv_pe.cpp, output_stage.hpp, tools/perf_model.py")
def moteur(out_dir):
    return plot_engine(load_engine(), out_dir)


# --- T13.10 : tuilage ---------------------------------------------------------------------

TILING_LAYERS = (0, 13)  # L00 : cin = 3 ; L13 : 13×13, une seule tuile spatiale


def load_tiling(net=NET, layers=TILING_LAYERS):
    """[(d, grille sans pliage, grille du noyau actuel)] pour les couches demandées."""
    pm = _tools("perf_model")
    convs = {d["layer"]: d for d in pm.conv_layers(need(ROOT / "model" / net / "manifest.json"))}
    k = pm.KERNEL
    out = []
    for i in layers:
        d = convs[i]
        out.append((d, pm.tile_grid(d), pm.tile_grid(d, fold=k["fold"], tile_pool=k["tile_pool"])))
    return out


def plot_tiling(rows, tn, tm, out_dir, name="tuilage"):
    from matplotlib.patches import Rectangle

    plt = st.plt()
    fig, axes = plt.subplots(len(rows), 2, figsize=(st.FULL, 4.0 * len(rows)),
                             gridspec_kw={"width_ratios": [1, 1.3]})
    for r, (d, g0, g) in enumerate(rows):
        ax = axes[r, 0]
        R, C = g["Rp"], g["Cp"]
        ax.add_patch(Rectangle((0, 0), C, R, fc="none", ec=st.INK2, lw=1.2))
        for i in range(g["n_r"]):
            for j in range(g["n_c"]):
                h = min(g["Pr"], R - i * g["Pr"])
                w = min(g["Pc"], C - j * g["Pc"])
                ax.add_patch(Rectangle((j * g["Pc"], i * g["Pr"]), w, h,
                                       fc=st.PALETTE[(i + j) % 2], alpha=0.25, ec=st.PALETTE[0],
                                       lw=0.6))
        ax.set_xlim(-0.02 * C, C * 1.02)
        ax.set_ylim(R * 1.02, -0.02 * R)
        ax.set_aspect("equal")
        ax.grid(False)
        pool = f", maxpool {g['pk']}/{g['ps']} fusionné" if d["pool_k"] else ""
        ax.set_title(f"L{d['layer']:02d} : sortie {R}×{C}{pool}\n{g['n_r']} × {g['n_c']} tuiles "
                     f"spatiales ({g['tr']}×{g['tc']} en entrée du pooling)", fontsize=8.5)
        ax.set_xlabel("colonnes")
        ax.set_ylabel("lignes")
        # voies
        ax = axes[r, 1]
        cases = [("sans pliage ni trim", g0, False), ("noyau actuel", g, True)]
        for y, (lab, gg, trim) in enumerate(cases):
            lanes = gg["lanes"]
            for t in range(gg["nti"]):
                used = min(tn, lanes - t * tn)
                x0 = t * tn
                ax.barh(y, used, left=x0, height=0.55, color=st.PALETTE[0], edgecolor="white")
                if used < tn:
                    ax.barh(y, tn - used, left=x0 + used, height=0.55, fc="white",
                            ec=st.PALETTE[7], hatch="///" if not trim else "",
                            ls="-" if not trim else ":", lw=0.8)
            busy = lanes / (gg["nti"] * tn)
            note = "; pliage : cin·k voies" if gg["folded"] else ""
            ax.text(gg["nti"] * tn + 1, y, f"{lab} : {lanes} voies utiles / {gg['nti']} × Tn "
                    f"= {100 * busy:.0f} %{note}", va="center", fontsize=7.5)
        ax.set_yticks([])
        ax.set_xlim(0, max(gg["nti"] for _, gg, _ in cases) * tn * 2.4 + 10)
        ax.set_xlabel(f"voies d'entrée (Tn = {tn}) ; hachuré : voies inutiles (sans trim)")
        ax.set_title(f"cin = {d['cin']}, cout = {d['cout']} → {g['n_m']} groupe(s) de Tm = {tm}"
                     f" ; {g['n_r'] * g['n_c'] * g['n_m'] * g['nti']} tuiles au total", fontsize=8.5)
        ax.grid(False)
        for s in ("left", "top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle("Tuilage du moteur unique (tools/perf_model.py:tile_grid)", fontsize=9.5)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("tuilage", "materiel", "T13.10",
        "Tuilage de deux couches : tuiles spatiales et voies d'entrée, avec le pliage de L00 "
        "(cin = 3) et la couche 13×13 couverte par une seule tuile.",
        "model/<net>/manifest.json, tools/perf_model.py:tile_grid, accel_config.hpp")
def tuilage(out_dir):
    pm = _tools("perf_model")
    return plot_tiling(load_tiling(), pm.TILES["tn"], pm.TILES["tm"], out_dir)


# --- T13.11 : streaming -------------------------------------------------------------------

def load_streaming(net=NET, board="kv260"):
    """Plan streaming (`stream_model.plan`) et cycles par couche du moteur unique."""
    sm, pm = _tools("stream_model"), _tools("perf_model")
    layers = pm.conv_layers(need(ROOT / "model" / net / "manifest.json"))
    b = sm.load_board(need(ROOT / "hw" / "boards" / f"{board}.yaml"))
    k = pm.KERNEL
    engine = [pm.layer_cycles(d, **pm.TILES, **{x: k[x] for x in ("width", "trim", "requant",
                                                                    "fold", "tile_pool")})
              ["overlapped"] for d in layers]
    return sm.plan(layers, b), engine


def plot_streaming(p, engine, out_dir, name="streaming"):
    plt = st.plt()
    rows = p["stages"]
    f = p["freq_hz"]
    fig, axes = plt.subplots(3, 1, figsize=(st.FULL, 8.4),
                             gridspec_kw={"height_ratios": [1.1, 1, 1.2]})
    x = np.arange(len(rows))
    labels = [f"L{r['layer']:02d}\n{r['pe']}×{r['simd']}" for r in rows]
    ax = axes[0]
    buf = np.array([r["buffer_bytes"] for r in rows]) / 1024
    wts = np.array([r["weight_bytes"] if r["weights"] == "puce" else 0 for r in rows]) / 1024
    ax.bar(x, buf, 0.6, color=st.PALETTE[0], hatch="///", ec="white",
           label="line buffer ou carte (Ko)")
    ax.bar(x, wts, 0.6, bottom=buf, color=st.PALETTE[3], hatch="///", ec="white",
           label="poids sur la puce (Ko)")
    for i, r in enumerate(rows):
        if r["weights"] == "ddr":
            ax.text(i, buf[i] * 1.15 + 1, "poids\nDDR", ha="center", fontsize=6.5, color=st.INK2)
    ax.set_yscale("log")
    ax.set_ylabel("mémoire par étage (Ko)")
    ax.set_xticks(x, labels)
    ax.legend(loc="upper left")
    ax.set_title(f"Streaming {p['board']} : un étage par conv (PE × SIMD), estimation "
                 f"stream_model — {p['dsp']} DSP / {p['dsp_budget']}, "
                 f"{p['onchip_bytes'] / 2**20:.2f} Mo sur puce")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.bar(x - 0.2, np.array([r["cycles"] for r in rows]) / 1e6, 0.4, color=st.PALETTE[1],
           hatch="///", ec="white", label="streaming : cycles de l'étage par image")
    ax.bar(x + 0.2, np.array(engine) / 1e6, 0.4, color=st.PALETTE[0], hatch="///", ec="white",
           label="moteur unique : cycles de la couche")
    ax.axhline(p["ii_cycles"] / 1e6, color=st.PALETTE[7], lw=1.2, ls="--")
    ax.text(len(rows) - 0.5, p["ii_cycles"] / 1e6 * 1.05, f"II = {p['ii_cycles'] / 1e6:.2f} M",
            ha="right", fontsize=7.5, color=st.PALETTE[7])
    ax.set_ylabel("Mcycles")
    ax.set_xticks(x, [f"L{r['layer']:02d}" for r in rows])
    ax.legend(loc="upper right")
    ax.grid(axis="x", visible=False)
    # chronogramme sur deux images
    ax = axes[2]
    t = 0.0
    tot = sum(engine)
    for img in range(2):
        for i, c in enumerate(engine):
            ax.broken_barh([(t / f * 1e3, c / f * 1e3)], (1.65, 0.6),
                           fc=st.PALETTE[i % 2 * 6], alpha=0.8, hatch="///")
            t += c
    start = 0.0
    for img in range(int(2 * tot / p["ii_cycles"]) + 1):
        if start / f * 1e3 > 2 * tot / f * 1e3:
            break
        for i, r in enumerate(rows):
            fill = sum(rr["delay"] for rr in rows[:i])
            ax.broken_barh([((start + fill) / f * 1e3, r["cycles"] / f * 1e3)],
                           (0.1 + 1.3 * i / len(rows), 1.2 / len(rows)),
                           fc=st.PALETTE[1 + img % 2 * 2], alpha=0.75, hatch="///")
        start += p["ii_cycles"]
    ax.set_yticks([0.75, 1.95], ["streaming\n(étages en parallèle)", "moteur unique\n"
                                 "(couche après couche)"])
    ax.set_xlim(0, 2 * tot / f * 1e3)
    ax.set_xlabel(f"temps (ms à {f / 1e6:.0f} MHz)")
    ax.set_title(f"Deux images : moteur unique {f / tot:.1f} img/s, streaming {p['fps']:.1f} "
                 f"img/s (latence ≈ {p['latency_ms']:.1f} ms) — estimations, hachuré", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("streaming", "materiel", "T13.11",
        "Architecture streaming face au moteur unique : mémoire et parallélisme par étage, "
        "cycles par couche, chronogramme sur deux images (estimations).",
        "tools/stream_model.py:plan, tools/perf_model.py, hw/boards/kv260.yaml")
def streaming(out_dir):
    return plot_streaming(*load_streaming(), out_dir)


# --- T13.12 : chaîne de vérification ------------------------------------------------------

CHAIN = [  # (stade, jalon qui vérifie le passage vers ce stade, critère)
    ("NumPy flottant\n(BN fusionnée)", None, ""),
    ("entier Python\n(IntNetwork)", "M4", "mAP {flottant} → {entier}"),
    ("golden C++", "M5", "0 écart à l'octet, {dumps} dumps"),
    ("C-sim du noyau", "M6", "0 écart par couche (tb_net)"),
    ("driver ARM\nbackend sim", "M7", "{csim_eq}"),
    ("carte KV260\n(backend uio)", "M8", "mAP {carte}"),
]
STATE_COLORS = {"vérifié": st.PALETTE[5], "vérifié sur PC": st.PALETTE[2], "à faire": st.GREY}


def chain_states(tracking):
    """[état] de chaque passage, lu dans le suivi du README (jamais codé en dur)."""
    from tools.figures.projet import PC_WORDS

    out = []
    for _, m, _ in CHAIN[1:]:
        n, done, note = tracking.get(m, (1, 0, ""))
        if done == n:
            out.append("vérifié")
        elif any(w in note for w in PC_WORDS) and "carte" not in m and m != "M8":
            out.append("vérifié sur PC")
        else:
            out.append("à faire")
    return out


def chain_facts(net=NET):
    st_ = {}
    p = ROOT / "build" / "m8" / net / "map_stades.json"
    if p.exists():
        st_ = {s["key"]: s for s in json.loads(p.read_text())["stages"]}

    def fmt(k):
        m = st_.get(k, {}).get("map")
        return f"{100 * m:.2f}" if m is not None else "à mesurer"

    dumps = len([d for d in (ROOT / "model" / net / "dumps").glob("*") if d.is_dir()])
    return {"flottant": fmt("flottant"), "entier": fmt("entier"), "carte": fmt("carte"),
            "dumps": dumps, "csim_eq": st_.get("csim", {}).get("eq", "détections identiques")}


def plot_chain(states, facts, out_dir, name="chaine_verif"):
    plt = st.plt()
    n = len(CHAIN)
    fig, ax = plt.subplots(figsize=(st.FULL, 2.9))
    st.schema_axes(ax, (0, 4 * n - 0.6), (0, 3.2))
    for i, (stage, _, _) in enumerate(CHAIN):
        last = i == n - 1
        color = STATE_COLORS[states[i - 1]] if i else st.PALETTE[0]
        st.block(ax, 4 * i, 1.2, 3.0, 1.3, stage, color, fill=0.18 if not last else 0.08,
                 ls="--" if states[i - 1 if i else 0] == "à faire" and i else "-", fontsize=8)
        if i:
            state = states[i - 1]
            crit = textwrap.fill(CHAIN[i][2].format(**facts), 20)
            st.arrow(ax, (4 * i - 1.0, 1.85), (4 * i, 1.85), color=STATE_COLORS[state], lw=2,
                     ls="--" if state == "à faire" else "-")
            ax.text(4 * i - 0.5, 0.95, crit, ha="center", va="top", fontsize=6.6, color=st.INK2,
                    wrap=True)
            ax.text(4 * i - 0.5, 2.75, f"{state}\n({CHAIN[i][1]})", ha="center", fontsize=6.8,
                    color=STATE_COLORS[state], weight="bold")
    ax.set_title("Chaîne de vérification bit-exact : chaque stade est comparé au précédent "
                 "(états lus dans le suivi de docs/tasks/README.md)", fontsize=9)
    return st.save(fig, out_dir, name)


@figure("chaine_verif", "materiel", "T13.12",
        "Chaîne de vérification bit-exact, du flottant NumPy à la carte : critère et état de "
        "chaque passage.",
        "tableau de suivi de docs/tasks/README.md, build/m8/<net>/map_stades.json, dumps")
def chaine_verif(out_dir):
    from tools.figures.projet import load_tracking

    return plot_chain(chain_states(load_tracking()), chain_facts(), out_dir)
