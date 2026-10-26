"""Figures de résultats (M13, section D) : mAP, quantification, cycles, état de l'art.

Chaque figure sépare la lecture de la source (`load_*`, données pures, testées contre la
source) et le tracé (`plot_*`). Les valeurs ne sont jamais recopiées à la main.
"""

import csv
import json
import math
import re
from pathlib import Path

import numpy as np

from tools.figures import ROOT, MissingSource, figure, need
from tools.figures import style as st

NET = "tiny-yolov2-voc"
VOC_IMAGES = 4952  # VOC2007 test : seule base des mAP publiées dans results/ (règle M12)
DEVKIT = ROOT / "data" / "VOCdevkit"


def _read_json(path):
    return json.loads(Path(path).read_text())


def _tools(name):
    """Module de tools/ (perf_model, roofline…), importé à la demande."""
    import importlib

    return importlib.import_module(f"tools.{name}")


# --- T13.13 : mAP aux stades -------------------------------------------------------------

STAGES = ("flottant", "entier", "csim", "carte")


def load_map_stades(path):
    """JSON de `tools/map_stades.py --json` : {classes, stages: [{key, name, map, aps, eq}]}."""
    d = _read_json(path)
    if d["images"] != VOC_IMAGES:
        raise MissingSource(f"{path} : {d['images']} images, pas {VOC_IMAGES}")
    return d


def plot_map_stades(d, out_dir, name="map_stades"):
    plt = st.plt()
    classes = d["classes"]
    done = [s for s in d["stages"] if s["map"] is not None]
    pending = [s for s in d["stages"] if s["map"] is None]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(st.FULL, 7.2),
                                  gridspec_kw={"height_ratios": [3, 2]})
    x = np.arange(len(classes))
    w = 0.8 / len(done)
    for k, s in enumerate(done):
        ax.bar(x + (k - (len(done) - 1) / 2) * w, np.array(s["aps"]) * 100, w * 0.92,
               color=st.STAGE_COLORS[s["key"]], hatch=st.STAGE_HATCH[s["key"]],
               edgecolor="white", linewidth=0.4,
               label=f"{s['name']} : mAP {s['map'] * 100:.2f}")
    from matplotlib.patches import Patch

    handles = ax.get_legend_handles_labels()[0]
    handles += [Patch(fc="white", ec=st.STAGE_COLORS[s["key"]], hatch=st.STAGE_HATCH[s["key"]],
                      label=f"{s['name']} : à mesurer") for s in pending]
    ax.set_xticks(x, classes, rotation=40, ha="right")
    ax.set_ylabel("AP (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"{d['net']} — AP par classe aux stades (VOC2007 test, {d['images']} images)")
    ax.legend(handles=handles, loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)

    ref = next(s for s in done if s["key"] == "flottant")
    ent = next(s for s in done if s["key"] == "entier")
    diff = (np.array(ent["aps"]) - np.array(ref["aps"])) * 100
    order = np.argsort(diff)
    ax2.bar(np.arange(len(classes)), diff[order],
            color=[st.PALETTE[7] if v < 0 else st.PALETTE[2] for v in diff[order]], width=0.7)
    ax2.axhline(0, color=st.INK2, lw=0.8)
    ax2.axhline(diff.mean(), color=st.INK2, lw=0.8, ls="--")
    ax2.text(len(classes) - 0.5, diff.mean(), f" moyenne {diff.mean():+.2f}", va="bottom",
             ha="right", fontsize=7, color=st.INK2)
    ax2.set_xticks(np.arange(len(classes)), [classes[i] for i in order], rotation=40,
                   ha="right")
    ax2.set_ylabel("Δ AP entier − flottant (points)")
    ax2.set_title("Effet de la quantification INT8 par classe (trié)")
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("map_stades", "resultats", "T13.13",
        "AP par classe aux stades flottant, entier et C-sim (la carte quand elle sera mesurée), "
        "et écart entier − flottant trié.",
        "build/m8/<net>/map_stades.json (tools/map_stades.py --json)")
def map_stades(out_dir):
    return plot_map_stades(load_map_stades(need(ROOT / "build" / "m8" / NET / "map_stades.json")),
                           out_dir)


# --- T13.14 : mAP selon le format numérique ------------------------------------------------

# (étiquette, fichier, clé de résultat, bits de poids, remarque)
FORMATS = [
    ("flottant", "build/m8/{net}/eval_float_int.json", "float", 32, ""),
    ("INT8 PTQ", "build/m8/{net}/eval_float_int.json", "int", 8, ""),
    ("uniform6 PTQ", "build/m9/map_{net}-uniform6-ptq.json", "int", 6, ""),
    ("pow2 mixed6 PTQ", "build/m9/map_{net}-pow2-ptq.json", "int", 6, ""),
    ("pow2 mixed6 ADMM", "build/m9/map_{net}-pow2-admm.json", "int", 6, "non convergé"),
    ("w4a4 PTQ", "build/m9/map_{net}-w4a4-ptq.json", "int", 4, ""),
    ("w4a4 QAT", "build/m9/map_{net}-w4a4-qat.json", "int", 4, "QAT court (600 it.)"),
]


def load_map_formats(net=NET, root=None):
    """[(étiquette, mAP, bits de poids, remarque)] des variantes évaluées sur 4 952 images."""
    root = root or ROOT
    rows = []
    for label, rel, key, bits, remark in FORMATS:
        p = Path(root) / rel.format(net=net)
        if not p.exists():
            continue
        d = _read_json(p)
        if d["images"] != VOC_IMAGES or key not in d["results"]:
            continue
        rows.append((label, d["results"][key]["map"], bits, remark))
    if not rows:
        raise MissingSource("aucune mAP 4 952 images dans build/m8 ni build/m9")
    return rows


def pareto(points):
    """Indices de la frontière (moins de bits, plus de mAP) : rend les non dominés."""
    keep = []
    for i, (b, m) in enumerate(points):
        if not any((b2 <= b and m2 > m) or (b2 < b and m2 >= m) for b2, m2 in points):
            keep.append(i)
    return keep


def plot_map_formats(rows, out_dir, name="map_formats"):
    plt = st.plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(st.FULL, 4.2),
                                  gridspec_kw={"width_ratios": [3, 2]})
    y = np.arange(len(rows))[::-1]
    for yi, (label, m, bits, remark) in zip(y, rows):
        weak = bool(remark)
        ax.barh(yi, m * 100, 0.62, color="white" if weak else st.PALETTE[0],
                edgecolor=st.PALETTE[0], hatch="///" if weak else "", linewidth=1)
        ax.text(m * 100 + 0.8, yi, f"{m * 100:.2f}" + (f"  ({remark})" if remark else ""),
                va="center", fontsize=8, color=st.INK)
    ax.set_yticks(y, [f"{r[0]}  [{r[2]} b]" for r in rows])
    ax.set_xlim(0, 75)
    ax.set_xlabel("mAP VOC2007 test (%)")
    ax.set_title("mAP par format numérique (4 952 images)")
    ax.grid(axis="y", visible=False)

    pts = [(r[2], r[1] * 100) for r in rows]
    front = sorted(pareto(pts), key=lambda i: pts[i][0])
    for k, ((b, m), r) in enumerate(zip(pts, rows)):
        ax2.plot(b, m, "o", ms=8, mfc="white" if r[3] else st.PALETTE[0],
                 mec=st.PALETTE[0], mew=1.4)
        close = any(abs(m - m2) < 2.5 and abs(b - b2) < 3 for j, (b2, m2) in enumerate(pts)
                    if j < k)
        ax2.annotate(r[0], (b, m), textcoords="offset points", fontsize=7, color=st.INK2,
                     xytext=(6, -11 if close else 3))
    ax2.plot([pts[i][0] for i in front], [pts[i][1] for i in front], color=st.PALETTE[1],
             lw=1.2, ls="--", label="frontière de Pareto")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks([4, 6, 8, 32], ["4", "6", "8", "32"])
    ax2.set_xlabel("bits par poids (coût mémoire et multiplieur)")
    ax2.set_ylabel("mAP (%)")
    ax2.set_title("Précision face au coût")
    ax2.plot([], [], "o", mfc="white", mec=st.PALETTE[0], label="non convergé / QAT court")
    ax2.legend(loc="lower right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("map_formats", "resultats", "T13.14",
        "mAP de Tiny-YOLOv2 selon le format des poids (flottant, INT8, puissances de 2, "
        "4 bits), et frontière précision / bits.",
        "build/m8/<net>/eval_float_int.json, build/m9/map_*.json")
def map_formats(out_dir):
    return plot_map_formats(load_map_formats(), out_dir)


# --- T13.15 : courbes précision-rappel ----------------------------------------------------

def load_pr(det_dir, samples, names=None):
    """{classe: (rappel, précision, AP VOC07)} des fichiers comp4 de `det_dir`."""
    from yolo.data.voc import VOC_CLASSES
    from yolo.infer.metrics import read_detections

    names = list(names or VOC_CLASSES)
    return pr_curves(read_detections(names, det_dir), samples, names)


def pr_curves(dets, samples, names):
    """{classe: (rappel, précision, AP VOC07)} par `yolo.infer.metrics.eval_class` ;
    `dets` : {c: (ids, scores, coins pixels)}, images hors de `samples` ignorées."""
    from yolo.infer.metrics import class_gts, eval_class, ignore_regions

    ids = {s["id"] for s in samples}
    out = {}
    ignore = ignore_regions(samples)
    for c, cls in enumerate(names):
        img, sc, bx = dets.get(c, ([], np.zeros(0), np.zeros((0, 4))))
        sc, bx = np.asarray(sc), np.asarray(bx).reshape(-1, 4)
        keep = [i for i, k in enumerate(img) if k in ids]
        rec, prec, ap = eval_class([img[i] for i in keep], sc[keep], bx[keep],
                                   class_gts(samples, c), use_07=True, ignore=ignore)
        out[cls] = (rec, prec, ap)
    return out


def voc07_points(rec, prec):
    """Les 11 points VOC07 : (r, max précision à rappel ≥ r)."""
    r = np.linspace(0, 1, 11)
    p = [prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in r]
    return r, np.array(p)


def plot_pr(curves, out_dir, name, title):
    """`curves` : {variante: {classe: (rec, prec, ap)}} ; grille 4 × 5, une courbe par variante."""
    plt = st.plt()
    variants = list(curves)
    classes = list(next(iter(curves.values())))
    fig, axes = plt.subplots(4, 5, figsize=(st.FULL, 8.4), sharex=True, sharey=True)
    for ax, cls in zip(axes.flat, classes):
        for k, v in enumerate(variants):
            rec, prec, ap = curves[v][cls]
            color = st.PALETTE[k]
            idx = np.unique(np.linspace(0, len(rec) - 1, 400).astype(int)) if len(rec) else []
            ax.plot(rec[idx], prec[idx], color=color, lw=1.1, label=v)  # tracé seul allégé
            if len(variants) == 1:
                r, p = voc07_points(rec, prec)
                ax.step(r, p, where="post", color=st.INK2, lw=0.8, ls="--")
                ax.plot(r, p, "o", ms=2.5, color=st.INK2)
        aps = "\n".join(f"{curves[v][cls][2] * 100:.1f}" for v in variants)
        ax.set_title(cls, fontsize=8)
        ax.text(0.03, 0.04, f"AP\n{aps}" if len(variants) == 1 else aps,
                transform=ax.transAxes, fontsize=6.5, color=st.INK2, va="bottom")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
    for ax in list(axes.flat)[len(classes):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("rappel")
    for ax in axes[:, 0]:
        ax.set_ylabel("précision")
    maps = ", ".join(f"{v} : mAP {np.mean([c[2] for c in curves[v].values()]) * 100:.2f}"
                     for v in variants)
    fig.suptitle(f"{title}\n{maps}", fontsize=9)
    if len(variants) > 1:
        fig.legend(*axes.flat[0].get_legend_handles_labels(), loc="lower center", ncol=4)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    return st.save(fig, out_dir, name)


def eval_dirs(root=None):
    return sorted(p for p in (Path(root or ROOT) / "build" / "eval").glob(f"{NET}-*")
                  if any(p.glob("comp4_det_test_*.txt")))


def test_samples(devkit=None):
    from yolo.data.voc import load_split

    devkit = devkit or DEVKIT
    need(Path(devkit) / "VOC2007" / "ImageSets" / "Main" / "test.txt")
    return load_split(devkit, 2007, "test")


@figure("pr", "resultats", "T13.15",
        "Courbes précision-rappel par classe (11 points VOC07) pour chaque prétraitement "
        "évalué, puis les quatre superposées.",
        "build/eval/<net>-<variante>/comp4_det_test_*.txt + data/VOCdevkit")
def pr(out_dir):
    dirs = eval_dirs()
    if not dirs:
        raise MissingSource("aucune détection dans build/eval/ (make eval-float)")
    samples = test_samples()
    curves = {d.name.removeprefix(NET + "-"): load_pr(d, samples) for d in dirs}
    paths = []
    for v, c in curves.items():
        paths += plot_pr({v: c}, out_dir, f"pr_{v}", f"{NET}, {v} — VOC2007 test")
    if len(curves) > 1:
        paths += plot_pr(curves, out_dir, "pr_variantes",
                         f"{NET} — prétraitements comparés (VOC2007 test)")
    return paths


# --- T13.16 : sensibilité par couche ------------------------------------------------------

def load_sensitivity(path):
    """sensitivity.csv de M10 : {format: {couche: perte de mAP (points)}}, et la référence."""
    out, ref = {}, None
    for r in csv.DictReader(Path(path).open()):
        if r["layer"] == "-":
            ref = float(r["map"])
            continue
        out.setdefault(r["kind"], {})[int(r["layer"])] = float(r["loss"])
    return ref, out


def load_fq_sensitivity(path):
    """eval_quant fq:<i> : {couche: perte de mAP (points)} quand cette couche seule est
    quantifiée, face au flottant ; plus (mAP flottante, mAP tout quantifié, images)."""
    d = _read_json(path)
    res = {k: v["map"] * 100 for k, v in d["results"].items()}
    per = {int(k[3:]): res["float"] - v for k, v in res.items()
           if k.startswith("fq:") and k != "fq:all"}
    return res["float"], res.get("fq:all"), d["images"], per


def plot_sensitivity(panels, out_dir, name="sensibilite"):
    """`panels` : [(titre, {format: {couche: perte}})]."""
    plt = st.plt()
    fig, axes = plt.subplots(len(panels), 1, figsize=(st.FULL, 3.0 * len(panels)),
                             squeeze=False)
    for ax, (title, series) in zip(axes[:, 0], panels):
        layers = sorted({i for s in series.values() for i in s})
        x = np.arange(len(layers))
        w = 0.8 / len(series)
        for k, (fmt, s) in enumerate(series.items()):
            ax.bar(x + (k - (len(series) - 1) / 2) * w, [s.get(i, np.nan) for i in layers],
                   w * 0.92, color=st.PALETTE[k], hatch=("", "//", "..", "xx")[k % 4],
                   edgecolor="white", linewidth=0.4, label=fmt)
        ax.axhline(0, color=st.INK2, lw=0.8)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi + 0.25 * (hi - lo))
        ax.set_xticks(x, [f"L{i:02d}" for i in layers])
        ax.set_ylabel("perte de mAP (points)")
        ax.set_title(title)
        ax.legend(loc="upper right", ncol=len(series))
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("sensibilite", "resultats", "T13.16",
        "Perte de mAP quand une seule couche est quantifiée (INT8 en fake-quant face au "
        "flottant ; uniform6, uniform4, mixed6 face à l'INT8). Sous-ensembles de VOC2007 test.",
        "build/quant/<net>/eval_test_*_fq*.json, build/m10/mixed/sensitivity.csv",
        subset=True)
def sensibilite(out_dir):
    panels = []
    fq = sorted((ROOT / "build" / "quant" / NET).glob("eval_test_*_float-fqall-*.json"))
    if fq:
        f, fall, n, per = load_fq_sensitivity(fq[-1])
        panels.append((f"INT8 seule couche face au flottant ({n} images, flottant {f:.2f}, "
                       f"tout INT8 {fall:.2f})", {"INT8 (fake-quant)": per}))
    sens = ROOT / "build" / "m10" / "mixed" / "sensitivity.csv"
    if sens.exists():
        ref, per = load_sensitivity(sens)
        panels.append((f"Poids basse précision, une couche à la fois, face à l'INT8 "
                       f"(500 images, INT8 {ref:.2f})", per))
    if not panels:
        raise MissingSource("ni eval_test_*_fq*.json ni build/m10/mixed/sensitivity.csv")
    return plot_sensitivity(panels, out_dir)


# --- T13.17 : calibration ------------------------------------------------------------------

def load_calib(path):
    """calib.json : [{id, act, candidates, mse, pick, scale, clip_rate}] par couche."""
    d = _read_json(path)
    return d["network"], d["layers"]


def plot_calibration(net, layers, samples, out_dir, name):
    """Histogramme log des activations par couche, seuil retenu (127 · s) et candidats."""
    plt = st.plt()
    n = len(layers)
    cols = 4
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(st.FULL, 2.1 * rows), squeeze=False)
    for ax, layer in zip(axes.flat, layers):
        clip = 127 * layer["scale"]
        v = samples.get(layer["id"]) if samples else None
        if v is not None:
            v = np.abs(v)
            hi = max(float(v.max()), clip) * 1.05
            ax.hist(v, bins=120, range=(0, hi), color=st.PALETTE[0], log=True, rasterized=True)
            sat = float((v > clip).mean())
        else:
            hi = max(layer["candidates"].values()) * 1.05
            sat = layer["clip_rate"]
        for cname, c in layer["candidates"].items():
            ax.axvline(c, color=st.GREY, lw=0.7, ls=":")
        ax.axvline(clip, color=st.PALETTE[7], lw=1.4)
        ax.set_xlim(0, hi)
        ax.set_title(f"L{layer['id']:02d} {layer['act']} — {layer['pick']}, "
                     f"saturé {sat * 100:.3f} %", fontsize=7.5, fontweight="normal")
        ax.tick_params(labelsize=6.5)
    for ax in list(axes.flat)[n:]:
        ax.set_visible(False)
    fig.suptitle(f"{net} — |activations| (log) et seuil de saturation retenu 127·s (rouge) ; "
                 "candidats en pointillé", fontsize=9)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def weight_scales_by_layer(net):
    """{id conv: échelles par canal max|W'_f|/127} du réseau à BN fusionnée (poids Darknet)."""
    from yolo.io.darknet_weights import load_darknet_weights
    from yolo.models.tiny_yolo import PRETRAINED, build
    from yolo.quant.fuse_bn import fuse_network
    from yolo.quant.quantize import weight_scales

    w = need(ROOT / "weights" / PRETRAINED[net])
    model = build(net)
    load_darknet_weights(model, w)
    fused = fuse_network(model)
    return {i: np.asarray(weight_scales(fused.params[i]["W"])).ravel()
            for i, layer in enumerate(fused.layers) if layer["type"] == "conv"}


def plot_weight_scales(net, scales, out_dir, name):
    plt = st.plt()
    ids = sorted(scales)
    fig, ax = plt.subplots(figsize=(st.FULL, 3.4))
    ax.boxplot([scales[i] for i in ids], showfliers=True,
               flierprops=dict(marker=".", ms=2, mec=st.MUTED),
               medianprops=dict(color=st.PALETTE[1]), boxprops=dict(color=st.PALETTE[0]))
    ax.set_xticks(range(1, len(ids) + 1), [f"L{i:02d}" for i in ids])
    ax.set_yscale("log")
    ax.set_ylabel("échelle s_w,f = max|W'_f| / 127")
    ax.set_title(f"{net} — échelles des poids par canal de sortie (BN fusionnée)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("calibration", "resultats", "T13.17",
        "Distribution des activations par couche et seuil de saturation retenu par la "
        "calibration ; échelles des poids par canal.",
        "build/quant/<net>/calib.json, build/m9/stats_<net>.npz, weights/")
def calibration(out_dir):
    paths = []
    for net in st.NETS:
        p = ROOT / "build" / "quant" / net / "calib.json"
        if not p.exists():  # calibration sur un autre jeu (calib-<jeu>.json, T11.1)
            p = next(iter(sorted(p.parent.glob("calib-*.json"))), p)
        if not p.exists():
            continue
        name, layers = load_calib(p)
        stats = ROOT / "build" / "m9" / f"stats_{net}.npz"
        samples = None
        if stats.exists():
            z = np.load(stats)
            samples = {int(k[1:]): z[k] for k in z.files}
        paths += plot_calibration(name, layers, samples, out_dir, f"calibration_{net}")
        try:
            paths += plot_weight_scales(net, weight_scales_by_layer(net), out_dir,
                                        f"echelles_poids_{net}")
        except MissingSource:
            pass
    if not paths:
        raise MissingSource("aucun build/quant/<net>/calib.json (make calibrate)")
    return paths


# --- T13.18 : erreur flottant → entier par couche -----------------------------------------

def layer_scales(manifest):
    """{id: échelle de la sortie entière} ; maxpool, upsample, route héritent de leur source."""
    scales = {}
    for i, layer in enumerate(manifest["layers"]):
        if "out_scale" in layer:
            scales[i] = layer["out_scale"]
        elif layer["type"] == "route":
            scales[i] = scales[layer["from"][0]]
        else:
            scales[i] = scales[i - 1]
    return scales


def snr_db(ref, got):
    err = float(np.sum((ref - got) ** 2))
    return math.inf if err == 0 else 10 * math.log10(float(np.sum(ref ** 2)) / err)


def load_layer_errors(net, images=None):
    """[(image, [(couche, SNR dB, écart max en pas de quantification)])] : sortie flottante du
    réseau à BN fusionnée (float64) face à la sortie entière déquantifiée des dumps."""
    from yolo.io.darknet_weights import load_darknet_weights
    from yolo.models.tiny_yolo import PRETRAINED, build
    from yolo.quant.fuse_bn import fuse_network

    model_dir = need(ROOT / "model" / net)
    manifest = _read_json(need(model_dir / "manifest.json"))
    dumps = sorted(p for p in (model_dir / "dumps").glob("*") if p.is_dir())
    if images:
        dumps = [d for d in dumps if d.name in images]
    if not dumps:
        raise MissingSource(f"model/{net}/dumps/ vide (make export)")
    model = build(net)
    load_darknet_weights(model, need(ROOT / "weights" / PRETRAINED[net]))
    fused = fuse_network(model)
    scales = layer_scales(manifest)
    out = []
    for d in dumps:
        x = np.load(d / "input.npy").astype(np.float64) * manifest["input"]["scale"]
        outs = fused.forward(x, train=False, all_outputs=True)
        rows = []
        for i, f in enumerate(outs):
            p = d / f"L{i:02d}.npy"
            if not p.exists():
                continue
            q = np.load(p).astype(np.float64) * scales[i]
            rows.append((i, snr_db(f, q), float(np.abs(f - q).max() / scales[i])))
        out.append((d.name, rows))
    return out


def load_golden_diffs(net):
    """[(image, [(couche, octets différents)])] de `compare_dumps.compare_layers`, ou None."""
    compare_layers = _tools("compare_dumps").compare_layers
    gold = ROOT / "build" / "golden" / "out" / net
    dumps = ROOT / "model" / net / "dumps"
    if not gold.exists():
        return None
    out = []
    for d in sorted(p for p in dumps.glob("*") if p.is_dir()):
        if (gold / d.name).exists():
            out.append((d.name, [(int(n[1:]), diff) for n, _, diff, _ in
                                 compare_layers(d, gold / d.name)]))
    return out or None


def plot_layer_errors(net, errors, golden, out_dir, name):
    plt = st.plt()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(st.FULL, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1.3]})
    markers = "osD^v"
    for k, (img, rows) in enumerate(errors):
        ids = [r[0] for r in rows]
        ax.plot(ids, [r[1] for r in rows], marker=markers[k % 5], ms=5, lw=1.2,
                color=st.PALETTE[k], label=f"image {img}")
    ax.set_ylabel("SNR flottant / entier (dB)")
    ax.set_title(f"{net} — erreur de quantification par couche (sortie entière déquantifiée "
                 "face au flottant à BN fusionnée)")
    ax.legend(loc="upper right", ncol=len(errors))
    ids = [r[0] for r in errors[0][1]]
    if golden:
        tot = {}
        for _, rows in golden:
            for i, diff in rows:
                tot[i] = tot.get(i, 0) + diff
        vals = [tot.get(i, 0) for i in ids]
        ax2.bar(ids, vals, color=st.PALETTE[2], width=0.6)
        ax2.set_ylim(0, max(1, max(vals)) * 1.2)
        ok = all(v == 0 for v in vals)
        ax2.text(0.5, 0.5, f"{'0 octet différent' if ok else 'ÉCARTS'} sur {len(golden)} "
                 f"images × {len(ids)} couches : entier Python == golden C++",
                 transform=ax2.transAxes, ha="center", va="center", fontsize=9,
                 color=st.PALETTE[5] if ok else st.PALETTE[7], fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "golden C++ non exécuté (make golden-check)",
                 transform=ax2.transAxes, ha="center", va="center", color=st.MUTED)
    ax2.set_ylabel("octets ≠")
    ax2.set_xticks(ids, [f"L{i:02d}" for i in ids], rotation=90)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("erreur_couches", "resultats", "T13.18",
        "SNR par couche entre le flottant et l'entier déquantifié, et écart entier Python ↔ "
        "golden C++ (0 partout : bit-exact).",
        "model/<net>/dumps/, build/golden/out/<net>/, weights/")
def erreur_couches(out_dir):
    paths = []
    for net in st.NETS:
        try:
            errors = load_layer_errors(net)
        except MissingSource:
            continue
        paths += plot_layer_errors(net, errors, load_golden_diffs(net), out_dir,
                                   f"erreur_couches_{net}")
    if not paths:
        raise MissingSource("ni dumps (make export) ni poids Darknet (make get-weights)")
    return paths


# --- T13.20 : cycles par couche -----------------------------------------------------------

CYCLE_PARTS = ("load_in", "load_w", "compute", "store")


def load_cycles(path):
    """cycles_conv.csv : {net: [dict par couche, valeurs entières]}."""
    out = {}
    for r in csv.DictReader(Path(path).open()):
        out.setdefault(r["net"], []).append(
            {k: (v if k == "net" else int(v)) for k, v in r.items()})
    return out


def plot_cycles(per_net, out_dir, name="cycles_couches", title=""):
    plt = st.plt()
    nets = list(per_net)
    fig, axes = plt.subplots(len(nets), 1, figsize=(st.FULL, 3.2 * len(nets)), squeeze=False)
    hatches = ("", "//", "", "..")
    for ax, net in zip(axes[:, 0], nets):
        rows = per_net[net]
        x = np.arange(len(rows))
        bottom = np.zeros(len(rows))
        for k, part in enumerate(CYCLE_PARTS):
            v = np.array([r[part] for r in rows]) / 1e6
            ax.bar(x, v, 0.7, bottom=bottom, color=st.PALETTE[k], hatch=hatches[k],
                   edgecolor="white", linewidth=0.5, label=part)
            bottom += v
        ax.plot(x, [r["overlapped"] / 1e6 for r in rows], "D", ms=6, color=st.INK,
                mfc="white", mew=1.4, label="recouvert (temps réel)", zorder=3)
        ax.set_xticks(x, [f"L{r['layer']:02d}" for r in rows])
        ax.set_ylabel("Mcycles")
        tot = sum(r["overlapped"] for r in rows)
        ax.set_title(f"{st.NET_LABELS.get(net, net)} — {tot / 1e6:.2f} Mcycles recouverts "
                     f"({tot / 200e3:.1f} ms à 200 MHz){title}")
        ax.grid(axis="x", visible=False)
    st.note(axes[0, 0], "projection C-sim (compteurs de count_cycles)", "upper left")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncol=5)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return st.save(fig, out_dir, name)


@figure("cycles_couches", "resultats", "T13.20",
        "Cycles par couche du noyau actuel, décomposés en chargements, calcul et stockage ; "
        "le losange donne le temps réel avec recouvrement.",
        "build/hls/cycles_conv.csv (make hls-cycles)")
def cycles_couches(out_dir):
    return plot_cycles(load_cycles(need(ROOT / "build" / "hls" / "cycles_conv.csv")), out_dir)


# --- T13.21 : cascade des optimisations M10 -----------------------------------------------

def scenario_done(kw, kernel):
    """Une piste est « faite » si chacune de ses options est dans le noyau actuel."""
    return ((kw["width"] in (1, kernel["width"])) and (not kw["trim"] or kernel["trim"])
            and kw["requant"] in (1, kernel["requant"]) and (not kw["fold"] or kernel["fold"])
            and kw["tile_pool"] in (None, kernel["tile_pool"]))


def load_cascade(net=NET):
    """(scénarios [(nom, ms, img/s, faite)], borne calcul M6 en ms, roofline KV260 en ms)."""
    pm = _tools("perf_model")
    need(ROOT / "model" / net / "manifest.json")
    _, rows = pm.scenario_table(net)
    steps = [(name, ms, fps, scenario_done(kw, pm.KERNEL))
             for (name, kw), (_, _, ms, fps, _, _) in zip(pm.SCENARIOS, rows)]
    bound = next(r[2] for r in rows if r[0].startswith("borne calcul M6"))
    return steps, bound, roofline_ms(net)


def roofline_ms(net, board="kv260"):
    """Temps du meilleur point roofline de la carte (results/roofline.md), ou None."""
    rl = _tools("roofline")
    try:
        b = rl.load_board(ROOT / "hw" / "boards" / f"{board}.yaml")
    except (ImportError, FileNotFoundError):
        return None
    from yolo.models.specs import NETWORKS

    layers = rl.conv_layers(NETWORKS["tiny-yolov2-voc"])
    best = rl.explore(layers, b)[0]
    return rl.evaluate(rl.conv_layers(NETWORKS[net if net in NETWORKS else "tiny-yolov2-voc"]),
                       *best["tiles"], b["freq_hz"], b["bw_bytes"])["time_s"] * 1e3


def plot_cascade(steps, bound, roof, out_dir, name="cascade_m10"):
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.FULL, 4.6))
    y = np.arange(len(steps))[::-1]
    ref = steps[0][1]
    for yi, (label, ms, fps, done) in zip(y, steps):
        ax.barh(yi, ms, 0.62, color=st.PALETTE[0] if done else "white",
                edgecolor=st.PALETTE[0], hatch="" if done else "///", linewidth=1)
        gain = "" if ms == ref else f"  (×{ref / ms:.1f} face à M6)"
        ax.text(ms + 2, yi, f"{ms:.1f} ms · {fps:.1f} img/s{gain}", va="center", fontsize=8)
    ax.axvline(bound, color=st.PALETTE[1], lw=1.2, ls="--")
    ax.text(bound, y[0] + 0.7, f" borne calcul M6 {bound:.1f} ms", color=st.INK2, fontsize=7.5)
    if roof:
        ax.axvline(roof, color=st.PALETTE[6], lw=1.2, ls=":")
        ax.text(roof, y[-1] - 0.85, f" roofline KV260 {roof:.1f} ms", color=st.INK2,
                fontsize=7.5)
    ax.set_yticks(y, [s[0] for s in steps])
    ax.set_xlim(0, max(s[1] for s in steps) * 1.35)
    ax.set_ylim(y[-1] - 1.2, y[0] + 1.2)
    ax.set_xlabel("temps de l'accélérateur par image (ms, 200 MHz)")
    ax.set_title("Tiny-YOLOv2 — pistes d'optimisation M10 (modèle de cycles == C-sim)")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    ax.legend(handles=[Patch(fc=st.PALETTE[0], ec=st.PALETTE[0], label="piste faite (noyau)"),
                       Patch(fc="white", ec=st.PALETTE[0], hatch="///",
                             label="piste projetée")], loc="lower right")
    st.note(ax, "projection C-sim, pas une mesure carte", "upper right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("cascade_m10", "resultats", "T13.21",
        "Temps par image de Tiny-YOLOv2 à chaque piste d'optimisation M10, face à la borne "
        "de calcul et au roofline KV260.",
        "tools/perf_model.py (SCENARIOS, scenario_table), tools/roofline.py")
def cascade_m10(out_dir):
    return plot_cascade(*load_cascade(), out_dir)


# --- T13.22 : roofline par couche ---------------------------------------------------------

def load_roofline_layers(net=NET, board="kv260"):
    """(carte, points [(couche, CTC, GOPS avant M10, GOPS noyau actuel)])."""
    pm, rl = _tools("perf_model"), _tools("roofline")
    try:
        b = rl.load_board(need(ROOT / "hw" / "boards" / f"{board}.yaml"))
    except ImportError as e:
        raise MissingSource(f"pyyaml absent ({e})") from e
    layers = pm.conv_layers(need(ROOT / "model" / net / "manifest.json"))
    pts = rl.layer_points(layers, b, before=lambda d: pm.layer_cycles(d)["overlapped"],
                          after=lambda d: pm.kernel_cycles(d)["overlapped"],
                          tiles=tuple(pm.TILES[k] for k in ("tm", "tn", "tr", "tc")))
    return b, pts


def plot_roofline_layers(board, pts, net, out_dir, name):
    plt = st.plt()
    tm, tn = _tools("perf_model").TILES["tm"], _tools("perf_model").TILES["tn"]
    peak = 2 * tm * tn * board["freq_hz"] / 1e9
    fig, ax = plt.subplots(figsize=(st.COL * 1.3, 4.8))
    ctc = np.array([p[1] for p in pts])
    x = np.logspace(np.log10(ctc.min() / 3), np.log10(ctc.max() * 3), 100)
    ax.plot(x, np.minimum(board["bw_bytes"] * x / 1e9, peak), color=st.INK2, lw=1.4)
    ax.text(x[3], board["bw_bytes"] * x[3] / 1e9 * 1.3,
            f"DDR {board['bw_bytes'] / 1e9:.1f} Go/s", fontsize=7.5, color=st.INK2, rotation=28)
    ax.text(x[-1], peak * 1.08, f"calcul Tm·Tn = {tm}×{tn} : {peak:.0f} GOPS", ha="right",
            fontsize=7.5, color=st.INK2)
    # PE paquetée à Tm 64 (M23, T23.3) : 2 MAC par DSP, même nombre de DSP, toit doublé.
    ax.plot(x, np.minimum(board["bw_bytes"] * x / 1e9, 2 * peak), color=st.INK2, lw=1.0,
            ls=st.PROJECTED["linestyle"], zorder=0)
    ax.text(x[-1], 2 * peak * 1.08, f"2 MAC/DSP, Tm {2 * tm} : {2 * peak:.0f} GOPS (projeté)",
            ha="right", fontsize=7.5, color=st.MUTED)
    for layer, c, g0, g1 in pts:
        ax.annotate("", (c, g1), (c, g0), arrowprops=dict(arrowstyle="->", color=st.GREY,
                                                         lw=0.9))
        ax.plot(c, g0, "o", ms=6, mfc="white", mec=st.PALETTE[1], mew=1.3)
        ax.plot(c, g1, "o", ms=7, mfc="white", mec=st.PALETTE[0], mew=1.8)
        ax.annotate(f"L{layer:02d}", (c, g1), textcoords="offset points", xytext=(5, 2),
                    fontsize=7, color=st.INK2)
    ax.plot([], [], "o", mfc="white", mec=st.PALETTE[1], label="noyau M6 (ports 8 bits)")
    ax.plot([], [], "o", mfc="white", mec=st.PALETTE[0], mew=1.8, label="noyau actuel (M10)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("intensité opérationnelle (opérations / octet DDR, modèle §10.2)")
    ax.set_ylabel("GOPS par couche")
    ax.set_title(f"{board['name']} — {net}, roofline par couche")
    ax.legend(loc="lower right")
    st.note(ax, "projection C-sim (marques creuses) ; mesure carte à venir", "upper left")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("roofline_couches", "resultats", "T13.22",
        "Roofline KV260 avec un point par couche de Tiny-YOLOv2, avant et après les "
        "optimisations M10.",
        "hw/boards/kv260.yaml, model/<net>/manifest.json, tools/perf_model.py")
def roofline_couches(out_dir):
    b, pts = load_roofline_layers()
    return plot_roofline_layers(b, pts, NET, out_dir, f"roofline_couches_{b['file']}")


# --- T13.23 : état de l'art ---------------------------------------------------------------

def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


PROJ_RE = re.compile(r"\s*\(projection C-sim(?:, ([^)]*))?\)")


def bench_net(modele):
    """Réseau de ce travail d'après la colonne `modele` (« YOLOv4-tiny COCO 416 »), ou None."""
    return next((n for n, lab in st.NET_LABELS.items() if modele.startswith(lab)), None)


def load_benchmarks(path):
    """Lignes de benchmarks.csv : valeurs numériques ou None, plus `projection` (bool),
    `base` (travail sans la mention de projection, commun à toutes les variantes de ce
    travail), `variant` (« 2 MAC/DSP, Tm 64 », vide pour le noyau M10) et `net`."""
    rows = []
    for r in csv.DictReader(Path(path).open()):
        d = {k: _num(r[k]) for k in ("fps", "latence_ms", "gops", "puissance_w", "dsp",
                                     "map_voc")}
        m = PROJ_RE.search(r["travail"])
        d.update(travail=r["travail"], modele=r["modele"], fpga=r["fpga"],
                 projection="projection" in r["travail"],
                 base=PROJ_RE.sub("", r["travail"]), variant=(m.group(1) or "") if m else "",
                 net=bench_net(r["modele"]) if m else None)
        if d["fps"] is None and d["latence_ms"]:
            d["fps"] = 1e3 / d["latence_ms"]
        rows.append(d)
    return rows


def _bench_tag(r):
    """Étiquette courte d'une ligne : « 2024-zhang » ou « ce travail* v4 · 2 MAC/DSP, Tm 64 »."""
    if not r["projection"]:
        return r["base"]
    tag = f"{r['base']}* {st.NET_SHORT.get(r['net'], '')}"
    return f"{tag} · {r['variant']}" if r["variant"] else tag


def plot_benchmarks(rows, out_dir, name="etat_art"):
    plt = st.plt()
    fig, axes = plt.subplots(2, 2, figsize=(st.FULL * 1.3, 9.0))
    axes = axes.ravel()
    labels = {}
    for r in rows:
        labels.setdefault(r["base"], len(labels))

    def color(r):
        if r["projection"] and r["net"] in st.NET_COLORS:
            return st.NET_COLORS[r["net"]]
        return st.PALETTE[labels[r["base"]] % 8]

    seen = {}

    def mark(ax, r, x, y):
        k = labels[r["base"]]
        c = color(r)
        ax.plot(x, y, "o^sDvP*X"[k % 8], ms=8, color=c, mfc="white" if r["projection"] else c,
                mew=1.6)
        n = seen[(id(ax), k)] = seen.get((id(ax), k), -1) + 1  # même travail : décalé
        ax.annotate(_bench_tag(r), (x, y), textcoords="offset points", xytext=(5, 3 - 10 * n),
                    fontsize=6.5, color=st.INK2)

    panels = [
        (axes[0], "gops", "puissance_w", "puissance (W)", "GOPS", "GOPS face à la puissance"),
        (axes[1], "fps", "map_voc", "mAP VOC (%)", "img/s", "Débit face à la précision"),
    ]
    missing = {}
    for ax, ky, kx, xl, yl, title in panels:
        for r in rows:
            if r[kx] is None or r[ky] is None:
                missing.setdefault(title, []).append(r["base"])
                continue
            mark(ax, r, r[kx], r[ky])
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(title)
    ax = axes[0]
    xs = np.linspace(0.5, 25, 50)
    for eff in (5, 10, 20, 40):
        ax.plot(xs, eff * xs, color=st.GREY, lw=0.8, ls=":", zorder=0)
        xe = min(24, 480 / eff)
        ax.text(xe, eff * xe, f"{eff} GOPS/W", fontsize=6.5, color=st.MUTED, ha="right",
                va="bottom")
    ax.set_xlim(0, 25)
    ax.set_ylim(0, 500)
    axes[1].set_yscale("log")
    axes[1].set_xlim(40, 70)

    def bars(ax, key, value, fmt, xlabel, title):
        ok = [r for r in rows if value(r) is not None]
        for r in rows:
            if r not in ok:
                missing.setdefault(title, []).append(r["base"])
        y = np.arange(len(ok))[::-1]
        for yi, r in zip(y, ok):
            v, c = value(r), color(r)
            ax.barh(yi, v, 0.6, color="white" if r["projection"] else c, edgecolor=c,
                    hatch="///" if r["projection"] else "", linewidth=1.2)
            ax.text(v * 1.02, yi, fmt.format(v), va="center", fontsize=7)
        ax.set_yticks(y, [f"{r['base']}{'*' if r['projection'] else ''}"
                          f"{' · ' + r['variant'] if r['variant'] else ''} · {r['modele'][:26]}"
                          for r in ok], fontsize=6.5)
        ax.set_xlabel(xlabel)
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max([value(r) for r in ok], default=1) * 1.18)

    bars(axes[2], "dsp", lambda r: r["gops"] / r["dsp"] if r["gops"] and r["dsp"] else None,
         "{:.2f}", "GOPS / DSP", "Efficacité par DSP")
    bars(axes[3], "fps", lambda r: r["fps"], "{:.1f}", "img/s (accélérateur seul pour ce travail)",
         "Débit")
    foot = ["Marque creuse, barre hachurée : projection (ce travail, C-sim, * dans les "
            "étiquettes ; couleur = réseau v2 / v3 / v4). Lignes sans valeur, omises du panneau :"]
    foot += [f"  {k} : {', '.join(sorted(set(v)))}" for k, v in missing.items()]
    fig.tight_layout(rect=(0, 0.018 * len(foot), 1, 1))
    fig.text(0.01, 0.005, "\n".join(foot), fontsize=6.5, color=st.MUTED, va="bottom")
    return st.save(fig, out_dir, name)


@figure("etat_art", "resultats", "T13.23",
        "Comparaison aux accélérateurs publiés : GOPS et puissance, débit et mAP, GOPS par DSP, "
        "débit (marque creuse : projection de ce travail, par réseau et variante M23).",
        "results/benchmarks.csv")
def etat_art(out_dir):
    return plot_benchmarks(load_benchmarks(need(ROOT / "results" / "benchmarks.csv")), out_dir)


# --- T13.24 : ressources face au budget KV260 ---------------------------------------------

BRAM18_BYTES = 18 * 1024 // 8
URAM_BYTES = 288 * 1024 // 8
EST, MES = "estimation", "synthèse"
# Rapports csynth de chaque variante (make hls-synth, hls-synth-pow2, hls-synth-stream). Le
# 4 bits n'a pas de noyau HLS propre : il reste une estimation (2 MAC par DSP).
SYNTH_DIRS = {"moteur unique INT8": "synth", "moteur unique pow2": "synth_pow2",
              "moteur unique 2 MAC/DSP": "synth_pack", "streaming W8A8": "synth_stream"}


def load_synth(board_file, step):
    """{"dsp", "lut", "mem"} mesurés par `csynth.xml` de hls/proj_<carte>_<step>, ou None."""
    d = ROOT / "hls" / f"proj_{board_file}_{step}" / "sol" / "syn" / "report"
    syn = _tools("hls_report").read_synth(d)
    if not syn or not syn["resources"]:
        return None

    def used(k):
        v = syn["resources"].get(k, ("0",))[0]
        return int(v) if v and v.strip().isdigit() else 0

    return {"dsp": used("DSP") or used("DSP48E"), "lut": used("LUT"),
            "mem": used("BRAM_18K") * BRAM18_BYTES + used("URAM") * URAM_BYTES}


def load_resources(board="kv260"):
    """Variantes [{name, dsp, mem, lut}] (chaque valeur : (nombre, EST | MES) ou None) et
    budget de la puce {dsp, mem, lut}. Une synthèse présente remplace l'estimation."""
    rl, pm = _tools("roofline"), _tools("perf_model")
    try:
        b = rl.load_board(need(ROOT / "hw" / "boards" / f"{board}.yaml"))
    except ImportError as e:
        raise MissingSource(f"pyyaml absent ({e})") from e
    t = pm.TILES
    mem = rl.bram18(t["tm"], t["tn"], t["tr"], t["tc"]) * BRAM18_BYTES
    rows = [
        {"name": "moteur unique INT8", "dsp": (rl.dsp(t["tm"], t["tn"], 1), EST),
         "mem": (mem, EST), "lut": None},
        # MAC par décalages : plus de DSP de multiplication, seule la requantification reste.
        {"name": "moteur unique pow2", "dsp": (rl.REQUANT_DSP * t["tm"], EST),
         "mem": (mem, EST), "lut": (rl.lut_pow2(t["tm"], t["tn"]), EST)},
        {"name": "moteur unique 4 bits", "dsp": (rl.dsp(t["tm"], t["tn"], 2), EST),
         "mem": (mem, EST), "lut": None},
    ]
    # PE paquetée (M23, T23.2-T23.3) : mêmes tuiles (make hls-synth-pack), puis les deux façons
    # de prendre le gain de perf_model.PACK_OPTIONS.
    for label, tm, tn in (("", t["tm"], t["tn"]), (", Tm 64", 2 * t["tm"], t["tn"]),
                          (", Tn 32", t["tm"], 32)):
        rows.append({"name": f"moteur unique 2 MAC/DSP{label}",
                     "dsp": (pm.kernel_dsp(tm, tn, pack=2), EST),
                     "mem": (rl.bram18(tm, tn, t["tr"], t["tc"]) * BRAM18_BYTES, EST),
                     "lut": None})
    budget_mem = None
    for wb in (8, 4):
        p = ROOT / "build" / "m9" / f"stream_plan_w{wb}.json"
        if p.exists():
            d = _read_json(p)
            rows.append({"name": f"streaming W{d['wbits']}A{d['abits']}",
                         "dsp": (d["dsp"], EST), "mem": (d["onchip_bytes"], EST), "lut": None})
            budget_mem = d["onchip_budget"] / rl.UTIL
    for r in rows:
        step = SYNTH_DIRS.get(r["name"])
        m = load_synth(b["file"], step) if step else None
        if m:
            r.update({k: (v, MES) for k, v in m.items()})
    if budget_mem is None:
        budget_mem = b["bram18"] * BRAM18_BYTES + b.get("uram", 0) * URAM_BYTES
    return b, rows, {"dsp": b["dsp"], "mem": budget_mem, "lut": b.get("lut")}


def plot_resources(board, rows, budget, out_dir, name="ressources"):
    plt = st.plt()
    panels = [(k, lab) for k, lab in (("dsp", "DSP"), ("mem", "mémoire sur puce"), ("lut", "LUT"))
              if budget.get(k)]
    fig, axes = plt.subplots(1, len(panels), figsize=(st.FULL, 0.45 * len(rows) + 1.9),
                             sharey=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for i, (ax, (k, lab)) in enumerate(zip(axes[0], panels)):
        tot = budget[k]
        for yi, r in zip(y, rows):
            if r.get(k) is None:
                ax.text(1.5, yi, "à mesurer", va="center", fontsize=7, color=st.MUTED,
                        style="italic")
                continue
            v, status = r[k]
            frac = v / tot * 100
            if status == MES:
                ax.barh(yi, frac, 0.6, color=st.PALETTE[i], edgecolor=st.PALETTE[i])
            else:
                ax.barh(yi, frac, 0.6, color="white", edgecolor=st.PALETTE[i], hatch="///",
                        linewidth=1.2)
            txt = f"{v / 1e6:.2f} Mo" if k == "mem" else f"{v:,.0f}".replace(",", " ")
            ax.text(min(frac, 118) + 1.5, yi, f"{txt} ({frac:.0f} %)", va="center",
                    fontsize=7)
        ax.axvline(100, color=st.INK2, lw=1.2)
        ax.axvline(80, color=st.MUTED, lw=0.9, ls="--")
        ax.text(80, y[-1] - 0.55, " 80 %", fontsize=7, color=st.MUTED, va="center")
        ax.set_xlim(0, 150)
        ax.set_xlabel(f"% du budget {board['name']}")
        total = f"{tot / 1e6:.2f} Mo" if k == "mem" else f"{tot:,.0f}".replace(",", " ")
        ax.set_title(f"{lab} — total {total}", fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0][0].set_yticks(y, [r["name"] for r in rows], fontsize=8)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.01, "Hachuré : estimation (modèles roofline, streaming, LUT par MAC de "
             "req_yolo.md, perf_model.kernel_dsp pour la PE paquetée) ; plein : rapport csynth "
             "(make hls-synth, hls-synth-pow2, hls-synth-pack, hls-synth-stream). « à mesurer » : "
             "aucune estimation.", fontsize=6.5,
             color=st.MUTED)
    return st.save(fig, out_dir, name)


@figure("ressources", "resultats", "T13.24",
        "DSP, mémoire sur puce et LUT des architectures (moteur unique INT8, pow2, 4 bits et "
        "2 MAC/DSP, streaming) face au budget de la KV260 ; hachuré : estimation, plein : "
        "synthèse.",
        "hw/boards/kv260.yaml, tools/roofline.py, build/m9/stream_plan_w*.json, "
        "hls/proj_kv260_synth*/sol/syn/report/csynth.xml")
def ressources(out_dir):
    b, rows, budget = load_resources()
    return plot_resources(b, rows, budget, out_dir)


# --- T13.25 : post-traitement matériel ----------------------------------------------------

HW_CAP = 256  # emplacements de la NMS sans tri (results/postproc_hw.md, eval_quant --hw-cap)


def load_hwpp(net=NET):
    """(mAP ARM, mAP matériel plafonné, mAP matériel sans plafond ou None, boîtes par image)."""
    d = _read_json(need(ROOT / "build" / "m9" / "hwpp_map.json"))
    if d["images"] != VOC_IMAGES:
        raise MissingSource(f"hwpp_map.json : {d['images']} images")
    nocap = ROOT / "build" / "m9" / "hwpp_map_nocap.json"
    m_nocap = _read_json(nocap)["results"]["int-hwpp"]["map"] if nocap.exists() else None
    counts = None
    jsonl = ROOT / "build" / "m8" / net / "int.jsonl"
    if jsonl.exists():
        counts = np.array([len(json.loads(line)["scores"])
                           for line in jsonl.read_text().splitlines() if line.strip()])
    return d["results"]["int"]["map"], d["results"]["int-hwpp"]["map"], m_nocap, counts


def plot_hwpp(m_arm, m_cap, m_nocap, counts, out_dir, name="hw_postproc"):
    plt = st.plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(st.FULL, 3.6),
                                  gridspec_kw={"width_ratios": [2, 3]})
    bars = [("ARM\n(tri + NMS)", m_arm), (f"matériel\n{HW_CAP} places", m_cap)]
    if m_nocap is not None:
        bars.append(("matériel\nsans plafond", m_nocap))
    for k, (lab, m) in enumerate(bars):
        ax.bar(k, m * 100, 0.6, color=st.PALETTE[k], hatch=("", "//", "..")[k])
        ax.text(k, m * 100 + 0.6, f"{m * 100:.2f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(bars)), [b[0] for b in bars], fontsize=7.5)
    ax.set_ylim(0, 70)
    ax.set_ylabel("mAP (%)")
    ax.set_title("mAP selon le post-traitement (4 952 images)")
    ax.grid(axis="x", visible=False)
    if counts is not None:
        ax2.hist(counts, bins=60, color=st.PALETTE[0])
        ax2.axvline(HW_CAP, color=st.PALETTE[7], lw=1.4)
        over = float((counts > HW_CAP).mean())
        ax2.text(HW_CAP, ax2.get_ylim()[1] * 0.9, f" plafond {HW_CAP}\n {over * 100:.1f} % "
                 "des images au-delà", fontsize=7.5, color=st.INK2)
        ax2.set_xlabel("boîtes gardées par image (modèle entier, seuil 0,005, après NMS)")
        ax2.set_ylabel("images")
        ax2.set_title("Boîtes par image face au plafond")
    else:
        ax2.set_visible(False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("hw_postproc", "resultats", "T13.25",
        "mAP avec le post-traitement matériel, avec et sans plafond de boîtes, et nombre de "
        "boîtes par image face au plafond.",
        "build/m9/hwpp_map.json, hwpp_map_nocap.json, build/m8/<net>/int.jsonl")
def hw_postproc(out_dir):
    return plot_hwpp(*load_hwpp(), out_dir)


# --- T13.26 : courbes d'entraînement ------------------------------------------------------

LOSS_PARTS = ("coord", "obj", "noobj", "cls")


def load_training(run_dir):
    """{loss: {colonne: tableau}, admm: {colonne: tableau} ou None} d'un `--out` de train.py."""
    run_dir = Path(run_dir)

    def cols(path):
        rows = list(csv.DictReader(path.open()))
        return {k: np.array([float(r[k]) for r in rows]) for k in rows[0]} if rows else None

    loss = cols(need(run_dir / "loss.csv"))
    if loss is None:
        raise MissingSource(f"{run_dir}/loss.csv vide")
    admm = run_dir / "admm.csv"
    return {"name": run_dir.name, "loss": loss, "admm": cols(admm) if admm.exists() else None}


def _window(n):
    return max(1, min(50, n // 10))


def plot_training(runs, out_dir, name):
    """Un run : perte, composantes, lr (et résidus ADMM) ; plusieurs : pertes superposées."""
    plt = st.plt()
    single = len(runs) == 1
    has_admm = single and runs[0]["admm"] is not None
    n = 3 + has_admm if single else 2
    ratios = ([3, 3, 1.2] + ([2.4] if has_admm else [])) if single else [3, 1.2]
    fig, axes = plt.subplots(n, 1, figsize=(st.FULL, 2.0 * n + 0.8), sharex=True,
                             gridspec_kw={"height_ratios": ratios})
    for k, r in enumerate(runs):
        L = r["loss"]
        w = _window(len(L["it"]))
        color = st.PALETTE[k]
        axes[0].plot(L["it"], L["loss"], color=color, alpha=0.18, lw=0.8)
        axes[0].plot(L["it"], st.smooth(L["loss"], w), color=color, lw=1.8,
                     label=f"{r['name']} (lissée sur {w})")
        axes[-1 - has_admm].plot(L["it"], L["lr"], color=color, lw=1.4)
    axes[0].set_ylabel("perte / image")
    axes[0].set_title("Perte d'entraînement" + (f" — {runs[0]['name']}" if single else ""))
    axes[0].legend(loc="upper right")
    if single:
        L = runs[0]["loss"]
        w = _window(len(L["it"]))
        for k, part in enumerate(LOSS_PARTS):
            axes[1].plot(L["it"], st.smooth(L[part], w), color=st.PALETTE[k + 1], lw=1.6,
                         ls=("-", "--", "-.", ":")[k], label=part)
        axes[1].set_ylabel("composante")
        axes[1].set_title("Composantes de la perte (lissées)")
        axes[1].legend(loc="upper right", ncol=4)
    lr_ax = axes[-1 - has_admm]
    lr_ax.set_ylabel("lr")
    lr_ax.set_yscale("log")
    if has_admm:
        A = runs[0]["admm"]
        res = [c for c in A if c.startswith("res_")]
        cmap = st.plt().get_cmap("viridis")
        for k, c in enumerate(res):
            axes[-1].plot(A["it"], A[c], color=cmap(k / max(1, len(res) - 1)), lw=1.2,
                          marker=".", label=f"L{int(c[4:]):02d}")
        axes[-1].set_ylabel("‖W − Z‖ / ‖W‖")
        axes[-1].set_title("ADMM : résidu primal par couche (doit tendre vers 0)")
        axes[-1].legend(loc="upper left", ncol=min(len(res), 9), fontsize=6.5)
    axes[-1].set_xlabel("itération")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def train_runs(root=None):
    return sorted(p for p in (Path(root or ROOT) / "build" / "train").glob("*")
                  if (p / "loss.csv").exists())


@figure("entrainement", "resultats", "T13.26",
        "Courbes d'entraînement : perte et composantes lissées, taux d'apprentissage, résidus "
        "ADMM ; puis tous les runs superposés.",
        "build/train/<run>/loss.csv, admm.csv (tools/train.py --out)")
def entrainement(out_dir):
    runs = []
    for p in train_runs():
        try:
            runs.append(load_training(p))
        except MissingSource:
            pass
    if not runs:
        raise MissingSource("aucun build/train/<run>/loss.csv")
    paths = []
    for r in runs:
        paths += plot_training([r], out_dir, f"entrainement_{r['name']}")
    if len(runs) > 1:
        paths += plot_training(runs, out_dir, "entrainement_runs")
    return paths


# --- Mode auto : table JSON d'une évaluation ----------------------------------------------

def plot_eval_json(d, out_dir, name):
    """JSON d'`eval_quant.py --out` : AP par classe de chaque variante, mAP en légende."""
    from yolo.data.voc import VOC_CLASSES

    names = d.get("classes") or VOC_CLASSES
    res = {k: v for k, v in d["results"].items() if len(v.get("aps", [])) == len(names)}
    if not res:
        raise MissingSource("pas d'AP par classe dans le JSON")
    plt = st.plt()
    fig, ax = plt.subplots(figsize=(st.FULL if len(names) <= 30 else st.FULL * 2, 3.8))
    x = np.arange(len(names))
    shown = list(res)[:8]  # au-delà, la palette ne suffit plus : la légende cite le reste
    w = 0.8 / len(shown)
    for k, v in enumerate(shown):
        ax.bar(x + (k - (len(shown) - 1) / 2) * w, np.array(res[v]["aps"]) * 100, w * 0.92,
               color=st.PALETTE[k], edgecolor="white", linewidth=0.3,
               label=f"{v} : mAP {res[v]['map'] * 100:.2f}")
    ax.set_xticks(x, names, rotation=40 if len(names) <= 30 else 90, ha="right",
                  fontsize=8 if len(names) <= 30 else 6)
    ax.set_ylabel("AP (%)")
    ax.set_ylim(0, 100)
    sub = "" if d.get("images") == VOC_IMAGES else " — sous-ensemble"
    ax.set_title(f"{d.get('net', '')}, {d.get('images', '?')} images{sub}")
    rest = [v for v in res if v not in shown]
    ax.legend(loc="upper left", ncol=min(4, len(shown)),
              title=f"+ {len(rest)} variantes non tracées" if rest else None)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_map_summary(rows, out_dir, name="map_resume", title=""):
    """`rows` : [(étiquette, mAP, images)] ; une barre par évaluation, triées par mAP."""
    if not rows:
        raise MissingSource("aucune mAP")
    plt = st.plt()
    rows = sorted(rows, key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(st.FULL, 0.28 * len(rows) + 1.3))
    y = np.arange(len(rows))
    for yi, (label, m, n) in zip(y, rows):
        full = n == VOC_IMAGES
        ax.barh(yi, m * 100, 0.62, color=st.PALETTE[0] if full else "white",
                edgecolor=st.PALETTE[0], hatch="" if full else "///", linewidth=1)
        ax.text(m * 100 + 0.5, yi, f"{m * 100:.2f}  ({n} images)", va="center", fontsize=7)
    ax.set_yticks(y, [r[0] for r in rows], fontsize=7)
    ax.set_xlim(0, 80)
    ax.set_xlabel("mAP (%)")
    ax.set_title(title or "mAP des évaluations (hachuré : sous-ensemble d'images)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


# --- T13.19 : détections côte à côte ------------------------------------------------------

DET_IMAGES = ("000004", "000014", "000025", "000058")  # test ; 000014 et 000058 : flottant ≠ entier
DET_CONF = 0.25  # seuil d'affichage (les JSONL d'évaluation descendent à 0,005)


def _thr(d, conf=DET_CONF):
    b, s, lab = (np.asarray(v) for v in d)
    keep = s > conf
    return b.reshape(-1, 4)[keep], s[keep], lab[keep]


def load_detections(ids=DET_IMAGES, root=None):
    """{id: {sample, vérité, flottant, entier, csim}} ; boîtes (cx, cy, w, h) normalisées
    dans l'image d'origine, prétraitement stretch (celui de `build/m8/`).

    Le flottant est recalculé par `pipeline.detect` ; entier et C-sim viennent de
    `int.jsonl` et `sim/dets_*.jsonl` (`tools/map_stades.read_jsonl`).
    """
    from PIL import Image

    from yolo.infer.pipeline import detect, preprocess

    from tools.figures.modeles import load_net
    from tools.map_stades import read_jsonl

    m8 = Path(root or ROOT) / "build" / "m8" / NET
    ints = read_jsonl([need(m8 / "int.jsonl")])
    sims = read_jsonl(sorted(m8.glob("sim/dets_*.jsonl")))
    if not sims:
        raise MissingSource(f"{m8.relative_to(ROOT)}/sim/dets_*.jsonl absent (make bench-sim)")
    samples = {s["id"]: s for s in test_samples()}
    model = load_net(NET)
    out = {}
    for iid in ids:
        s = samples[iid]
        with Image.open(s["image"]) as img:
            x, wh = preprocess(img, 416, "stretch")
        out[iid] = {"sample": s,
                    "vérité": (s["boxes"][~s["difficult"]], np.ones((~s["difficult"]).sum()),
                               s["labels"][~s["difficult"]]),
                    "flottant": detect(model, x[None], [wh], "stretch", DET_CONF)[0],
                    "entier": _thr(ints[iid]), "csim": _thr(sims[iid])}
    return out


def same_detections(a, b):
    return all(np.array_equal(np.asarray(u), np.asarray(v)) for u, v in zip(a, b))


def plot_detections(dets, out_dir, name="detections"):
    from PIL import Image

    from yolo.data.voc import VOC_CLASSES
    from yolo.infer.boxes import cxcywh_to_xyxy

    plt = st.plt()
    cols = [("vérité", st.PALETTE[5]), ("flottant", st.STAGE_COLORS["flottant"]),
            ("entier", st.STAGE_COLORS["entier"]), ("csim", st.STAGE_COLORS["csim"])]
    fig, axes = plt.subplots(len(dets), len(cols), figsize=(st.FULL, 2.2 * len(dets) + 0.4))
    for r, (iid, d) in enumerate(dets.items()):
        s = d["sample"]
        with Image.open(s["image"]) as img:
            arr = np.asarray(img.convert("RGB"))
        for c, (key, color) in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(arr)
            b, sc, lab = d[key]
            xyxy = cxcywh_to_xyxy(np.asarray(b).reshape(-1, 4)) * np.tile([s["width"], s["height"]], 2)
            texts = [VOC_CLASSES[k] if key == "vérité" else f"{VOC_CLASSES[k]} {v:.2f}"
                     for k, v in zip(lab, sc)]
            st.draw_boxes(ax, xyxy, texts, color=color, lw=1.4)
            ax.set_xlim(0, s["width"])
            ax.set_ylim(s["height"], 0)
            label = {"csim": "C-sim"}.get(key, key)
            if key == "csim":
                label += " (= entier)" if same_detections(d["csim"], d["entier"]) else " (≠ entier !)"
            st.image_axes(ax, f"{iid} : {label}, {len(b)}" if c == 0 else f"{label}, {len(b)}")
    fig.suptitle(f"{NET}, VOC2007 test, stretch, score > {DET_CONF}", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return st.save(fig, out_dir, name)


@figure("detections", "resultats", "T13.19",
        "Vérité terrain, flottant, entier et C-sim sur quatre images de VOC2007 test : les "
        "colonnes entier et C-sim sont identiques.",
        "weights/, build/m8/<net>/int.jsonl et sim/dets_*.jsonl, data/VOCdevkit")
def detections(out_dir):
    return plot_detections(load_detections(), out_dir)


# --- T13.53 : balayages lot × sous-ensemble (docs/tasks/resultats-balayages.md) -----------

BALAYAGES_MD = "docs/tasks/resultats-balayages.md"


def _md_number(cell):
    """« **36,27** » → 36.27 ; None si la cellule n'est pas un nombre."""
    t = cell.replace("*", "").replace("`", "").replace(" ", "").replace(" ", "")
    try:
        return float(t.replace(",", "."))
    except ValueError:
        return None


def _md_tables(text):
    """Tables Markdown d'un texte : [(en-têtes, lignes)], cellules nettoyées du gras."""
    tables, cur = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            cur.append([c.strip() for c in line.strip().strip("|").split("|")])
        elif cur:
            if len(cur) > 2:
                head = [c.lower() for c in cur[0]]
                rows = [[c.replace("**", "").strip() for c in r] for r in cur[2:]]
                tables.append((head, rows))
            cur = []
    return tables


def load_balayages(path):
    """Tables de resultats-balayages.md : {jeu: {"sweep": [run…], "infer": [modèle…]}}.

    Une section `## <jeu>` par jeu ; table de balayage repérée par sa colonne « lot »
    (run, lot, images, perte finale, mAP), table d'inférence par « modèle » (modèle,
    classes évaluées, mAP). Les autres sections (conditions, analyse) sont ignorées."""
    text = Path(need(path)).read_text()
    out = {}
    for block in text.split("\n## ")[1:]:
        title, _, body = block.partition("\n")
        sweep, infer = [], []
        for head, rows in _md_tables(body):
            col = {h: i for i, h in enumerate(head)}
            mcol = next((i for h, i in col.items() if h.startswith("map")), None)
            if mcol is None:
                continue
            if "lot" in col:
                for r in rows:
                    sweep.append({"run": r[col["run"]], "batch": int(r[col["lot"]]),
                                  "subset": 0 if r[col["images"]] == "tout"
                                  else int(r[col["images"]]),
                                  "loss": _md_number(r[col["perte finale"]]),
                                  "map": _md_number(r[mcol])})
            elif "modèle" in col:
                ccol = next((i for h, i in col.items() if h.startswith("classes")), None)
                for r in rows:
                    classes = r[ccol].split()[0] if ccol is not None else ""
                    infer.append({"model": r[col["modèle"]].replace("`", ""),
                                  "classes": int(classes) if classes.isdigit() else None,
                                  "map": _md_number(r[mcol])})
        if sweep or infer:
            out[title.strip()] = {"sweep": sweep, "infer": infer}
    if not out:
        raise MissingSource(f"aucune table de balayage dans {path}")
    return out


SUBSET_STYLE = {0: dict(label="tout le split", color=st.PALETTE[0], hatch="", fill=True),
                500: dict(label="500 images", color=st.PALETTE[1], hatch="///", fill=False)}


def _subset_style(s):
    return SUBSET_STYLE.get(s, dict(label=f"{s} images", color=st.PALETTE[2], hatch="..",
                                    fill=False))


def _grid(plt, n, height, per_row=3):
    """Panneaux d'une figure par jeu : une ligne jusqu'à 4 jeux, sinon `per_row` par ligne ;
    rend (figure, axes utilisés), les cases vides masquées."""
    cols = n if n <= 4 else per_row
    rows = -(-n // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(st.FULL, height * rows), squeeze=False)
    flat = list(axes.flat)
    for ax in flat[n:]:
        ax.set_visible(False)
    return fig, flat[:n]


def plot_balayage_map(data, out_dir, name="balayage_map"):
    """mAP par lot, une barre par sous-ensemble, un panneau par jeu ; meilleur run marqué."""
    plt = st.plt()
    sets = [k for k, v in data.items() if v["sweep"]]
    fig, axes = plt.subplots(1, len(sets), figsize=(st.FULL, 3.4), squeeze=False)
    for ax, ds in zip(axes[0], sets):
        runs = data[ds]["sweep"]
        batches = sorted({r["batch"] for r in runs})
        subsets = sorted({r["subset"] for r in runs}, key=lambda s: (s == 0, s))
        best = max(runs, key=lambda r: r["map"])
        x = np.arange(len(batches))
        w = 0.8 / len(subsets)
        top = max(r["map"] for r in runs)
        for k, s in enumerate(subsets):
            sty = _subset_style(s)
            for xi, b in zip(x, batches):
                r = next((r for r in runs if r["batch"] == b and r["subset"] == s), None)
                if r is None:
                    continue
                xb = xi + (k - (len(subsets) - 1) / 2) * w
                ax.bar(xb, r["map"], w * 0.92, color=sty["color"] if sty["fill"] else "white",
                       edgecolor=sty["color"], hatch=sty["hatch"], linewidth=1.2,
                       label=sty["label"] if xi == 0 else None)
                star = " ★" if r is best else ""
                ax.text(xb, r["map"] + top * 0.02, f"{r['map']:.2f}{star}".replace(".", ","),
                        ha="center", va="bottom", fontsize=7, color=st.INK,
                        fontweight="bold" if r is best else "normal")
        ax.set_xticks(x, [str(b) for b in batches])
        ax.set_xlabel("lot")
        ax.set_axisbelow(True)
        ax.set_ylim(0, top * 1.18)
        ax.set_ylabel("mAP (%)")
        ax.set_title(f"{ds}\nmeilleur : {best['run']}", fontsize=9)
        ax.grid(axis="x", visible=False)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.07))
    fig.tight_layout()
    fig.text(0.99, -0.02, "50 images, palier R (non publiable)", ha="right", va="top",
             fontsize=7, color=st.MUTED)
    return st.save(fig, out_dir, name)


def _label_offset(r, runs):
    """Étiquette au-dessus à droite, ou en dessous si un voisin proche la masquerait."""
    span_x = (max(q["loss"] for q in runs) - min(q["loss"] for q in runs)) or 1
    span_y = (max(q["map"] for q in runs) - min(q["map"] for q in runs)) or 1
    for q in runs:
        if (q is not r and abs(q["loss"] - r["loss"]) < 0.08 * span_x
                and 0 <= q["map"] - r["map"] < 0.08 * span_y):
            return (5, -11)
    return (5, 4)


def plot_balayage_perte(data, out_dir, name="balayage_perte"):
    """Perte finale face à la mAP : une perte basse sur 500 images ne fait pas une bonne mAP."""
    plt = st.plt()
    sets = [k for k, v in data.items() if v["sweep"]]
    fig, flat = _grid(plt, len(sets), 3.4)
    for ax, ds in zip(flat, sets):
        runs = data[ds]["sweep"]
        for s in sorted({r["subset"] for r in runs}, key=lambda s: (s == 0, s)):
            sty = _subset_style(s)
            pts = [r for r in runs if r["subset"] == s]
            ax.scatter([r["loss"] for r in pts], [r["map"] for r in pts], s=48, zorder=3,
                       facecolor=sty["color"] if sty["fill"] else "white",
                       edgecolor=sty["color"], linewidth=1.6, label=sty["label"])
            for r in pts:
                ax.annotate(r["run"], (r["loss"], r["map"]), xytext=_label_offset(r, runs),
                            textcoords="offset points", fontsize=7, color=st.INK2)
        ax.set_xlabel("perte finale")
        ax.set_ylabel("mAP (%)")
        ax.set_title(ds)
        ax.margins(x=0.18, y=0.15)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.04))
    fig.text(0.99, 0.0, "perte finale après 600 itérations ; 50 images, palier R", ha="right",
             va="top", fontsize=7, color=st.MUTED)
    fig.tight_layout()
    return st.save(fig, out_dir, name)


def plot_balayage_modeles(data, out_dir, name="balayage_modeles"):
    """mAP des poids publiés (hors domaine pour VisDrone) face au meilleur run affiné."""
    plt = st.plt()
    sets = [k for k, v in data.items() if v["infer"]]
    fig, flat = _grid(plt, len(sets), 2.6)
    for ax, ds in zip(flat, sets):
        rows = data[ds]["infer"][::-1]
        top = max(r["map"] for r in rows)
        y = np.arange(len(rows))
        for yi, r in zip(y, rows):
            color = st.NET_COLORS.get(r["model"], st.PALETTE[2])
            ax.barh(yi, r["map"], 0.6, color=color)
            ax.text(r["map"] + top * 0.02, yi, f"{r['map']:.2f}".replace(".", ","),
                    va="center", fontsize=7, color=st.INK)
        labels = [f"{r['model'].split()[0]}\n({r['classes']} classes)" if r["classes"]
                  else r["model"] for r in rows]
        ax.set_yticks(y, labels, fontsize=7)
        ax.set_xlim(0, top * 1.2)
        ax.set_axisbelow(True)
        ax.set_xlabel("mAP (%)")
        ax.set_title(ds)
        ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.99, 0.01, "50 images, palier R ; hors domaine : classes communes seulement",
             ha="right", va="bottom", fontsize=7, color=st.MUTED)
    return st.save(fig, out_dir, name)


@figure("balayage", "resultats", "T13.53",
        "Balayages lot × sous-ensemble, un panneau par jeu : mAP par lot, perte finale face à la "
        "mAP, et run affiné face aux poids publiés. 50 images, palier R.",
        f"{BALAYAGES_MD} (tables des notebooks _sweep et _infer)", subset=True,
        dest=ROOT / "docs" / "tasks" / "figures")
def balayage(out_dir):
    data = load_balayages(ROOT / BALAYAGES_MD)
    return (plot_balayage_map(data, out_dir) + plot_balayage_perte(data, out_dir)
            + plot_balayage_modeles(data, out_dir))


# --- T13.55 : two MACs per DSP, the options of T23.3 ---------------------------------------

def _present_nets():
    nets = [n for n in st.NETS if (ROOT / "model" / n / "manifest.json").exists()]
    if not nets:
        raise MissingSource("no model/<net>/manifest.json (make export)")
    return nets


def load_pack_options(nets=None, board="kv260"):
    """({net: rows of perf_model.pack_table}, w_buf flip-flops per option, board budget)."""
    pm, rl = _tools("perf_model"), _tools("roofline")
    nets = nets or _present_nets()
    try:
        b = rl.load_board(need(ROOT / "hw" / "boards" / f"{board}.yaml"))
    except ImportError as e:
        raise MissingSource(f"pyyaml absent ({e})") from e
    ff = [rl.wbuf_ff(o["tm"], o["tn"]) for _, o in pm.PACK_OPTIONS]
    return {net: pm.pack_table(net) for net in nets}, ff, b


def plot_pack_options(tables, ff, board, out_dir, name="pack_options"):
    plt = st.plt()
    nets = list(tables)
    fig, axes = plt.subplots(1, len(nets), figsize=(st.FULL * 1.25, 3.6), sharey=True,
                             squeeze=False)
    axes = axes[0]
    budget = {"dsp": rl_util(board["dsp"]), "ff": rl_util(board.get("ff"))}
    for ax, net in zip(axes, nets):
        rows = tables[net]
        y = np.arange(len(rows))[::-1]
        c = st.NET_COLORS[net]
        xmax = max(r[6] for r in rows) * 1.75
        for yi, (opt, tm, tn, dpe, dtot, cyc, ms, cb, gops, other), f in zip(y, rows, ff):
            packed = yi != y[0]
            ax.barh(yi, ms, 0.6, color="white" if packed else c, edgecolor=c,
                    hatch="///" if packed else "", linewidth=1.2)
            ax.plot([cb, cb], [yi - 0.36, yi + 0.36], color=st.INK, lw=1.6)
            over = [k for k, v in (("DSP", dtot), ("FF", f)) if budget[k.lower()]
                    and v > budget[k.lower()]]
            txt = f"{ms:.1f} ms · {1e3 / ms:.1f} img/s\n{dtot} DSP · w_buf {f / 1e3:.0f} k FF"
            ax.text(ms + xmax * 0.02, yi, txt + (f"  > 80 % {'+'.join(over)}" if over else ""),
                    va="center", fontsize=6.5, color=st.PALETTE[7] if over else st.INK2)
        ax.set_xlim(0, xmax)
        ax.set_xlabel("accelerator time per image (ms, 200 MHz)")
        ax.set_title(st.NET_LABELS[net], fontsize=9)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(rows))[::-1], [r[0] for r in rows], fontsize=7.5)
    fig.suptitle("Two int8 MACs per DSP: same tiles (half the DSPs), Tm 64 (same DSPs) or Tn 32",
                 fontsize=10, weight="bold")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.text(0.01, 0.01, "Cycle model == C-sim (tools/perf_model.py:pack_table), projection, "
             "not a board measurement. Black tick: compute bound. Hatched: packed PE "
             f"(ACC_PACK=2). Budget: 80 % of the {board['name']} DSP and flip-flops "
             "(w_buf in registers).", fontsize=6.5, color=st.MUTED)
    return st.save(fig, out_dir, name)


def rl_util(v):
    return None if v is None else _tools("roofline").UTIL * v


@figure("pack_options", "resultats", "T13.55",
        "Time per image of each network with the packed PE (T23.3): M10 kernel, same tiles, "
        "Tm 64 and Tn 32, with the compute bound, DSP and w_buf flip-flops.",
        "tools/perf_model.py (PACK_OPTIONS, pack_table), tools/roofline.py (wbuf_ff), "
        "hw/boards/kv260.yaml")
def pack_options(out_dir):
    return plot_pack_options(*load_pack_options(), out_dir)


# --- T13.56 : Winograd F(2×2, 3×3) against packing ----------------------------------------

def load_winograd(nets=None):
    """[tools/winograd.py:net_study(net)] of the exported networks."""
    w = _tools("winograd")
    return [w.net_study(net) for net in (nets or _present_nets())]


def plot_winograd(rows, out_dir, name="winograd"):
    plt = st.plt()
    fig, axes = plt.subplots(1, 3, figsize=(st.FULL * 1.25, 3.8),
                             gridspec_kw={"width_ratios": [1, 1, 0.8]})
    x = np.arange(len(rows))
    panels = [
        (axes[0], "Compute bound", [("direct", "direct_ms", "", 3),
                                    ("Winograd", "wino_ms", "//", 4),
                                    ("packing Tm 64", "pack_ms", "..", 5)]),
        (axes[1], "With the loads (first order)",
         [("Winograd, offline weights", "wino_off_ms", "//", 4),
          ("Winograd, on the fly", "wino_fly_ms", "xx", 4),
          ("packing Tm 64", "pack_t_ms", "..", 5)]),
    ]
    for ax, title, series in panels:
        w = 0.8 / len(series)
        for k, (lab, key, hatch, ci) in enumerate(series):
            xs = x + (k - (len(series) - 1) / 2) * w
            vals = [r[key] for r in rows]
            ax.bar(xs, vals, w * 0.92, color="white", edgecolor=st.PALETTE[ci], hatch=hatch,
                   linewidth=1.2, label=lab)
            for xi, v in zip(xs, vals):
                ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.3)
        ax.set_xticks(x, [st.NET_SHORT.get(r["net"], r["net"]) for r in rows])
        ax.set_ylabel("ms per image (200 MHz)")
        ax.set_title(title, fontsize=9)
        ax.legend(loc="upper left", fontsize=6.5)
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(r[key] for r in rows for _, key, *_ in series) * 1.3)
    ax = axes[2]
    w = 0.38
    direct = [r["w_direct"] / 1e6 for r in rows]
    wino = [r["w_wino"] * 12 / 8 / 1e6 for r in rows]  # 12-bit transformed weights
    ax.bar(x - w / 2, direct, w * 0.92, color=st.PALETTE[3], label="direct (int8, 3×3)")
    ax.bar(x + w / 2, wino, w * 0.92, color="white", edgecolor=st.PALETTE[4], hatch="//",
           linewidth=1.2, label="Winograd (12 bits, 4×4)")
    for xi, a, b in zip(x, direct, wino):
        ax.text(xi + w / 2, b + 0.5, f"×{b / a:.2f}", ha="center", fontsize=6.5)
    ax.set_xticks(x, [st.NET_SHORT.get(r["net"], r["net"]) for r in rows])
    ax.set_ylabel("weight bytes of the 3×3 convs (MB)")
    ax.set_title("Weight footprint", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(axis="x", visible=False)
    fig.suptitle("Winograd F(2×2, 3×3) against two MACs per DSP (T23.5)", fontsize=10,
                 weight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, "Projection from the perf_model counters (tools/winograd.py), M10 tiles "
             "Tm 32 × Tn 24; v2 Tiny-YOLOv2 VOC, v3 Tiny-YOLOv3 COCO, v4 YOLOv4-tiny COCO.",
             fontsize=6.5, color=st.MUTED)
    return st.save(fig, out_dir, name)


@figure("winograd", "resultats", "T13.56",
        "Winograd F(2×2, 3×3) against the packed PE: compute bound, time with the loads and "
        "weight footprint of the 3×3 convs, per network.",
        "tools/winograd.py (net_study), model/<net>/manifest.json and weights.bin")
def winograd(out_dir):
    return plot_winograd(load_winograd(), out_dir)


# --- T13.57 : COCO val2017 AP (M24) --------------------------------------------------------

COCO_METRICS = ("AP", "AP50", "AP75", "APs", "APm", "APl")
COCO_NETS = {"YOLOv4-tiny": "tiny-yolov4-coco", "Tiny-YOLOv3": "tiny-yolov3-coco"}


def _md_cells(line):
    return [c.strip().strip("*").strip() for c in line.strip().strip("|").split("|")]


def _md_num(v):
    try:
        return float(v)
    except ValueError:
        return None


def load_map_coco(path):
    """results/map_coco.md: float AP rows [{net, prep, AP…}], published AP50 {net: value} and
    the YOLOv4-tiny stages [{stage, AP…}] (None where still to run)."""
    text = Path(path).read_text()
    floats, stages, table, net = [], [], None, None
    for line in text.splitlines():
        if not line.startswith("|"):
            table = None
            continue
        cells = _md_cells(line)
        if cells[0] in ("Network", "Stage"):
            table = cells
            continue
        if table is None or set(cells[0]) <= {"-"}:
            continue
        vals = {m: _md_num(cells[table.index(m)]) for m in COCO_METRICS if m in table}
        if table[0] == "Network":
            name = re.sub(r"\s*\(.*\)", "", cells[0]).strip("* ")
            net = COCO_NETS.get(name, net)
            floats.append({"net": net, "prep": cells[1], **vals})
        else:
            stages.append({"stage": re.sub(r"\s*\(.*\)", "", cells[0]), **vals})
    published = {COCO_NETS[n]: float(v) for n, v in
                 re.findall(r"\*\*(YOLOv4-tiny|Tiny-YOLOv3) ([\d.]+)\*\*", text)}
    if not floats:
        raise MissingSource(f"{path}: no AP table")
    return floats, published, stages


def plot_map_coco(floats, published, stages, out_dir, name="map_coco"):
    plt = st.plt()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(st.FULL * 1.15, 4.0),
                                  gridspec_kw={"width_ratios": [2.2, 1]})
    x = np.arange(len(COCO_METRICS))
    w = 0.8 / len(floats)
    for k, r in enumerate(floats):
        xs = x + (k - (len(floats) - 1) / 2) * w
        c = st.NET_COLORS.get(r["net"], st.MUTED)
        stretch = r["prep"] == "stretch"
        ax.bar(xs, [r[m] or 0 for m in COCO_METRICS], w * 0.92, color=c if stretch else "white",
               edgecolor=c, hatch="" if stretch else "//", linewidth=1.1,
               label=f"{st.NET_LABELS.get(r['net'], r['net'])}, {r['prep']}")
        for xi, m in zip(xs, COCO_METRICS):
            if r[m] is not None:
                ax.text(xi, r[m] + 0.5, f"{r[m]:.1f}", ha="center", fontsize=5.8, rotation=90)
    i50 = COCO_METRICS.index("AP50")
    for net, v in published.items():
        ax.plot([i50 - 0.45, i50 + 0.45], [v, v], color=st.NET_COLORS[net], ls="--", lw=1.3)
        ax.text(i50 + 0.47, v, f"published {v:g}\n(test-dev)", fontsize=6, va="center",
                color=st.NET_COLORS[net])
    ax.set_xticks(x, COCO_METRICS)
    ax.set_ylabel("COCO val2017 (%)")
    ax.set_ylim(0, max(r["AP50"] or 0 for r in floats) * 1.25 + 4)
    ax.set_title("Float, Darknet weights, 416", fontsize=9)
    ax.legend(loc="upper right", fontsize=6.5)
    ax.grid(axis="x", visible=False)
    c = st.NET_COLORS["tiny-yolov4-coco"]
    xs = np.arange(len(stages))
    for xi, r in zip(xs, stages):
        if r.get("AP") is None and r.get("AP50") is None:
            ax2.text(xi, 2, "to run", ha="center", va="bottom", fontsize=7.5, color=st.MUTED,
                     style="italic")
            continue
        for k, (m, off) in enumerate((("AP", -0.2), ("AP50", 0.2))):
            v = r.get(m)
            if v is None:
                continue
            ax2.bar(xi + off, v, 0.38, color=c if k else "white", edgecolor=c, linewidth=1.1)
            ax2.text(xi + off, v + 0.5, f"{v:.1f}", ha="center", fontsize=6.5)
    ax2.set_xticks(xs, [r["stage"].replace("FPGA, ", "") for r in stages], fontsize=7)
    ax2.set_ylim(0, max([r.get("AP50") or 0 for r in stages] + [1]) * 1.2)
    ax2.set_xlim(-0.6, len(stages) - 0.4)
    ax2.set_title("YOLOv4-tiny stages: AP (hollow), AP50 (solid)", fontsize=9)
    ax2.grid(axis="x", visible=False)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.text(0.01, 0.01, "Source: results/map_coco.md (python/yolo/infer/coco_eval.py, equal to "
             "pycocotools to 1e-4). Solid: stretch resize, hatched: letterbox.", fontsize=6.5,
             color=st.MUTED)
    return st.save(fig, out_dir, name)


@figure("map_coco", "resultats", "T13.57",
        "COCO val2017 AP of YOLOv4-tiny and Tiny-YOLOv3 (stretch and letterbox) with the "
        "published AP50, and the float / integer / C-sim stages of YOLOv4-tiny.",
        "results/map_coco.md")
def map_coco(out_dir):
    return plot_map_coco(*load_map_coco(need(ROOT / "results" / "map_coco.md")), out_dir)


# --- T13.58: fault sensitivity (T25.1) ----------------------------------------------------

FAULT_SITES = ("weight", "bias", "act", "acc")


def load_faults(path):
    """report.json of `tools/fault_campaign.py report`: {baseline, images, ber, layer, bit,
    ranking, abft} (mAP as fractions)."""
    rep = _read_json(path)
    if rep.get("baseline") is None or not (rep["ber"] or rep["layer"] or rep["bit"]):
        raise MissingSource(f"{path}: no baseline or no grid")
    return rep


def plot_faults(rep, out_dir, name="faults"):
    plt = st.plt()
    panels = [g for g in ("ber", "layer", "bit") if rep[g]]
    fig, axes = plt.subplots(len(panels), 1, figsize=(st.FULL, 2.9 * len(panels)),
                             squeeze=False)
    base = rep["baseline"] * 100
    for ax, g in zip(axes[:, 0], panels):
        rows = rep[g]
        sites = [s for s in FAULT_SITES if any(a["site"] == s for a in rows)]
        if g == "ber":
            for k, s in enumerate(sites):
                pts = sorted((a["rate"], a["map"] * 100, a["std"] * 100) for a in rows
                             if a["site"] == s)
                r, m, e = map(np.array, zip(*pts))
                ax.errorbar(r, m, yerr=e, color=st.PALETTE[k], marker="os^D"[k], ms=4,
                            capsize=2, lw=1.2, label=s)
            ax.set_xscale("log")
            ax.axhline(base, color=st.INK2, lw=0.8, ls="--")
            ax.text(ax.get_xlim()[0], base, f" fault-free {base:.2f}", va="bottom", fontsize=7,
                    color=st.INK2)
            ax.set_xlabel("bit error rate (per stored bit)")
            ax.set_ylabel("mAP (%)")
            ax.set_title("mAP against the bit error rate, per fault site (mean ± std of the "
                         "seeds)")
        elif g == "layer":
            layers = sorted({a["layer"] for a in rows})
            rate = max(a["rate"] for a in rows)
            x = np.arange(len(layers))
            w = 0.8 / len(sites)
            for k, s in enumerate(sites):
                d = {a["layer"]: a["map"] * 100 - base for a in rows
                     if a["site"] == s and a["rate"] == rate}
                ax.bar(x + (k - (len(sites) - 1) / 2) * w, [d.get(i, np.nan) for i in layers],
                       w * 0.92, color=st.PALETTE[k], hatch=("", "//", "..", "xx")[k],
                       edgecolor="white", linewidth=0.4, label=s)
            ax.axhline(0, color=st.INK2, lw=0.8)
            ax.set_xticks(x, [f"L{i:02d}" for i in layers])
            ax.set_ylabel("mAP change (points)")
            ax.set_title(f"Faults in one layer at a BER of {rate:.0e}")
            ax.grid(axis="x", visible=False)
        else:
            for k, s in enumerate(sites):
                pts = sorted((a["bit"], a["map"] * 100 - base) for a in rows if a["site"] == s)
                b, d = zip(*pts)
                ax.plot(b, d, color=st.PALETTE[k], marker="os^D"[k], ms=4, lw=1.2, label=s)
            ax.axhline(0, color=st.INK2, lw=0.8)
            ax.set_xlabel("bit position (7: sign of an int8, 31: sign of an int32)")
            ax.set_ylabel("mAP change (points)")
            ax.set_title("One bit flip at a given position (per image for act / acc, per "
                         "model for weight / bias)")
        ax.legend(loc="lower left", ncol=len(sites), fontsize=7)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.text(0.01, 0.005, f"{rep['images']} images of VOC2007 test (ranking only, M12 rule); "
             "integer model, faults of yolo.quant.faults (T25.0).", fontsize=6.5,
             color=st.MUTED)
    return st.save(fig, out_dir, name)


@figure("faults", "resultats", "T13.58",
        "mAP of the integer Tiny-YOLOv2 under bit flips: against the bit error rate per fault "
        "site, per layer, and per bit position. Subset of VOC2007 test.",
        "build/m25/faults/report.json (else build/m25/smoke/report.json)", subset=True)
def faults(out_dir):
    path = ROOT / "build" / "m25" / "faults" / "report.json"
    if not path.exists():
        path = ROOT / "build" / "m25" / "smoke" / "report.json"
    return plot_faults(load_faults(need(path)), out_dir)


# --- T13.61: miss rate against FPPI (M27, T27.2) -------------------------------------------

def load_miss_fppi(paths):
    """`oppoint.json` files of `tools/person_safety.py oppoint` → [{title, stages: {stage:
    {fppi, mr, lamr, targets, repo}}}], one panel each."""
    out = []
    for p in paths:
        d = json.loads(Path(p).read_text())
        out.append({"title": f"{st.NET_LABELS.get(d['net'], d['net'])} — {d['dataset']} "
                             f"{d['split']}", "stages": d["stages"]})
    return out


def plot_miss_fppi(panels, out_dir, name="miss_fppi"):
    plt = st.plt()
    fig, axes = plt.subplots(1, len(panels), figsize=(st.COL * len(panels) * 0.85, 4.4),
                             squeeze=False)
    colors = {"float": st.STAGE_COLORS["flottant"], "int": st.STAGE_COLORS["entier"],
              "csim": st.STAGE_COLORS["csim"]}
    names = {"float": "float", "int": "integer", "csim": "C-sim"}
    for ax, pnl in zip(axes[0], panels):
        for key, s in pnl["stages"].items():
            f, mr = np.array(s["curve"]["fppi"]), np.array(s["curve"]["mr"])
            keep = f > 0
            ax.plot(f[keep], mr[keep], color=colors.get(key, st.INK2),
                    ls="-" if key == "float" else "--",
                    label=f"{names.get(key, key)}: LAMR {100 * s['lamr']:.1f} %")
            for thr, r in s.get("repo", {}).items():
                x = max(r["fppi"], 1e-3)  # FPPI 0: pinned to the left edge
                ax.plot(x, 1 - r["recall"], "o", ms=5, mfc="white",
                        mec=colors.get(key, st.INK2), clip_on=False)
                if key == "float":
                    ax.annotate(f"thr {thr}" + (" (FPPI 0)" if r["fppi"] == 0 else ""),
                                (x, 1 - r["recall"]), fontsize=7, color=st.INK2,
                                xytext=(4, 4), textcoords="offset points")
            for t, r in s.get("targets", {}).items():
                if r["fppi"] > 0:
                    ax.plot(r["fppi"], 1 - r["recall"], "s", ms=5,
                            color=colors.get(key, st.INK2))
                    if key == "float":
                        ax.annotate(f"recall {float(t):g}", (r["fppi"], 1 - r["recall"]),
                                    fontsize=7, color=st.INK2, xytext=(4, -10),
                                    textcoords="offset points")
        ax.axvspan(1e-2, 1.0, color=st.GRID, alpha=0.35, lw=0)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(1e-3, 10)
        ax.set_ylim(5e-3, 1.05)
        ax.set_xlabel("false positives per image (FPPI)")
        ax.set_ylabel("miss rate")
        ax.set_title(pnl["title"])
        ax.legend(loc="lower left")
        st.note(ax, "shaded: LAMR range; ○ repo thresholds,\n■ target recall (check half)",
                loc="lower right")
    fig.tight_layout()
    return st.save(fig, out_dir, name)


@figure("miss_fppi", "resultats", "T13.61",
        "Miss rate against false positives per image (log-log), float and integer, with the "
        "repo thresholds (0.005, 0.5) and the thresholds chosen for recall 0.95 / 0.99.",
        "build/m27/<dataset>-<split>/<net>/oppoint.json (tools/person_safety.py oppoint)")
def miss_fppi(out_dir):
    paths = sorted((ROOT / "build" / "m27").glob("*/*/oppoint.json"))
    if not paths:
        raise MissingSource("build/m27/*/*/oppoint.json absent")
    return plot_miss_fppi(load_miss_fppi(paths), out_dir)
