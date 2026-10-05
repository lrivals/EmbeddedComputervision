"""Performance, puissance et ressources de ce travail → results/benchmarks.csv (T8.1, §10.4).

    python tools/bench_report.py                                  # projection seule
    python tools/bench_report.py --net tiny-yolov2-voc \\
        --times 'build/m8/tiny-yolov2-voc/board/times_*.csv' \\
        --layer-csv build/m8/tiny-yolov2-voc/board/layers.csv \\
        --vivado build/vivado/kv260 --board-power 6.1 --idle-power 4.9

Mesures (protocole : results/protocole.md) : `times` de `yolo_bench --images` sur la carte
(≥ 1 000 images), temps par couche de `run_compare --csv`, `utilization.rpt` / `power.rpt`
de `make vivado-build`, puissance de la carte lue par `yolo_bench --power` (INA260 du SOM).
Sans mesures, seule la **projection** du modèle de cycles (`tools/perf_model.py`, égal aux
compteurs C-sim) est écrite, marquée comme telle.

Définitions : FPS = 1 000 / latence moyenne de bout en bout (images traitées l'une après
l'autre, sans recouvrement des étages) ; GOPS = 2 × MACs / temps accélérateur ; efficacité =
cycles théoriques (MACs / (Tm·Tn)) / cycles mesurés (temps accélérateur × f).
Écrit aussi `results/mesures.md` (étages, moyenne et p99, efficacité par couche).
"""

import argparse
import csv
import glob
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402
from perf_model import FREQ_HZ, TILES, conv_layers, layer_cycles, macs  # noqa: E402

COLUMNS = ["travail", "modele", "fpga", "format", "fps", "latence_ms", "gops", "puissance_w",
           "perimetre_puissance", "lut", "dsp", "bram", "frequence_mhz", "map_voc", "source"]
OURS = "ce travail"
MODEL_NAME = {"tiny-yolov2-voc": "Tiny-YOLOv2 VOC 416", "tiny-yolov3-coco": "Tiny-YOLOv3 COCO 416"}
FPGA = "Kria KV260 (XCK26)"
FORMAT = "INT8 (W8A8, par canal)"
MIN_IMAGES = 1000

# Travaux publiés, §10.4 de la spécification (`bin/pdb bench`) ; n/r = non rapporté → vide.
PUBLISHED = [
    ["2023-zhai", "YOLOv3-tiny élagué, 1 module", "Zynq XC7Z035", "INT16", "91.65", "",
     "67.91", "12.51", "unspec.", "38228", "144", "132.5", "", "", "Table 8 (high)"],
    ["2023-zhai", "YOLOv3-tiny élagué, 2 modules", "Zynq XC7Z035", "INT16", "168.72", "",
     "124.01", "15.18", "unspec.", "91108", "294", "263", "", "", "Table 9 (high)"],
    ["2024-zhang", "YOLOv2-Tiny 416, lot 1", "Kintex-7 325T", "INT8", "", "14.49", "406",
     "12.3", "board (puce estimée Vivado 7,8 W)", "134900", "687", "551", "", "",
     "Table 3, §5.3 (medium)"],
    ["2024-zhang", "YOLOv3-Tiny 416, lot 1", "Kintex-7 325T", "INT8", "", "15.2", "401",
     "12.3", "board", "136500", "687", "553", "", "", "Tables 4-5 (high)"],
    ["2023-montgomerie-corcoran", "YOLOv3-Tiny 416", "VCU110", "W8A16", "", "", "418.9",
     "15.4", "unspec.", "", "", "", "", "", "Table III (medium)"],
    ["2019-ding", "Tiny-YOLOv2, ADMM hétérogène", "Virtex-7 690T", "FFT + puissances de 2",
     "314.2", "", "", "21", "unspec.", "", "", "", "", "", "Tables 1-2 (high)"],
    ["2026-fata", "YOLOv3-tiny élagué à 70 % (DPU Vitis AI)", "Kria KV260 (DPU)", "INT8 QAT",
     "24.3", "", "60.18", "2.13", "unspec.", "", "", "", "", "", "Tables 7-10 (medium)"],
    ["2025-kim", "YOLOv2 complet (prototype)", "Zybo Z7-20", "INT16", "0.083", "12000", "",
     "", "", "", "", "", "", "", "§10.4 : ≈ 12 s par image, pas une référence de performance"],
]


def fmt(x, nd=2):
    return "" if x is None else f"{x:.{nd}f}"


def p99(v):
    """Quantile « nearest rank », comme yolo_bench."""
    v = np.sort(np.asarray(v, dtype=np.float64))
    return float(v[max(1, int(np.ceil(0.99 * len(v)))) - 1]) if len(v) else 0.0


def read_times(pattern):
    rows = []
    for p in sorted(glob.glob(pattern)):
        rows += list(csv.DictReader(open(p)))
    cols = ["pre_ms", "load_ms", "acc_ms", "post_ms"]
    t = {c: np.array([float(r[c]) for r in rows]) for c in cols}
    t["total_ms"] = sum(t[c] for c in cols) if rows else np.array([])
    return t, len(rows)


def read_utilization(path):
    """LUT, DSP, BRAM (tuiles BRAM36) de `report_utilization` (UltraScale+)."""
    text = Path(path).read_text()

    def used(label):
        m = re.search(r"\|\s*" + re.escape(label) + r"\s*\|\s*([\d.]+)\s*\|", text)
        return m.group(1) if m else ""
    return {"lut": used("CLB LUTs") or used("Slice LUTs"), "dsp": used("DSPs"),
            "bram": used("Block RAM Tile")}


def read_power(path):
    """Puissance puce estimée par Vivado (`report_power`) : totale, dynamique, PS."""
    text = Path(path).read_text()

    def watts(label):
        m = re.search(r"\|\s*" + re.escape(label) + r"\s*\|\s*([\d.]+)", text)
        return float(m.group(1)) if m else None
    return {"total": watts("Total On-Chip Power (W)"), "dynamic": watts("Dynamic (W)"),
            "static": watts("Device Static (W)"), "ps": watts("PS8")}


def per_layer(layers, layer_csv, net):
    """Efficacité par couche : cycles théoriques / cycles mesurés (moyenne sur les images)."""
    acc = {}
    for r in csv.DictReader(open(layer_csv)):
        if r["net"] == net:
            acc.setdefault(int(r["layer"]), []).append(float(r["seconds"]))
    out = []
    for d in layers:
        if d["layer"] not in acc:
            continue
        t = float(np.mean(acc[d["layer"]]))
        th = macs(d) / (TILES["tm"] * TILES["tn"])
        pred = layer_cycles(d)["overlapped"]
        out.append((d["layer"], macs(d), 1e3 * t, t * FREQ_HZ, pred, th / (t * FREQ_HZ)))
    return out


def update_csv(path, ours):
    """Lignes publiées + lignes « ce travail » remplacées ; le reste est conservé."""
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    keep = [r for r in rows if not r["travail"].startswith(OURS)
            and r["travail"] not in {p[0] for p in PUBLISHED}]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLUMNS)
        for p in PUBLISHED:
            w.writerow(p)
        for r in keep:
            w.writerow([r[c] for c in COLUMNS])
        for r in ours:
            w.writerow([r.get(c, "") for c in COLUMNS])


def read_map(net):
    """mAP du stade FPGA (sinon entier) de results/map_stades.md, si le réseau correspond."""
    p = ROOT / "results" / "map_stades.md"
    if not p.exists() or net not in p.read_text():
        return ""
    vals = {}
    for line in p.read_text().splitlines():
        m = re.match(r"\| (.+?) \| \*\*([\d.]+)\*\* \|", line)
        if m:
            vals[m.group(1)] = m.group(2)
    for key in ("FPGA, KV260", "FPGA, C-sim", "Entier Python"):
        for name, v in vals.items():
            if name.startswith(key):
                return v
    return ""


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc", choices=sorted(MODEL_NAME))
    ap.add_argument("--times", default=None, help="motif glob des times*.csv de yolo_bench")
    ap.add_argument("--layer-csv", type=Path, default=None, help="run_compare --csv")
    ap.add_argument("--vivado", type=Path, default=None, help="dossier des .rpt de vivado-build")
    ap.add_argument("--board-power", type=float, default=None, help="W en charge (yolo_bench)")
    ap.add_argument("--idle-power", type=float, default=None, help="W au repos (yolo_bench)")
    ap.add_argument("--csv", type=Path, default=ROOT / "results" / "benchmarks.csv")
    ap.add_argument("--md", type=Path, default=ROOT / "results" / "mesures.md")
    args = ap.parse_args()

    md = ["# Mesures de ce travail (T8.1)", "",
          "Généré par `python tools/bench_report.py` (`make bench-report`) ; protocole : "
          "[protocole.md](protocole.md). Kria KV260, tuiles Tm = 32, Tn = 24, Tr = Tc = 13, "
          f"{FREQ_HZ / 1e6:.0f} MHz.", ""]

    # Projection (toujours) : modèle de cycles == compteurs C-sim, chaque réseau.
    ours = []
    md += ["## Projection (modèle de cycles, accélérateur seul)", "",
           "`tools/perf_model.py`, ports m_axi de 8 bits, II = 1, profondeurs de pipeline et "
           "pilotage ARM ignorés : borne basse du temps accélérateur mesurable.", "",
           "| Réseau | GMAC | Mcycles | ms | img/s | GOPS | efficacité |", "|---|---|---|---|---|---|---|"]
    for net in MODEL_NAME:
        layers = conv_layers(ROOT / "model" / net / "manifest.json")
        n_macs = sum(macs(d) for d in layers)
        cyc = sum(layer_cycles(d)["overlapped"] for d in layers)
        t = cyc / FREQ_HZ
        eff = n_macs / (TILES["tm"] * TILES["tn"]) / cyc
        md.append(f"| {net} | {n_macs / 1e9:.3f} | {cyc / 1e6:.2f} | {1e3 * t:.1f} | "
                  f"{1 / t:.2f} | {2 * n_macs / t / 1e9:.1f} | {eff * 100:.1f} % |")
        ours.append({"travail": f"{OURS} (projection C-sim)", "modele": MODEL_NAME[net],
                     "fpga": FPGA, "format": FORMAT, "fps": fmt(1 / t), "latence_ms": fmt(1e3 * t, 1),
                     "gops": fmt(2 * n_macs / t / 1e9, 1), "perimetre_puissance": "à mesurer",
                     "frequence_mhz": f"{FREQ_HZ / 1e6:.0f}",
                     "map_voc": read_map(net) if net.endswith("voc") else "",
                     "source": "tools/perf_model.py (accélérateur seul, ports 8 bits) ; "
                               "ressources après synthèse"})
    md.append("")

    # Mesures sur la carte, si fournies.
    layers = conv_layers(ROOT / "model" / args.net / "manifest.json")
    n_macs = sum(macs(d) for d in layers)
    pattern = args.times or str(ROOT / "build" / "m8" / args.net / "board" / "times_*.csv")
    t, n = read_times(pattern)
    res = read_utilization(args.vivado / "utilization.rpt") if args.vivado else {}
    pw = read_power(args.vivado / "power.rpt") if args.vivado else {}
    md += ["## Mesures sur la carte", ""]
    if n == 0:
        md += [f"À mesurer : aucun fichier `{Path(pattern).relative_to(ROOT)}` "
               "(KV260 non disponible). Procédure : [protocole.md](protocole.md).", ""]
    else:
        if n < MIN_IMAGES:
            print(f"attention : {n} images < {MIN_IMAGES} (critère T8.1)")
        md += [f"{args.net}, {n} images ({pattern}).", "",
               "| Étage | moyenne (ms) | p99 (ms) |", "|---|---|---|"]
        for c, name in [("pre_ms", "prétraitement (décodage, stretch, quantification)"),
                        ("load_ms", "copie de l'entrée en DDR"), ("acc_ms", "accélérateur"),
                        ("post_ms", "post-traitement (décodage + NMS)"),
                        ("total_ms", "**bout en bout**")]:
            md.append(f"| {name} | {t[c].mean():.3f} | {p99(t[c]):.3f} |")
        acc_s = t["acc_ms"].mean() / 1e3
        eff = n_macs / (TILES["tm"] * TILES["tn"]) / (acc_s * FREQ_HZ)
        gops = 2 * n_macs / acc_s / 1e9
        fps = 1e3 / t["total_ms"].mean()
        md += ["", f"FPS (bout en bout, séquentiel) **{fps:.2f}** ; GOPS (accélérateur) "
               f"**{gops:.1f}** ; efficacité **{eff * 100:.1f} %**.", ""]
        perim, watts = [], ""
        if args.board_power is not None:
            watts = fmt(args.board_power)
            perim.append("carte : SOM mesuré (INA260)"
                         + (f", repos {args.idle_power:.2f} W" if args.idle_power else ""))
        if pw.get("total") is not None:
            perim.append(f"puce estimée Vivado {pw['total']:.2f} W")
            watts = watts or fmt(pw["total"])
        ours.append({"travail": OURS, "modele": MODEL_NAME[args.net], "fpga": FPGA,
                     "format": FORMAT, "fps": fmt(fps), "latence_ms": fmt(t["total_ms"].mean()),
                     "gops": fmt(gops, 1), "puissance_w": watts,
                     "perimetre_puissance": " ; ".join(perim) or "à mesurer",
                     **res, "frequence_mhz": f"{FREQ_HZ / 1e6:.0f}",
                     "map_voc": read_map(args.net) if args.net.endswith("voc") else "",
                     "source": f"yolo_bench, {n} images, results/protocole.md"})
        if pw:
            md += ["Puissance puce (Vivado `report_power`) : "
                   + ", ".join(f"{k} {v:.3f} W" for k, v in pw.items() if v is not None), ""]
        if res:
            md += [f"Ressources (implémentation) : LUT {res['lut']}, DSP {res['dsp']}, "
                   f"BRAM36 {res['bram']}.", ""]
    if args.layer_csv:
        md += ["## Par couche", "", "| Couche | MMAC | ms | cycles mesurés | cycles modèle | "
               "efficacité |", "|---|---|---|---|---|---|"]
        for lid, mm, ms, cyc, pred, e in per_layer(layers, args.layer_csv, args.net):
            md.append(f"| L{lid:02d} | {mm / 1e6:.1f} | {ms:.3f} | {cyc:,.0f} | {pred:,} | "
                      f"{e * 100:.1f} % |")
        md.append("")

    update_csv(args.csv, ours)
    args.md.write_text("\n".join(md))
    print("\n".join(md))
    print(f"→ {args.csv}, {args.md}")


if __name__ == "__main__":
    main()
