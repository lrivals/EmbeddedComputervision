"""Rapport de l'accélérateur HLS → results/hls_report.md (T6.5, §10.4).

    python tools/hls_report.py                 # carte kv260
    python tools/hls_report.py --board zybo-z7-20

Sources, toutes optionnelles (une section absente est signalée « à produire ») :
- estimation C-sim par couche : `build/hls/cycles_conv.csv` (`make hls-cycles`, testbench
  `hls/tb/tb_conv.cpp`, compteurs `accel::sim_cycles`) ;
- synthèse : `hls/proj_<carte>_synth/sol/syn/report/*_csynth.xml` (`make hls-synth`) —
  LUT, FF, DSP, BRAM, URAM, période estimée, II des boucles pipelinées, DSP de l'étage de
  sortie (requantification) ;
- co-simulation : `hls/proj_<carte>_cosim/sol/sim/report/yolo_conv_cosim.rpt`
  (`make hls-cosim`) — statut et latence min / moyenne / max par appel du noyau.
"""

import argparse
import csv
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIPELINED_LOOPS = ("mac", "load_in", "load_w", "requant", "prepool", "pool")


def read_tcl_config(board):
    cfg = {}
    for line in (ROOT / "hls" / "configs" / f"{board}.tcl").read_text().splitlines():
        m = re.match(r"\s*set\s+(\w+)\s+(\S+)", line)
        if m:
            cfg[m.group(1)] = m.group(2)
    return cfg


def read_cycles(path):
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def text(node, path, default="—"):
    e = node.find(path)
    return e.text.strip() if e is not None and e.text else default


def read_synth(report_dir):
    top = report_dir / "csynth.xml"
    if not top.exists():
        return None
    root = ET.parse(top).getroot()
    res = root.find("AreaEstimates/Resources")
    avail = root.find("AreaEstimates/AvailableResources")
    out = {
        "period": text(root, "PerformanceEstimates/SummaryOfTimingAnalysis/EstimatedClockPeriod"),
        "target": text(root, "UserAssignments/TargetClockPeriod"),
        "resources": {r.tag: (r.text, text(avail, r.tag)) for r in res} if res is not None else {},
        "loops": [],
        "store_dsp": None,
    }
    for xml in sorted(report_dir.glob("*_csynth.xml")):
        mod = ET.parse(xml).getroot()
        name = xml.stem.removesuffix("_csynth")
        if "store_tile" in name:
            out["store_dsp"] = text(mod, "AreaEstimates/Resources/DSP")
        loops = mod.find("PerformanceEstimates/SummaryOfLoopLatency")
        if loops is None:
            continue
        for loop in loops.iter():
            if any(loop.tag.startswith(p) for p in PIPELINED_LOOPS):
                out["loops"].append((name, loop.tag, text(loop, "PipelineII"),
                                     text(loop, "PipelineDepth")))
    return out


def read_cosim(path):
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 5 and cells[0].lower() in ("verilog", "vhdl"):
            return {"rtl": cells[0], "status": cells[1], "min": cells[2], "avg": cells[3],
                    "max": cells[4]}
    return {"status": "rapport illisible"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", default="kv260")
    ap.add_argument("--cycles", type=Path, default=ROOT / "build" / "hls" / "cycles_conv.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "hls_report.md")
    args = ap.parse_args()

    cfg = read_tcl_config(args.board)
    freq = 1e3 / float(cfg["CLOCK_NS"])  # MHz
    tm, tn = int(cfg["TM"]), int(cfg["TN"])
    rows = read_cycles(args.cycles)
    synth = read_synth(ROOT / "hls" / f"proj_{args.board}_synth" / "sol" / "syn" / "report")
    cosim = read_cosim(ROOT / "hls" / f"proj_{args.board}_cosim" / "sol" / "sim" / "report"
                       / "yolo_conv_cosim.rpt")

    md = [
        "# Accélérateur HLS — rapport (T6.5)",
        "",
        f"Généré par `python tools/hls_report.py --board {args.board}` (`make hls-report`). "
        f"Carte `{args.board}` : part `{cfg['PART']}`, horloge {freq:.0f} MHz, "
        f"tuiles Tm = {tm}, Tn = {tn}, Tr = {cfg['TR']}, Tc = {cfg['TC']} "
        f"([ADR 0003](../docs/adr/0003-choix-carte.md)).",
        "",
        "## Synthèse",
        "",
    ]
    if synth is None:
        md += ["À produire : `make hls-synth` (Vitis HLS non installé sur la machine de "
               "développement).", ""]
    else:
        md += [f"Période estimée {synth['period']} ns pour une cible de {synth['target']} ns.",
               "", "| Ressource | Utilisée | Disponible |", "|---|---|---|"]
        md += [f"| {k} | {u} | {a} |" for k, (u, a) in synth["resources"].items()]
        md += ["", f"DSP de l'étage de sortie (requantification 32 × 31 bits) : "
               f"{synth['store_dsp'] or '—'}.", "",
               "| Module | Boucle | II | Profondeur |", "|---|---|---|---|"]
        md += [f"| {m} | `{lp}` | {ii} | {d} |" for m, lp, ii, d in synth["loops"]]
        md.append("")

    md += ["## Co-simulation RTL", ""]
    if cosim is None:
        md += ["À produire : `make hls-cosim` (une image complète de Tiny-YOLOv2, chaque "
               "couche comparée au golden).", ""]
    else:
        md += ["| RTL | Statut | Latence min | moyenne | max (cycles par appel) |",
               "|---|---|---|---|---|",
               f"| {cosim.get('rtl', '—')} | {cosim['status']} | {cosim.get('min', '—')} | "
               f"{cosim.get('avg', '—')} | {cosim.get('max', '—')} |", ""]

    md += ["## Cycles par couche (estimation C-sim)", ""]
    if not rows:
        md += ["À produire : `make hls-cycles`.", ""]
    else:
        md += [
            "Compteurs du noyau en C-sim (`accel::sim_cycles`, II = 1, profondeurs de pipeline "
            "ignorées). *Calcul* = K²·Tr·Tc par (tuile, ti) : la borne du toit de calcul ; "
            "*ping-pong* = chargement de ti + 1 recouvert par le calcul de ti, stockage de la "
            "tuile k − 1 recouvert par la tuile k ; *séquentiel* = sans recouvrement. Les "
            "chargements lisent un octet par cycle (ports m_axi de 8 bits).",
            "",
        ]
        for net in dict.fromkeys(r["net"] for r in rows):
            sel = [r for r in rows if r["net"] == net]
            md += [f"### {net}", "",
                   "| Couche | K | Cin → Cout | Entrée | MMAC | calcul | ping-pong | séquentiel "
                   "| ms (ping-pong) | efficacité MAC |",
                   "|---|---|---|---|---|---|---|---|---|---|"]
            tot = {"macs": 0, "compute": 0, "overlapped": 0, "sequential": 0}
            for r in sel:
                macs, comp, ovl, seq = (int(r[k]) for k in ("macs", "compute", "overlapped",
                                                            "sequential"))
                for k, v in zip(tot, (macs, comp, ovl, seq)):
                    tot[k] += v
                md.append(
                    f"| L{int(r['layer']):02d} | {r['k']} | {r['cin']} → {r['cout']} | "
                    f"{r['h']}×{r['w']} | {macs / 1e6:.1f} | {comp:,} | {ovl:,} | {seq:,} | "
                    f"{ovl / (freq * 1e3):.2f} | {macs / (ovl * tm * tn):.1%} |")
            md += [f"| **total** | | | | {tot['macs'] / 1e6:.0f} | {tot['compute']:,} | "
                   f"{tot['overlapped']:,} | {tot['sequential']:,} | "
                   f"**{tot['overlapped'] / (freq * 1e3):.1f}** | "
                   f"{tot['macs'] / (tot['overlapped'] * tm * tn):.1%} |", "",
                   f"Borne calcul seule : {tot['compute'] / (freq * 1e3):.1f} ms ; "
                   f"le reste est le temps de chargement non recouvert.", ""]

    if rows:
        md += [
            "## Lecture",
            "",
            "- Le noyau est **limité par les chargements**, pas par le calcul : avec des ports "
            "de 8 bits, charger `in_buf` (Tn·IR·IC octets) et `w_buf` (Tm·Tn·K² octets) prend "
            "plus de cycles que les K²·Tr·Tc cycles de calcul d'une ti, même en ping-pong.",
            "- Pistes (M8/M9) : ports m_axi larges (64-128 bits) avec poids réordonnés par le "
            "driver dans l'ordre des tuiles ; réutilisation de `in_buf` entre tuiles `to` "
            "(boucle to la plus interne, entrée inchangée) ; Tn adapté à la couche 0 (cin = 3).",
            "",
        ]
    args.out.write_text("\n".join(md))
    print(f"→ {args.out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
