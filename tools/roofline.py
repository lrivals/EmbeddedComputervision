"""Explorateur roofline : tuiles (Tm, Tn, Tr, Tc) par carte candidate (T5.6, §10.2).

    python tools/roofline.py                       # toutes les cartes de hw/boards/
    python tools/roofline.py --boards hw/boards/kv260.yaml --net tiny-yolov2-voc

Modèle du moteur unique couche par couche de 2015-zhang, tel que le golden C++
(`cpp/golden/include/golden/conv.hpp`) l'exécute, pour chaque conv (M = cout, N = cin,
R × C = sortie avant pooling, K, S = 1) :

- cycles = ⌈M/Tm⌉ ⌈N/Tn⌉ ⌈R/Tr⌉ ⌈C/Tc⌉ · K² Tr Tc (boucle interne II = 1, Tm × Tn MAC par
  cycle) ; toit de calcul = 2 · MACs / cycles · f ;
- trafic DDR (octets, int8) : B_in = Tn (S Tr + K − S)(S Tc + K − S) et B_w = Tm Tn K² lus
  ⌈M/Tm⌉ ⌈N/Tn⌉ ⌈R/Tr⌉ ⌈C/Tc⌉ fois, B_out = Tm Tr Tc écrit ⌈M/Tm⌉ ⌈R/Tr⌉ ⌈C/Tc⌉ fois ;
  CTC = opérations / octets ;
- double tampon : temps de la couche = max(cycles / f, octets / BW effective) ; performance
  atteignable = min(toit de calcul, CTC × BW) [2015-zhang] ;
- ressources : DSP = Tm Tn / (MAC int8 par DSP) + 4 Tm (requantification 32 × 31 bits) ;
  BRAM18 en ping-pong (×2) : `in_buf` en Tn bancs de 8 bits, `out_buf` en Tm bancs de 32 bits
  (un banc occupe au moins une BRAM18) ; `w_buf` (Tm Tn K² octets, entièrement partitionné)
  en registres. Points retenus : ≤ 80 % des DSP et des BRAM18.

Hypothèses hors base : les formules B_in, B_w, B_out sont celles du §10.2 (relecture de
`2015-zhang#014.0` non faite, `bin/pdb` absent du dépôt), l'efficacité DDR et l'horloge sont
dans `hw/boards/<carte>.yaml`. Sorties : `results/roofline.md`, `results/roofline_<carte>.png`.
"""

import argparse
import itertools
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from yolo.models.specs import NETWORKS, infer_shapes  # noqa: E402

K_MAX, S = 3, 1
BRAM18_BITS = 18 * 1024
BRAM18_WIDTH = 36
UTIL = 0.8
REQUANT_DSP = 4  # produit acc · M0 (32 × 31 bits) en DSP48 par canal de sortie
# PE à décalages (ACC_WMODE_POW2, T9.2.4) : deux décaleurs 7:1 sur 8 bits et un additionneur
# 15 bits par MAC, ordre de grandeur hors base (results/req_yolo.md). Estimation, pas mesure.
POW2_LUT_PER_MAC = 40
TM_TN = (1, 2, 4, 8, 12, 16, 24, 32, 48, 64)
TR_TC = (2, 4, 7, 8, 13, 16, 26, 52)


def conv_layers(net):
    """[(M, N, R, C, K)] des convs du réseau (sortie avant pooling, stride 1)."""
    out = []
    for layer, (shape_in, shape_out) in zip(net["layers"], infer_shapes(net)):
        if layer["type"] == "conv":
            out.append((shape_out[0], shape_in[0], shape_out[1], shape_out[2], layer["k"]))
    return np.array(out, dtype=np.int64)


def total_macs(layers):
    m, n, r, c, k = layers.T
    return int((m * n * r * c * k * k).sum())


def buffer_sizes(tm, tn, tr, tc, k=K_MAX, s=S):
    """§10.2 : (B_in, B_w, B_out) en éléments."""
    return tn * (s * tr + k - s) * (s * tc + k - s), tm * tn * k * k, tm * tr * tc


def bram18(tm, tn, tr, tc):
    def banks(n, width, depth):
        return n * max(math.ceil(width / BRAM18_WIDTH), math.ceil(width * depth / BRAM18_BITS))

    depth_in = (S * tr + K_MAX - S) * (S * tc + K_MAX - S)
    return 2 * (banks(tn, 8, depth_in) + banks(tm, 32, tr * tc))


def dsp(tm, tn, macs_per_dsp=1):
    return math.ceil(tm * tn / macs_per_dsp) + REQUANT_DSP * tm


def lut_pow2(tm, tn):
    """LUT estimées des MAC de la PE à décalages (hors contrôle, chargeurs et requantification)."""
    return POW2_LUT_PER_MAC * tm * tn


def evaluate(layers, tm, tn, tr, tc, freq_hz, bw_bytes):
    """Réseau entier : dict(cycles, bytes, ops, time_s, comp_roof, ctc, attainable)."""
    m, n, r, c, k = layers.T
    trips = -(-m // tm) * -(-n // tn) * -(-r // tr) * -(-c // tc)
    cycles = trips * k * k * tr * tc
    b_in = tn * (S * tr + k - S) * (S * tc + k - S)
    b_w = tm * tn * k * k
    b_out = tm * tr * tc
    nbytes = trips * (b_in + b_w) + (-(-m // tm) * -(-r // tr) * -(-c // tc)) * b_out
    ops = 2 * m * n * r * c * k * k
    time_s = np.maximum(cycles / freq_hz, nbytes / bw_bytes)
    tot_ops, tot_cycles, tot_bytes = ops.sum(), cycles.sum(), nbytes.sum()
    comp_roof = tot_ops / tot_cycles * freq_hz
    ctc = tot_ops / tot_bytes
    return {
        "cycles": int(tot_cycles), "bytes": int(tot_bytes), "ops": int(tot_ops),
        "time_s": float(time_s.sum()), "comp_roof": float(comp_roof), "ctc": float(ctc),
        "attainable": float(tot_ops / time_s.sum()),
    }


def layer_points(convs, board, before, after, tiles=(32, 24, 13, 13)):
    """Roofline par couche (T13.22) : [(id, CTC, GOPS avant, GOPS après)].

    `convs` : dicts de `perf_model.conv_layers` ; CTC du modèle §10.2 aux tuiles `tiles` ;
    `before`, `after` : d → cycles de la couche (par exemple noyau M6 et noyau actuel).
    """
    out = []
    for d in convs:
        r = d["h"] + 2 * d["pad"] - d["k"] + 1
        c = d["w"] + 2 * d["pad"] - d["k"] + 1
        arr = np.array([(d["cout"], d["cin"], r, c, d["k"])], dtype=np.int64)
        e = evaluate(arr, *tiles, board["freq_hz"], board["bw_bytes"])
        ops = 2 * total_macs(arr)
        out.append((d["layer"], e["ctc"], ops * board["freq_hz"] / before(d) / 1e9,
                    ops * board["freq_hz"] / after(d) / 1e9))
    return out


def ideal_time_s(layers, tm, tn, freq_hz):
    """Ordre de grandeur du §10.2 : MACs / (Tm Tn) cycles, efficacité 100 %."""
    return total_macs(layers) / (tm * tn) / freq_hz


def load_board(path):
    import yaml

    b = yaml.safe_load(Path(path).read_text())
    b["file"] = Path(path).stem
    b["freq_hz"] = b["freq_mhz"] * 1e6
    b["bw_bytes"] = b["ddr_bw_gbps"] * 1e9 * b["ddr_efficiency"]
    return b


def explore(layers, board):
    """Tous les points (Tm, Tn, Tr, Tc) qui tiennent sur la carte, triés par temps."""
    points = []
    for tm, tn, tr, tc in itertools.product(TM_TN, TM_TN, TR_TC, TR_TC):
        d, b = dsp(tm, tn, board["int8_macs_per_dsp"]), bram18(tm, tn, tr, tc)
        if d > UTIL * board["dsp"] or b > UTIL * board["bram18"]:
            continue
        e = evaluate(layers, tm, tn, tr, tc, board["freq_hz"], board["bw_bytes"])
        points.append({"tiles": (tm, tn, tr, tc), "dsp": d, "bram18": b, **e})
    points.sort(key=lambda p: (p["time_s"], p["dsp"], p["bram18"]))
    return points


def plot(points, board, net_name, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ctc = np.array([p["ctc"] for p in points])
    roof = np.array([p["comp_roof"] for p in points]) / 1e9
    ax.scatter(ctc, roof, s=6, alpha=0.35, color="#7a8ca3", label="points (toit de calcul)")
    x = np.logspace(np.log10(ctc.min() / 2), np.log10(ctc.max() * 2), 100)
    ax.plot(x, board["bw_bytes"] * x / 1e9, color="#c0504d",
            label=f"toit DDR {board['bw_bytes'] / 1e9:.1f} Go/s")
    best = points[0]
    peak = 2 * best["tiles"][0] * best["tiles"][1] * board["freq_hz"] / 1e9
    ax.axhline(peak, color="#2a6f3e", ls="--", lw=1,
               label=f"toit de calcul Tm·Tn du meilleur point : {peak:.0f} GOPS")
    ax.scatter([best["ctc"]], [best["attainable"] / 1e9], s=60, color="#2a6f3e", zorder=3,
               label="meilleur Tm,Tn,Tr,Tc = {},{},{},{}".format(*best["tiles"]))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_ylim(roof.min() / 2, roof.max() * 3)
    ax.set_xlabel("CTC (opérations / octet DDR)")
    ax.set_ylabel("GOPS")
    ax.set_title(f"{board['name']} — {net_name} à {board['freq_mhz']} MHz")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(boards, nets, out_dir):
    ref = conv_layers(NETWORKS["tiny-yolov2-voc"])
    lines = [
        "# Roofline des cartes candidates (T5.6)",
        "",
        "Généré par `python tools/roofline.py` (`make roofline`) ; modèle et hypothèses dans "
        "l'en-tête du script. Ressources : `hw/boards/*.yaml`. Points retenus : ≤ 80 % des DSP "
        "et des BRAM18 ; meilleur point = temps minimal pour Tiny-YOLOv2.",
        "",
        "Ordre de grandeur du §10.2 : Tm = Tn = 16, 200 MHz, efficacité 100 % → "
        f"{total_macs(ref) / 1e9:.2f} GMAC / 256 = "
        f"{ideal_time_s(ref, 16, 16, 200e6) * 1e3:.1f} ms (Tiny-YOLOv2).",
        "",
        "| Carte | DSP | BRAM18 | BW eff. (Go/s) | f (MHz) | Tm, Tn, Tr, Tc | DSP util. | "
        "BRAM18 util. | CTC (op/o) | toit calcul (GOPS) | atteint (GOPS) | "
        + " | ".join(f"{n} (ms)" for n in nets) + " |",
        "|---|---|---|---|---|---|---|---|---|---|---|" + "---|" * len(nets),
    ]
    for board in boards:
        layers = {n: conv_layers(NETWORKS[n]) for n in nets}
        points = explore(layers[nets[0]], board)
        if not points:
            lines.append(f"| {board['name']} | aucune tuile ne tient |" + " |" * 10)
            continue
        best = points[0]
        times = [evaluate(layers[n], *best["tiles"], board["freq_hz"], board["bw_bytes"])
                 ["time_s"] * 1e3 for n in nets]
        lines.append(
            f"| {board['name']} | {board['dsp']} | {board['bram18']} | "
            f"{board['bw_bytes'] / 1e9:.2f} | {board['freq_mhz']} | "
            "{}, {}, {}, {} | ".format(*best["tiles"])
            + f"{best['dsp']} | {best['bram18']} | {best['ctc']:.1f} | "
            f"{best['comp_roof'] / 1e9:.1f} | {best['attainable'] / 1e9:.1f} | "
            + " | ".join(f"{t:.1f}" for t in times) + " |")
        png = out_dir / f"roofline_{board['file']}.png"
        plot(points, board, nets[0], png)
    lines += ["", "Graphiques : " + ", ".join(f"`roofline_{b['file']}.png`" for b in boards),
              "", "Roofline par couche, avant et après M10 : "
              "`results/figures/resultats/roofline_couches_kv260.png` "
              "(`python -m tools.figures roofline_couches`, T13.22).", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--boards", nargs="*", type=Path,
                    default=sorted((ROOT / "hw" / "boards").glob("*.yaml")))
    ap.add_argument("--net", nargs="*", default=["tiny-yolov2-voc", "tiny-yolov3-voc"],
                    choices=sorted(NETWORKS))
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    args = ap.parse_args()
    if not args.boards:
        sys.exit("aucune carte dans hw/boards/")
    boards = [load_board(p) for p in args.boards]
    args.out.mkdir(parents=True, exist_ok=True)
    text = report(boards, args.net, args.out)
    (args.out / "roofline.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
