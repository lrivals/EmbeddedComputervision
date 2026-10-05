"""Modèle de ressources et de débit d'une architecture streaming (T9.4.1, §10.1, §10.2).

    python tools/stream_model.py                          # Tiny-YOLOv2, KV260, poids 8 bits
    python tools/stream_model.py --wbits 4 --json build/m9/stream_plan.json

Un étage matériel par conv (maxpool fusionné), chaînés en pipeline [2018-venieris#004.0] :

- **Calcul** : l'étage l déroule PE canaux de sortie et SIMD canaux d'entrée (repliement
  de type FINN, hors base) : cycles_l = H·W·K²·(C_in/SIMD)·(C_out/PE) par image, PE | C_out,
  SIMD | C_in. DSP_l = PE·SIMD / (MAC int8 par DSP) + 4·PE (requantification 32 × 31 bits).
- **Fenêtre glissante** : line buffer de (K − 1)·W·C_in mots
  [2023-montgomerie-corcoran#011.0], plus une ligne W·C_out pour un maxpool fusionné.
- **Poids** : sur la puce (K²·C_in·C_out·b/8 octets) tant que la mémoire le permet — SATAY
  garde tous ses paramètres sur la puce [2023-montgomerie-corcoran#006.0]. Sinon (hybride,
  hors base) l'étage garde **toute sa carte d'entrée** (H·W·C_in) et lit ses poids une fois
  par image en DDR, groupe PE par groupe PE (ordre « PE extérieur », T10.8). Il émet donc
  ses sorties canal par canal (CHW) : un étage suivant lui aussi en DDR les range telles
  quelles dans sa carte d'entrée ; sinon l'étage garde **sa carte de sortie** (H·W·C_out) et
  l'émet en HWC après le dernier groupe (un maxpool fusionné impose aussi la carte). Son temps
  est au moins octets de poids / octets DDR par cycle, et sa latence de remplissage est une
  image entière.
- **FIFO** entre étages (T10.8) : une ligne de sortie de l'étage producteur (W'·C_out, après
  pooling), émise d'un bloc à la fin de chaque ligne : le producteur ne bloque pas tant que
  le consommateur lit plus vite qu'une ligne par ligne. Chaîne linéaire : pas d'interblocage
  quelle que soit la profondeur, elle ne règle que le débit.
- **Allocation** : tous les poids sur la puce, puis les plus gros en DDR jusqu'à tenir dans
  le budget mémoire (cartes des étages en DDR comprises), puis
  parallélisme doublé (au diviseur suivant) sur l'étage le plus lent tant que les DSP le
  permettent (80 % des DSP, 80 % de BRAM + URAM).
- **Débit** : II = max_l cycles_l ; **latence** ≈ II + Σ_l remplissage_l, avec
  remplissage_l = cycles_l·(pad + 1 + pool)/H pour un étage à line buffer (lignes d'entrée
  attendues avant la première ligne de sortie), cycles_l pour un étage à carte entière.

Estimation hors base, à confirmer par synthèse (Vitis absent) ; comparée au moteur unique
de `tools/perf_model.py` dans `results/streaming.md`.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from perf_model import conv_layers, net_cycles  # noqa: E402
from roofline import load_board  # noqa: E402

REQUANT_DSP = 4
BUDGET = 0.8
URAM_BITS = 288 * 1024
BRAM18_BITS = 18 * 1024


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def next_div(n, cur):
    nxt = [d for d in divisors(n) if d > cur]
    return nxt[0] if nxt else None


def onchip_bytes(board):
    bits = board["bram18"] * BRAM18_BITS + board.get("uram", 0) * URAM_BITS
    return int(bits * BUDGET / 8)


def weight_bytes(d, wbits):
    return d["k"] * d["k"] * d["cin"] * d["cout"] * wbits / 8


def line_buffer_bytes(d, abits):
    lb = (d["k"] - 1) * d["w"] * d["cin"]
    if d["pool_k"]:
        lb += d["w"] * d["cout"]  # une ligne de sortie conv pour le maxpool
    return lb * abits / 8


def frame_bytes(d, abits, out_frame=True):
    """Carte d'entrée d'un étage à poids en DDR (PE extérieur), et sa carte de sortie si
    `out_frame` (étage suivant à line buffer, ou maxpool fusionné)."""
    return d["h"] * d["w"] * (d["cin"] + (d["cout"] if out_frame else 0)) * abits / 8


def out_row(d):
    """Valeurs d'une ligne de sortie (après maxpool fusionné de stride 2)."""
    w = d["w"] // 2 if d["pool_k"] and d["pool_s"] == 2 else d["w"]
    return w * d["cout"]


def fifo_depths(layers):
    """Profondeur (mots de 8 bits) du flux qui suit chaque étage, sauf le dernier (la tête)."""
    return [out_row(d) for d in layers[:-1]]


def stage_cycles(d, pe, simd, ddr_weights=False, wbits=8, ddr_bpc=math.inf):
    comp = d["h"] * d["w"] * d["k"] ** 2 * (d["cin"] // simd) * (d["cout"] // pe)
    if ddr_weights:
        comp = max(comp, math.ceil(weight_bytes(d, wbits) / ddr_bpc))
    return comp


def stage_dsp(pe, simd, macs_per_dsp=1):
    return math.ceil(pe * simd / macs_per_dsp) + REQUANT_DSP * pe


def plan(layers, board, wbits=8, abits=8):
    """Allocation (placement des poids puis parallélisme) ; rend un dict.

    `wbits` : bits des poids, commun ou {id conv: bits} (précision mixte, T10.10) ; un étage
    à poids ≤ 4 bits fait deux MAC par DSP.
    """
    mem = onchip_bytes(board)
    dsp_budget = int(board["dsp"] * BUDGET)
    wb_of = (lambda d: wbits.get(d["layer"], 8)) if isinstance(wbits, dict) else (lambda d: wbits)

    def mpd_of(d):
        return board.get("int8_macs_per_dsp", 1) * (2 if wb_of(d) <= 4 else 1)
    freq = board["freq_mhz"] * 1e6
    ddr_bpc = board["ddr_bw_gbps"] * 1e9 * board.get("ddr_efficiency", 1.0) / freq

    st = [{"layer": d["layer"], "d": d, "pe": 1, "simd": 1, "ddr": True} for d in layers]
    fifo = sum(fifo_depths(layers)) * abits / 8
    used = sum(line_buffer_bytes(d, abits) for d in layers) + fifo
    # Tous les poids sur la puce, puis les plus gros passent en DDR (l'étage garde alors ses
    # cartes d'entrée et de sortie) jusqu'à tenir dans le budget.
    def out_frame(k):
        nxt = st[k + 1] if k + 1 < len(st) else None
        return st[k]["d"]["pool_k"] > 0 or nxt is None or not nxt["ddr"]

    def total():
        return used + sum(frame_bytes(s["d"], abits, out_frame(k)) if s["ddr"]
                          else weight_bytes(s["d"], wb_of(s["d"])) for k, s in enumerate(st))

    for s in st:
        s["ddr"] = False
    for s in sorted(st, key=lambda s: -weight_bytes(s["d"], wb_of(s["d"]))):
        if total() <= mem:
            break
        s["ddr"] = True
    if total() > mem:
        raise ValueError("mémoire sur puce insuffisante même avec poids en DDR")

    def cyc(s):
        return stage_cycles(s["d"], s["pe"], s["simd"], s["ddr"], wb_of(s["d"]), ddr_bpc)

    def dsp_total():
        return sum(stage_dsp(s["pe"], s["simd"], mpd_of(s["d"])) for s in st)

    blocked = set()
    while True:
        cand = [s for s in st if s["layer"] not in blocked]
        if not cand:
            break
        s = max(cand, key=cyc)
        d = s["d"]
        best = None
        for key, n in (("simd", d["cin"]), ("pe", d["cout"])):
            nv = next_div(n, s[key])
            if nv is None:
                continue
            trial = dict(s, **{key: nv})
            mpd = mpd_of(d)
            extra = stage_dsp(trial["pe"], trial["simd"], mpd) - stage_dsp(s["pe"], s["simd"], mpd)
            if cyc(trial) < cyc(s) and (best is None or extra < best[0]):
                best = (extra, key, nv)
        if best is None or dsp_total() + best[0] > dsp_budget:
            blocked.add(s["layer"])
            continue
        s[best[1]] = best[2]

    rows = []
    for k, s in enumerate(st):
        d = s["d"]
        c = cyc(s)
        delay = c if s["ddr"] else c * (d["pad"] + 1 + (1 if d["pool_k"] else 0)) / d["h"]
        rows.append({
            "layer": s["layer"], "k": d["k"], "cin": d["cin"], "cout": d["cout"],
            "h": d["h"], "w": d["w"], "pe": s["pe"], "simd": s["simd"],
            "dsp": stage_dsp(s["pe"], s["simd"], mpd_of(d)), "cycles": c,
            "weights": "ddr" if s["ddr"] else "puce", "wbits": wb_of(d),
            "weight_bytes": weight_bytes(d, wb_of(d)),
            "buffer_bytes": (frame_bytes(d, abits, out_frame(k)) if s["ddr"]
                             else line_buffer_bytes(d, abits)),
            "out_frame": bool(s["ddr"] and out_frame(k)),
            "ddr_bytes": weight_bytes(d, wb_of(d)) if s["ddr"] else 0, "delay": delay,
        })
    ii = max(r["cycles"] for r in rows)
    lat = ii + sum(r["delay"] for r in rows)
    macs = sum(d["h"] * d["w"] * d["k"] ** 2 * d["cin"] * d["cout"] for d in layers)
    onchip = fifo + sum(r["buffer_bytes"] + (r["weight_bytes"] if r["weights"] == "puce" else 0)
                        for r in rows)
    return {
        "board": board["name"], "freq_hz": freq, "wbits": wbits, "abits": abits,
        "stages": rows, "ii_cycles": ii, "latency_cycles": lat,
        "fps": freq / ii, "latency_ms": 1e3 * lat / freq,
        "dsp": sum(r["dsp"] for r in rows), "dsp_budget": dsp_budget,
        "onchip_bytes": onchip, "onchip_budget": mem, "fifo_depths": fifo_depths(layers),
        "fifo_bytes": fifo,
        "ddr_bytes_per_frame": sum(r["ddr_bytes"] for r in rows),
        "macs": macs, "gops": 2 * macs * freq / ii / 1e9,
        "mac_efficiency": macs / (ii * sum(r["pe"] * r["simd"] for r in rows)),
    }


def markdown(p, engine_cycles):
    f = p["freq_hz"]
    lines = [
        "| conv | K | C_in → C_out | H×W | PE × SIMD | DSP | cycles | poids | tampon (Ko) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in p["stages"]:
        lines.append(f"| L{r['layer']:02d} | {r['k']} | {r['cin']} → {r['cout']} | "
                     f"{r['h']}×{r['w']} | {r['pe']} × {r['simd']} | {r['dsp']} | "
                     f"{r['cycles']:_} | {r['weights']} ({r['weight_bytes'] / 1024:_.0f} Ko) | "
                     f"{r['buffer_bytes'] / 1024:_.1f} |".replace("_", " "))
    lines += [
        "",
        f"- II = {p['ii_cycles']:_} cycles → **{p['fps']:.1f} img/s** à {f / 1e6:.0f} MHz ; "
        f"latence ≈ {p['latency_ms']:.1f} ms".replace("_", " "),
        f"- DSP {p['dsp']} / {p['dsp_budget']} (80 %) ; mémoire sur puce "
        f"{p['onchip_bytes'] / 2**20:.2f} / {p['onchip_budget'] / 2**20:.2f} Mo ; "
        f"DDR {p['ddr_bytes_per_frame'] / 2**20:.2f} Mo de poids par image",
        f"- {p['gops']:.1f} GOPS, efficacité MAC {100 * p['mac_efficiency']:.1f} %",
        f"- moteur unique (perf_model, ports 8 bits) : {engine_cycles:_} cycles → ".replace("_", " ")
        + f"{f / engine_cycles:.1f} img/s ; streaming × {engine_cycles / p['ii_cycles']:.1f} "
        "en débit",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--board", default="kv260")
    ap.add_argument("--wbits", type=int, default=8)
    ap.add_argument("--abits", type=int, default=8)
    ap.add_argument("--json", type=Path, default=None, help="plan complet (JSON)")
    args = ap.parse_args()
    layers = conv_layers(ROOT / "model" / args.net / "manifest.json")
    board = load_board(ROOT / "hw" / "boards" / f"{args.board}.yaml")
    p = plan(layers, board, args.wbits, args.abits)
    print(markdown(p, net_cycles(layers)))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(p, indent=1) + "\n")


if __name__ == "__main__":
    main()
