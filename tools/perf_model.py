"""Modèle de cycles du noyau HLS et pistes d'optimisation chiffrées (T8.1, T8.3).

    python tools/perf_model.py                       # scénarios, Tiny-YOLOv2 et v3
    python tools/perf_model.py --check build/hls/cycles_conv.csv   # == compteurs C-sim

Réplique en Python de `count_cycles` (hls/kernels/conv_pe.cpp) : même parcours des tuiles
(row → col → to, tuiles partielles, maxpool fusionné), mêmes hypothèses (II = 1, profondeurs
de pipeline ignorées, deux bundles de chargement en parallèle, ping-pong ti / tuile). Avec
`width = 1` octet/cycle et `trim = False`, les cycles sont **identiques** à ceux de
`make hls-cycles` (vérifié par `--check` et `python/tests/test_perf_model.py`), à un détail
près : `tb_conv` écrit la carte avant pooling de **toute** conv poolée (`prepool_all`), le
programme du driver seulement quand le manifest la demande (L08 de Tiny-YOLOv3). Les
scénarios suivent le programme réel.

Paramètres des pistes (hors base, à confirmer par synthèse et co-sim) :
- `width` : octets transférés par cycle sur les ports m_axi (1 = ports 8 bits actuels ;
  8 = 64 bits ; 16 = 128 bits, poids réordonnés par le driver dans l'ordre des tuiles) ;
  s'applique aux chargements et aux écritures de sortie ;
- `trim` : ne charger que les canaux d'entrée valides de la dernière tuile ti (Tn = 24 pour
  cin = 3 en L00 : 21 canaux chargés pour rien) ;
- `requant` : canaux requantifiés par cycle dans l'étage de sortie (1 aujourd'hui : un seul
  multiplieur 32 × 31 bits, 4 DSP, partagé par les Tm canaux ; `output_stage.hpp`).
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TILES = {"tm": 32, "tn": 24, "tr": 13, "tc": 13}  # KV260 (ADR 0003, hls/configs/kv260.tcl)
K_MAX = 3
FREQ_HZ = 200e6
NETS = ("tiny-yolov2-voc", "tiny-yolov3-coco")


def _cdiv(a, b):
    return -(-a // b)


def conv_layers(manifest):
    """Convs du manifest : k, cin, cout, entrée (h, w), pad, pooling fusionné, prépool."""
    m = json.loads(Path(manifest).read_text())
    out = []
    prev = tuple(m["input"]["shape"][2:])  # (N, C, H, W)
    for layer in m["layers"]:
        t = layer["type"]
        if t == "conv":
            src = prev
            pool = layer.get("fused_pool")
            d = {"layer": layer["id"], "k": layer["k"], "pad": layer["pad"], "cin": layer["cin"],
                 "cout": layer["cout"], "h": src[0], "w": src[1],
                 "pool_k": pool["k"] if pool else 0, "pool_s": pool["s"] if pool else 0,
                 "prepool": layer.get("prepool_out") is not None}
            out.append(d)
        if "out_shape" in layer:  # entrée de la couche suivante (route comprise)
            prev = tuple(layer["out_shape"][1:])
    return out


def layer_cycles(d, tm=TILES["tm"], tn=TILES["tn"], tr=TILES["tr"], tc=TILES["tc"],
                 width=1, trim=False, requant=1, prepool_all=False):
    """Cycles d'une conv : dict load_in, load_w, compute, store, sequential, overlapped."""
    k, cin, cout = d["k"], d["cin"], d["cout"]
    kk = k * k
    ir, ic = tr + K_MAX - 1, tc + K_MAX - 1
    R = d["h"] + 2 * d["pad"] - k + 1
    C = d["w"] + 2 * d["pad"] - k + 1
    pooled = d["pool_k"] > 0
    pk, ps = (d["pool_k"], d["pool_s"]) if pooled else (1, 1)
    Rp = R if ps == 1 else (R - pk) // ps + 1
    Cp = C if ps == 1 else (C - pk) // ps + 1
    Pr, Pc = (tr - pk) // ps + 1, (tc - pk) // ps + 1
    n_r, n_c, n_m = _cdiv(Rp, Pr), _cdiv(Cp, Pc), _cdiv(cout, tm)
    nti = _cdiv(cin, tn)

    # Chargements par ti : Tn·IR·IC et Tm·Tn·K² octets (ou les seuls canaux valides).
    tns = [min(tn, cin - t * tn) if trim else tn for t in range(nti)]
    s = dict(load_in=0, load_w=0, compute=0, store=0, sequential=0, overlapped=0)
    prev_store = 0
    for ir_ in range(n_r):
        for ic_ in range(n_c):
            for im in range(n_m):
                row, col = ir_ * Pr * ps, ic_ * Pc * ps
                tr_n = min((Pr - 1) * ps + pk, R - row)
                tc_n = min((Pc - 1) * ps + pk, C - col)
                np_r = min(Pr, Rp - ir_ * Pr)
                np_c = min(Pc, Cp - ic_ * Pc)
                tm_n = min(tm, cout - im * tm)
                comp = kk * tr * tc
                lds = [(_cdiv(n * ir * ic, width), _cdiv(tm * n * kk, width)) for n in tns]
                own = (min(tr_n, Pr * ps) * min(tc_n, Pc * ps)
                       if pooled and (d["prepool"] or prepool_all) else 0)
                # Étage de sortie : passe de requantification TR·TC par groupe de `requant`
                # canaux (un multiplieur aujourd'hui), puis écritures canal par canal.
                st = _cdiv(tm_n, requant) * tr * tc + tm_n * _cdiv(own + np_r * np_c, width)
                s["load_in"] += sum(a for a, _ in lds)
                s["load_w"] += sum(b for _, b in lds)
                s["compute"] += nti * comp
                s["store"] += st
                s["sequential"] += sum(a + b + comp for a, b in lds) + st
                ld = [max(a, b) for a, b in lds]  # deux bundles m_axi en parallèle
                tile = ld[0] + sum(max(x, comp) for x in ld[1:]) + comp
                first = ir_ == 0 and ic_ == 0 and im == 0
                s["overlapped"] += tile if first else max(tile, prev_store)
                prev_store = st
    s["overlapped"] += prev_store  # stockage de la dernière tuile
    return s


def macs(d):
    R = d["h"] + 2 * d["pad"] - d["k"] + 1
    C = d["w"] + 2 * d["pad"] - d["k"] + 1
    return d["k"] ** 2 * d["cin"] * d["cout"] * R * C


def net_cycles(layers, **kw):
    return sum(layer_cycles(d, **kw)["overlapped"] for d in layers)


SCENARIOS = [
    ("actuel (ports 8 bits)", dict(width=1)),
    ("canaux valides seulement (trim)", dict(width=1, trim=True)),
    ("ports 64 bits", dict(width=8)),
    ("ports 64 bits + trim", dict(width=8, trim=True)),
    ("ports 128 bits + trim", dict(width=16, trim=True)),
    ("ports 128 bits + trim + requant ×8 (+28 DSP)", dict(width=16, trim=True, requant=8)),
]


def scenario_table(net, freq_hz=FREQ_HZ, tiles=TILES):
    """Lignes (nom, cycles, ms, FPS accélérateur, GOPS, efficacité MAC)."""
    layers = conv_layers(ROOT / "model" / net / "manifest.json")
    total_macs = sum(macs(d) for d in layers)
    ideal = total_macs / (tiles["tm"] * tiles["tn"])
    rows = []
    for name, kw in SCENARIOS:
        cyc = net_cycles(layers, **tiles, **kw)
        t = cyc / freq_hz
        rows.append((name, cyc, 1e3 * t, 1 / t, 2 * total_macs / t / 1e9, ideal / cyc))
    comp = sum(layer_cycles(d, **tiles)["compute"] for d in layers)
    t = comp / freq_hz
    rows.append(("borne calcul (chargements gratuits)", comp, 1e3 * t, 1 / t,
                 2 * total_macs / t / 1e9, ideal / comp))
    return total_macs, rows


def check(path):
    """Compare aux compteurs C-sim de `make hls-cycles` ; rend le nombre d'écarts."""
    bad = 0
    want = list(csv.DictReader(Path(path).open()))
    layers = {net: {d["layer"]: d for d in conv_layers(ROOT / "model" / net / "manifest.json")}
              for net in {r["net"] for r in want}}
    for r in want:
        # tb_conv écrit la carte avant pooling de toute conv poolée (test du chemin prépool)
        got = layer_cycles(layers[r["net"]][int(r["layer"])], prepool_all=True)
        for key, v in got.items():
            if int(r[key]) != v:
                print(f"{r['net']} L{int(r['layer']):02d} {key} : {v} au lieu de {r[key]}")
                bad += 1
    print(f"{len(want)} couches comparées, {bad} écarts")
    return bad


def markdown(nets=NETS):
    out = []
    for net in nets:
        total, rows = scenario_table(net)
        out += [f"### {net} ({total / 1e9:.2f} GMAC, Tm·Tn = {TILES['tm'] * TILES['tn']}, "
                f"{FREQ_HZ / 1e6:.0f} MHz)", "",
                "| Scénario | Mcycles | ms (accélérateur) | img/s | GOPS | efficacité MAC |",
                "|---|---|---|---|---|---|"]
        out += [f"| {n} | {c / 1e6:.2f} | {ms:.1f} | {fps:.1f} | {g:.1f} | {e * 100:.1f} % |"
                for n, c, ms, fps, g, e in rows]
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=Path, default=None, help="cycles_conv.csv de make hls-cycles")
    args = ap.parse_args()
    if args.check:
        sys.exit(1 if check(args.check) else 0)
    print(markdown())


if __name__ == "__main__":
    main()
