"""Modèle de cycles du noyau HLS et pistes d'optimisation chiffrées (T8.1, T8.3, M10).

    python tools/perf_model.py                       # scénarios, Tiny-YOLOv2 et v3
    python tools/perf_model.py --check build/hls/cycles_conv.csv   # == compteurs C-sim

Réplique en Python de `count_cycles` (hls/kernels/conv_pe.cpp) : même parcours des tuiles
(row → col → to, tuiles partielles, maxpool fusionné), mêmes hypothèses (II = 1, profondeurs
de pipeline ignorées, deux bundles de chargement en parallèle, ping-pong ti / tuile). Les
cycles sont **identiques** à ceux de `make hls-cycles` (vérifié par `--check` et
`python/tests/test_perf_model.py`), à un détail près : `tb_conv` écrit la carte avant pooling
de **toute** conv poolée (`prepool_all`), le programme du driver seulement quand le manifest la
demande (L08 de Tiny-YOLOv3). Les scénarios suivent le programme réel.

Paramètres (défauts de `layer_cycles` = noyau M6, ports 8 bits ; `KERNEL` = noyau actuel,
lu dans hls/kernels/accel_config.hpp) :
- `width` : octets par mot sur les ports m_axi (1 = 8 bits ; 8 = 64 bits, T10.1). Lecture
  ligne par ligne : chaque (voie, ligne) de `in_buf` coûte `cdiv(IC + width − 1, width)` mots
  (assez pour tout alignement de la ligne en DDR) ; une tuile de poids, réordonnée par le
  driver en bloc contigu, `cdiv(Tm·n·K², width)` mots (moitié moins d'octets pour une conv à
  poids 4 bits, champ `wbits` du manifest, T10.10) ; une ligne de sortie de n octets,
  `cdiv(n + width − 1, width)` mots ;
- `trim` : ne charger que les canaux d'entrée valides de la dernière tuile ti (T10.3) ;
- `requant` : canaux requantifiés par cycle dans l'étage de sortie (T10.2) ;
- `fold` : pliage de la première conv (cin·k ≤ Tn) : les voies portent (canal, ligne du
  noyau), la conv devient 1 × k (T10.4) ;
- `tile_pool` : Tr = Tc des convs suivies d'un maxpool de stride 2 (None : Tr, Tc ; T10.4).
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
K_MAX = 3
FREQ_HZ = 200e6
NETS = ("tiny-yolov2-voc", "tiny-yolov3-coco")


def kernel_config(path=ROOT / "hls" / "kernels" / "accel_config.hpp"):
    """Valeurs par défaut des `#define ACC_*` du noyau (tuiles KV260, ports, requant…)."""
    found = dict(re.findall(r"#define (ACC_\w+) (\d+)", Path(path).read_text()))
    v = {k: int(x) for k, x in found.items()}
    return {"tm": v["ACC_TM"], "tn": v["ACC_TN"], "tr": v["ACC_TR"], "tc": v["ACC_TC"],
            "width": v["ACC_WORD_BYTES"], "trim": bool(v["ACC_TRIM"]),
            "requant": v["ACC_RQ"], "fold": bool(v["ACC_FOLD"]),
            "tile_pool": v["ACC_TILE_POOL"] or None}


KERNEL = kernel_config()
TILES = {k: KERNEL[k] for k in ("tm", "tn", "tr", "tc")}  # KV260 (ADR 0003)
M6 = dict(width=1, trim=False, requant=1, fold=False, tile_pool=None)


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
                 "prepool": layer.get("prepool_out") is not None,
                 "wbits": layer.get("wbits", 8)}
            out.append(d)
        if "out_shape" in layer:  # entrée de la couche suivante (route comprise)
            prev = tuple(layer["out_shape"][1:])
    return out


def layer_tiling(d, tn=TILES["tn"], tr=TILES["tr"], tc=TILES["tc"], fold=False,
                 tile_pool=None):
    """Choix du driver (`driver::conv_desc`) : tuile (tr, tc) et pliage de la conv."""
    if tile_pool and d["pool_k"] > 0 and d["pool_s"] == 2:
        tr = tc = tile_pool
    folded = fold and d["k"] > 1 and d["cin"] * d["k"] <= tn
    return tr, tc, folded


def layer_cycles(d, tm=TILES["tm"], tn=TILES["tn"], tr=TILES["tr"], tc=TILES["tc"],
                 width=1, trim=False, requant=1, prepool_all=False, fold=False,
                 tile_pool=None):
    """Cycles d'une conv : dict load_in, load_w, compute, store, sequential, overlapped."""
    tr, tc, folded = layer_tiling(d, tn, tr, tc, fold, tile_pool)
    k, cin, cout = d["k"], d["cin"], d["cout"]
    kh = 1 if folded else k  # lignes du noyau parcourues par le calcul
    lanes = cin * k if folded else cin
    ir, ic = tr + K_MAX - 1, tc + K_MAX - 1
    R = d["h"] + 2 * d["pad"] - k + 1
    C = d["w"] + 2 * d["pad"] - k + 1
    pooled = d["pool_k"] > 0
    pk, ps = (d["pool_k"], d["pool_s"]) if pooled else (1, 1)
    Rp = R if ps == 1 else (R - pk) // ps + 1
    Cp = C if ps == 1 else (C - pk) // ps + 1
    Pr, Pc = (tr - pk) // ps + 1, (tc - pk) // ps + 1
    n_r, n_c, n_m = _cdiv(Rp, Pr), _cdiv(Cp, Pc), _cdiv(cout, tm)
    nti = _cdiv(lanes, tn)

    def words(n):  # une ligne de n octets, alignement quelconque
        return _cdiv(n + width - 1, width)

    # Chargements par ti : voies × lignes × mots par ligne, et un bloc de poids contigu.
    tns = [min(tn, lanes - t * tn) if trim else tn for t in range(nti)]
    # Poids 4 bits (T10.10) : deux par octet, Tm pair.
    wdiv = 2 if d.get("wbits", 8) == 4 else 1
    lds = [(n * ir * words(ic), _cdiv(tm * n * kh * k // wdiv, width)) for n in tns]
    comp = kh * k * tr * tc
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
                pre = pooled and (d["prepool"] or prepool_all)
                own = min(tr_n, Pr * ps) * words(min(tc_n, Pc * ps)) if pre else 0
                # Étage de sortie : passe de requantification Tr·Tc par groupe de `requant`
                # canaux, puis écritures canal par canal, ligne par ligne.
                st = _cdiv(tm_n, requant) * tr * tc + tm_n * (own + np_r * words(np_c))
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


def kernel_cycles(d, **kw):
    """Cycles d'une conv avec le noyau actuel (`KERNEL`)."""
    return layer_cycles(d, **{**KERNEL, **kw})


P64 = dict(M6, width=8)
SCENARIOS = [
    ("M6 : ports 8 bits", M6),
    ("canaux valides seulement (trim)", dict(M6, trim=True)),
    ("ports 64 bits", P64),
    ("ports 64 bits + trim", dict(P64, trim=True)),
    ("ports 128 bits + trim", dict(M6, width=16, trim=True)),
    ("ports 64 bits + trim + requant ×8 (+28 DSP)", dict(P64, trim=True, requant=8)),
    ("… + pliage de L00", dict(P64, trim=True, requant=8, fold=True)),
    ("… + pliage + tuiles 12 (convs poolées)",
     dict(P64, trim=True, requant=8, fold=True, tile_pool=12)),
    ("… + pliage + tuiles 14 (convs poolées)",
     dict(P64, trim=True, requant=8, fold=True, tile_pool=14)),
]


def scenario_table(net, freq_hz=FREQ_HZ, tiles=TILES):
    """Lignes (nom, cycles, ms, FPS accélérateur, GOPS, efficacité MAC)."""
    layers = conv_layers(ROOT / "model" / net / "manifest.json")
    total_macs = sum(macs(d) for d in layers)
    ideal = total_macs / (tiles["tm"] * tiles["tn"])
    rows = []

    def row(name, cyc):
        t = cyc / freq_hz
        rows.append((name, cyc, 1e3 * t, 1 / t, 2 * total_macs / t / 1e9, ideal / cyc))

    for name, kw in SCENARIOS:
        row(name, net_cycles(layers, **tiles, **kw))
    row(f"**noyau actuel** ({kernel_label()})",
        sum(kernel_cycles(d)["overlapped"] for d in layers))
    row("borne calcul M6 (chargements gratuits)",
        sum(layer_cycles(d, **tiles)["compute"] for d in layers))
    row("borne calcul du noyau actuel", sum(kernel_cycles(d)["compute"] for d in layers))
    return total_macs, rows


def kernel_label(k=KERNEL):
    parts = [f"ports {8 * k['width']} bits"]
    if k["trim"]:
        parts.append("trim")
    if k["requant"] > 1:
        parts.append(f"requant ×{k['requant']}")
    if k["fold"]:
        parts.append("pliage L00")
    if k["tile_pool"]:
        parts.append(f"tuiles {k['tile_pool']} poolées")
    return ", ".join(parts)


def check(path):
    """Compare aux compteurs C-sim de `make hls-cycles` ; rend le nombre d'écarts."""
    bad = 0
    want = list(csv.DictReader(Path(path).open()))
    layers = {net: {d["layer"]: d for d in conv_layers(ROOT / "model" / net / "manifest.json")}
              for net in {r["net"] for r in want}}
    for r in want:
        # Configuration du noyau qui a produit la ligne (colonnes écrites par tb_conv).
        kw = dict(tm=int(r["tm"]), tn=int(r["tn"]), tr=int(r["tr"]), tc=int(r["tc"]),
                  width=int(r["width"]), trim=bool(int(r["trim"])), requant=int(r["rq"]),
                  fold=bool(int(r["fold"])))
        # tb_conv écrit la carte avant pooling de toute conv poolée (test du chemin prépool)
        got = layer_cycles(layers[r["net"]][int(r["layer"])], prepool_all=True, **kw)
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


def manifest_summary(path, freq_hz=FREQ_HZ):
    """Noyau actuel sur un export quelconque (élagué, précision mixte : T10.10, T10.11)."""
    layers = conv_layers(path)
    total = sum(macs(d) for d in layers)
    cyc = sum(kernel_cycles(d)["overlapped"] for d in layers)
    t = cyc / freq_hz
    return {"manifest": str(path), "macs": total, "cycles": cyc, "ms": 1e3 * t, "fps": 1 / t,
            "gops": 2 * total / t / 1e9,
            "mac_efficiency": total / (TILES["tm"] * TILES["tn"]) / cyc}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", type=Path, default=None, help="cycles_conv.csv de make hls-cycles")
    ap.add_argument("--manifest", type=Path, action="append", default=[],
                    help="cycles du noyau actuel pour cet export (répétable, JSON en sortie)")
    ap.add_argument("--no-figures", action="store_true",
                    help="pas de figure dans build/perf/figures/ (M13 ; aussi YOLO_FIGURES=0)")
    args = ap.parse_args()
    if args.check:
        sys.exit(1 if check(args.check) else 0)
    if args.manifest:
        print(json.dumps([manifest_summary(m) for m in args.manifest], indent=1))
        return
    print(markdown())
    if not args.no_figures:
        sys.path.insert(0, str(ROOT))
        from tools.figures.auto import after_run

        after_run("perf-model", ROOT / "build" / "perf")


if __name__ == "__main__":
    main()
