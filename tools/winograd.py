"""Winograd F(2×2, 3×3) against two MACs per DSP — T23.5 study → results/winograd.md.

    python tools/winograd.py                 # Tiny-YOLOv2 VOC and Tiny-YOLOv3 COCO exports

Integer-exact variant: G is multiplied by 2 so that the transformed weights stay integers,

    U = G' g G'ᵀ = 4·G g Gᵀ,   V = Bᵀ d B,   4·Y = Aᵀ (Σ_c U ⊙ V) A,

and the output is divided by 4 exactly (4·Y is a multiple of 4 by construction). The
int32 accumulator of the contract still holds: the intermediate sums may exceed 32 bits
in the worst case, but the final value fits and two's-complement wrap-around is exact.

Compute bound: the M10 kernel compute cycles (tools/perf_model.py) of each 3×3 conv are
scaled by 16/36 multiplications per output and by the 2×2 alignment of the output tile
(odd maps waste a row and a column); L00 (folded, T10.4) and the 1×1 convs do not change.
Time with loads, first order and the same estimator for every option: Σ over the convs of
max(compute, weight load, input load, output stage) from the perf_model counters, the
weight load of a Winograd 3×3 conv ×16·12/72 when the weights are transformed offline.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import perf_model as pm  # noqa: E402

BT = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]], dtype=np.int64)
G2 = np.array([[2, 0, 0], [1, 1, 1], [1, -1, 1], [0, 0, 2]], dtype=np.int64)  # 2·G
AT = np.array([[1, 1, 1, 0], [0, 1, -1, -1]], dtype=np.int64)
QMAX = 127
NETS = ("tiny-yolov2-voc", "tiny-yolov3-coco")


def transform_weights(g):
    """U = G' g G'ᵀ for (…, 3, 3) int filters → (…, 4, 4)."""
    return G2 @ np.asarray(g, dtype=np.int64) @ G2.T


def transform_input(d):
    """V = Bᵀ d B for (…, 4, 4) int tiles."""
    return BT @ np.asarray(d, dtype=np.int64) @ BT.T


def conv_tile(g, d):
    """2×2 outputs of a 3×3 conv (valid) over cin: g (cin, 3, 3), d (cin, 4, 4) → (2, 2)."""
    m = (transform_weights(g) * transform_input(d)).sum(axis=0)
    y4 = AT @ m @ AT.T
    assert (y4 % 4 == 0).all()
    return y4 // 4


def direct_tile(g, d):
    out = np.zeros((2, 2), dtype=np.int64)
    for r in range(2):
        for c in range(2):
            out[r, c] = (np.asarray(g, np.int64) * np.asarray(d, np.int64)[:, r:r + 3, c:c + 3]).sum()
    return out


def bits(v):
    """Signed bits needed for |v| ≤ max."""
    return int(np.max(np.abs(v))).bit_length() + 1


def bounds(qmax=QMAX):
    """Worst-case magnitudes: transformed weight, transformed input, product."""
    u = int(np.abs(G2).sum(axis=1).max() ** 2 * qmax)
    v = int(np.abs(BT).sum(axis=1).max() ** 2 * qmax)
    return u, v, u * v


WBYTES = 16 * 12 / 72  # transformed 3×3 weights (16 of 12 bits) against 9 bytes


def first_order(kc, comp, wload=1.0):
    return max(comp, kc["load_w"] * wload, kc["load_in"], kc["store"])


def layer_study(d, kc):
    """(direct compute cycles, Winograd compute cycles) of one conv at the M10 tiles."""
    comp = kc["compute"]
    if d["k"] != 3 or pm.layer_tiling(d, fold=pm.KERNEL["fold"],
                                      tile_pool=pm.KERNEL["tile_pool"])[2]:
        return comp, comp  # 1×1 or folded L00: no Winograd
    R = d["h"] + 2 * d["pad"] - 2
    C = d["w"] + 2 * d["pad"] - 2
    align = (-(-R // 2) * 2) * (-(-C // 2) * 2) / (R * C)
    return comp, comp * 16 / 36 * align


def net_study(net):
    from yolo.io.export import load_model

    model_dir = ROOT / "model" / net
    layers = pm.conv_layers(model_dir / "manifest.json")
    qm, _ = load_model(model_dir)
    macs = {"3x3": 0, "1x1": 0, "L00": 0}
    direct = wino = wino_off = wino_fly = pack_t = 0.0
    umax = 0
    w_direct = w_wino = 0
    for d in layers:
        kc = pm.kernel_cycles(d)
        a, b = layer_study(d, kc)
        direct += a
        wino += b
        wino_off += first_order(kc, b, WBYTES if b < a else 1.0)
        wino_fly += first_order(kc, b)
        kp = pm.kernel_cycles(d, tm=2 * pm.TILES["tm"])
        pack_t += first_order(kp, kp["compute"])
        m = pm.macs(d)
        if d["layer"] == 0:
            macs["L00"] += m
        elif d["k"] == 3:
            macs["3x3"] += m
            u = transform_weights(qm.convs[d["layer"]].qW)
            umax = max(umax, int(np.abs(u).max()))
            w_direct += d["cin"] * d["cout"] * 9
            w_wino += d["cin"] * d["cout"] * 16
        else:
            macs["1x1"] += m
    pack = sum(pm.kernel_cycles(d, tm=2 * pm.TILES["tm"])["compute"] for d in layers)
    return {"net": net, "macs": macs, "direct_ms": direct / pm.FREQ_HZ * 1e3,
            "wino_ms": wino / pm.FREQ_HZ * 1e3, "pack_ms": pack / pm.FREQ_HZ * 1e3,
            "umax": umax, "w_direct": w_direct, "w_wino": w_wino,
            "wino_off_ms": wino_off / pm.FREQ_HZ * 1e3,
            "wino_fly_ms": wino_fly / pm.FREQ_HZ * 1e3, "pack_t_ms": pack_t / pm.FREQ_HZ * 1e3}


def self_check(rng, n=2000, cin=8):
    """Winograd == direct on random int8 tiles and on the extreme ones."""
    for _ in range(n):
        g = rng.integers(-QMAX, QMAX + 1, (cin, 3, 3))
        d = rng.integers(-QMAX, QMAX + 1, (cin, 4, 4))
        assert (conv_tile(g, d) == direct_tile(g, d)).all()
    for s in (-1, 1):
        g = np.full((cin, 3, 3), QMAX) * s
        d = np.full((cin, 4, 4), QMAX)
        assert (conv_tile(g, d) == direct_tile(g, d)).all()


def report(rows):
    u, v, p = bounds()
    wbuf = pm.TILES["tm"] * pm.TILES["tn"]
    lines = [
        "# Winograd F(2×2, 3×3) against two MACs per DSP (T23.5)",
        "",
        "Generated by `python tools/winograd.py`; method and assumptions in the script header. "
        "Packing: [dsp_packing.md](dsp_packing.md). Base: 2023-zhai uses Winograd on FPGA "
        "[2023-zhai#004.0].",
        "",
        "![Winograd against packing](figures/resultats/winograd.png)",
        "",
        "Figure: `python -m tools.figures winograd` (T13.56), which also covers YOLOv4-tiny.",
        "",
        "## Integer-exact variant",
        "",
        "- G′ = 2G = [[2, 0, 0], [1, 1, 1], [1, −1, 1], [0, 0, 2]]: U = G′ g G′ᵀ = 4·G g Gᵀ "
        "stays an integer; the output 4·Y = Aᵀ(Σ U ⊙ V)A is divided by 4 exactly (checked on "
        "random and extreme int8 tiles).",
        f"- Transformed weights: |U| ≤ 3·3·127 = {u} → **{bits(u)} bits** signed "
        f"(exports: max |U| = " + ", ".join(f"{r['umax']} ({r['net']})" for r in rows) + ").",
        f"- Transformed inputs: |V| ≤ 2·2·127 = {v} → **{bits(v)} bits** signed.",
        f"- Product U·V ≤ {p} → {bits(p)} bits: one DSP48E2 (27 × 18) per product. Two such "
        f"products in one DSP would need A ≥ {bits(u)} + {bits(p)} bits > 27: **packing and "
        "Winograd do not combine** with 8-bit DSP inputs.",
        "- Accumulator: the final 4·Y is bounded by 4 × the direct sum, so int32 with "
        "wrap-around stays exact even when intermediate sums exceed 32 bits.",
        "",
        "## Multiplications, compute bound, weights",
        "",
        "Compute bound at the M10 tiles (Tm 32, Tn 24, 768 multipliers). Winograd on the 3×3 "
        "convs except L00 (folded); packing = Tm 64 on the same DSPs (perf_model, T23.3).",
        "",
        "| Network | MACs 3×3 / 1×1 / L00 (G) | compute bound: direct / Winograd / packing (ms) "
        "| with loads: Winograd offline / on the fly / packing (ms) "
        "| 3×3 weight bytes, direct → Winograd |",
        "|---|---|---|---|---|",
    ]
    for r in rows:
        m = r["macs"]
        wb = bits(u)
        lines.append(
            f"| {r['net']} | {m['3x3'] / 1e9:.2f} / {m['1x1'] / 1e9:.2f} / {m['L00'] / 1e9:.2f} "
            f"| {r['direct_ms']:.1f} / {r['wino_ms']:.1f} / {r['pack_ms']:.1f} "
            f"| {r['wino_off_ms']:.1f} / {r['wino_fly_ms']:.1f} / {r['pack_t_ms']:.1f} | "
            f"{r['w_direct'] / 1e6:.2f} MB → {r['w_wino'] * wb / 8 / 1e6:.2f} MB |")
    lines += [
        "",
        "## Resources and bandwidth",
        "",
        f"- `w_buf`: 16 transformed weights of {bits(u)} bits instead of 9 of 8 bits, i.e. "
        f"×{16 * bits(u) / 72:.2f}. In registers (ping-pong) at Tm·Tn = {wbuf}: "
        f"{2 * wbuf * 16 * bits(u)} FF against {2 * wbuf * 72} today, more than the "
        "234,240 of the XCK26. The weights would have to move to BRAM/LUTRAM, or the tiles "
        "shrink. Packing at Tm 64 is also beyond the 80 % budget (221,184 FF); its feasible "
        "point is Tn 32 (147,456 FF, [roofline.md](roofline.md)).",
        "- The first-order estimator is optimistic: for packing at Tm 64, the exact model "
        "(perf_model, tile-level overlap) gives 24.7 ms on Tiny-YOLOv2, not 19.8.",
        f"- DDR weight traffic of the 3×3 convs ×{16 * bits(u) / 72:.2f} if the weights are "
        "transformed offline (the weight-load-bound layers get worse). Transforming them on "
        "the fly keeps the traffic but adds one 4×4 transform per (too, tii) pair.",
        "- Inputs: same traffic; one Bᵀ d B transform (32 additions on 10 bits) per lane and "
        "per 2×2 output tile, and the output transform (24 additions) per output channel.",
        "- Odd maps (13 × 13 from L13 on) waste a row and a column of each 2×2 tile (counted "
        "in the table).",
        "",
    ]
    wins = [r["net"] for r in rows if min(r["wino_off_ms"], r["wino_fly_ms"]) < r["pack_t_ms"]]
    cb = "; ".join(f"{r['net']}: {r['wino_ms']:.1f} against {r['pack_ms']:.1f} ms"
                   for r in rows)
    lt = "; ".join(f"{r['net']}: {r['wino_off_ms']:.1f} (offline) or {r['wino_fly_ms']:.1f} "
                   f"(on the fly) against {r['pack_t_ms']:.1f} ms" for r in rows)
    lines += [
        "## Decision",
        "",
        "Winograd removes 2.25× of the multiplications of the 3×3 convs only; packing doubles "
        "the products of every conv, 1×1 included, on the same PE and schedule (T23.2, "
        f"C-sim == golden). On the compute bound alone, Winograd is slightly ahead ({cb}). "
        "With the loads, it is not: a Winograd PE consumes its weights 2.25× faster, and "
        f"transformed weights are 2.67× wider ({lt}). On top of that come a new PE, the "
        "transforms, and a `w_buf` larger than the flip-flops of the board.",
        "",
        ("**Winograd is not implemented**: it does not beat packing once the loads are "
         "counted, and the two cannot be combined with 8-bit DSP inputs." if not wins else
         f"Winograd beats packing on {', '.join(wins)}: to be implemented (T23.5)."),
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "winograd.md")
    ap.add_argument("--net", nargs="*", default=list(NETS))
    args = ap.parse_args()
    self_check(np.random.default_rng(0))
    rows = []
    for net in args.net:
        if not (ROOT / "model" / net / "manifest.json").exists():
            print(f"{net}: no export (make export), skipped")
            continue
        rows.append(net_study(net))
    text = report(rows)
    args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
