"""Report of the faults in the HLS buffers (T25.3, T25.7), from hls/tb/tb_fault.

    build/hls/tb_fault --net tiny-yolov2-voc --net tiny-yolov3-coco --flips 20 --seed 1 \\
        --csv build/m25/hls/campaign.csv
    build/hls/tb_fault … --flips 3 --csv build/m25/hls/dump.csv --dump build/m25/hls/dump
    python tools/hls_faults.py                     # Markdown tables for results/faults.md

Per buffer (in, w, out, q, cs): flips, share that changed the layer output, share detected
by the ABFT lane, silent corruptions (changed, not detected), detections with the output
unchanged (masked by the requantization or the pooling, or a flip of the checksum lane).
With the dumps: for each weight or input flip that changed the output, the output bytes it changed against those of the **persistent** fault of the same element (a
corrupted weight in DDR, or a corrupted activation of the previous layer: the T25.0 sites
`weight` and `act`), which reaches every tile: the share masked by tile reloading.
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

BUFS = ("in", "w", "out", "q", "cs")
OUT = ROOT / "build" / "m25" / "hls"


def campaign_table(rows):
    agg = defaultdict(lambda: defaultdict(int))
    for r in rows:
        for key in (("all", r["buf"]), (r["net"], r["buf"])):
            a = agg[key]
            ch, det = int(r["changed"]) > 0, int(r["abft"]) > 0
            a["n"] += 1
            a["changed"] += ch
            a["detected"] += det
            a["det_changed"] += ch and det
            a["silent"] += ch and not det
            a["false"] += det and not ch
    nets = sorted({r["net"] for r in rows})
    lines = ["| Network | Buffer | Flips | Output changed | Detected (ABFT) | Changed and "
             "detected | Silent corruption | Detected, output unchanged |",
             "|---|---|---|---|---|---|---|---|"]
    for net in nets + ["all"]:
        for b in BUFS:
            a = agg[(net, b)]
            if not a["n"]:
                continue
            pc = lambda k: f"{100 * a[k] / a['n']:.0f} %"  # noqa: E731
            det = (f"{100 * a['det_changed'] / a['changed']:.0f} %" if a["changed"] else "—")
            lines.append(f"| {net} | {b} | {a['n']} | {pc('changed')} | {pc('detected')} | "
                         f"{det} | {pc('silent')} | {pc('false')} |")
    return "\n".join(lines)


def _persistent(x, qc, r, pool, tn, fold):
    """Output bytes changed by the persistent fault of the element of a `w` / `in` flip."""
    from yolo.quant import faults as F
    from yolo.quant import int_layers as il

    k = qc.qW.shape[-1]
    pad = (k - 1) // 2
    W0, x0 = qc.qW.astype(np.int64), x.astype(np.int64)
    W, x = W0.copy(), x0.copy()
    a, b, c, d, bit = (int(r[n]) for n in ("a", "b", "c", "d", "bit"))
    if r["buf"] == "w":
        ch, rr = F.hls_lane(int(r["ti"]) * tn + b, k, fold)
        i = rr if fold else c
        W[int(r["to"]) + a, ch, i, d] = F._flip8(W[int(r["to"]) + a, ch, i, d], bit)
    else:
        ch, ki = F.hls_lane(int(r["ti"]) * tn + a, k, fold)
        ir, ic = int(r["row"]) - pad + b + (ki or 0), int(r["col"]) - pad + c
        if not (0 <= ir < x.shape[1] and 0 <= ic < x.shape[2]):
            return None  # a padding zero: no element in DDR
        x[ch, ir, ic] = F._flip8(x[ch, ir, ic], bit)

    def out(W_, x_):
        acc = il.conv_acc(x_[None], W_, qc.qb, wrap=True)
        y = il.conv_out(acc, qc.M0, qc.n, qc.act, qc.qmax)
        pk, ps = pool
        return y if pk == 1 and ps == 1 else (y[:, :, ::ps, ::ps] if pk == 1
                                               else il.maxpool_int(y, pk, ps))

    return int((out(W, x) != out(W0, x0)).sum())


def masking_table(rows, dump):
    from yolo.io.export import load_model

    tn = int(next(ln for ln in (ROOT / "hls/kernels/accel_config.hpp").read_text().splitlines()
                  if ln.startswith("#define ACC_TN")).split()[-1])
    cache, res = {}, defaultdict(list)
    for r in rows:
        if r["buf"] not in ("w", "in") or int(r["changed"]) == 0:
            continue
        net, layer = r["net"], int(r["layer"])
        if net not in cache:
            d = ROOT / "model" / net
            cache[net] = (load_model(d)[0], json.loads((d / "manifest.json").read_text()))
        qm, man = cache[net]
        e = next(x for x in man["layers"] if x["id"] == layer)
        pool = e.get("fused_pool")
        pk_ps = (pool["k"], pool["s"]) if pool else ((1, 2) if e["s"] == 2 else (1, 1))
        dumps = ROOT / "model" / net / "dumps" / r["image"]
        x = np.load(dumps / ("input.npy" if layer == 0 else f"L{layer - 1:02d}.npy"))[0]
        qc = qm.convs[layer]
        k = qc.qW.shape[-1]
        p = _persistent(x, qc, r, pk_ps, tn, k > 1 and qc.qW.shape[1] * k <= tn)
        if p:
            res[r["buf"]].append((int(r["changed"]), p))
    lines = ["| Buffer | Flips compared | Bytes changed, buffer flip (mean) | Bytes changed, "
             "persistent fault (mean) | Share masked by tile reloading |",
             "|---|---|---|---|---|"]
    for b in ("w", "in"):
        v = np.array(res[b], np.float64)
        if not len(v):
            continue
        lines.append(f"| {b} | {len(v)} | {v[:, 0].mean():.1f} | {v[:, 1].mean():.1f} | "
                     f"{100 * (1 - v[:, 0].sum() / v[:, 1].sum()):.0f} % |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dir", type=Path, default=OUT)
    args = ap.parse_args()
    camp = args.dir / "campaign.csv"
    if camp.exists():
        rows = [r for r in csv.DictReader(camp.open()) if r.get("abft") not in (None, "")]
        print(campaign_table(rows))
        print()
    dump = args.dir / "dump.csv"
    if dump.exists():
        print(masking_table(list(csv.DictReader(dump.open())), args.dir / "dump"))


if __name__ == "__main__":
    main()
