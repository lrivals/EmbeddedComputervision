"""Model ↔ circuit equivalence matrix, per network and per layer (T28.0).

    python tools/equivalence.py run             # N images per net → results/equivalence.json
    python tools/equivalence.py run --n 50 --nets tiny-yolov2-voc
    python tools/equivalence.py report          # matrix section of results/equivalence.md
    python tools/equivalence.py --check         # CI: committed dumps still reproduce, doc up to date

Stages and level reached between consecutive stages (CoDANN II §3.6.1):

- float → fake-quant: **proximity**, max |float − fake-quant| per layer in LSB of the layer
  scale s_y, and **statistical**, the mAP gap of the full test sets (cited from
  `results/map_*.md`, not recomputed: hours of evaluation);
- fake-quant → integer: share of int8 values where round(fake-quant / s_y) ≠ integer, max
  gap in LSB (float accumulation order and round-half cases);
- integer (Python) → golden C++ → HLS C-sim: **pointwise**, byte-identical on every layer.

The integer stage is the exported model (`model/<net>/`); the fake-quant stage is rebuilt from
`build/quant/<net>/calib.json` (the export has no weight scales), and `run` checks that this
rebuilt model gives the same integers as the export. Images: the first N of the test split of
each net's dataset (VOC2007 test, COCO val2017), letterbox as in `export_model.py`. Dumps go to
`build/equivalence/model/<net>/dumps/` (manifest and blobs linked from `model/<net>/`), where
`golden_run` and `tb_net` read them. Cost: about 15 s of C-sim per image and net on one core,
spread over `--jobs`; 20 images × 3 nets ≈ 3 min on 16 cores.

C-sim → co-sim and co-sim → board stay "pending" until Vitis and the board
(docs/tasks/deferred/M28-equivalence.md).
"""

import argparse
import datetime
import json
import multiprocessing as mp
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from yolo.quant.calibrate import scale_owners  # noqa: E402

DOC = ROOT / "results" / "equivalence.md"
DATA = ROOT / "results" / "equivalence.json"
OUT = ROOT / "build" / "equivalence"
GOLDEN = ROOT / "build" / "golden" / "golden_run"
TB_NET = ROOT / "build" / "hls" / "tb_net"
BEGIN, END = "<!-- equivalence:begin -->", "<!-- equivalence:end -->"

NETS = {"tiny-yolov2-voc": "voc", "tiny-yolov3-coco": "coco", "tiny-yolov4-coco": "coco"}
# Statistical level: full test sets, from the committed results (float → integer stage).
MAP = {
    "tiny-yolov2-voc": ("VOC2007 test, 4,952 images, mAP (stretch)", 56.30, 55.66,
                        "results/map_stades.md"),
    "tiny-yolov3-coco": None,
    "tiny-yolov4-coco": ("COCO val2017, 5,000 images, AP50 (stretch)", 41.4, 40.1,
                         "results/map_coco.md"),
}


# --------------------------------------------------------------------------- run
_W = {}


def _models(net):
    """(fused float64, fake-quant QuantModel from calib.json, IntNetwork, exported model)."""
    if net not in _W:
        from calibrate import load_fused

        from yolo.io.export import load_model
        from yolo.quant.calibrate import load_scales
        from yolo.quant.int_model import IntNetwork, QuantModel

        fused = load_fused(net, dtype=np.float64)
        qm = QuantModel.from_fused(fused, *load_scales(ROOT / "build" / "quant" / net /
                                                       "calib.json"))
        _W[net] = fused, qm, IntNetwork(qm, "int64"), load_model(ROOT / "model" / net)[0]
    return _W[net]


def _local_conv(fused, qm, qx, ints, i):
    """Conv i of the fake-quant model fed with the dequantized integer input: int8 output."""
    from yolo.layers.activations import leaky_forward
    from yolo.layers.conv import conv_forward
    from yolo.quant import int_layers as il
    from yolo.quant.quantize import quantize

    c, layer = qm.convs[i], fused.layers[i]
    j = fused.inputs[i][0]
    x = (qx if j < 0 else ints[j]).astype(np.float64) * qm.scale_of(j)  # −1: the input
    W = c.qW * c.sw[:, None, None, None]
    y, _ = conv_forward(x, W, fused.params[i]["b"], s=layer["s"])
    if layer["act"] == "leaky":
        y, _ = leaky_forward(y, il.LEAKY_MUL / 2**il.LEAKY_SHIFT)
    return quantize(y, c.sy, c.qmax)


def _image(task):
    """Dumps of one image (exported model) and its per-layer float/fake-quant/integer gaps."""
    from export_model import _preprocessed, write_dumps

    from yolo.data.letterbox import input_size
    from yolo.quant.int_model import fake_quant_forward
    from yolo.quant.quantize import quantize, quantize_input

    net, s, out_dir = task
    fused, qm, inet, qm_exp = _models(net)
    write_dumps(qm_exp, [s], out_dir)
    x32 = _preprocessed(s, input_size(qm.net, None), qm.net["input"][0])[0][None]
    x = x32.astype(np.float64)  # quantized from float32, as in write_dumps
    flt = fused.forward(x, train=False, all_outputs=True)
    fq = fake_quant_forward(fused, qm, x, all_outputs=True)
    qx = quantize_input(x32)
    ints = inet.forward(qx, all_outputs=True)
    d = out_dir / "dumps" / s["id"]
    same = all(np.array_equal(o, np.load(d / f"L{i:02d}.npy")) for i, o in enumerate(ints))
    rows = []
    for i, layer in enumerate(qm.layers):
        sc = qm.scale_of(i)
        o = scale_owners(qm.layers)[i]
        qmax = qm.convs[o].qmax if o >= 0 else 127
        inside = np.abs(flt[i]) <= sc * qmax  # values the int8 grid can represent
        e = np.abs(flt[i] - fq[i])[inside] / sc
        g = np.abs(quantize(fq[i], sc, qmax) - ints[i])
        r = {"fq_err": float(e.max(initial=0)), "fq_sum": float(e.sum()), "fq_n": int(e.size),
             "clipped": int((~inside).sum()),
             "int_ndiff": int((g != 0).sum()), "int_dmax": int(g.max(initial=0)),
             "size": int(g.size), "local_ndiff": 0, "local_dmax": 0}
        if layer["type"] == "conv":
            gl = np.abs(_local_conv(fused, qm, qx, ints, i) - ints[i])
            r["local_ndiff"], r["local_dmax"] = int((gl != 0).sum()), int(gl.max(initial=0))
        rows.append(r)
    return same, rows


def _python_stages(net, samples, out_dir, pool):
    """Per-layer gaps summed (counts) or maxed (errors) over the images."""
    res = pool.map(_image, [(net, s, out_dir) for s in samples])
    per = [dict(r) for r in res[0][1]]
    for _, rows in res[1:]:
        for p, r in zip(per, rows):
            for k, v in r.items():
                p[k] = max(p[k], v) if k in ("fq_err", "int_dmax", "local_dmax") else p[k] + v
    types = [layer["type"] for layer in _models(net)[1].layers]
    return per, types, sum(same for same, _ in res)


def _golden(task):
    net, img_dir, out = task
    r = subprocess.run([str(GOLDEN), "run", str(ROOT / "model" / net),
                        str(img_dir / "input.npy"), str(out)], capture_output=True, text=True)
    if r.returncode:
        return img_dir.name, None
    from compare_dumps import compare_layers

    return img_dir.name, {name: nd for name, _, nd, _ in compare_layers(img_dir, out)}


def _csim(task):
    model_dir, net, image = task
    r = subprocess.run([str(TB_NET), "--model", str(model_dir), "--net", net, "--image", image],
                       capture_output=True, text=True)
    bad = {}
    for line in r.stdout.splitlines():
        m = re.match(r"\s+L(\d+) \S+ vs golden : (\d+) / \d+", line)
        if m:
            bad[int(m.group(1))] = int(m.group(2))
    m = re.search(r"couches vérifiées :([\d ]*)", r.stdout)
    ids = [int(v) for v in m.group(1).split()] if m else []
    return image, {"ok": r.returncode == 0, "checked": ids, "bad": bad}


def run_net(net, n, n_csim, pool):
    from yolo.data import datasets

    ds = NETS[net]
    samples = datasets.load(ds, None, datasets.DATASETS[ds].test, "test")[:n]
    model_dir = OUT / "model"
    out_dir = model_dir / net
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / "model" / net).iterdir():  # manifest + blobs; dumps are ours
        link = out_dir / f.name
        if f.name != "dumps" and not link.exists():
            link.symlink_to(f)
    t0 = time.time()
    per, types, calib_same = _python_stages(net, samples, out_dir, pool)
    t_py = time.time() - t0
    ids = [s["id"] for s in samples]

    t0 = time.time()
    gold = dict(pool.map(_golden, [(net, out_dir / "dumps" / i, OUT / "golden" / net / i)
                                   for i in ids]))
    t_gold = time.time() - t0
    t0 = time.time()
    csim = dict(pool.map(_csim, [(model_dir, net, i) for i in ids[:n_csim]]))
    t_csim = time.time() - t0

    for i, p in enumerate(per):
        name = f"L{i:02d}"
        p["golden_ndiff"] = sum(g.get(name, 0) for g in gold.values() if g is not None)
        p["csim_ndiff"] = sum(c["bad"].get(i, 0) for c in csim.values())
        # 0: the kernel never writes this map to DDR (pre-pool map of a fused maxpool, a
        # route view consumed in place); the layers after it are checked.
        p["csim_images"] = sum(i in c["checked"] for c in csim.values())
        p["type"] = types[i]
    return {
        "dataset": f"{ds} {datasets.DATASETS[ds].test}", "images": ids,
        "n_csim": len(csim), "calib_equals_export": calib_same,
        "golden_failed": sorted(k for k, g in gold.items() if g is None),
        "csim_failed": sorted(k for k, c in csim.items() if not c["ok"]),
        "csim_layers_checked": sum(len(c["checked"]) for c in csim.values()),
        "seconds": {"python": round(t_py), "golden": round(t_gold), "csim": round(t_csim)},
        "layers": per,
    }


def cmd_run(args):
    for exe in (GOLDEN, TB_NET):
        if not exe.exists():
            sys.exit(f"{exe.relative_to(ROOT)} missing: make golden-check csim-gcc")
    data = json.loads(DATA.read_text()) if DATA.exists() else {"nets": {}}
    with mp.Pool(args.jobs) as pool:
        for net in args.nets.split(","):
            print(f"{net}: {args.n} images, C-sim on {min(args.n, args.n_csim)}", flush=True)
            r = run_net(net, args.n, args.n_csim, pool)
            data["nets"][net] = r
            print(f"  {r['seconds']}  calib == export on {r['calib_equals_export']} / "
                  f"{len(r['images'])}", flush=True)
    data["date"] = datetime.date.today().isoformat()
    DATA.write_text(json.dumps(data, indent=1) + "\n")
    print(f"{DATA.relative_to(ROOT)} written")


# --------------------------------------------------------------------------- report
def _cell(ndiff, total):
    return "**0**" if ndiff == 0 else f"{ndiff} / {total}"


def csim_cell(p, r):
    if p["csim_images"] == 0:
        return "not in DDR"
    return _cell(p["csim_ndiff"], p["size"] // len(r["images"]) * p["csim_images"])


def render(data):
    nets = data["nets"]
    out = [BEGIN, "", "## T28.0: equivalence matrix", "",
           f"Generated by `python tools/equivalence.py run` on {data['date']}; "
           "`python tools/equivalence.py report` rewrites this section, `--check` (in "
           "`make ci`) fails if it is stale.", "",
           "### Summary per stage", "",
           "| Network | Images | float → fake-quant | fake-quant → integer | integer → golden "
           "| golden → C-sim | C-sim → co-sim | co-sim → board |",
           "|---|---|---|---|---|---|---|---|"]
    for net, r in nets.items():
        L = r["layers"]
        fq = max(p["fq_err"] for p in L)
        fqm = sum(p["fq_sum"] for p in L) / sum(p["fq_n"] for p in L)
        nd = sum(p["int_ndiff"] for p in L)
        tot = sum(p["size"] for p in L)
        dmax = max(p["int_dmax"] for p in L)
        ld = max(p["local_dmax"] for p in L)
        lnd = sum(p["local_ndiff"] for p in L)
        ltot = sum(p["size"] for p in L if p["type"] == "conv")
        g = sum(p["golden_ndiff"] for p in L) + len(r["golden_failed"])
        c = sum(p["csim_ndiff"] for p in L) + len(r["csim_failed"])
        stat = MAP[net]
        st = (f"statistical: {stat[2] - stat[1]:+.2f} ({stat[0]})" if stat
              else "statistical: not measured")
        out.append(
            f"| {net} | {len(r['images'])} {r['dataset']} (C-sim: first {r['n_csim']}) "
            f"| proximity: mean {fqm:.2f} LSB, max {fq:.0f} LSB; {st} "
            f"| proximity: per layer ≤ {ld} LSB ({100 * lnd / ltot:.3f} % of conv values); "
            f"end to end ≤ {dmax} LSB ({100 * nd / tot:.2f} %) "
            f"| pointwise: {'**identical**' if g == 0 else f'{g} gaps'} "
            f"| pointwise: {'**identical**' if c == 0 else f'{c} gaps'} | pending | pending |")
    out += ["",
            "- **float → fake-quant**: max |float − fake-quant| over the images, in LSB of the "
            "layer scale s_y, on the values the int8 grid can represent (the clipped ones, "
            "calibration at p99.99, are counted apart in the per-layer tables). The error "
            "accumulates through the layers; the statistical gap is the one that counts for "
            "the task.",
            "- **fake-quant → integer**, two measures. *Per layer*: the fake-quant conv fed "
            "with the same integer input as the integer model; the gap comes only from the "
            "requantization (M0/2ⁿ against a float scale) and the rounded leaky, so it "
            "stays at 1 LSB on rare values. *End to end*: both models run from the "
            "image; the one-LSB gaps propagate and grow through the layers. This is the "
            "stage where the equivalence is only *bounded proximity*; the bound is "
            "measured, not proved.",
            "- **integer → golden → C-sim**: every byte of every layer output equal, plus "
            "the C-sim kernel ABFT check silent. `tb_net` compares the maps that the kernel "
            "writes to DDR; \"not in DDR\" marks a pre-pool map of a fused maxpool or a "
            "route consumed in place, covered by the next layer that is in DDR. The rebuilt fake-quant model gives the same "
            "integers as the export on every image ("
            + ", ".join(f"{r['calib_equals_export']} / {len(r['images'])}"
                        for r in nets.values()) + ").",
            "- **Wider evidence of the same pointwise level**: `results/map_stades.md` "
            "(Tiny-YOLOv2, C-sim detections == integer on 4,952 / 4,952 VOC images) and "
            "the C-sim smoke run of `results/map_coco.md` (YOLOv4-tiny, 10 / 10).",
            "- **pending**: co-sim needs Vitis, the board row is T28.5 "
            "(docs/tasks/deferred/M28-equivalence.md).",
            "- **Failure head (T39.5)**, a side output on the integer features (no golden "
            "C++ model): float → integer is *statistical* (AUROC of the failure prediction, "
            "results/failure_prediction.md); integer → C-sim is *pointwise*, the score equal "
            "on every image (`hls/tb/tb_failure.cpp`, `tools/failure_prediction.py stages "
            "--check`, `python/tests/test_failure.py`); co-sim and board pending (T39.6).", ""]
    for net, r in nets.items():
        out += [f"### {net}", "",
                f"{len(r['images'])} images ({r['dataset']}, first of the split); golden on "
                f"all, C-sim on the first {r['n_csim']} ({r['csim_layers_checked']} layer "
                f"outputs checked by `tb_net`). Run time: Python {r['seconds']['python']} s, "
                f"golden {r['seconds']['golden']} s, C-sim {r['seconds']['csim']} s.", "",
                "| Layer | Type | float → fq (mean / max LSB) | clipped | fq → int per layer "
                "(values ≠, max LSB) | fq → int end to end | int → golden | golden → C-sim |",
                "|---|---|---|---|---|---|---|---|"]
        for i, p in enumerate(r["layers"]):
            loc = (f"{p['local_ndiff']}, {p['local_dmax']}" if p["type"] == "conv" else "—")
            out.append(f"| L{i:02d} | {p['type']} | {p['fq_sum'] / p['fq_n']:.2f} / "
                       f"{p['fq_err']:.1f} | {p['clipped']} | {loc} "
                       f"| {p['int_ndiff']} / {p['size']}, {p['int_dmax']} "
                       f"| {_cell(p['golden_ndiff'], p['size'])} "
                       f"| {csim_cell(p, r)} |")
        out.append("")
    out.append(END)
    return "\n".join(out)


def splice(doc, section):
    if BEGIN in doc:
        a, b = doc.index(BEGIN), doc.index(END) + len(END)
        return doc[:a] + section + doc[b:]
    anchor = "## T28.1:"
    i = doc.index(anchor)
    return doc[:i] + section + "\n\n" + doc[i:]


def cmd_report(args):
    DOC.write_text(splice(DOC.read_text(), render(json.loads(DATA.read_text()))))
    print(f"{DOC.relative_to(ROOT)}: matrix section written")


# --------------------------------------------------------------------------- post-processing
POST_NETS = ("tiny-yolov2-voc", "tiny-yolov3-coco")  # yolo_post: no scale_x_y / new_coords
POST_CAPS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def candidates(heads, conf):
    from yolo.infer import hw_postproc as hp

    out = []
    for h in heads:
        hp.decode_head(h, conf, out)
    return out


def greedy_nms(cands, p, q):
    """Classic NMS (§8.2) in integers: by descending score, same class, IoU > p/q."""
    from yolo.infer import hw_postproc as hp

    keep = []
    for b in sorted(cands, key=lambda c: (-c[4], c)):
        s = np.array([k for k in keep if k[5] == b[5]], np.int64).reshape(-1, 6)
        if not len(s) or not hp.overlaps(b, s[:, 0], s[:, 1], s[:, 2], s[:, 3], p, q).any():
            keep.append(b)
    return keep


def postproc_props(cands, iou=0.45, perms=0, seed=0):
    """T28.6 properties of the no-sort NMS on one candidate stream; raises on a violation.

    - capacity: for every cap, overflow = 0 ⟺ slots used with unlimited slots ≤ cap, and then
      the result equals the unlimited one (no candidate dropped below capacity, an overflow
      is always counted);
    - order: the stream sorted by (score ↓, box) gives the classic NMS, whatever the order the
      cells came in;
    - `perms` random permutations of the raw stream: how many change the result (the raw
      no-sort NMS is order dependent by design, [2024-zhang#014.2]).
    """
    from yolo.infer import hw_postproc as hp

    p, q = hp.iou_fraction(iou)
    st = {}
    full, ov = hp.nms_nosort(cands, p, q, cap=len(cands) + 1, stats=st)
    assert ov == 0
    for cap in POST_CAPS:
        out, ov = hp.nms_nosort(cands, p, q, cap=cap)
        assert (ov == 0) == (st["slots"] <= cap), (cap, ov, st["slots"])
        if ov == 0:
            assert out == full, cap
    key = lambda c: (-c[4], c)  # noqa: E731
    ref = sorted(hp.nms_nosort(sorted(cands, key=key), p, q, cap=len(cands) + 1)[0])
    assert ref == sorted(greedy_nms(cands, p, q))
    rng = np.random.default_rng(seed)
    changed = 0
    for _ in range(perms):
        sh = [cands[i] for i in rng.permutation(len(cands))]
        got = hp.nms_nosort(sorted(sh, key=key), p, q, cap=len(sh) + 1)[0]
        assert sorted(got) == ref
        changed += sorted(hp.nms_nosort(sh, p, q, cap=len(sh) + 1)[0]) != sorted(full)
    return {"cands": len(cands), "slots": st["slots"], "kept": len(full), "changed": changed,
            "differs_from_greedy": sorted(full) != ref}


def post_images(net):
    """Dump folders of `net`: those of `run` (build/equivalence) or the committed ones."""
    d = OUT / "model" / net / "dumps"
    if not d.exists():
        d = ROOT / "model" / net / "dumps"
    return sorted(x for x in d.iterdir() if (x / "input.npy").exists())


def head_data(net, img):
    from yolo.infer import hw_postproc as hp
    from yolo.io.export import load_model
    from yolo.quant.lut import HeadLuts

    qm, m = load_model(ROOT / "model" / net)
    luts = {e["id"]: HeadLuts(e["scale"]) for e in m["layers"] if "lut_offset" in e}
    ids = [hid for hid, _ in hp.heads(qm.net)]
    outs = {h: np.load(img / f"L{h:02d}.npy")[None] for h in ids}
    return hp.make_heads(outs, qm.net, luts), [img / f"L{h:02d}.npy" for h in ids]


def cmd_postproc(args):
    from yolo.infer import hw_postproc as hp

    for net in POST_NETS:
        imgs = post_images(net)
        tot = {"images": len(imgs), "golden_equal": 0, "cands": 0, "overflow_256": 0,
               "changed": 0, "perms": 0, "differs_from_greedy": 0, "max_slots": 0}
        for img in imgs:
            heads, files = head_data(net, img)
            for conf in (0.005, 0.25):
                boxes, ov = hp.run(heads, conf, 0.45)
                r = subprocess.run([str(GOLDEN), "hwpp", str(ROOT / "model" / net), repr(conf),
                                    "0.45", *map(str, files)], capture_output=True, text=True,
                                   check=True).stdout.splitlines()
                gb = [tuple(int(v) for v in ln.split()[1:]) for ln in r if ln.startswith("box")]
                gov = [int(ln.split()[1]) for ln in r if ln.startswith("overflow")][0]
                tot["golden_equal"] += gb == boxes and gov == ov
            c = candidates(heads, 0.005)
            s = postproc_props(c, perms=args.perms)
            tot["cands"] += s["cands"]
            tot["max_slots"] = max(tot["max_slots"], s["slots"])
            tot["overflow_256"] += s["slots"] > 256
            tot["changed"] += s["changed"]
            tot["perms"] += args.perms
            tot["differs_from_greedy"] += s["differs_from_greedy"]
        print(net, json.dumps(tot))


# --------------------------------------------------------------------------- check
def check_dumps(nets, model_root=ROOT / "model"):
    """The committed dumps of model/<net>/ still come out of the exported model (tier R)."""
    from yolo.io.export import load_model
    from yolo.quant.int_model import IntNetwork

    bad = []
    for net in nets:
        d = model_root / net
        if not (d / "manifest.json").exists():
            bad.append(f"{net}: export missing")
            continue
        inet = IntNetwork(load_model(d)[0], "f64")
        for img in sorted((d / "dumps").iterdir()):
            outs = inet.forward(np.load(img / "input.npy"), all_outputs=True)
            for i, o in enumerate(outs):
                ref = np.load(img / f"L{i:02d}.npy")
                if not np.array_equal(o, ref):
                    bad.append(f"{net} {img.name} L{i:02d}: {int((o != ref).sum())} values")
    return bad


def cmd_check(args):
    bad = check_dumps(NETS)
    for b in bad:
        print(f"dump mismatch: {b}")
    stale = not DATA.exists() or splice(DOC.read_text(), render(
        json.loads(DATA.read_text()))) != DOC.read_text()
    if stale:
        print(f"{DOC.relative_to(ROOT)} is stale: python tools/equivalence.py report")
    if not bad and not stale:
        print("equivalence: committed dumps reproduce, matrix up to date")
    sys.exit(1 if bad or stale else 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run")
    p.add_argument("--nets", default=",".join(NETS))
    p.add_argument("--n", type=int, default=20, help="images per net (golden on all)")
    p.add_argument("--n-csim", type=int, default=20, help="images per net in C-sim")
    p.add_argument("--jobs", type=int, default=16)
    sub.add_parser("report")
    p = sub.add_parser("postproc", help="T28.6: equivalence and properties of the NMS")
    p.add_argument("--perms", type=int, default=20, help="random stream orders per image")
    args = ap.parse_args()
    if args.check:
        cmd_check(args)
    elif args.cmd == "run":
        cmd_run(args)
    elif args.cmd == "report":
        cmd_report(args)
    elif args.cmd == "postproc":
        cmd_postproc(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
