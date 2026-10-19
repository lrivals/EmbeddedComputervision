"""T25.1: fault sensitivity campaign of the integer model (mAP under bit flips).

    # cost of each grid before running it (M12 sizing)
    python tools/fault_campaign.py plan --subset 500 --jobs 16
    # mAP against the BER, per site (weight, bias, act, acc), 3 seeds
    python tools/fault_campaign.py run --grid ber --subset 500 --out build/m25/faults/ber.jsonl
    # per layer, and per bit position (one flip per image / per model)
    python tools/fault_campaign.py run --grid layer --out build/m25/faults/layer.jsonl
    python tools/fault_campaign.py run --grid bit --out build/m25/faults/bit.jsonl
    # tables (markdown) and report.json for the figure (python -m tools.figures faults)
    python tools/fault_campaign.py report build/m25/faults/*.jsonl
    # ABFT checksum widths and cost of the extra lane (T25.2)
    python tools/fault_campaign.py widths model/tiny-yolov2-voc model/tiny-yolov3-coco

Fault model: `yolo.quant.faults` (T25.0). Same preprocessing and thresholds as
`eval_quant.py --variants int --resize stretch` (conf 0.005, NMS 0.45, VOC07 mAP): the
`baseline` row (no fault) gives back its mAP on the same subset (first N images of the split).
Each run also counts, per image, whether the heads differ from the fault-free heads
(`changed`) and whether the ABFT check (T25.2) saw an error (`detected`); `sdc` = changed and
not detected (silent data corruption).

- `weight` / `bias` faults are drawn once per run (persistent, the same corrupted model for
  every image); `act` / `acc` faults are drawn again for each image (transient), seeded by
  (seed, image index): a run is reproducible whatever the number of jobs.
- The quantized inputs and the fault-free heads of the subset are cached once in
  `build/m25/cache/<net>_<N>/` (500 images: about 260 MB).
- `run` appends one JSON line per run to `--out` and skips the runs already there (resume).
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from yolo.quant import faults as F  # noqa: E402

SITES = ("weight", "bias", "act", "acc")
BERS = (1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
LAYER_BERS = (1e-7, 1e-5)
BITS = {"weight": tuple(range(8)), "act": tuple(range(8)),
        "bias": tuple(range(0, 32, 4)) + (31,), "acc": tuple(range(0, 32, 4)) + (31,)}
CONF, IOU = 0.005, 0.45

_W = {}


# ------------------------------------------------------------------------- run specs
def spec_key(spec):
    """Unique key of a run (resume)."""
    return json.dumps({k: spec[k] for k in sorted(spec)}, sort_keys=True)


def faults_of(spec):
    """Faults of a run spec ({} : baseline, no fault)."""
    if spec.get("site") is None:
        return []
    f = F.Fault(spec["site"], spec["kind"], spec.get("layer"), spec.get("bit"),
                spec.get("rate", 0.0))
    return [f] * spec.get("flips", 1)


def grid(name, convs, sites=SITES, bers=BERS, layer_bers=LAYER_BERS, bits=None, seeds=(0,),
         flips=100):
    """Run specs of a grid (`ber`, `layer`, `bit`), the baseline first."""
    out = [{"grid": "baseline", "site": None, "seed": 0}]
    for seed in seeds:
        for site in sites:
            if name == "ber":
                out += [{"grid": name, "site": site, "kind": "ber", "rate": r, "seed": seed}
                        for r in bers]
            elif name == "layer":
                out += [{"grid": name, "site": site, "kind": "ber", "layer": i, "rate": r,
                         "seed": seed} for i in convs for r in layer_bers]
            elif name == "bit":
                out += [{"grid": name, "site": site, "kind": "flip", "bit": b, "flips": flips,
                         "seed": seed} for b in (bits or BITS)[site]]
            else:
                raise ValueError(f"unknown grid {name!r}")
    return out


# ------------------------------------------------------------------------- model, cache
def load_qm(net, weights=None, calib=None, model_dir=None):
    """Integer model: from an export (`model_dir`), or as `eval_quant.py` builds it."""
    from yolo.io.export import load_model
    from yolo.quant.calibrate import load_scales
    from yolo.quant.int_model import QuantModel

    if model_dir:
        return load_model(model_dir)[0]
    from calibrate import load_fused, net_tag
    from yolo.models.tiny_yolo import PRETRAINED

    weights = weights or ROOT / "weights" / PRETRAINED[net]
    calib = calib or ROOT / "build" / "quant" / net_tag(net) / "calib.json"
    return QuantModel.from_fused(load_fused(net, weights, dtype=np.float64),
                                 *load_scales(calib))


def _prep_image(task):
    from PIL import Image

    from yolo.infer.pipeline import preprocess
    from yolo.quant.quantize import quantize_input

    path, size, channels = task
    with Image.open(path) as img:
        x, _ = preprocess(img, size, "stretch", channels=channels)
    return quantize_input(x)


def _ref_heads(task):  # fault-free heads of image `task`
    if "error" in _W:
        raise _W["error"]
    return _W["inet"].forward(np.asarray(_W["inputs"][task:task + 1]))


def prepare(cache, samples, qm, size, jobs):
    """Cache of the quantized inputs (`inputs.npy`), their sizes (`meta.json`) and the
    fault-free heads (`head_<id>.npy`)."""
    cache = Path(cache)
    if (cache / "meta.json").exists():
        return
    cache.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ch = qm.net["input"][0]
    tasks = [(s["image"], size, ch) for s in samples]
    if jobs <= 1:
        xs = [_prep_image(t) for t in tasks]
    else:
        with mp.get_context("spawn").Pool(jobs) as pool:
            xs = pool.map(_prep_image, tasks, chunksize=4)
    np.save(cache / "inputs.npy", np.stack(xs))
    print(f"inputs: {len(xs)} images  {time.time() - t0:.0f} s", flush=True)
    if jobs <= 1:
        _setup(qm, cache, abft=False)
        heads = [_ref_heads(k) for k in range(len(xs))]
    else:
        with mp.get_context("spawn").Pool(jobs, initializer=_init,
                                          initargs=(qm, cache, False)) as pool:
            heads = pool.map(_ref_heads, range(len(xs)), chunksize=4)
    for hid in heads[0]:
        np.save(cache / f"head_{hid}.npy", np.concatenate([h[hid] for h in heads]))
    (cache / "meta.json").write_text(json.dumps(
        {"ids": [s["id"] for s in samples], "width": [s["width"] for s in samples],
         "height": [s["height"] for s in samples], "net": qm.net["name"]}) + "\n")
    print(f"cache {cache}: {len(xs)} images  {time.time() - t0:.0f} s", flush=True)


# ------------------------------------------------------------------------- workers
def _init(*args):
    try:
        _setup(*args)
    except Exception as exc:  # noqa: BLE001  (raised by the first task, not respawned)
        _W["error"] = exc


def _setup(qm, cache, abft=True, conf=CONF, iou=IOU):
    from yolo.quant.abft import AbftMonitor
    from yolo.quant.int_model import IntNetwork, head_luts

    cache = Path(cache)
    inputs = np.load(cache / "inputs.npy", mmap_mode="r")
    inet = IntNetwork(qm)
    mon = AbftMonitor(qm) if abft else None
    if mon is not None:
        inet.hooks.append(mon)
    refs = {i: np.load(cache / f"head_{i}.npy", mmap_mode="r") for i in qm.outputs
            if (cache / f"head_{i}.npy").exists()}
    _W.clear()
    _W.update(qm=qm, inet=inet, mon=mon, luts=head_luts(qm), inputs=inputs, refs=refs,
              sizes=F.conv_sizes(qm, inputs.shape[1:]), conf=conf, iou=iou, run=None)


def _image(task):
    """One image of one run → (labels, scores, boxes (cx, cy, w, h) normalized, flips,
    changed, detected)."""
    from yolo.infer.nms import postprocess_int

    if "error" in _W:
        raise _W["error"]
    key, spec, k = task
    if _W["run"] != key:  # persistent faults: drawn once per run and worker
        from yolo.quant.int_model import IntNetwork

        faults = faults_of(spec)
        fqm, _W["nparam"] = F.inject_params(_W["qm"], faults, spec["seed"])
        _W["inj"] = F.FaultInjector(faults, spec["seed"], _W["sizes"])
        _W["finet"] = IntNetwork(fqm)
        _W["finet"].wrap_acc = True
        _W["finet"].hooks = [_W["inj"], *_W["inet"].hooks]
        _W["run"] = key
    inj, mon = _W["inj"], _W["mon"]
    inj.start(k)
    if mon is not None:
        mon.reset()
    heads = _W["finet"].forward(np.asarray(_W["inputs"][k:k + 1]))
    changed = any(not np.array_equal(heads[i], _W["refs"][i][k:k + 1]) for i in _W["refs"])
    boxes, scores, labels = postprocess_int(heads, _W["qm"].net, _W["luts"], _W["conf"],
                                            _W["iou"])[0]
    flips = sum(inj.flips.values()) + _W["nparam"]
    detected = bool(mon is not None and mon.detected)
    return (np.asarray(labels).tolist(), np.asarray(scores, np.float64).tolist(),
            np.asarray(boxes, np.float64).reshape(-1, 4).tolist(), flips, changed, detected)


# ------------------------------------------------------------------------- one run
def score(results, samples, meta, n_classes):
    """mAP (VOC07) and counters of a run from the per-image results."""
    from yolo.infer.metrics import evaluate, to_voc_pixels

    per = {c: ([], [], []) for c in range(n_classes)}
    flips = changed = detected = sdc = n_det = 0
    for k, (labels, scores, boxes, fl, ch, de) in enumerate(results):
        px = to_voc_pixels(boxes, meta["width"][k], meta["height"][k])
        for c, s, b in zip(labels, scores, px):
            per[c][0].append(meta["ids"][k])
            per[c][1].append(s)
            per[c][2].append(b)
        n_det += len(labels)
        flips += fl
        changed += ch
        detected += de
        sdc += ch and not de
    dets = {c: (d[0], np.array(d[1]), np.array(d[2]).reshape(-1, 4)) for c, d in per.items()}
    _, m = evaluate(dets, samples, n_classes, use_07=True)
    n = max(len(results), 1)
    return {"map": float(m), "images": len(results), "flips": int(flips),
            "changed": changed / n, "detected": detected / n, "sdc": sdc / n,
            "dets_per_image": n_det / n}


def done_keys(out):
    if not Path(out).exists():
        return set()
    return {json.loads(line)["key"] for line in Path(out).read_text().splitlines()
            if line.strip()}


def campaign(specs, samples, meta, n_classes, out, mapper):
    """Runs every spec not yet in `out`; `mapper(tasks)` → per-image results in order."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(out)
    todo = [s for s in specs if spec_key(s) not in done]
    print(f"{len(specs)} runs, {len(specs) - len(todo)} already done", flush=True)
    rows = []
    for r, spec in enumerate(todo, 1):
        t0 = time.time()
        key = spec_key(spec)
        res = mapper([(key, spec, k) for k in range(len(samples))])
        row = {"key": key, **spec, **score(res, samples, meta, n_classes),
               "seconds": round(time.time() - t0, 1)}
        with out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        rows.append(row)
        what = " ".join(f"{k}={spec[k]}" for k in ("site", "layer", "bit", "rate")
                        if spec.get(k) is not None)
        print(f"[{r}/{len(todo)}] {spec['grid']} {what} seed={spec['seed']}: "
              f"mAP {row['map'] * 100:.2f}  changed {row['changed'] * 100:.0f} %  "
              f"detected {row['detected'] * 100:.0f} %  ({row['seconds']:.0f} s)", flush=True)
    return rows


# ------------------------------------------------------------------------- report
def aggregate(rows):
    """Mean and standard deviation over the seeds, per (grid, site, layer, bit, rate)."""
    groups = {}
    for r in rows:
        k = (r["grid"], r.get("site"), r.get("layer"), r.get("bit"), r.get("rate"))
        groups.setdefault(k, []).append(r)
    out = []
    for (g, site, layer, bit, rate), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        maps = np.array([r["map"] for r in rs])
        out.append({"grid": g, "site": site, "layer": layer, "bit": bit, "rate": rate,
                    "seeds": len(rs), "map": float(maps.mean()),
                    "std": float(maps.std(ddof=1)) if len(rs) > 1 else 0.0,
                    **{k: float(np.mean([r[k] for r in rs]))
                       for k in ("changed", "detected", "sdc", "flips")}})
    return out


def report(rows):
    """{baseline, ber, layer, bit, abft, ranking} for the tables and the figure."""
    agg = aggregate(rows)
    base = [a for a in agg if a["grid"] == "baseline"]
    rep = {"baseline": base[0]["map"] if base else None,
           "images": rows[0]["images"] if rows else 0}
    for g in ("ber", "layer", "bit"):
        rep[g] = [a for a in agg if a["grid"] == g]
    # Ranking of the layers: mean mAP drop over the sites and rates of the layer grid.
    if rep["layer"] and rep["baseline"] is not None:
        drop = {}
        for a in rep["layer"]:
            drop.setdefault(a["layer"], []).append(rep["baseline"] - a["map"])
        rep["ranking"] = sorted(([int(i), float(np.mean(d))] for i, d in drop.items()),
                                key=lambda t: -t[1])
    else:
        rep["ranking"] = []
    # ABFT: share of the runs' changed images that were detected, per site.
    abft = {}
    for r in rows:
        if r.get("site") is None:
            continue
        a = abft.setdefault(r["site"], {"changed": 0.0, "detected": 0.0, "sdc": 0.0, "n": 0})
        for k in ("changed", "detected", "sdc"):
            a[k] += r[k] * r["images"]
        a["n"] += r["images"]
    rep["abft"] = {s: {"changed": a["changed"] / a["n"], "detected": a["detected"] / a["n"],
                       "sdc": a["sdc"] / a["n"],
                       "detected_given_changed": 1 - a["sdc"] / a["changed"]
                       if a["changed"] else None}
                   for s, a in abft.items()}
    return rep


def _pm(a):
    s = f"{a['map'] * 100:.2f}"
    if a["seeds"] > 1:
        s += f" ± {a['std'] * 100:.2f}"
    return s


def markdown(rep):
    """Markdown tables of a report (for results/faults.md)."""
    base = rep["baseline"]
    lines = [f"Fault-free mAP: **{base * 100:.2f}** ({rep['images']} images)." if base
             is not None else "No baseline row.", ""]
    if rep["ber"]:
        rates = sorted({a["rate"] for a in rep["ber"]})
        lines += ["| Site | " + " | ".join(f"BER {r:.0e}" for r in rates) + " |",
                  "|---|" + "---|" * len(rates)]
        for site in SITES:
            cells = {a["rate"]: _pm(a) for a in rep["ber"] if a["site"] == site}
            if cells:
                lines.append(f"| {site} | " + " | ".join(cells.get(r, "") for r in rates)
                             + " |")
        lines.append("")
    if rep["layer"]:
        cols = sorted({(a["site"], a["rate"]) for a in rep["layer"]},
                      key=lambda t: (SITES.index(t[0]), t[1]))
        lines += ["| Conv | " + " | ".join(f"{s} {r:.0e}" for s, r in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for i in sorted({a["layer"] for a in rep["layer"]}):
            cells = {(a["site"], a["rate"]): f"{(a['map'] - base) * 100:+.2f}"
                     for a in rep["layer"] if a["layer"] == i}
            lines.append(f"| L{i:02d} | " + " | ".join(cells.get(c, "") for c in cols) + " |")
        lines += ["", "mAP change in points (negative: loss). Ranking (mean loss): "
                  + ", ".join(f"L{i:02d} ({d * 100:.2f})" for i, d in rep["ranking"]), ""]
    if rep["bit"]:
        sites = [s for s in SITES if any(a["site"] == s for a in rep["bit"])]
        bits = sorted({a["bit"] for a in rep["bit"]})
        lines += ["| Bit | " + " | ".join(sites) + " |", "|---|" + "---|" * len(sites)]
        for b in bits:
            cells = {a["site"]: f"{(a['map'] - base) * 100:+.2f}" for a in rep["bit"]
                     if a["bit"] == b}
            lines.append(f"| {b} | " + " | ".join(cells.get(s, "") for s in sites) + " |")
        lines += ["", "mAP change in points for one flip at that bit (per image for act / acc, "
                  "per model for weight / bias).", ""]
    if rep["abft"]:
        lines += ["| Site | Images changed | Flagged by ABFT | Changed and flagged | Silent "
                  "(changed, not flagged) |", "|---|---|---|---|---|"]
        for s in SITES:
            if s in rep["abft"]:
                a = rep["abft"][s]
                dg = a["detected_given_changed"]
                lines.append(f"| {s} | {a['changed'] * 100:.1f} % | {a['detected'] * 100:.1f} % "
                             f"| {'—' if dg is None else f'{dg * 100:.1f} %'} | "
                             f"{a['sdc'] * 100:.1f} % |")
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------- main
def _samples(args):
    from yolo.data import datasets
    from yolo.models.tiny_yolo import load_cfg

    samples = datasets.load_args(args)
    if args.subset:
        samples = samples[:args.subset]
    view = datasets.eval_view(args.dataset, load_cfg(args.net)["classes"])
    return datasets.remap(samples, view.gt_lut), len(view.names)


def _cache_dir(args, n):
    return ROOT / "build" / "m25" / "cache" / f"{args.net}_{n}"


def _parse_floats(s):
    return tuple(float(v) for v in s.split(",")) if s else None


def _context(args):
    from yolo.data.letterbox import input_size
    from yolo.models.tiny_yolo import load_cfg

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    samples, n_classes = _samples(args)
    qm = load_qm(args.net, args.weights, args.calib, args.model_dir)
    cache = _cache_dir(args, len(samples))
    prepare(cache, samples, qm, input_size(load_cfg(args.net), None), args.jobs)
    meta = json.loads((cache / "meta.json").read_text())
    if meta["ids"] != [s["id"] for s in samples]:
        raise SystemExit(f"{cache}: other images, delete the cache")
    return samples, n_classes, qm, cache, meta


def cmd_run(args):
    samples, n_classes, qm, cache, meta = _context(args)
    sites = tuple(args.sites.split(","))
    specs = grid(args.grid, sorted(qm.convs), sites, _parse_floats(args.bers) or BERS,
                 _parse_floats(args.layer_bers) or LAYER_BERS, None,
                 tuple(int(s) for s in args.seeds.split(",")), args.flips)
    if args.layers:
        keep = {int(i) for i in args.layers.split(",")}
        specs = [s for s in specs if s.get("layer") is None or s["layer"] in keep]
    abft = not args.no_abft
    if args.jobs <= 1:
        _setup(qm, cache, abft)
        campaign(specs, samples, meta, n_classes, args.out, lambda t: [_image(x) for x in t])
        return
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init,
                                      initargs=(qm, cache, abft)) as pool:
        campaign(specs, samples, meta, n_classes, args.out,
                 lambda t: pool.map(_image, t, chunksize=max(1, len(t) // (4 * args.jobs))))


def cmd_plan(args):
    samples, _, qm, cache, _ = _context(args)
    _setup(qm, cache, abft=not args.no_abft)
    spec = {"grid": "ber", "site": "acc", "kind": "ber", "rate": 1e-6, "seed": 0}
    _image(("plan", spec, 0))
    t0 = time.time()
    for k in range(min(5, len(samples))):
        _image(("plan", spec, k))
    per = (time.time() - t0) / min(5, len(samples))
    run_s = per * len(samples) / max(args.jobs, 1) * 1.15  # 15 %: post-processing, mAP
    print(f"{per * 1000:.0f} ms per image and job, {len(samples)} images, {args.jobs} jobs: "
          f"≈ {run_s:.0f} s per run")
    seeds = tuple(int(s) for s in args.seeds.split(","))
    for g in ("ber", "layer", "bit"):
        n = len(grid(g, sorted(qm.convs), seeds=seeds))
        print(f"  {g:6s} {n:5d} runs  ≈ {n * run_s / 3600:.1f} h")


TM, TN = 32, 24  # KV260 tiles (hls/kernels/accel_config.hpp)


def widths_table(model_dir):
    """Markdown rows: checksum width and extra MACs of the ABFT lane per conv (T25.2)."""
    from count_macs import layer_cost

    from yolo.io.export import load_model
    from yolo.models.specs import infer_shapes
    from yolo.quant.abft import widths

    qm, _ = load_model(model_dir)
    shapes = infer_shapes(qm.net)
    rows, macs, extra = [], 0, 0
    for w in widths(qm):
        ins, outs = shapes[w["id"]]
        m = layer_cost(qm.layers[w["id"]], ins, outs)[1]
        macs += m
        extra += m / w["cout"]
        rows.append(f"| L{w['id']:02d} | {w['cout']} | {w['cs_max']:,} | {w['cs_bits']} | "
                    f"{m / w['cout'] / 1e6:.2f} M (+{w['extra'] * 100:.2f} %) |")
    bits = max(w["cs_bits"] for w in widths(qm))
    head = [f"**{qm.net['name']}**: checksum filters up to {bits} bits (signed); lane "
            f"+{extra / 1e6:.1f} M MACs on {macs / 1e6:.0f} M (+{extra / macs * 100:.2f} %) "
            f"if computed per conv, +{100 / TM:.1f} % (one lane of Tm = {TM}, i.e. {TN} more "
            "multipliers) in the tiled engine.", "",
            "| Conv | F | max abs(Σ_f w_f) | Bits | Lane MACs |", "|---|---|---|---|---|"]
    return "\n".join(head + rows)


def cmd_widths(args):
    for d in args.model_dirs:
        print(widths_table(d) + "\n")


def cmd_report(args):
    rows = [json.loads(line) for p in args.files for line in Path(p).read_text().splitlines()
            if line.strip()]
    rep = report(rows)
    out = args.out or Path(args.files[0]).parent / "report.json"
    Path(out).write_text(json.dumps(rep, indent=1) + "\n")
    print(markdown(rep))
    print(f"report: {out}")


def main():
    from yolo.data import datasets

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "plan"):
        p = sub.add_parser(name)
        p.add_argument("--net", default="tiny-yolov2-voc")
        p.add_argument("--weights", type=Path, default=None)
        p.add_argument("--calib", type=Path, default=None)
        p.add_argument("--model-dir", type=Path, default=None, help="exported integer model")
        datasets.add_args(p)
        p.add_argument("--subset", type=int, default=500)
        p.add_argument("--jobs", type=int, default=16)
        p.add_argument("--seeds", default="0,1,2")
        p.add_argument("--no-abft", action="store_true", help="no ABFT check (faster)")
        if name == "run":
            p.add_argument("--grid", choices=("ber", "layer", "bit"), required=True)
            p.add_argument("--sites", default=",".join(SITES))
            p.add_argument("--bers", default=None, help=f"ber grid (default {BERS})")
            p.add_argument("--layer-bers", default=None,
                           help=f"layer grid (default {LAYER_BERS})")
            p.add_argument("--layers", default=None, help="layer grid: these convs only")
            p.add_argument("--flips", type=int, default=100,
                           help="bit grid: flips per image (act, acc) or per model")
            p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("widths", help="ABFT checksum widths and lane cost (T25.2)")
    p.add_argument("model_dirs", nargs="+")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    {"run": cmd_run, "plan": cmd_plan, "report": cmd_report, "widths": cmd_widths}[args.cmd](args)


if __name__ == "__main__":
    main()
