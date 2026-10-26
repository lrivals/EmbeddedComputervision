"""Person detection measured as a safety function (M27): misses at fixed FPPI, three stages,
operating point, degraded conditions, failures per hour, unflagged misses.

    # 1. detections down to 0.001, float and integer (T27.1), and degraded (T27.3)
    python tools/person_safety.py infer --net tiny-yolov4-coco --dataset prewid --split test
    python tools/person_safety.py infer --net tiny-yolov4-coco --dataset prewid --split test \
        --degrade all
    # 2. C-sim stage on a subset (C-sim == integer is bit-exact; the column confirms it)
    python tools/person_safety.py csim --net tiny-yolov4-coco --dataset prewid --split test
    # 3. tables of results/person_safety.md
    python tools/person_safety.py report --dataset prewid --split test \
        --nets tiny-yolov3-coco,tiny-yolov4-coco
    python tools/person_safety.py oppoint --dataset prewid --split test --net tiny-yolov4-coco
    python tools/person_safety.py degrade-report --dataset prewid --split test \
        --net tiny-yolov4-coco
    python tools/person_safety.py deel --events build/m27/prewid-test/tiny-yolov4-coco/events_int.json
    python tools/person_safety.py unflagged --dataset prewid --split test --net tiny-yolov4-coco

Detections are stored per image in `build/m27/<dataset>-<split>/<net>/<stage>[__<degradation>].jsonl`
(`image`, VOC pixel corners, scores, labels in the model's classes), so the metrics are
recomputed without inference. Metrics: `yolo.infer.metrics.person_metrics` (VOC matching at
IoU 0.5, `difficult` and ignore regions as in the mAP). The video events (T27.4) come from
`tools/dreaded_event.py`.
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import degrade as D  # noqa: E402
from yolo.data import datasets  # noqa: E402
from yolo.data.letterbox import input_size, parse_size  # noqa: E402
from yolo.infer import metrics as M  # noqa: E402
from yolo.infer.nms import IOU_THR  # noqa: E402
from yolo.infer.pipeline import MODES  # noqa: E402
from yolo.models.tiny_yolo import PRETRAINED, load_cfg  # noqa: E402

BUILD = ROOT / "build" / "m27"
RESULTS = ROOT / "results" / "person_safety.md"
CONF = 0.001
STAGES = ("float", "int")
STAGE_NAMES = {"float": "float", "int": "integer", "csim": "C-sim"}
REPO_THRESHOLDS = (0.005, 0.5)  # threshold of the mAP, of the demo
TARGET_RECALLS = (0.95, 0.99)
# IEC 61508-1 table 3, high demand or continuous mode: SIL n holds while the average
# frequency of dangerous failure stays below these upper bounds (per hour).
SIL_PFH = {1: 1e-5, 2: 1e-6, 3: 1e-7}
ALL_DATASETS = sorted(datasets.DATASETS) + sorted(datasets.EVAL_ONLY)


# ------------------------------------------------------------------------------- files
def run_dir(dataset, split, net, out=BUILD):
    tag = datasets.tag(dataset, split) or f"voc-{split.replace(':', '')}"
    return out / tag / Path(net).stem


def dets_path(dataset, split, net, stage, spec=None, out=BUILD):
    suffix = f"__{spec.replace(':', '-')}" if spec else ""
    return run_dir(dataset, split, net, out) / f"{stage}{suffix}.jsonl"


def read_dets(path):
    """{image: (VOC corners (n, 4), scores (n,), labels (n,))}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["image"]] = (np.asarray(d["boxes"], np.float64).reshape(-1, 4),
                               np.asarray(d["scores"], np.float64),
                               np.asarray(d["labels"], np.int64))
    return out


def write_section(key, title, body, path=None):
    """Replaces (or appends) the section between `<!-- m27:key -->` markers of `path`
    (default `RESULTS`)."""
    path = path or RESULTS
    start, end = f"<!-- m27:{key} -->", f"<!-- /m27:{key} -->"
    text = path.read_text() if path.exists() else HEADER
    block = f"{start}\n## {title}\n\n{body.rstrip()}\n{end}"
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    print(f"→ {path} ({key})")


HEADER = """# Person detection as a safety function (M27)

The detector measured the way a safety function is judged: a missed person is the dreaded
event, a false alarm an unwanted stop
([docs/tasks/M27-safety-metrics.md](../docs/tasks/M27-safety-metrics.md)). Tools:
`tools/person_safety.py`, `tools/degrade.py`, `tools/dreaded_event.py`; metrics in
`python/yolo/infer/metrics.py`. Matching: VOC rule at IoU 0.5, `difficult` boxes and ignore
regions neither hit nor false alarm; detections down to 0.001.

Notation: **FPPI** false positives per image; **R@0.1** recall at the lowest threshold whose
FPPI ≤ 0.1; **MR** = 1 − recall; **LAMR** log-average miss rate over FPPI ∈ [10⁻², 10⁰]
(Caltech, Dollár 2012; lower is better).
"""


# ---------------------------------------------------------------------------- inference
_W = {}


def _init(net, weights, calib, conf, iou, stages, seed):
    import eval_quant

    eval_quant._init(net, weights, calib, conf, iou)
    _W.update(eval_quant._W, stages=stages, seed=seed)


def _run(task):
    from PIL import Image

    from yolo.infer.metrics import to_voc_pixels
    from yolo.infer.pipeline import detect, detect_int, preprocess

    if "error" in _W:
        raise _W["error"]
    image_id, path, width, height, size, mode, channels, spec, boxes = task
    with Image.open(path) as im:
        img = im.convert("RGB")
        if spec:
            img = Image.fromarray(D.apply(np.asarray(img), spec, D.rng_for(_W["seed"], image_id),
                                          boxes))
        x, wh = preprocess(img, size, mode, channels=channels)
    out = {}
    for stage in _W["stages"]:
        if stage == "float":
            dets = detect(_W["fused"], x[None], [wh], mode, _W["conf"], _W["iou"])
        else:
            dets = detect_int(_W["inet"], _W["luts"], x[None], [wh], mode, _W["conf"],
                              _W["iou"])
        b, s, lab = dets[0]
        px = to_voc_pixels(np.asarray(b, np.float64).reshape(-1, 4), width, height)
        out[stage] = {"image": image_id, "boxes": px.tolist(),
                      "scores": [float(v) for v in s], "labels": [int(c) for c in lab]}
    return out


def net_defaults(args):
    from calibrate import net_tag

    if args.weights is None and args.net in PRETRAINED:
        args.weights = ROOT / "weights" / PRETRAINED[args.net]
    args.calib = args.calib or ROOT / "build" / "quant" / net_tag(args.net) / "calib.json"


def load_samples(dataset, split, data_root=None, subset=0):
    s = datasets.load(dataset, data_root, split)
    return s[:subset] if subset else s


def cmd_infer(args):
    net_defaults(args)
    split = args.split or datasets.info(args.dataset).test
    samples = load_samples(args.dataset, split, args.data_root, args.subset)
    cfg = load_cfg(args.net)
    size, channels = input_size(cfg, args.size), cfg["input"][0]
    stages = args.stages.split(",")
    specs = [None] if not args.degrade else (
        D.all_specs() if args.degrade == "all" else args.degrade.split(","))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.blas_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.blas_threads))
    ctx = mp.get_context("spawn")
    with ctx.Pool(args.jobs, initializer=_init,
                  initargs=(args.net, args.weights, args.calib, args.conf, args.iou, stages,
                            args.seed)) as pool:
        for spec in specs:
            paths = {st: dets_path(args.dataset, split, args.net, st, spec, args.out)
                     for st in stages}
            if not args.force and all(p.exists() for p in paths.values()):
                print(f"{spec or 'clean'}: already there ({paths[stages[0]].parent})")
                continue
            paths[stages[0]].parent.mkdir(parents=True, exist_ok=True)
            tasks = [(s["id"], s["image"], s["width"], s["height"], size, args.resize,
                      channels, spec, D.sample_boxes(s) if spec else None) for s in samples]
            files = {st: open(p.with_suffix(".tmp"), "w") for st, p in paths.items()}  # noqa: SIM115
            t0 = time.time()
            for k, res in enumerate(pool.imap(_run, tasks, chunksize=2), 1):
                for st, d in res.items():
                    files[st].write(json.dumps(d) + "\n")
                if k % 200 == 0 or k == len(tasks):
                    el = time.time() - t0
                    print(f"{spec or 'clean'} {k}/{len(tasks)}  {el:5.0f} s "
                          f"({el / k * 1000:.0f} ms/image)", flush=True)
            for st, f in files.items():
                f.close()
                paths[st].with_suffix(".tmp").replace(paths[st])
                print(f"→ {paths[st]}")


# ------------------------------------------------------------------------------ metrics
def eval_samples(dataset, samples, n_classes):
    """(samples reduced to the evaluated person class 0, model class of person)."""
    view = datasets.eval_view(dataset, n_classes)
    if "person" not in view.names:
        raise SystemExit(f"{dataset}: no person class for a {n_classes}-class model")
    p = view.names.index("person")
    lut = np.where(view.gt_lut == p, 0, -1)
    det_class = int(np.nonzero(view.det_lut == p)[0][0])
    return datasets.remap(samples, lut), det_class


def person_dets(dets, samples, det_class):
    """(ids, scores, corners) of the person class over `samples` (images absent from `dets`
    raise: a partial run is not a measurement)."""
    ids, scores, boxes = [], [], []
    for s in samples:
        b, sc, lab = dets[s["id"]]
        m = lab == det_class
        ids += [s["id"]] * int(m.sum())
        scores.append(sc[m])
        boxes.append(b[m])
    return (ids, np.concatenate(scores) if scores else np.zeros(0),
            np.concatenate(boxes).reshape(-1, 4) if boxes else np.zeros((0, 4)))


def stage_metrics(dataset, split, net, stage, samples=None, spec=None, data_root=None,
                  out=BUILD, **kw):
    """`person_metrics` of a stored run; `samples` restricts the images (C-sim subset,
    camera)."""
    path = dets_path(dataset, split, net, stage, spec, out)
    if not path.exists():
        return None
    dets = read_dets(path)
    if samples is None:
        samples = datasets.load(dataset, data_root, split)
    samples = [s for s in samples if s["id"] in dets]
    ev, det_class = eval_samples(dataset, samples, load_cfg(net)["classes"])
    return M.person_metrics(person_dets(dets, ev, det_class), ev, **kw)


def pct(v):
    return "–" if v is None or v != v else f"{100 * v:.1f}"


def frac(found_total):
    f, t = found_total
    return f"{100 * f / t:.1f} ({f}/{t})" if t else "–"


def bin_labels(edges, unit=""):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        if math.isinf(b):
            out.append(f"≥ {a:g}{unit}")
        elif a == 0:
            out.append(f"< {b:g}{unit}")
        else:
            out.append(f"{a:g}-{b:g}{unit}")
    return out


def occl_labels(edges):
    return [f"{100 * a:.0f}-{min(100, 100 * b):.0f} %" for a, b in zip(edges[:-1], edges[1:])]


def section_tag(dataset, split):
    return datasets.tag(dataset, split) or "voc"


def has_empty(samples):
    return any(len(s["labels"]) == 0 for s in samples)


def cmd_report(args):
    split = args.split or datasets.info(args.dataset).test
    samples = datasets.load(args.dataset, args.data_root, split)
    nets = args.nets.split(",")
    stages = args.stages.split(",")
    rows, hrows, orows, loss, summary = [], [], [], [], {}
    show_empty = has_empty(samples)
    for net in nets:
        got = {}
        for st in stages:
            if st == "csim":
                continue
            m = stage_metrics(args.dataset, split, net, st, samples, out=args.out)
            if m is None:
                print(f"{net} {st}: no run ({dets_path(args.dataset, split, net, st, None, args.out)})")
                continue
            got[st] = m
            me = stage_metrics(args.dataset, split, net, st, samples, out=args.out,
                               empty_only=True) if show_empty else None
            rows.append(metric_row(net, STAGE_NAMES[st], m, me))
            hrows.append([net, STAGE_NAMES[st]] + [frac(x) for x in m["at"][0.1]["height"]])
            if not np.isnan(m["occlusion"]).all():
                orows.append([net, STAGE_NAMES[st]] + [frac(x) for x in m["at"][0.1]["occlusion"]])
            summary[f"{net}/{st}"] = summarize(m)
        if "float" in got and "int" in got:
            loss.append([net, "quantization (integer − float)", "all",
                         delta(got["int"], got["float"])])
        if "csim" in stages:
            cs = csim_subset(args.dataset, split, net, args.out)
            if cs is not None:
                sub = [s for s in samples if s["id"] in cs]
                mi = stage_metrics(args.dataset, split, net, "int", sub, out=args.out)
                mc = stage_metrics(args.dataset, split, net, "csim", sub, out=args.out)
                if mi is not None and mc is not None:
                    pad = ["–", "–"] if show_empty else []
                    rows.append(metric_row(net, f"C-sim ({len(sub)} images)", mc, None) + pad)
                    rows.append(metric_row(net, f"integer ({len(sub)} images)", mi, None) + pad)
                    loss.append([net, "engine (C-sim − integer)", f"{len(sub)} images",
                                 delta(mc, mi)])
    head = ["Net", "Stage", "Images", "Persons", "R@0.1", "R@0.01", "MR@0.1", "LAMR",
            "thr@0.1"]
    if show_empty:
        head += ["R@0.1 (FPPI on empty)", "empty frames"]
    body = [f"{args.dataset} `{split}`, {len(samples)} images. Recall and miss rate in %; "
            "per-bin recall at the R@0.1 threshold of each row.", "",
            table(head, rows), "", "**Recall per box height** (pixels of the original image, "
            "Caltech bins), at 0.1 FPPI", "",
            table(["Net", "Stage"] + bin_labels(M.HEIGHT_EDGES, " px"), hrows)]
    if orows:
        body += ["", "**Recall per occlusion** (1 − visible / full box), at 0.1 FPPI", "",
                 table(["Net", "Stage"] + occl_labels(M.OCCLUSION_EDGES), orows)]
    if loss:
        body += ["", "**Loss per stage** (difference of R@0.1 and LAMR, points)", "",
                 table(["Net", "Step", "Images", "ΔR@0.1 / ΔLAMR"], loss)]
    if args.per_camera and any("camera" in s for s in samples):
        body += ["", per_camera(args, split, samples, nets, stages)]
    key = f"stages-{section_tag(args.dataset, split)}"
    out = run_dir(args.dataset, split, nets[0], args.out).parent / "report.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(body))
    if args.write:
        write_section(key, f"Three stages: {args.dataset} {split} (T27.1)", "\n".join(body))


def per_camera(args, split, samples, nets, stages):
    rows = []
    for cam in sorted({s["camera"] for s in samples if "camera" in s}):
        sub = [s for s in samples if s.get("camera") == cam]
        for net in nets:
            for st in stages:
                if st == "csim":
                    continue
                m = stage_metrics(args.dataset, split, net, st, sub, out=args.out)
                if m is not None:
                    me = stage_metrics(args.dataset, split, net, st, sub, out=args.out,
                                       empty_only=True) if has_empty(sub) else None
                    rows.append([cam] + metric_row(net, STAGE_NAMES[st], m, me))
    head = ["Camera", "Net", "Stage", "Images", "Persons", "R@0.1", "R@0.01", "MR@0.1",
            "LAMR", "thr@0.1", "R@0.1 (FPPI on empty)", "empty frames"]
    return "**Per camera**\n\n" + table(head, rows)


def metric_row(net, stage, m, m_empty):
    a1, a2 = m["at"][0.1], m["at"][0.01]
    row = [net, stage, m["n_images"], m["npos"], pct(a1["recall"]), pct(a2["recall"]),
           pct(1 - a1["recall"]), pct(m["lamr"]), f"{a1['thr']:.3g}"]
    if m_empty is not None:
        row += [pct(m_empty["at"][0.1]["recall"]), m_empty["n_images"]]
    return row


def delta(a, b):
    return (f"{100 * (a['at'][0.1]['recall'] - b['at'][0.1]['recall']):+.1f} / "
            f"{100 * (a['lamr'] - b['lamr']):+.1f}")


def summarize(m):
    c = m["curve"]
    keep = thin(c["fppi"])
    return {"npos": m["npos"], "n_images": m["n_images"], "lamr": m["lamr"],
            "at": {str(f): {"recall": v["recall"], "thr": v["thr"]} for f, v in m["at"].items()},
            "curve": {k: [float(x) for x in np.asarray(c[k])[keep]] for k in ("thr", "fppi", "mr")}}


def thin(fppi, n=400):
    """Indices of at most ~n points spread over log FPPI (for the figures)."""
    f = np.asarray(fppi)
    if len(f) <= n:
        return np.arange(len(f))
    grid = np.logspace(-4, math.log10(max(f[-1], 1e-3)), n)
    idx = np.unique(np.r_[0, np.searchsorted(f, grid, side="right") - 1, len(f) - 1])
    return idx[idx >= 0]


def table(head, rows):
    out = ["| " + " | ".join(map(str, head)) + " |", "|" + "---|" * len(head)]
    out += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(out)


# ------------------------------------------------------------------------------- C-sim
def csim_subset(dataset, split, net, out=BUILD):
    p = dets_path(dataset, split, net, "csim", None, out)
    return set(read_dets(p)) if p.exists() else None


def cmd_csim(args):
    """`sim/dets_*.jsonl` of `bench_sim.sh` (normalised boxes) → VOC corners in the stage
    file `csim.jsonl`, checked image by image against the integer stage."""
    from map_stades import read_jsonl

    from yolo.infer.metrics import to_voc_pixels

    split = args.split or datasets.info(args.dataset).test
    sim = sorted((args.dir / "sim").glob("dets_*.jsonl"))
    if not sim:
        raise SystemExit(f"no {args.dir}/sim/dets_*.jsonl: run make_inputs.py and "
                         "bench_sim.sh first (commands in the module docstring)")
    raw = read_jsonl(sim)
    by_id = {s["id"]: s for s in datasets.load(args.dataset, args.data_root, split)}
    ref = read_dets(dets_path(args.dataset, split, args.net, "int", None, args.out))
    path = dets_path(args.dataset, split, args.net, "csim", None, args.out)
    same = diff = 0
    with open(path, "w") as f:
        for i, (b, sc, lab) in raw.items():
            s = by_id[i]
            px = to_voc_pixels(np.asarray(b, np.float64).reshape(-1, 4), s["width"], s["height"])
            f.write(json.dumps({"image": i, "boxes": px.tolist(), "scores": list(sc),
                                "labels": list(lab)}) + "\n")
            rb, rs, rl = ref[i]
            keep = rs >= args.conf - 1e-12
            ok = (np.array_equal(rl[keep], np.asarray(lab, np.int64))
                  and np.array_equal(rs[keep], np.asarray(sc, np.float64))
                  and np.array_equal(rb[keep], px))
            same, diff = same + ok, diff + (not ok)
    print(f"C-sim: {len(raw)} images, {same} identical to the integer stage, {diff} different "
          f"(integer detections ≥ {args.conf}) → {path}")
    if diff:
        raise SystemExit(1)


# --------------------------------------------------------------------- operating point
def block_halves(samples, block):
    """Calibration / check halves: blocks of `block` consecutive frames of a sequence
    (camera, sequence, or id order), alternating, so neighbouring frames stay together."""
    def key(s):
        return (str(s.get("camera", s.get("sequence", ""))), s.get("frame", 0), s["id"])

    order = sorted(samples, key=key)
    cal, chk = [], []
    groups = {}
    for s in order:
        groups.setdefault(key(s)[0], []).append(s)
    k = 0
    for g in groups.values():
        for i in range(0, len(g), block):
            (cal if k % 2 == 0 else chk).extend(g[i:i + block])
            k += 1
    return cal, chk


def cmd_oppoint(args):
    split = args.split or datasets.info(args.dataset).test
    samples = datasets.load(args.dataset, args.data_root, split)
    cal, chk = block_halves(samples, args.block)
    targets = [float(t) for t in args.targets.split(",")]
    rows, place, curves = [], [], {}
    for st in args.stages.split(","):
        m_all = stage_metrics(args.dataset, split, args.net, st, samples, out=args.out)
        if m_all is None:
            print(f"{st}: no run")
            continue
        m_cal = stage_metrics(args.dataset, split, args.net, st, cal, out=args.out)
        m_chk = stage_metrics(args.dataset, split, args.net, st, chk, out=args.out)
        curves[st] = summarize(m_all)
        for t in targets:
            thr = M.threshold_for_recall(m_cal["curve"], t)
            if thr is None:
                rows.append([STAGE_NAMES[st], pct(t),
                             f"never reached (max {pct(m_cal['curve']['recall'][-1])} at "
                             f"{CONF})", "–", "–", "–"])
                continue
            r_cal, f_cal = M.point_at_threshold(m_cal["curve"], thr)
            r_chk, f_chk = M.point_at_threshold(m_chk["curve"], thr)
            rows.append([STAGE_NAMES[st], pct(t), f"{thr:.4g}", f"{pct(r_cal)} / {f_cal:.3f}",
                         f"{pct(r_chk)} / {f_chk:.3f}", "yes" if r_chk >= t else "**no**"])
            curves[st].setdefault("targets", {})[str(t)] = {"thr": thr, "recall": r_chk,
                                                            "fppi": f_chk}
        for thr in REPO_THRESHOLDS:
            r, f = M.point_at_threshold(m_all["curve"], thr)
            place.append([STAGE_NAMES[st], thr, pct(r), f"{f:.3f}"])
            curves[st].setdefault("repo", {})[str(thr)] = {"recall": r, "fppi": f}
    out = run_dir(args.dataset, split, args.net, args.out) / "oppoint.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.dataset, "split": split, "net": args.net,
                               "stages": curves}, indent=1) + "\n")
    body = [f"{args.net} on {args.dataset} `{split}`: threshold chosen on the calibration "
            f"half ({len(cal)} images), checked on the other half ({len(chk)} images); halves "
            f"made of alternating blocks of {args.block} consecutive frames per sequence, so "
            "neighbouring frames do not leak across. Recall in %.", "",
            table(["Stage", "Target recall", "Threshold", "Calibration recall / FPPI",
                   "Check recall / FPPI", "Target held"], rows), "",
            "**Repo thresholds on the curve** (all images)", "",
            table(["Stage", "Threshold", "Recall", "FPPI"], place), "",
            f"Curves: `{out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}`; figure "
            "`python -m tools.figures miss_fppi`."]
    print("\n".join(body))
    if args.write:
        write_section(f"oppoint-{section_tag(args.dataset, split)}-{Path(args.net).stem}",
                      f"Asymmetric operating point: {args.net}, {args.dataset} {split} (T27.2)",
                      "\n".join(body))


# ------------------------------------------------------------------------- degradations
def cmd_degrade_report(args):
    split = args.split or datasets.info(args.dataset).test
    samples = datasets.load(args.dataset, args.data_root, split)
    specs = D.all_specs() if args.specs == "all" else args.specs.split(",")
    stages = args.stages.split(",")
    clean = {st: stage_metrics(args.dataset, split, args.net, st, samples, out=args.out)
             for st in stages}
    rows = []
    for spec in [None] + specs:
        row = [spec or "clean"]
        for st in stages:
            m = clean[st] if spec is None else stage_metrics(args.dataset, split, args.net, st,
                                                             samples, spec, out=args.out)
            if m is None or clean[st] is None:
                row += ["–", "–"]
                continue
            r_fixed, f_fixed = M.point_at_threshold(m["curve"], clean[st]["at"][0.1]["thr"])
            row += [pct(m["at"][0.1]["recall"]), f"{pct(r_fixed)} / {f_fixed:.2f}"]
        if len(stages) == 2 and "–" not in row[1::2]:
            row.append(f"{float(row[3]) - float(row[1]):+.1f}")
        rows.append(row)
    head = ["Condition"]
    for st in stages:
        head += [f"R@0.1 {STAGE_NAMES[st]}", f"{STAGE_NAMES[st]} at clean thr: R / FPPI"]
    if len(stages) == 2:
        head.append("integer − float (R@0.1)")
    body = [f"{args.net} on {args.dataset} `{split}` ({len(samples)} images), seed "
            f"{args.seed}. **R@0.1**: recall at 0.1 FPPI with the threshold re-chosen on the "
            "degraded images; **at clean thr**: recall and FPPI when the threshold stays at "
            "the clean 0.1-FPPI operating point (what a deployed system would see). "
            "Degradations of `tools/degrade.py` (severity 1-3).", "", table(head, rows), "",
            "These six degradations are the ODD axes proposed to T31.0: lighting (low light), "
            "airborne particles (fog / dust), camera or subject motion (motion blur), sensor "
            "noise, compression (JPEG), partial occlusion by plant equipment."]
    print("\n".join(body))
    if args.write:
        write_section(f"degrade-{section_tag(args.dataset, split)}-{Path(args.net).stem}",
                      f"Degraded conditions: {args.net}, {args.dataset} {split} (T27.3)", "\n".join(body))


# ------------------------------------------------------------------------- DEEL (T27.5)
def failures_per_hour(fps, presence, p_frame, n_frames, measured_per_hour=None,
                      measured_presence=1.0):
    """Dangerous failures per hour: the person in the zone stays unreported for `n_frames`
    consecutive frames. Independence: a run of n misses starts at a present frame with
    probability (1 − p)·pⁿ (≈ pⁿ), over fps·3600·presence present frames per hour.
    `measured_per_hour`: rate of such runs measured on video (T27.4), per hour of video with
    a presence share `measured_presence`, rescaled to `presence`."""
    present = fps * 3600.0 * presence
    out = {"present_frames_per_hour": present,
           "independent": present * (1 - p_frame) * p_frame ** n_frames}
    if measured_per_hour is not None:
        out["measured"] = measured_per_hour * presence / max(measured_presence, 1e-12)
    return out


def sil_gap(rate):
    """Orders of magnitude above each SIL bound (positive: the bound is missed)."""
    return {k: (math.log10(rate / v) if rate > 0 else -math.inf) for k, v in SIL_PFH.items()}


def cmd_deel(args):
    ev = json.loads(Path(args.events).read_text())
    fps = args.fps or ev["fps"]
    n = args.n or max(1, int(math.ceil(args.reaction_s * fps)))
    p = ev["frame_fn_rate"] if args.p_frame is None else args.p_frame
    meas = ev["per_hour"].get(str(n)) if str(n) in ev["per_hour"] else None
    r = failures_per_hour(fps, args.presence, p, n, meas, ev.get("presence", 1.0))
    rows = []
    for name in ("independent", "measured"):
        if name in r:
            gap = sil_gap(r[name])
            rows.append([name, f"{r[name]:.3g}"] + [f"{gap[k]:+.1f}" for k in SIL_PFH])
    a = [["Frame rate", f"{fps:g} frames/s", ev.get("fps_source", "argument")],
         ["Presence in the zone", f"{100 * args.presence:g} % of the time",
          "assumption (`--presence`)"],
         ["Per-frame miss probability p", f"{p:.4g}",
          f"measured ({ev['net']} {ev['stage']}, threshold {ev['thr']:.3g}, "
          f"{ev['dataset']} {ev['split']}, {ev['mode']} mode)"],
         ["Frames unreported before harm (N)", f"{n}",
          f"`--n`, or reaction budget {args.reaction_s:g} s × fps"],
         ["Measured N-run rate on video", f"{meas:.3g} /h of video" if meas is not None
          else "–", f"T27.4, presence {100 * ev.get('presence', 1.0):.0f} % of the frames"],
         ["Independence", "misses independent from frame to frame", "row `independent`"],
         ["Measured correlation", "runs counted on consecutive frames", "row `measured`"],
         ["Every miss of N frames is dangerous", "no detection by other means", "conservative"],
         ["SIL bounds", "PFH < 10⁻⁵ / 10⁻⁶ / 10⁻⁷ per hour for SIL 1 / 2 / 3",
          "IEC 61508-1 table 3, high demand or continuous mode"]]
    body = ["DEEL (Delseny 2021, §4.2.4.4) redone with the measured rates: dangerous failure "
            "= a person present in the zone and not reported for N consecutive frames.", "",
            table(["Assumption", "Value", "Source"], a), "",
            table(["Model", "Dangerous failures / h", "vs SIL 1", "vs SIL 2", "vs SIL 3"],
                  rows), "",
            "Gaps in orders of magnitude (log10 of rate / bound; positive = bound missed). "
            f"Present frames per hour: {r['present_frames_per_hour']:.3g}."]
    print("\n".join(body))
    if args.write:
        write_section(f"deel-{ev['dataset']}", "From per-image errors to failures per hour "
                      "(T27.5)", "\n".join(body))


# ------------------------------------------------------------------- unflagged (T27.7)
def read_flags(path):
    """{image: alarm} from a JSONL of {image, alarm}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            d = json.loads(line)
            out[d["image"]] = bool(d["alarm"])
    return out


def episode_outcomes(eps, n, flags, deadline):
    """Per presence episode of at least `n` frames (T27.4): failed = it holds a miss run of
    length ≥ n; flagged = every such run has an alarm in its first `deadline` frames."""
    failed, flagged = [], []
    for e in eps:
        if len(e["images"]) < n:
            continue
        long_runs = [(a, L) for a, L in e["runs"] if L >= n]
        failed.append(bool(long_runs))
        flagged.append(bool(long_runs) and all(
            any(flags.get(i, False) for i in e["images"][a:a + min(L, deadline)])
            for a, L in long_runs))
    return failed, flagged


def cmd_unflagged(args):
    split = args.split or datasets.info(args.dataset).test
    samples = datasets.load(args.dataset, args.data_root, split)
    m = stage_metrics(args.dataset, split, args.net, args.stage, samples, out=args.out)
    if m is None:
        raise SystemExit("no run: person_safety.py infer first")
    thr = m["at"][args.fppi]["thr"] if args.fppi in m["at"] else \
        M.recall_at_fppi(m["curve"], args.fppi)[1]
    ev, _ = eval_samples(args.dataset, samples, load_cfg(args.net)["classes"])
    img_of = np.concatenate([[s["id"]] * len(s["labels"]) for s in ev]) if ev else np.zeros(0)
    keep = ~np.isnan(m["gt_score"])
    missed = (m["gt_score"] < thr)[keep]
    imgs = img_of[keep]
    monitors = [("detector alone", {})] + [(Path(f).stem, read_flags(f)) for f in args.flags]
    rows = []
    for name, flags in monitors:
        flagged = np.array([flags.get(i, False) for i in imgs], bool)
        r = M.unflagged_miss_rate(missed, flagged)
        alarm = np.mean([flags.get(s["id"], False) for s in samples]) if flags else 0.0
        rows.append([name, "person", f"{r['k']} / {r['n']}", f"{100 * r['rate']:.2f}",
                     f"[{100 * r['ci'][0]:.2f}, {100 * r['ci'][1]:.2f}]", pct(alarm)])
    if args.events:
        eps = json.loads(Path(args.events).read_text())["episode_list"]
        for name, flags in monitors:
            failed, flagged = episode_outcomes(eps, args.event_n, flags, args.deadline)
            r = M.unflagged_miss_rate(failed, flagged)
            rows.append([name, f"episode (N = {args.event_n})", f"{r['k']} / {r['n']}",
                         f"{100 * r['rate']:.2f}",
                         f"[{100 * r['ci'][0]:.2f}, {100 * r['ci'][1]:.2f}]", "–"])
    body = [f"{args.net} {STAGE_NAMES[args.stage]} on {args.dataset} `{split}`, operating "
            f"point {args.fppi} FPPI (threshold {thr:.3g}). Unflagged miss: a person missed "
            "by the detector **and** in an image the monitor does not flag (per person); a "
            "presence episode of at least N frames holding a run of N or more missed frames "
            f"not flagged within its first {args.deadline} frame(s) (per episode, T27.4); "
            "over all the persons (episodes). 95 % Clopper-Pearson interval; the alarm rate of each monitor is given "
            "apart, never folded in.", "",
            table(["Monitor", "Unit", "Unflagged / total", "Rate %", "95 % CI", "Alarms %"],
                  rows)]
    if not args.flags:
        body += ["", "Monitor rows pending: T29.1 (`tools/ood_monitor.py compare`), T39.1 "
                 "(`tools/failure_prediction.py`) and M40, at the same alarm rate; pass their "
                 "per-image alarms with `--flags`."]
    print("\n".join(body))
    if args.write:
        write_section(f"unflagged-{section_tag(args.dataset, split)}-{Path(args.net).stem}",
                      f"Unflagged dangerous misses: {args.net}, {args.dataset} {split} (T27.7)",
                      "\n".join(body))


# ---------------------------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, net=True):
        p.add_argument("--dataset", choices=ALL_DATASETS, default="prewid")
        p.add_argument("--data-root", type=Path, default=None)
        p.add_argument("--split", default=None)
        p.add_argument("--out", type=Path, default=BUILD, help="root of the runs")
        if net:
            p.add_argument("--net", default="tiny-yolov4-coco")
        return p

    p = common(sub.add_parser("infer", help="detections of each stage (T27.1, T27.3)"))
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--stages", default="float,int")
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--iou", type=float, default=IOU_THR)
    p.add_argument("--resize", choices=MODES, default="stretch")
    p.add_argument("--size", type=parse_size, default=None)
    p.add_argument("--degrade", default=None, help="`all` or kind:sev[,kind:sev…]")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--blas-threads", type=int, default=1)
    p.add_argument("--force", action="store_true")

    p = common(sub.add_parser("csim", help="C-sim stage from bench_sim.sh outputs"))
    p.add_argument("--dir", type=Path, required=True, help="make_inputs.py folder (ids.txt, sim/)")
    p.add_argument("--conf", type=float, default=CONF, help="threshold given to yolo_bench")

    p = common(sub.add_parser("report", help="three-stage tables (T27.1)"), net=False)
    p.add_argument("--nets", default="tiny-yolov3-coco,tiny-yolov4-coco")
    p.add_argument("--stages", default="float,int,csim")
    p.add_argument("--per-camera", action="store_true")
    p.add_argument("--write", action="store_true", help="section of results/person_safety.md")

    p = common(sub.add_parser("oppoint", help="threshold for a target recall (T27.2)"))
    p.add_argument("--stages", default="float,int")
    p.add_argument("--targets", default=",".join(map(str, TARGET_RECALLS)))
    p.add_argument("--block", type=int, default=25)
    p.add_argument("--write", action="store_true")

    p = common(sub.add_parser("degrade-report", help="recall under degradations (T27.3)"))
    p.add_argument("--stages", default="float,int")
    p.add_argument("--specs", default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--write", action="store_true")

    p = sub.add_parser("deel", help="dangerous failures per hour (T27.5)")
    p.add_argument("--events", type=Path, required=True, help="JSON of dreaded_event.py")
    p.add_argument("--fps", type=float, default=None, help="default: the one of the events")
    p.add_argument("--presence", type=float, default=0.1, help="share of time a person is in the zone")
    p.add_argument("--reaction-s", type=float, default=0.5,
                   help="time a person may stay unreported before harm")
    p.add_argument("--n", type=int, default=None, help="N frames (overrides --reaction-s)")
    p.add_argument("--p-frame", type=float, default=None)
    p.add_argument("--write", action="store_true")

    p = common(sub.add_parser("unflagged", help="unflagged dangerous misses (T27.7)"))
    p.add_argument("--stage", default="int")
    p.add_argument("--fppi", type=float, default=0.1)
    p.add_argument("--flags", nargs="*", default=[], help="JSONL {image, alarm} per monitor")
    p.add_argument("--events", type=Path, default=None, help="JSON of dreaded_event.py")
    p.add_argument("--event-n", type=int, default=3)
    p.add_argument("--deadline", type=int, default=1, help="frames of an event a flag may come in")
    p.add_argument("--write", action="store_true")

    args = ap.parse_args(argv)
    {"infer": cmd_infer, "csim": cmd_csim, "report": cmd_report, "oppoint": cmd_oppoint,
     "degrade-report": cmd_degrade_report, "deel": cmd_deel,
     "unflagged": cmd_unflagged}[args.cmd](args)


if __name__ == "__main__":
    main()
