"""M36: explanation maps of the detector at the float, integer and golden stages.

    python tools/explain.py map --net tiny-yolov2-voc --image 000004 --methods gradcam,drise
    python tools/explain.py sanity   --n 20 --jobs 20   # T36.1: IG completeness, Adebayo
    python tools/explain.py stages   --n 20 --jobs 20   # T36.2: float / int / golden D-RISE
    python tools/explain.py intattr  --n 20 --jobs 20   # T36.3: exact split, integer LRP
    python tools/explain.py evaluate --n 50 --jobs 20   # T36.4: pointing, deletion, insertion
    python tools/explain.py misses   --n 500 --jobs 20  # T36.5: causes of the missed persons
    python tools/explain.py misses   --dataset exdark   # same on ExDark (keys <net>@exdark)
    python tools/explain.py concepts --jobs 20          # T36.6: dissection, TCAV
    python tools/explain.py cka      --jobs 20          # T36.8: float / int, checkpoints
    python tools/explain.py record   --run build/explain/record/clean  # T36.7 summary
    python tools/explain.py report                      # results/explain.md, PNGs

Images: VOC2007 test in file order (the first `--n`), `stretch` preprocessing at 416 as
`tools/ood_monitor.py`, so the normalised boxes are those of the original image. Networks:
Tiny-YOLOv2 VOC, Tiny-YOLOv3 and v4 COCO (VOC classes mapped to COCO,
`datasets.class_lut`). Explained box of an image: the float detection of best score at the
operating point (conf 0.25, NMS 0.45) that matches a ground truth of its class (IoU ≥ 0.5);
images without one are skipped. Integer model: `QuantModel` of the calibration
`build/quant/<net>/calib.json`, the one of the exports `model/<net>/` run by the golden.

Every command writes `build/explain/<command>.json`; `report` turns them into the tables
of `results/explain.md` (between `<!-- explain:<section>:begin/end -->`) and the PNGs of
`results/explain/`.
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
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

NETS = ("tiny-yolov2-voc", "tiny-yolov3-coco", "tiny-yolov4-coco")
OUT = ROOT / "build" / "explain"
RESULTS = ROOT / "results" / "explain.md"
PNG = ROOT / "results" / "explain"
SIZE = 416
CONF, IOU, MATCH = 0.25, 0.45, 0.5
GRAD_METHODS = ("saliency", "smoothgrad", "ig_black", "ig_blur", "gradcam")
DRISE_MASKS = 300
GOLDEN_MASKS = 100
DEPTHS = (1, 3, None)  # cascading randomization: top conv, top 3, every conv

_W = {}


# ------------------------------------------------------------------------ models
def models(name):
    """(float folded net (float32), QuantModel, IntNetwork, head LUTs), cached per process."""
    if name not in _W:
        from calibrate import load_fused

        from yolo.explain.targets import explain_net
        from yolo.quant.calibrate import load_scales
        from yolo.quant.int_model import IntNetwork, QuantModel, head_luts

        qm = QuantModel.from_fused(load_fused(name, dtype=np.float64),
                                   *load_scales(ROOT / "build" / "quant" / name / "calib.json"))
        _W[name] = (explain_net(name, dtype=np.float32), qm, IntNetwork(qm), head_luts(qm))
    return _W[name]


def samples(n, split="2007:test", only_class=None):
    from yolo.data import datasets

    s = datasets.load("voc", split=split)
    if only_class is not None:
        s = [x for x in s if (np.asarray(x["labels"]) == only_class).any()]
    return s[:n]


def load_image(path):
    from PIL import Image

    from yolo.infer.pipeline import preprocess

    with Image.open(path) as img:
        x, wh = preprocess(img, SIZE, "stretch")
    return x[None], wh


def gt_of(name, s, dataset="voc"):
    """Ground truth of sample `s` of `dataset` in the model's classes (`MAPPINGS`):
    (boxes (k, 4), labels (k,))."""
    from yolo.data import datasets

    net, _, _, _ = models(name)
    lut = datasets.class_lut(dataset, net.net["classes"])
    lab = lut[np.asarray(s["labels"], dtype=np.int64)]
    keep = (lab >= 0) & ~np.asarray(s["difficult"], dtype=bool)
    return np.asarray(s["boxes"], dtype=np.float64)[keep], lab[keep]


def pick(name, s, x):
    """Target of the best float detection matching a ground truth, or None."""
    from yolo.explain.targets import detections
    from yolo.infer.boxes import iou

    net = models(name)[0]
    boxes, labels = gt_of(name, s)
    if not len(boxes):
        return None
    for d in detections(net.forward(x, train=False), net.net, CONF, IOU):
        k = labels == d.cls
        if k.any() and (iou(np.array([d.box]), boxes[k])[0] >= MATCH).any():
            return d
    return None


def stage(name, kind):
    from yolo.explain import perturb as P

    net, qm, inet, luts = models(name)
    if kind == "float":
        return P.float_stage(net)
    if kind == "int":
        return P.int_stage(inet, luts)
    if kind == "golden":
        return P.golden_stage(ROOT / "model" / name, qm, luts, jobs=1)
    raise ValueError(kind)


def _timed(fn, *a, **kw):
    t0 = time.perf_counter()
    r = fn(*a, **kw)
    r["seconds"] = time.perf_counter() - t0
    return r


def _abs(m):
    return np.abs(m)


# ------------------------------------------------------------------ T36.1 sanity
def _sanity(task):
    from yolo.explain import gradient as G
    from yolo.explain.maps import rank_corr

    name, s = task
    net = models(name)[0]
    x, _ = load_image(s["image"])
    t = pick(name, s, x)
    if t is None:
        return None
    out = {"id": s["id"], "methods": {}}
    n_conv = len(G.conv_ids(net))
    for m in GRAD_METHODS:
        r = _timed(G.METHODS[m], net, x, t)
        ref = _abs(r["map"])
        row = {"passes": r["passes"], "seconds": r["seconds"], "rand": {}}
        if "completeness" in r:
            row["completeness"] = r["completeness"]
        for d in DEPTHS:
            k = n_conv if d is None else d
            rnet = G.randomize_from_top(net, k, rng=k)
            row["rand"][str(d or "all")] = rank_corr(ref, _abs(G.METHODS[m](rnet, x, t)["map"]))
        out["methods"][m] = row
    return out


def cmd_sanity(args):
    res = run_pool(args, _sanity, {n: [(n, s) for s in samples(args.n)] for n in args.nets})
    save("sanity", res)


# ------------------------------------------------------------------ T36.2 stages
def _stages(task):
    from yolo.explain import perturb as P
    from yolo.explain.maps import rank_corr

    name, s, golden = task
    x, _ = load_image(s["image"])
    t = pick(name, s, x)
    if t is None:
        return None
    out = {"id": s["id"]}
    masks = P.rise_masks(DRISE_MASKS, SIZE, SIZE, rng=0)
    maps = {}
    for kind in ("float", "int"):
        st = stage(name, kind)
        maps[("drise", kind)] = _timed(P.drise, st, x, t.box, t.cls, masks=masks)
        maps[("occlusion", kind)] = _timed(P.occlusion, st, x, t.box, t.cls)
    for m in ("drise", "occlusion"):
        out[m] = {"spearman_float_int": rank_corr(maps[(m, "float")]["map"],
                                                  maps[(m, "int")]["map"]),
                  "seconds": {k: maps[(m, k)]["seconds"] for k in ("float", "int")},
                  "passes": maps[(m, "float")]["passes"]}
    if golden:
        g = masks[:GOLDEN_MASKS]
        a = _timed(P.drise, stage(name, "int"), x, t.box, t.cls, masks=g)
        b = _timed(P.drise, stage(name, "golden"), x, t.box, t.cls, masks=g)
        out["golden"] = {"masks": len(g), "weights_equal": bool(np.array_equal(a["weights"],
                                                                                b["weights"])),
                         "map_equal": bool(np.array_equal(a["map"], b["map"])),
                         "seconds": b["seconds"]}
    return out


def cmd_stages(args):
    tasks = {n: [(n, s, k < args.golden) for k, s in enumerate(samples(args.n))]
             for n in args.nets}
    save("stages", run_pool(args, _stages, tasks))


# ----------------------------------------------------------------- T36.3 intattr
def _intattr(task):
    from yolo.explain import gradient as G
    from yolo.explain import int_attr as IA
    from yolo.explain.maps import rank_corr
    from yolo.quant.int_model import IntNetwork
    from yolo.quant.quantize import quantize_input

    name, s = task
    net, qm, _, _ = models(name)
    x, _ = load_image(s["image"])
    t = pick(name, s, x)
    if t is None:
        return None
    qx = quantize_input(x)
    hook = IA.KeepInputs()
    inet = IntNetwork(qm)
    inet.hooks.append(hook)
    outs = inet.forward(qx, all_outputs=True)
    bad = IA.check_exact(hook, qm.convs, n=16)
    out = {"id": s["id"], "exact_checked": 16 * len(bad), "exact_bad": sum(bad.values())}
    t0 = time.perf_counter()
    r_eps = IA.lrp_int(inet, qx, t, "eps", outs=outs)
    out["seconds_int"] = time.perf_counter() - t0
    r_zp = IA.lrp_int(inet, qx, t, "zplus", outs=outs)
    f_eps = IA.lrp_float(net, x.astype(np.float64), t, "eps")
    cam = G.gradcam(net, x, t)["map"]
    out.update({
        "conservation_eps": {str(k): v for k, v in r_eps.conservation.items()},
        "conservation_zplus": {str(k): v for k, v in r_zp.conservation.items()},
        "total_eps": r_eps.total, "total_zplus": r_zp.total,
        "spearman_int_float_lrp": rank_corr(r_eps.map, f_eps.map),
        "spearman_int_lrp_gradcam": rank_corr(r_eps.map, cam),
        "spearman_zplus_gradcam": rank_corr(r_zp.map, cam),
    })
    return out


def cmd_intattr(args):
    save("intattr", run_pool(args, _intattr, {n: [(n, s) for s in samples(args.n)]
                                              for n in args.nets}))


# ---------------------------------------------------------------- T36.4 evaluate
EVAL_METHODS = (("saliency", "float"), ("smoothgrad", "float"), ("ig_black", "float"),
                ("ig_blur", "float"), ("gradcam", "float"), ("occlusion", "float"),
                ("drise", "float"), ("lrp_float", "float"), ("lrp_int", "int"),
                ("drise", "int"), ("random", "float"), ("centre", "float"),
                ("random", "int"), ("centre", "int"))


def make_map(name, method, kind, x, t):
    """{map, passes, seconds} of one method at one stage."""
    from yolo.explain import evaluate as E
    from yolo.explain import gradient as G
    from yolo.explain import int_attr as IA
    from yolo.explain import perturb as P
    from yolo.quant.quantize import quantize_input

    net, _, inet, _ = models(name)
    if method in G.METHODS:
        return _timed(G.METHODS[method], net, x, t)
    if method == "occlusion":
        return _timed(P.occlusion, stage(name, kind), x, t.box, t.cls)
    if method == "drise":
        return _timed(P.drise, stage(name, kind), x, t.box, t.cls,
                      masks=P.rise_masks(DRISE_MASKS, SIZE, SIZE, rng=0))
    if method == "lrp_float":
        return _timed(lambda: {"map": IA.lrp_float(net, x.astype(np.float64), t).map,
                               "passes": 2})
    if method == "lrp_int":
        return _timed(lambda: {"map": IA.lrp_int(inet, quantize_input(x), t).map,
                               "passes": 2})
    if method == "random":
        return {"map": E.random_map((SIZE, SIZE), rng=0), "passes": 0, "seconds": 0.0}
    if method == "centre":
        return {"map": E.centre_map((SIZE, SIZE)), "passes": 0, "seconds": 0.0}
    raise ValueError(method)


def _evaluate(task):
    from yolo.explain import evaluate as E

    name, s = task
    x, (w, h) = load_image(s["image"])
    t = pick(name, s, x)
    if t is None:
        return None
    out = {"id": s["id"], "bin": E.size_bin(t.box, w, h), "rows": {}}
    for method, kind in EVAL_METHODS:
        r = make_map(name, method, kind, x, t)
        sc = E.scores(stage(name, kind), x, r["map"], t.box, t.cls)
        out["rows"][f"{method}@{kind}"] = {**sc, "passes": r["passes"], "seconds": r["seconds"]}
    return out


def cmd_evaluate(args):
    """Saved after each network, and a network already in `evaluate.json` with the same
    `--n` is skipped: a crash (OOM) loses one network, not the run."""
    path = OUT / "evaluate.json"
    res = json.loads(path.read_text()) if path.exists() else {}
    meta = res.pop("_n", {})
    for n in args.nets:
        if n in res and meta.get(n) == args.n:
            print(f"{n}: already in {path.name}, skipped", flush=True)
            continue
        res.update(run_pool(args, _evaluate, {n: [(n, s) for s in samples(args.n)]}))
        meta[n] = args.n
        save("evaluate", {**res, "_n": meta})


# ------------------------------------------------------------------ T36.5 misses
def person_class(name):
    from yolo.data import datasets
    from yolo.data.voc import CLASS_INDEX

    return int(datasets.class_lut("voc", models(name)[0].net["classes"])[CLASS_INDEX["person"]])


def _misses(task):
    from yolo.explain import misses as M
    from yolo.explain.targets import grid_sizes
    from yolo.quant.quantize import quantize_input

    name, s, dataset = task
    net, qm, inet, luts = models(name)
    x, (w, h) = load_image(s["image"])
    boxes, labels = gt_of(name, s, dataset)
    pc = person_class(name)
    fo = M.float_outputs(net.forward(x, train=False), net.net, CONF, IOU)
    io = M.int_outputs(inet.forward(quantize_input(x)), qm.net, luts, CONF, IOU)
    grids = grid_sizes(net.net)
    rows = []
    for b in boxes[labels == pc]:
        from yolo.explain.evaluate import size_bin

        rows.append({"id": s["id"], "box": [float(v) for v in b], "bin": size_bin(b, w, h),
                     "float": M.classify_miss(net.net, tuple(b), pc, fo, grids, CONF),
                     "int": M.classify_miss(qm.net, tuple(b), pc, io, grids, CONF,
                                            float_out=fo)})
    return rows


def person_samples(dataset, n):
    """First `n` test images of `dataset` with a non-difficult object mapped to `person`."""
    from yolo.data import datasets
    from yolo.data.voc import CLASS_INDEX

    if dataset == "voc":
        return samples(n, only_class=CLASS_INDEX["person"])
    src = [c for c, v in datasets.MAPPINGS[(dataset, "voc")].items() if v == "person"]
    idx = [datasets.info(dataset).classes.index(c) for c in src]
    out = [s for s in datasets.load(dataset)
           if np.isin(np.asarray(s["labels"])[~np.asarray(s["difficult"], dtype=bool)],
                      idx).any()]
    return out[:n]


def cmd_misses(args):
    """`--dataset voc`: `misses.json` keyed by network; another dataset (ExDark) adds its
    keys `<net>@<dataset>` to the same file, the VOC entries are kept."""
    tasks = {n: [(n, s, args.dataset) for s in person_samples(args.dataset, args.n)]
             for n in args.nets}
    res = run_pool(args, _misses, tasks)
    res = {n: [r for rows in v for r in rows] for n, v in res.items()}
    if args.dataset != "voc":
        old = json.loads((OUT / "misses.json").read_text()) if (OUT / "misses.json").exists() else {}
        res = {**old, **{f"{n}@{args.dataset}": v for n, v in res.items()}}
    save("misses", res)


# ---------------------------------------------------------------- T36.6 concepts
CONCEPT_N, RANDOM_SETS, CLASS_N = 30, 20, 50


def voc_flags(s, tag):
    """Per-object flag `tag` (`truncated`, `occluded`) of a VOC sample, from its XML."""
    import xml.etree.ElementTree as ET

    xml = Path(s["image"]).parents[1] / "Annotations" / f"{s['id']}.xml"
    root = ET.parse(xml).getroot()
    return [(o.findtext("name"), o.findtext(tag, "0").strip() == "1")
            for o in root.iter("object")]


def _acts(task):
    """Layer output (C, h, w) of one (degraded) image, and optionally δs/δA of the target."""
    from yolo.explain import concepts as C
    from yolo.explain.targets import explain, miss_target

    name, layer, path, degrade, box, cls = task
    net = models(name)[0]
    x, _ = load_image(path)
    if degrade:
        x = getattr(C, degrade)(x)
    if box is None:
        return net.forward(x, train=False, all_outputs=True)[layer][0], None
    e = explain(net, x, miss_target(net.net, box, cls), keep=[layer])
    return e.acts[layer][0], e.dlayers[layer][0]


def cmd_concepts(args):
    from yolo.data.voc import CLASS_INDEX
    from yolo.explain import concepts as C
    from yolo.explain.evaluate import size_bin

    name, layer = args.nets[0], args.layer
    person = CLASS_INDEX["person"]
    test = samples(args.n)
    pool_ = mp.get_context("spawn").Pool(args.jobs, initializer=_quiet)
    t0 = time.time()
    acts = pool_.map(_acts, [(name, layer, s["image"], None, None, None) for s in test],
                     chunksize=4)
    A = np.stack([a for a, _ in acts]).astype(np.float32)
    iou = C.dissect(A, [s["boxes"] for s in test], [s["labels"] for s in test], 20)
    res = {"net": name, "layer": layer, "images": len(test),
           "top": {k: v for k, v in C.top_channels(iou, 3).items()},
           "unmatched_share": C.unmatched_share(iou), "best_iou": iou.max(axis=0).tolist()}
    print(f"dissection: {time.time() - t0:.0f} s, unmatched {res['unmatched_share']:.2f}")

    # TCAV sets, all from VOC2007 trainval (disjoint from the class images of the test split)
    rng = np.random.default_rng(0)
    train = samples(10 ** 6, split="2007:trainval")
    idx = rng.permutation(len(train))
    pool_imgs = [train[i] for i in idx]
    randoms = [pool_imgs[k * CONCEPT_N:(k + 1) * CONCEPT_N] for k in range(RANDOM_SETS)]
    rest = pool_imgs[RANDOM_SETS * CONCEPT_N:]
    plain = rest[:CONCEPT_N]
    small = [s for s in rest if (np.asarray(s["labels"]) == person).any()
             and all(size_bin(b, s["width"], s["height"]) == "small"
                     for b, lab in zip(s["boxes"], s["labels"]) if lab == person)][:CONCEPT_N]
    occl = [s for s in rest if any(n == "person" and f for n, f in voc_flags(s, "truncated"))
            ][:CONCEPT_N]
    concept_sets = {"dark": (plain, "dark"), "fog": (plain, "fog"), "small": (small, None),
                    "occluded": (occl, None)}

    def pooled(ss, degrade=None):
        r = pool_.map(_acts, [(name, layer, s["image"], degrade, None, None) for s in ss],
                      chunksize=2)
        return C.pooled(np.stack([a for a, _ in r]))

    rand_p = [pooled(r) for r in randoms]
    cls_imgs = [s for s in test if (np.asarray(s["labels"]) == person).any()][:CLASS_N]
    pc = person_class(name)
    gr = pool_.map(_acts, [(name, layer, s["image"], None,
                            tuple(s["boxes"][np.asarray(s["labels"]) == person][0]), pc)
                           for s in cls_imgs], chunksize=2)
    grads = np.stack([g for _, g in gr])
    res["tcav"] = {}
    for cname, (ss, degrade) in concept_sets.items():
        r = C.tcav(pooled(ss, degrade), rand_p, grads)
        res["tcav"][cname] = {"images": len(ss), "score": r["score"], "p": r["p"], "t": r["t"],
                              "random_mean": float(np.mean(r["random_scores"]))}
        print(f"TCAV {cname}: {r['score']:.2f} p={r['p']:.3g}")
    pool_.close()
    res["seconds"] = time.time() - t0
    save("concepts", res)


# -------------------------------------------------------------------- T36.8 CKA
def _probe_acts(task):
    """Pooled outputs (2 × 2 grid) of every conv, float and integer (dequantized)."""
    from yolo.explain.cka import pool
    from yolo.quant.quantize import quantize_input

    name, path = task
    net, qm, inet, _ = models(name)
    x, _ = load_image(path)
    fa = net.forward(x, train=False, all_outputs=True)
    ia = inet.forward(quantize_input(x), all_outputs=True)
    convs = sorted(qm.convs)
    return ([pool(fa[i], 2)[0] for i in convs],
            [pool(ia[i].astype(np.float64) * qm.scale_of(i), 2)[0] for i in convs])


STRIP = 4  # probe images of the Grad-CAM strip across the checkpoints (T36.8)


def _snap_acts(task):
    """Pooled conv outputs of a snapshot on the probe, and the Grad-CAM maps (104 × 104) of
    the first ground truth of the first `STRIP` probe images (a fixed miss target)."""
    from yolo.data import datasets
    from yolo.explain.cka import pool
    from yolo.explain.gradient import gradcam
    from yolo.explain.maps import resize
    from yolo.explain.targets import miss_target
    from yolo.models.tiny_yolo import build
    from yolo.quant.fuse_bn import fuse_network

    name, snap, paths = task
    net = build(name)
    with np.load(snap) as d:
        for key in d.files:
            kind, i, k = key.split("/")
            (net.params if kind == "param" else net.state)[int(i)][k][...] = d[key]
    f = fuse_network(net, dtype=np.float32)
    convs = [i for i, layer in enumerate(f.layers) if layer["type"] == "conv"]
    xs = np.concatenate([load_image(p)[0] for p in paths])
    acts = f.forward(xs, train=False, all_outputs=True)
    reps = [pool(acts[i], 2) for i in convs]
    by_path = {s["image"]: s for s in samples(len(paths))}
    lut = datasets.class_lut("voc", f.net["classes"])
    cams = []
    for k, p in enumerate(paths[:STRIP]):
        s = by_path[p]
        t = miss_target(f.net, tuple(s["boxes"][0]), int(lut[s["labels"][0]]))
        cams.append(resize(gradcam(f, xs[k:k + 1], t)["map"], 104, 104))
    return reps, np.stack(cams)


def cmd_cka(args):
    from yolo.explain.cka import linear_cka

    probe = [s["image"] for s in samples(args.n)]
    res = {"probe": len(probe), "float_int": {}}
    with mp.get_context("spawn").Pool(args.jobs, initializer=_quiet) as pool_:
        for name in args.nets:
            r = pool_.map(_probe_acts, [(name, p) for p in probe], chunksize=2)
            convs = sorted(models(name)[1].convs)
            res["float_int"][name] = {
                str(c): linear_cka(np.stack([a[0][k] for a in r]),
                                   np.stack([a[1][k] for a in r]))
                for k, c in enumerate(convs)}
            low = min(res["float_int"][name].items(), key=lambda kv: kv[1])
            print(f"{name}: CKA float/int min {low[1]:.4f} at L{low[0]}")
        if args.run:
            snaps = sorted(Path(args.run).glob("snap_*.npz"))
            name = args.nets[0]
            got = pool_.map(_snap_acts, [(name, s, probe[:args.n]) for s in snaps])
            per = [r for r, _ in got]
            np.savez_compressed(OUT / "cka_gradcam.npz", maps=np.stack([c for _, c in got]),
                                its=np.array([int(s.stem.split("_")[1]) for s in snaps]),
                                images=np.array(probe[:STRIP]))
            last = per[-1]
            res["checkpoints"] = {
                "run": str(args.run), "its": [int(s.stem.split("_")[1]) for s in snaps],
                "cka_to_last": [[linear_cka(p[k], last[k]) for k in range(len(last))]
                                for p in per]}
    save("cka", res)


# ----------------------------------------------------------------- T36.7 record
def auroc(score, positive):
    from yolo.infer.ood import auroc as _auroc

    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    return _auroc(score[~positive], score[positive])


def cmd_record(args):
    from yolo.explain.recorder import summary

    out = {}
    for run in args.runs:
        run = Path(run)
        with np.load(run / "record.npz") as d:
            a = {k: d[k] for k in d.files}
        s = summary(a["loss_img"])
        loss_csv = np.genfromtxt(run / "loss.csv", delimiter=",", names=True)
        train_s = float(np.sum(loss_csv["seconds"]))
        row = {"epochs": int(a["loss_img"].shape[0]), "images": int(a["loss_img"].shape[1]),
               "record_seconds": float(a["record_seconds"]), "train_seconds": train_s,
               "overhead": float(a["record_seconds"]) / train_s,
               "never_learned": int(s["never_learned"].sum()),
               "forgotten": int((s["forgetting"] > 0).sum()),
               "dead_last": a["dead"][-1].tolist() if len(a["dead"]) else [],
               "update_last": a["update"][-1].tolist() if len(a["update"]) else []}
        bad_f = run / "corrupted.json"
        if bad_f.exists():
            bad = np.zeros(row["images"], bool)
            bad[json.loads(bad_f.read_text())] = True
            row["corrupted"] = int(bad.sum())
            row["auroc_never_learned"] = auroc(s["never_learned"].astype(float), bad)
            row["auroc_mean_loss"] = auroc(s["mean"], bad)
            row["never_learned_precision"] = (float(bad[s["never_learned"]].mean())
                                              if s["never_learned"].any() else 0.0)
        out[run.name] = row
        print(run.name, {k: v for k, v in row.items() if not isinstance(v, list)})
    save("record", out)


# ---------------------------------------------------------------------- map / PNG
def overlay(x, maps, box, path, title=""):
    """PNG: the image and each map over it, the box drawn."""
    from tools.figures.style import plt

    from yolo.explain.maps import normalize

    p = plt()
    img = np.asarray(x[0]).transpose(1, 2, 0)
    fig, axes = p.subplots(1, len(maps) + 1, figsize=(3 * (len(maps) + 1), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (name, m) in zip(axes, [("image", None), *maps.items()]):
        ax.imshow(img)
        if m is not None:
            ax.imshow(normalize(m), cmap="jet", alpha=0.5)
        if box is not None:
            cx, cy, bw, bh = box
            h, w = img.shape[:2]
            ax.add_patch(p.Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h,
                                     fill=False, color="w", lw=1.5))
        ax.set_title(name, fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    p.close(fig)
    return path


def cmd_map(args):
    from yolo.data import datasets

    name = args.nets[0]
    all_s = {s["id"]: s for s in datasets.load("voc", split="2007:test")}
    ids = args.image.split(",") if args.image else list(all_s)[:args.n]
    for i in ids:
        s = all_s[i]
        x, _ = load_image(s["image"])
        t = pick(name, s, x)
        if t is None:
            print(f"{i}: no matched detection")
            continue
        maps = {}
        for m in args.methods.split(","):
            method, _, kind = m.partition("@")
            maps[m] = make_map(name, method, kind or "float", x, t)["map"]
        print(overlay(x, maps, t.box, OUT / "maps" / name / f"{i}.png",
                      f"{name} {i} class {t.cls} score {t.score:.2f}"))


# ----------------------------------------------------------------------- plumbing
def _quiet():
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _call(args):
    fn, task = args
    try:
        return fn(task)
    except Exception as exc:  # one bad image must not lose the run
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_pool(args, fn, tasks):
    """{net: [result]} of `fn` over the tasks of each net, in a spawn pool."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    out = {}
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.jobs, initializer=_quiet) as pool_:
        for name, ts in tasks.items():
            res = pool_.map(_call, [(fn, t) for t in ts], chunksize=1)
            errs = [r["error"] for r in res if isinstance(r, dict) and "error" in r]
            for e in errs[:3]:
                print(f"  {name}: {e}")
            out[name] = [r for r in res if r is not None and not (isinstance(r, dict)
                                                                   and "error" in r)]
            print(f"{name}: {len(out[name])} results, {len(errs)} errors, "
                  f"{time.time() - t0:.0f} s", flush=True)
    return out


def save(cmd, res):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{cmd}.json"
    path.write_text(json.dumps(res, indent=1, default=float))
    print(f"→ {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("map", "sanity", "stages", "intattr", "evaluate", "misses", "concepts",
                 "cka", "record", "report"):
        p = sub.add_parser(name)
        p.add_argument("--nets", default=",".join(NETS) if name not in ("map", "concepts")
                       else NETS[0], type=lambda v: v.split(","))
        p.add_argument("--n", type=int, default={"evaluate": 50, "misses": 500,
                                                 "concepts": 200, "cka": 64}.get(name, 20))
        p.add_argument("--jobs", type=int, default=min(20, os.cpu_count() or 1))
        if name == "map":
            p.add_argument("--image", default="", help="VOC ids, comma-separated")
            p.add_argument("--methods", default="gradcam,ig_blur,drise,lrp_int@int")
        if name == "stages":
            p.add_argument("--golden", type=int, default=5,
                           help="images whose int map is checked against the golden")
        if name == "misses":
            p.add_argument("--dataset", default="voc",
                           help="test split of a dataset with a person mapping (exdark)")
        if name == "concepts":
            p.add_argument("--layer", type=int, default=13)
        if name == "cka":
            p.add_argument("--run", type=Path, default=None,
                           help="recorder run with snap_*.npz (T36.7)")
        if name == "record":
            p.add_argument("--runs", nargs="+", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "report":
        from explain_report import report

        return report()
    {"map": cmd_map, "sanity": cmd_sanity, "stages": cmd_stages, "intattr": cmd_intattr,
     "evaluate": cmd_evaluate, "misses": cmd_misses, "concepts": cmd_concepts,
     "cka": cmd_cka, "record": cmd_record}[args.cmd](args)


if __name__ == "__main__":
    main()
