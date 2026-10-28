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
    # T25.6: also the share of images that lose a fault-free person
    python tools/fault_campaign.py run --grid layer --criterion person --out …
    # T25.9: faults in the integer failure head (T39.5) on the stored features of T39.0
    python tools/fault_campaign.py monitor build/m39/labels/tiny-yolov4-coco_prewid-lab.npz
    # T25.8: critical faults in one video frame, seen by temporal coherence and by the ABFT
    python tools/fault_campaign.py temporal --dataset mot17 --split train --points 300

T25.4, `--grid protect`: every site faulty at once at each BER, minus what a protection
scheme (`SCHEMES`) masks. A fault seen by the ABFT check is assumed reloaded and recomputed
without fault, a single flip per ECC word corrected, and a duplicated layer recomputed on a
mismatch, so a masked fault is simply not injected.

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

T25.6, `--criterion person`: each run also compares, per image, the person boxes (score ≥
`--person-conf`) of the faulty output with those of the fault-free output. A fault-free person
with no faulty box at IoU ≥ 0.5 is **lost**, and an image with a lost person is **critical**.
Extra boxes and boxes moved (matched below IoU 0.9) are tolerable and counted apart. The
reference is the fault-free model, not the ground truth: without a fault, critical = 0.
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
PROTECT_BERS = (1e-5, 1e-4)
SENSITIVE = 3  # T25.4: the first convs protected by the selective schemes (T25.1 ranking)
# Scheme → {site: "all" | "sel"}: the sites masked on every conv or on the sensitive ones.
# ECC (SECDED on the weight and bias words) masks the parameters; the ABFT check (T25.2)
# sees the weight, bias and accumulator faults; duplication sees act and acc.
SCHEMES = {
    "none": {},
    "ecc": {"weight": "all", "bias": "all"},
    "abft-sel": {"weight": "sel", "bias": "sel", "acc": "sel"},
    "abft": {"weight": "all", "bias": "all", "acc": "all"},
    "dup-sel": {"act": "sel", "acc": "sel"},
    "ecc+dup-sel": {"weight": "all", "bias": "all", "act": "sel", "acc": "sel"},
    "abft+dup-sel": {"weight": "all", "bias": "all", "acc": "all", "act": "sel"},
}
PERSON_CONF = 0.25  # T25.6: operating threshold of the person criterion (Darknet default)
MATCH_IOU, MOVED_IOU = 0.5, 0.9

_W = {}


# ------------------------------------------------------------------------- run specs
def spec_key(spec):
    """Unique key of a run (resume)."""
    return json.dumps({k: spec[k] for k in sorted(spec)}, sort_keys=True)


def faults_of(spec):
    """Faults of a run spec ({} : baseline, no fault)."""
    if "hit" in spec:  # protect grid: a BER on every (site, conv) left unmasked
        return [F.Fault(site, "ber", layer, rate=spec["rate"]) for site, layers in
                sorted(spec["hit"].items()) for layer in layers]
    if spec.get("site") is None:
        return []
    f = F.Fault(spec["site"], spec["kind"], spec.get("layer"), spec.get("bit"),
                spec.get("rate", 0.0))
    return [f] * spec.get("flips", 1)


def protected_hit(scheme, convs, sites=SITES, sensitive=SENSITIVE):
    """{site: convs left faulty} under a protection scheme."""
    sel = set(sorted(convs)[:sensitive])
    masks = SCHEMES[scheme]
    return {site: [c for c in sorted(convs) if not (
        masks.get(site) == "all" or (masks.get(site) == "sel" and c in sel))] for site in sites}


def grid(name, convs, sites=SITES, bers=BERS, layer_bers=LAYER_BERS, bits=None, seeds=(0,),
         flips=100):
    """Run specs of a grid (`ber`, `layer`, `bit`, `protect`), the baseline first."""
    out = [{"grid": "baseline", "site": None, "seed": 0}]
    if name == "protect":
        return out + [{"grid": name, "site": "all", "scheme": sc, "kind": "ber", "rate": r,
                       "seed": seed, "hit": protected_hit(sc, convs, sites)}
                      for seed in seeds for sc in SCHEMES for r in bers]
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


def person_match(ref, out, iou_thr=MATCH_IOU, moved_iou=MOVED_IOU):
    """T25.6: fault-free person boxes `ref` against the faulty ones `out` ((cx, cy, w, h)),
    matched one to one by decreasing IoU → (lost, extra, moved): `ref` boxes with no `out` box
    at IoU ≥ `iou_thr`, `out` boxes left unmatched, matched pairs below `moved_iou`."""
    from yolo.infer.boxes import iou

    ref, out = np.reshape(ref, (-1, 4)), np.reshape(out, (-1, 4))
    if not len(ref) or not len(out):
        return len(ref), len(out), 0
    m = iou(ref, out)
    pairs = sorted(((m[i, j], i, j) for i, j in zip(*np.nonzero(m >= iou_thr))), reverse=True)
    used_r, used_o, moved = set(), set(), 0
    for v, i, j in pairs:
        if i in used_r or j in used_o:
            continue
        used_r.add(i)
        used_o.add(j)
        moved += v < moved_iou
    return len(ref) - len(used_r), len(out) - len(used_o), moved


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


def _setup(qm, cache, abft=True, conf=CONF, iou=IOU, person=None, person_conf=PERSON_CONF):
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
              sizes=F.conv_sizes(qm, inputs.shape[1:]), conf=conf, iou=iou, run=None,
              person=person, person_conf=person_conf, ref_persons={})


def _image(task):
    """One image of one run → (labels, scores, boxes (cx, cy, w, h) normalized, flips,
    changed, detected, persons): `persons` = (fault-free persons, lost, extra, moved) with
    the person criterion (T25.6), None otherwise."""
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
    persons = None
    if _W["person"] is not None:
        if k not in _W["ref_persons"]:
            ref = {i: np.asarray(_W["refs"][i][k:k + 1]) for i in _W["refs"]}
            _W["ref_persons"][k] = _persons(*postprocess_int(ref, _W["qm"].net, _W["luts"],
                                                             _W["conf"], _W["iou"])[0])
        ref = _W["ref_persons"][k]
        persons = (len(ref), *person_match(ref, _persons(boxes, scores, labels)))
    return (np.asarray(labels).tolist(), np.asarray(scores, np.float64).tolist(),
            np.asarray(boxes, np.float64).reshape(-1, 4).tolist(), flips, changed, detected,
            persons)


def _persons(boxes, scores, labels):  # person boxes above the operating threshold
    keep = (np.asarray(labels) == _W["person"]) & (np.asarray(scores) >= _W["person_conf"])
    return np.asarray(boxes, np.float64).reshape(-1, 4)[keep]


# ------------------------------------------------------------------------- one run
def score(results, samples, meta, n_classes, det_lut=None):
    """mAP (VOC07) and counters of a run from the per-image results. `det_lut`: model class
    → dataset class (−1: not scored), when the dataset is not the model's (`eval_view`)."""
    from yolo.infer.metrics import evaluate, to_voc_pixels

    per = {c: ([], [], []) for c in range(n_classes)}
    flips = changed = detected = sdc = n_det = 0
    pers = np.zeros(5)  # images with persons, critical images, persons, lost, moved
    extra = 0
    for k, (labels, scores, boxes, fl, ch, de, *rest) in enumerate(results):
        if rest and rest[0] is not None:
            n_ref, lost, ext, mov = rest[0]
            pers += (n_ref > 0, lost > 0, n_ref, lost, mov)
            extra += ext
        px = to_voc_pixels(boxes, meta["width"][k], meta["height"][k])
        for c, s, b in zip(labels, scores, px):
            if det_lut is not None:
                c = int(det_lut[c])
                if c < 0:
                    continue
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
    out = {"map": float(m), "images": len(results), "flips": int(flips),
           "changed": changed / n, "detected": detected / n, "sdc": sdc / n,
           "dets_per_image": n_det / n}
    if results and len(results[0]) > 6 and results[0][6] is not None:
        out.update(person_images=int(pers[0]), critical=pers[1] / max(pers[0], 1),
                   lost=pers[3] / max(pers[2], 1), moved=pers[4] / max(pers[2], 1),
                   extra=extra / n)
    return out


def done_keys(out):
    if not Path(out).exists():
        return set()
    return {json.loads(line)["key"] for line in Path(out).read_text().splitlines()
            if line.strip()}


def campaign(specs, samples, meta, n_classes, out, mapper, det_lut=None):
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
        row = {"key": key, **spec, **score(res, samples, meta, n_classes, det_lut),
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
        k = (r["grid"], r.get("site"), r.get("layer"), r.get("bit"), r.get("rate"),
             r.get("scheme"))
        groups.setdefault(k, []).append(r)
    out = []
    for (g, site, layer, bit, rate, scheme), rs in sorted(groups.items(),
                                                          key=lambda kv: str(kv[0])):
        maps = np.array([r["map"] for r in rs])
        a = {"grid": g, "site": site, "layer": layer, "bit": bit, "rate": rate,
             "seeds": len(rs), "map": float(maps.mean()),
             "std": float(maps.std(ddof=1)) if len(rs) > 1 else 0.0,
             **{k: float(np.mean([r[k] for r in rs]))
                for k in ("changed", "detected", "sdc", "flips")}}
        if scheme is not None:
            a["scheme"] = scheme
        if all("critical" in r for r in rs):  # person criterion (T25.6)
            a.update({k: float(np.mean([r[k] for r in rs]))
                      for k in ("critical", "lost", "moved", "extra")})
            a["critical_std"] = (float(np.std([r["critical"] for r in rs], ddof=1))
                                 if len(rs) > 1 else 0.0)
        out.append(a)
    return out


def report(rows):
    """{baseline, ber, layer, bit, abft, ranking} for the tables and the figure."""
    agg = aggregate(rows)
    base = [a for a in agg if a["grid"] == "baseline"]
    rep = {"baseline": base[0]["map"] if base else None,
           "images": rows[0]["images"] if rows else 0}
    for g in ("ber", "layer", "bit", "protect"):
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
    # T25.6: the same ranking by the share of critical images (person lost).
    crit = {}
    for a in rep["layer"]:
        if "critical" in a:
            crit.setdefault(a["layer"], []).append(a["critical"])
    rep["person_ranking"] = sorted(([int(i), float(np.mean(c))] for i, c in crit.items()),
                                   key=lambda t: -t[1])
    rep["person"] = any("critical" in a for g in ("ber", "layer", "bit") for a in rep[g])
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
    if any(s in rep["abft"] for s in SITES):  # the protect grid has only site "all"
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
    if rep.get("protect"):
        rates = sorted({a["rate"] for a in rep["protect"]})
        lines += ["| Scheme | Faulty sites | " + " | ".join(f"BER {r:.0e}" for r in rates)
                  + " |", "|---|---|" + "---|" * len(rates)]
        for sc, masks in SCHEMES.items():
            cells = {a["rate"]: _pm(a) for a in rep["protect"] if a["scheme"] == sc}
            if cells:
                left = ", ".join(f"{s}{'' if s not in masks else ' (not the first convs)'}"
                                 for s in SITES if masks.get(s) != "all")
                lines.append(f"| {sc} | {left or '—'} | "
                             + " | ".join(cells.get(r, "") for r in rates) + " |")
        lines += ["", f"mAP with every site faulty at once at that BER, minus the masked "
                  f"sites (sel: the first {SENSITIVE} convs).", ""]
    if rep.get("person"):
        lines += person_markdown(rep)
    return "\n".join(lines)


def _crit(a):
    s = f"{a['critical'] * 100:.1f}"
    if a["seeds"] > 1:
        s += f" ± {a['critical_std'] * 100:.1f}"
    return s


def person_markdown(rep):
    """T25.6 tables: critical images (a fault-free person lost) per layer and per bit, and
    the two rankings of the layers."""
    lines = []
    layer = [a for a in rep["layer"] if "critical" in a]
    if layer:
        cols = sorted({(a["site"], a["rate"]) for a in layer},
                      key=lambda t: (SITES.index(t[0]), t[1]))
        lines += ["| Conv | " + " | ".join(f"{s} {r:.0e}" for s, r in cols) + " |",
                  "|---|" + "---|" * len(cols)]
        for i in sorted({a["layer"] for a in layer}):
            cells = {(a["site"], a["rate"]): _crit(a) for a in layer if a["layer"] == i}
            lines.append(f"| L{i:02d} | " + " | ".join(cells.get(c, "") for c in cols) + " |")
        rank_map = [i for i, _ in rep["ranking"]]
        lines += ["", "Critical images in % (a fault-free person lost, among the images with a "
                  "person).", "",
                  "| Rank | By critical images | By mAP loss |", "|---|---|---|"]
        for r, (i, c) in enumerate(rep["person_ranking"]):
            j = rank_map[r] if r < len(rank_map) else None
            lines.append(f"| {r + 1} | L{i:02d} ({c * 100:.1f} %) | "
                         + ("—" if j is None else f"L{j:02d}") + " |")
        lines.append("")
    bit = [a for a in rep["bit"] if "critical" in a]
    if bit:
        sites = [s for s in SITES if any(a["site"] == s for a in bit)]
        lines += ["| Bit | " + " | ".join(sites) + " |", "|---|" + "---|" * len(sites)]
        for b in sorted({a["bit"] for a in bit}):
            cells = {a["site"]: _crit(a) for a in bit if a["bit"] == b}
            lines.append(f"| {b} | " + " | ".join(cells.get(s, "") for s in sites) + " |")
        lines += ["", "Critical images in % for one flip at that bit.", ""]
    rows = [a for g in ("ber", "layer", "bit") for a in rep[g] if "critical" in a]
    if rows:
        lines += ["| Site | Critical images | Persons lost | Persons moved | Extra boxes / "
                  "image |", "|---|---|---|---|---|"]
        for s in SITES:
            rs = [a for a in rows if a["site"] == s]
            if rs:
                lines.append(f"| {s} | " + " | ".join(
                    f"{np.mean([a[k] for a in rs]) * (100 if k != 'extra' else 1):.2f}"
                    + (" %" if k != "extra" else "")
                    for k in ("critical", "lost", "moved", "extra")) + " |")
        lines += ["", "Mean over every run of the site (all grids).", ""]
    return lines


# ------------------------------------------------------------------------- main
def _view(args):
    from yolo.data import datasets
    from yolo.models.tiny_yolo import load_cfg

    return datasets.eval_view(args.dataset, load_cfg(args.net)["classes"])


def _person_index(args):
    """Model class of `person` (the labels of `_image` are the model's classes)."""
    if args.criterion != "person":
        return None
    view = _view(args)
    if "person" not in view.names:
        raise SystemExit(f"--criterion person: no person class in {args.dataset}")
    return int(np.flatnonzero(np.asarray(view.det_lut) == view.names.index("person"))[0])


def _samples(args):
    from yolo.data import datasets
    from yolo.models.tiny_yolo import load_cfg

    samples = datasets.load_args(args)
    if args.subset:
        samples = samples[:args.subset]
    view = datasets.eval_view(args.dataset, load_cfg(args.net)["classes"])
    return datasets.remap(samples, view.gt_lut), len(view.names)


def _cache_dir(args, n):
    tag = "" if args.dataset == "voc" else f"_{args.dataset}"
    return ROOT / "build" / "m25" / "cache" / f"{args.net}{tag}_{n}"


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
    bers = PROTECT_BERS if args.grid == "protect" else BERS
    specs = grid(args.grid, sorted(qm.convs), sites, _parse_floats(args.bers) or bers,
                 _parse_floats(args.layer_bers) or LAYER_BERS, None,
                 tuple(int(s) for s in args.seeds.split(",")), args.flips)
    if args.layers:
        keep = {int(i) for i in args.layers.split(",")}
        specs = [s for s in specs if s.get("layer") is None or s["layer"] in keep]
    setup = (qm, cache, not args.no_abft, CONF, IOU, _person_index(args), args.person_conf)
    if args.jobs <= 1:
        _setup(*setup)
        campaign(specs, samples, meta, n_classes, args.out, lambda t: [_image(x) for x in t],
                 _view(args).det_lut)
        return
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init, initargs=setup) as pool:
        campaign(specs, samples, meta, n_classes, args.out,
                 lambda t: pool.map(_image, t, chunksize=max(1, len(t) // (4 * args.jobs))),
                 _view(args).det_lut)


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


# ------------------------------------------------------------------------- monitor (T25.9)
MON_BERS = (1e-5, 1e-4, 1e-3)


def monitor_runs(ih, p, flag0, thr, missed, trials=50, bers=MON_BERS):
    """T25.9 rows: per site, layer and bit (one flip, per image for act / acc, per head for
    weight / bias over `trials` draws) and per BER, the shares of the images flagged without
    fault that are no longer flagged (silenced), of those with a missed person (silenced_miss),
    of the unflagged images now flagged (false alarm), and of the silenced ones that the ABFT
    check sees."""
    rows = []
    for (site, layer) in F.head_sizes(ih):
        width = F.WIDTH[site]
        bits = BITS[site] if width == 32 else tuple(range(width))
        specs = [("flip", b, 0.0) for b in bits] + [("ber", None, r) for r in bers]
        for kind, bit, rate in specs:
            acc = np.zeros(4)  # silenced, silenced with a miss, false alarms, silenced+ABFT
            for t in range(trials if site in F.PERSISTENT or kind == "ber" else 1):
                s, det = F.head_forward(ih, p, [F.Fault(site, kind, layer, bit, rate)], seed=t)
                flag = s > thr
                sil = flag0 & ~flag
                acc += (sil.sum(), (sil & missed).sum(), (~flag0 & flag).sum(),
                        (sil & det).sum())
                n = t + 1
            rows.append({"site": site, "layer": layer, "kind": kind, "bit": bit, "rate": rate,
                         "draws": n, "silenced": acc[0] / (n * max(flag0.sum(), 1)),
                         "silenced_miss": acc[1] / (n * max((flag0 & missed).sum(), 1)),
                         "false_alarm": acc[2] / (n * max((~flag0).sum(), 1)),
                         "abft_silenced": acc[3] / acc[0] if acc[0] else None})
    return rows


def monitor_markdown(name, rows, info):
    lines = [f"`{name}`: {info['images']} images, {info['flagged']} flagged without fault "
             f"(5 % alarm rate on val), {info['flagged_miss']} of them with a missed person; "
             f"head {info['memory']:,} B.", "",
             "| Site | Layer | Bit / BER | Silenced | Silenced (with a miss) | False alarms | "
             "Silenced caught by ABFT |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        if not (r["silenced"] or r["false_alarm"]):
            continue
        what = f"bit {r['bit']}" if r["kind"] == "flip" else f"BER {r['rate']:.0e}"
        ab = "—" if r["abft_silenced"] is None else f"{r['abft_silenced'] * 100:.0f} %"
        lines.append(f"| {r['site']} | {'hidden' if r['layer'] == 0 else 'output'} | {what} | "
                     f"{r['silenced'] * 100:.2f} % | {r['silenced_miss'] * 100:.2f} % | "
                     f"{r['false_alarm'] * 100:.2f} % | {ab} |")
    lines += ["", "Rows with no effect at all are left out. One flip per image (act, acc) or "
              f"per head over {info['trials']} draws (weight, bias); BER over {info['trials']} "
              "draws.", ""]
    return "\n".join(lines)


def cmd_monitor(args):
    from failure_labels import Labels
    from failure_prediction import head_path, load_head

    from yolo.infer.failure import alarm_threshold
    from yolo.quant.int_failure import IntFailureHead

    out = ROOT / "build" / "m25" / "monitor"
    out.mkdir(parents=True, exist_ok=True)
    for path in args.labels:
        L = Labels(path)
        h, _ = load_head(head_path(L, "float", "miss"))
        p = L.feat_int.astype(np.int64)
        ih = IntFailureHead.from_float(h, L.scales, p[L.split == 0])
        s0, det0 = F.head_forward(ih, p)
        assert np.array_equal(s0, ih.score(p)) and not det0.any()
        thr = alarm_threshold(s0[L.split == 1], 0.05)
        flag0, missed = s0 > thr, L.miss_int > 0
        t0 = time.time()
        rows = monitor_runs(ih, p, flag0, thr, missed, args.trials)
        info = {"images": len(p), "flagged": int(flag0.sum()),
                "flagged_miss": int((flag0 & missed).sum()), "memory": ih.memory_bytes,
                "trials": args.trials, "seconds": round(time.time() - t0, 1)}
        (out / f"{L.name}.json").write_text(json.dumps({"info": info, "rows": rows},
                                                       indent=1) + "\n")
        print(monitor_markdown(L.name, rows, info))
        print(f"→ {out / f'{L.name}.json'} ({info['seconds']:.0f} s)")


# ------------------------------------------------------------------------- temporal (T25.8)
# Transient faults of one frame, among the critical ones of T25.6 (VOC person criterion).
TEMPORAL_FAULTS = (
    {"site": "act", "kind": "ber", "layer": 0, "rate": 1e-5},
    {"site": "acc", "kind": "ber", "layer": 0, "rate": 1e-5},
    {"site": "acc", "kind": "ber", "layer": 2, "rate": 1e-5},
    {"site": "act", "kind": "flip", "bit": 7, "flips": 100},
    {"site": "acc", "kind": "flip", "bit": 24, "flips": 100},
    {"site": "acc", "kind": "flip", "bit": 31, "flips": 100},
)
STABLE = (1, 2, 3)  # previous frames a person track must span to raise the alarm
TRACK_IOU = 0.3  # association between frames, as `temporal.vote`


def stable_tracks(prev, m, iou_thr=TRACK_IOU):
    """Boxes of the last previous frame that link back, frame by frame (IoU ≥ `iou_thr`), over
    the `m` last frames of `prev` (list of (n, 4) person boxes (cx, cy, w, h), oldest first)."""
    from yolo.infer.boxes import iou

    if len(prev) < m:
        raise ValueError(f"need {m} previous frames, got {len(prev)}")
    keep = []
    for b in np.reshape(prev[-1], (-1, 4)):
        cur = b
        for j in range(2, m + 1):
            other = np.reshape(prev[-j], (-1, 4))
            if not len(other):
                break
            ov = iou(cur[None], other)[0]
            if ov.max() < iou_thr:
                break
            cur = other[int(np.argmax(ov))]
        else:
            keep.append(b)
    return np.reshape(keep, (-1, 4))


def temporal_alarm(prev, cur, m, iou_thr=TRACK_IOU):
    """T25.8 coherence check at one frame: alarm when a person track present in each of the
    `m` previous frames has no box at IoU ≥ `iou_thr` in the current frame `cur`."""
    from yolo.infer.boxes import iou

    tracks, cur = stable_tracks(prev, m, iou_thr), np.reshape(cur, (-1, 4))
    if not len(tracks):
        return False
    if not len(cur):
        return True
    return bool((iou(tracks, cur).max(axis=1) < iou_thr).any())


def temporal_points(samples, person, n, window, seed=0):
    """`n` injection frames (indices into `samples`) with a ground-truth person and
    `window` − 1 contiguous previous frames, drawn at random; and their windows."""
    from dreaded_event import contiguous_runs

    pos = {s["id"]: i for i, s in enumerate(samples)}
    cand = [[pos[s["id"]] for s in run[t - window + 1:t + 1]]
            for run in contiguous_runs(samples) for t in range(window - 1, len(run))
            if (np.asarray(run[t]["labels"]) == person).any()]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(cand), min(n, len(cand)), replace=False)
    return [cand[i] for i in sorted(pick)]


def temporal_markdown(name, rows, info):
    ms = info["stable"]
    lines = [f"`{name}`: {info['points']} injection frames ({info['frames']} frames with "
             f"their {info['window'] - 1} previous ones), seeds {info['seeds']}. "
             "False alarms of the coherence check without fault, per frame: "
             + ", ".join(f"m = {m}: {info['false_alarm'][str(m)] * 100:.1f} %" for m in ms)
             + ".", "",
             "| Fault | Critical | " + " | ".join(f"Temporal m = {m}" for m in ms)
             + f" | ABFT | Both (m = {info['both_m']}) | Either | Missed |",
             "|---" * (6 + len(ms)) + "|"]
    for r in rows:
        f = r["fault"]
        what = (f"{f['site']} L{f['layer']:02d} BER {f['rate']:.0e}" if f["kind"] == "ber"
                else f"{f['site']} bit {f['bit']} × {f['flips']}")
        c = max(r["critical"], 1)
        pct = lambda v: f"{v / c * 100:.0f} %"  # noqa: E731
        lines.append(f"| {what} | {r['critical']} / {r['runs']} | "
                     + " | ".join(pct(r["temporal"][str(m)]) for m in ms)
                     + f" | {pct(r['abft'])} | {pct(r['both'])} | {pct(r['either'])} | "
                     f"{pct(r['critical'] - r['either'])} |")
    lines += ["", "Shares of the critical runs (a fault-free person lost in the faulty frame). "
              f"Temporal m: a person track linked over the m previous fault-free frames "
              f"(IoU ≥ {TRACK_IOU}) has no box in the faulty frame.", ""]
    return "\n".join(lines)


def cmd_temporal(args):
    from yolo.data.letterbox import input_size
    from yolo.models.tiny_yolo import load_cfg

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    args.criterion = "person"
    person = _person_index(args)
    samples, _ = _samples(args)
    window = max(STABLE) + 1
    points = temporal_points(samples, person, args.points, window)
    frames = sorted({i for w in points for i in w})
    sel = [samples[i] for i in frames]
    k_of = {i: k for k, i in enumerate(frames)}
    qm = load_qm(args.net, args.weights, args.calib, args.model_dir)
    name = f"{args.net}_{args.dataset}-{datasets_split(args)}"
    cache = ROOT / "build" / "m25" / "cache" / f"temporal_{name}_{len(points)}"
    prepare(cache, sel, qm, input_size(load_cfg(args.net), None), args.jobs)
    setup = (qm, cache, True, CONF, IOU, person, args.person_conf)
    seeds = tuple(int(s) for s in args.seeds.split(","))
    base = {"grid": "baseline", "site": None, "seed": 0}
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init, initargs=setup) as pool:
        def run(spec, ks):
            return pool.map(_image, [(spec_key(spec), spec, k) for k in ks],
                            chunksize=max(1, len(ks) // (4 * args.jobs)))

        def persons(res):
            lab, sc, bx = (np.asarray(v) for v in res[:3])
            keep = (lab == person) & (sc >= args.person_conf)
            return np.reshape(bx, (-1, 4))[keep]

        ref = [persons(r) for r in run(base, range(len(frames)))]
        prev = [[ref[k_of[i]] for i in w[:-1]] for w in points]
        cur = [k_of[w[-1]] for w in points]
        fa = {str(m): float(np.mean([temporal_alarm(p, ref[k], m) for p, k in zip(prev, cur)]))
              for m in STABLE}
        rows = []
        for f in TEMPORAL_FAULTS:
            acc = {"runs": 0, "critical": 0, "abft": 0, "both": 0, "either": 0,
                   "temporal": {str(m): 0 for m in STABLE}}
            for seed in seeds:
                spec = {"grid": "temporal", "seed": seed, **f}
                for p, r in zip(prev, run(spec, cur)):
                    acc["runs"] += 1
                    if not r[6][1]:  # no fault-free person lost
                        continue
                    acc["critical"] += 1
                    seen = {m: temporal_alarm(p, persons(r), m) for m in STABLE}
                    for m in STABLE:
                        acc["temporal"][str(m)] += seen[m]
                    acc["abft"] += r[5]
                    acc["both"] += r[5] and seen[args.both_m]
                    acc["either"] += r[5] or seen[args.both_m]
            rows.append({"fault": f, **acc})
            print(f"{f}: {acc}  {time.time() - t0:.0f} s", flush=True)
    info = {"points": len(points), "frames": len(frames), "window": window, "seeds": list(seeds),
            "stable": list(STABLE), "both_m": args.both_m, "false_alarm": fa,
            "person_conf": args.person_conf, "seconds": round(time.time() - t0, 1)}
    out = ROOT / "build" / "m25" / "temporal"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"info": info, "rows": rows}, indent=1) + "\n")
    print(temporal_markdown(name, rows, info))
    print(f"→ {out / f'{name}.json'} ({info['seconds']:.0f} s)")


def datasets_split(args):
    from yolo.data import datasets

    return datasets.split_of(args)


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
            p.add_argument("--grid", choices=("ber", "layer", "bit", "protect"),
                           required=True)
            p.add_argument("--sites", default=",".join(SITES))
            p.add_argument("--bers", default=None, help=f"ber grid (default {BERS})")
            p.add_argument("--layer-bers", default=None,
                           help=f"layer grid (default {LAYER_BERS})")
            p.add_argument("--layers", default=None, help="layer grid: these convs only")
            p.add_argument("--flips", type=int, default=100,
                           help="bit grid: flips per image (act, acc) or per model")
            p.add_argument("--out", type=Path, required=True)
            p.add_argument("--criterion", choices=("map", "person"), default="map",
                           help="person: also critical images, a fault-free person lost (T25.6)")
            p.add_argument("--person-conf", type=float, default=PERSON_CONF,
                           help="operating threshold of the person criterion")
    p = sub.add_parser("widths", help="ABFT checksum widths and lane cost (T25.2)")
    p.add_argument("model_dirs", nargs="+")
    p = sub.add_parser("monitor", help="faults in the failure head of T39.1 (T25.9)")
    p.add_argument("labels", nargs="+", help="labels of T39.0 (build/m39/labels/*.npz)")
    p.add_argument("--trials", type=int, default=50)
    p = sub.add_parser("temporal", help="one-frame critical faults on video (T25.8)")
    p.add_argument("--net", default="tiny-yolov2-voc")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=None, help="exported integer model")
    datasets.add_args(p)
    p.add_argument("--subset", type=int, default=0)
    p.add_argument("--points", type=int, default=300, help="injection frames")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--person-conf", type=float, default=PERSON_CONF)
    p.add_argument("--both-m", type=int, default=2, choices=STABLE,
                   help="temporal check used in the Both and Either columns")
    p = sub.add_parser("report")
    p.add_argument("files", nargs="+")
    p.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    {"run": cmd_run, "plan": cmd_plan, "report": cmd_report, "widths": cmd_widths,
     "monitor": cmd_monitor, "temporal": cmd_temporal}[args.cmd](args)


if __name__ == "__main__":
    main()
