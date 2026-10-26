"""T39.0: failure labels and frozen features for the learned failure monitor (M39).

    python tools/failure_labels.py label --net tiny-yolov3-coco --set coco:val2017 --n 500
    python tools/failure_labels.py label --net tiny-yolov3-coco --set prewid:test
    python tools/failure_labels.py label --net build/m11/cfg/tiny-yolov3-crowdhuman.cfg \
        --weights <final.weights> --calib build/quant/tiny-yolov3-crowdhuman/calib.json \
        --set crowdhuman:val
    python tools/failure_labels.py report          # section of results/failure_prediction.md

Per image, float and integer stages (`IntNetwork`), `stretch` preprocessing at the input
size of the cfg, threshold `--conf` and NMS 0.45:

- per person (ground truth of the person class, not `difficult`): missed or not, its cause
  (`yolo.explain.misses.classify_miss`, the causes of T36.5; at the integer stage the float
  outputs are given, so "quantization" is a cause) and its TCP (`failure.person_tcp`);
- per image: "at least one missed person" (the KGFP criterion), the top person score, the
  pooled multi-scale features (`failure.feature_layers`: float mean / max, and the integer
  `pool_int_shift`), and a fixed split fit / val / test (60 / 20 / 20, seed 0);
- the person detections down to 0.005 with their correctness (IoU ≥ 0.5, greedy by score):
  the ECE of T39.4.

**Operating point**: the labels depend on it. Until T27.2 chooses it, `--conf` defaults to
0.25 (the threshold of the demo and of M36) and the files say `provisional`. A new threshold
means new labels.

`--set`: `<dataset>:<split>`; `coco` keeps the images with at least one person (the COCO
persons), the other sets keep every image (empty PREWID frames are failure-free images).
`--checkpoint`: a `Trainer` checkpoint of `--net` (T39.2), BN folded after loading.
Output: `build/m39/labels/<model>_<dataset>-<split>.npz`.
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

OUT = ROOT / "build" / "m39"
LABELS = OUT / "labels"
RESULTS = ROOT / "results" / "failure_prediction.md"
CONF, IOU, LOW_CONF, MATCH = 0.25, 0.45, 0.005, 0.5
PERSONS_ONLY = ("coco",)

_W = {}


# ------------------------------------------------------------------------ models
def load_models(net, weights=None, calib=None, checkpoint=None):
    """(float folded net (float32), QuantModel, IntNetwork, head LUTs)."""
    from calibrate import load_fused, net_tag

    from yolo.quant.calibrate import load_scales
    from yolo.quant.int_model import IntNetwork, QuantModel, head_luts

    calib = Path(calib) if calib else ROOT / "build" / "quant" / net_tag(net) / "calib.json"
    if checkpoint:
        fused64, fused32 = (checkpoint_fused(net, checkpoint, d) for d in (np.float64,
                                                                           np.float32))
    else:
        fused64, fused32 = (load_fused(net, weights, dtype=d) for d in (np.float64,
                                                                        np.float32))
    qm = QuantModel.from_fused(fused64, *load_scales(calib))
    inet = IntNetwork(qm)
    return fused32, qm, inet, head_luts(qm)


def checkpoint_fused(net, path, dtype):
    from yolo.backend import copy_into
    from yolo.models.tiny_yolo import build
    from yolo.quant.fuse_bn import fuse_network

    m = build(net, dtype=np.float64)
    with np.load(path) as data:
        for key in data.files:
            kind, _, rest = key.partition("/")
            if kind in ("param", "state"):
                i, k = rest.split("/")
                copy_into((m.params if kind == "param" else m.state)[int(i)][k], data[key])
    return fuse_network(m, dtype=dtype)


def person_index(spec, dataset):
    """(person class of the model, dataset → model class table)."""
    from yolo.data import datasets

    lut = datasets.class_lut(dataset, spec["classes"])
    src = datasets.DATASETS[dataset].classes
    return int(lut[list(src).index("person")]), lut


def set_samples(spec, name, n=None, seed=0):
    """Samples of `<dataset>:<split>`, the first n of a seeded draw if n is given."""
    from yolo.data import datasets

    ds, split = name.split(":", 1)
    pc, lut = person_index(spec, ds)
    s = datasets.load(ds, split=split)
    if ds in PERSONS_ONLY:
        s = [x for x in s if len(x["labels"]) and (lut[np.asarray(x["labels"])] == pc).any()]
    if n is not None and n < len(s):
        idx = np.sort(np.random.default_rng(seed).choice(len(s), n, replace=False))
        s = [s[i] for i in idx]
    return s, ds, pc, lut


# ------------------------------------------------------------------------ labelling
def _init(net, weights, calib, checkpoint, conf, dataset):
    from yolo.infer.failure import feature_layers

    fused, qm, inet, luts = load_models(net, weights, calib, checkpoint)
    pc, lut = person_index(qm.net, dataset)
    _W.update(fused=fused, qm=qm, inet=inet, luts=luts, conf=conf, pc=pc, lut=lut,
              layers=feature_layers(qm.net))


def greedy_correct(dets, gt, match_iou=MATCH):
    """Correct flag of each detection (sorted by score): IoU ≥ `match_iou` with a ground
    truth not taken by a better detection."""
    from yolo.infer.boxes import iou

    boxes = np.asarray(dets, np.float64).reshape(-1, 4)
    out = np.zeros(len(boxes), bool)
    if not len(gt) or not len(boxes):
        return out
    ov = iou(boxes, np.asarray(gt, np.float64))
    free = np.ones(len(gt), bool)
    for k in range(len(boxes)):
        cand = np.where(free, ov[k], -1.0)
        j = int(cand.argmax())
        if cand[j] >= match_iou:
            out[k], free[j] = True, False
    return out


def label_image(s):
    from PIL import Image

    from yolo.data.letterbox import input_size
    from yolo.explain import misses as M
    from yolo.infer import failure as F
    from yolo.infer.nms import filter_and_nms
    from yolo.infer.pipeline import preprocess
    from yolo.quant.quantize import quantize_input

    fused, qm, inet, luts = _W["fused"], _W["qm"], _W["inet"], _W["luts"]
    conf, pc, layers = _W["conf"], _W["pc"], _W["layers"]
    with Image.open(s["image"]) as img:
        x, _ = preprocess(img.convert("RGB"), input_size(qm.net), "stretch")
    x = x[None]
    fouts = fused.forward(x, train=False, all_outputs=True)
    iouts = inet.forward(quantize_input(x), all_outputs=True)
    fo = M.float_outputs({h: fouts[h] for h in qm.outputs}, fused.net, conf, IOU)
    io = M.int_outputs({h: iouts[h] for h in qm.outputs}, qm.net, luts, conf, IOU)
    grids = {h: tuple(fouts[h].shape[-2:]) for h in qm.outputs}
    lab = _W["lut"][np.asarray(s["labels"], np.int64)] if len(s["labels"]) else np.zeros(0, int)
    keep = (lab == pc) & ~np.asarray(s["difficult"], bool)
    gt = np.asarray(s["boxes"], np.float64).reshape(-1, 4)[keep]
    rec = {"id": str(s["id"]), "persons": []}
    for b in gt:
        box = tuple(float(v) for v in b)
        rec["persons"].append({
            "box": box,
            "float": M.classify_miss(fused.net, box, pc, fo, grids, conf),
            "int": M.classify_miss(qm.net, box, pc, io, grids, conf, float_out=fo),
            "tcp_float": F.person_tcp(fused.net, box, pc, fo, grids),
            "tcp_int": F.person_tcp(qm.net, box, pc, io, grids)})
    for stage, out in (("float", fo), ("int", io)):
        rec[f"top_{stage}"] = float(out.scores[:, pc].max())
        b, sc, _ = filter_and_nms(out.boxes, out.scores[:, [pc]], LOW_CONF, IOU)
        rec[f"dets_{stage}"] = (np.asarray(sc, np.float64),
                                greedy_correct(b, gt))
    rec["feat_float"] = F.features_float(fouts, layers)[0].astype(np.float32)
    rec["feat_int"] = F.features_int(iouts, layers)[0].astype(np.int32)
    return rec


def _quiet():
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")


def label_path(net, set_name, out=LABELS):
    from calibrate import net_tag

    return out / f"{net_tag(net)}_{set_name.replace(':', '-')}.npz"


def cmd_label(args):
    from yolo.infer import failure as F

    _quiet()
    tmp = load_models(args.net, args.weights, args.calib, args.checkpoint)
    spec, qm = tmp[1].net, tmp[1]
    samples, ds, pc, _ = set_samples(spec, args.set, args.n, args.seed)
    layers = F.feature_layers(spec)
    dst = args.out or label_path(args.net, args.set)
    dst.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init,
                                      initargs=(args.net, args.weights, args.calib,
                                                args.checkpoint, args.conf, ds)) as pool:
        recs = pool.map(label_image, samples, chunksize=2)
    print(f"{args.set}: {len(recs)} images, {time.time() - t0:.0f} s", flush=True)
    shapes = [tuple(o.shape) for o in tmp[0].forward(
        np.zeros((1, *spec["input"]), np.float32), train=False, all_outputs=True)]
    hws = [shapes[i][2] * shapes[i][3] for i in layers]
    chans = [shapes[i][1] for i in layers]
    scales = F.dequant_scales([qm.scale_of(i) for i in layers], chans, hws)
    save_labels(dst, recs, {
        "net": args.net, "weights": str(args.weights or ""), "checkpoint": str(args.checkpoint
                                                                             or ""),
        "set": args.set, "conf": args.conf, "iou": IOU, "provisional": args.conf == CONF,
        "person_class": pc, "layers": layers, "channels": chans, "hw": hws, "n": len(recs)},
        scales)
    print(f"→ {dst}")


def save_labels(dst, recs, meta, scales):
    from yolo.infer.failure import image_targets, split_of

    img, pers, dets = {}, {}, {}
    p_img, p_box, cf, ci, tf, ti = [], [], [], [], [], []
    for k, r in enumerate(recs):
        for p in r["persons"]:
            p_img.append(k)
            p_box.append(p["box"])
            cf.append(p["float"] or "")
            ci.append(p["int"] or "")
            tf.append(p["tcp_float"])
            ti.append(p["tcp_int"])
    for stage in ("float", "int"):
        causes = [[p[stage] for p in r["persons"]] for r in recs]
        tcps = [[p[f"tcp_{stage}"] for p in r["persons"]] for r in recs]
        t = [image_targets(c, s) for c, s in zip(causes, tcps)]
        img[f"miss_{stage}"] = np.array([a for a, _ in t], bool)
        img[f"tcp_{stage}"] = np.array([b for _, b in t])
        img[f"top_{stage}"] = np.array([r[f"top_{stage}"] for r in recs])
        d_img = np.concatenate([np.full(len(r[f"dets_{stage}"][0]), k)
                                for k, r in enumerate(recs)]).astype(np.int64)
        dets[f"d_image_{stage}"] = d_img
        dets[f"d_score_{stage}"] = np.concatenate([r[f"dets_{stage}"][0] for r in recs])
        dets[f"d_correct_{stage}"] = np.concatenate([r[f"dets_{stage}"][1] for r in recs])
    pers = {"p_image": np.array(p_img, np.int64), "p_box": np.array(p_box).reshape(-1, 4),
            "p_cause_float": np.array(cf, dtype="U16"), "p_cause_int": np.array(ci, dtype="U16"),
            "p_tcp_float": np.array(tf), "p_tcp_int": np.array(ti)}
    np.savez_compressed(
        dst, ids=np.array([r["id"] for r in recs]), split=split_of(len(recs)),
        n_persons=np.array([len(r["persons"]) for r in recs]),
        feat_float=np.stack([r["feat_float"] for r in recs]),
        feat_int=np.stack([r["feat_int"] for r in recs]), scales=scales,
        meta=np.array(json.dumps(meta)), **img, **pers, **dets)


class Labels:
    """A labels file: arrays as attributes, `meta` decoded, per-stage helpers."""

    def __init__(self, path):
        self.path = Path(path)
        with np.load(path) as f:
            self.__dict__.update({k: f[k] for k in f.files})
        self.meta = json.loads(str(self.meta))

    @property
    def name(self):
        return self.path.stem

    def missed(self, stage):
        return getattr(self, f"p_cause_{stage}") != ""

    def features(self, stage):
        """Float features of the stage (integer ones dequantized)."""
        return (self.feat_float.astype(np.float64) if stage == "float"
                else self.feat_int.astype(np.float64) * self.scales)

    def select(self, split):
        """(image mask, person mask, person image index remapped to the subset)."""
        m = self.split == SPLIT_CODE[split]
        pm = m[self.p_image]
        remap = np.cumsum(m) - 1
        return m, pm, remap[self.p_image[pm]]


SPLIT_CODE = {"fit": 0, "val": 1, "test": 2}


# --------------------------------------------------------------------------- report
def write_section(key, title, body, path=RESULTS):
    """Replaces (or appends) the section between `<!-- m39:key -->` markers."""
    start, end = f"<!-- m39:{key} -->", f"<!-- /m39:{key} -->"
    text = path.read_text() if path.exists() else HEADER
    block = f"{start}\n## {title}\n\n{body.rstrip()}\n{end}"
    if start in text:
        a, b = text.index(start), text.index(end) + len(end)
        text = text[:a] + block + text[b:]
    else:
        text = text.rstrip() + "\n\n" + block + "\n"
    path.write_text(text)
    print(f"→ {path} ({key})")


HEADER = """# Learned failure prediction (M39)

A monitor trained to flag the images where the detector misses a person
([docs/tasks/M39-failure-prediction.md](../docs/tasks/M39-failure-prediction.md)). Tools:
`tools/failure_labels.py` (labels and features), `tools/failure_prediction.py` (head, stages,
calibration), `tools/ood_monitor.py compare --labels` (against the monitors of T29.0),
`tools/kgfp_baseline.py`.
"""


def pct(v):
    return "–" if v != v else f"{100 * v:.1f} %"


def cmd_report(args):
    from yolo.explain.misses import CAUSES

    files = sorted(LABELS.glob("*.npz"))
    if not files:
        sys.exit(f"no labels in {LABELS}")
    rows, causes = [], []
    for f in files:
        L = Labels(f)
        for stage in ("float", "int"):
            missed = L.missed(stage)
            for split in ("fit", "val", "test", "all"):
                m = np.ones(len(L.ids), bool) if split == "all" else L.split == SPLIT_CODE[split]
                pm = m[L.p_image]
                fails = getattr(L, f"miss_{stage}")[m]
                rows.append(f"| {L.name} | {stage} | {split} | {m.sum()} | {pm.sum()} | "
                            f"{pct(missed[pm].mean() if pm.any() else float('nan'))} | "
                            f"{pct(fails.mean())} | "
                            f"{(~fails).sum() / max(fails.sum(), 1):.1f} |")
            c = getattr(L, f"p_cause_{stage}")[missed]
            causes.append(f"| {L.name} | {stage} | {missed.sum()} | " + " | ".join(
                pct((c == k).mean() if len(c) else float("nan")) for k in CAUSES) + " |")
    meta = Labels(files[0]).meta
    prov = (", **provisional**: the operating point of T27.2 is not chosen yet; a new "
            "threshold means new labels" if meta["provisional"] else "")
    body = f"""Threshold {meta['conf']} (NMS {meta['iou']}){prov}. A person is missed when no
detection of the person class kept by the NMS has IoU ≥ 0.5 with it (`misses.matched`, as T36.5); an image fails when one of its persons is missed.
Split fit / val / test 60 / 20 / 20, seed 0.

| Labels | Stage | Split | Images | Persons | Missed persons | Images with a miss \
| Negatives per positive |
|---|---|---|---|---|---|---|---|
""" + "\n".join(rows) + f"""

The last column is the class imbalance T39.1 trains against (failure-free images per failing
image): the positive weight of `imbalance="weight"`.

Causes of the missed persons (T36.5, fixed order of tests):

| Labels | Stage | Missed | {' | '.join(CAUSES)} |
|---|---|---|{'---|' * len(CAUSES)}
""" + "\n".join(causes)
    write_section("labels", "Failure labels (T39.0)", body)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("label")
    p.add_argument("--net", default="tiny-yolov3-coco")
    p.add_argument("--weights", type=Path, default=None)
    p.add_argument("--calib", type=Path, default=None)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--set", default="coco:val2017")
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--conf", type=float, default=CONF,
                   help="operating point (T27.2; provisional default 0.25)")
    p.add_argument("--jobs", type=int, default=min(16, os.cpu_count() or 1))
    p.add_argument("--out", type=Path, default=None)
    sub.add_parser("report")
    args = ap.parse_args()
    {"label": cmd_label, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
