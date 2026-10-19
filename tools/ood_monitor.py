"""T25.5: out-of-distribution monitor on the backbone features of Tiny-YOLOv2.

    python tools/ood_monitor.py extract --jobs 16   # features + integer detections
    python tools/ood_monitor.py fit                 # Mahalanobis baseline + autoencoder
    python tools/ood_monitor.py eval                # AUROC, score / mAP per group, cost

Features: output of L13 (last backbone conv, 1024 × 13 × 13) of the integer model
(`IntNetwork`, int8 map dequantized by its scale), `stretch` preprocessing as
`eval_quant.py`. Sets (`build/m25/ood/feat_<set>.npz`):

- `fit`: 2,000 images of VOC2007 trainval drawn with seed 0 (training of the monitor);
- `voc`: the first 500 images of VOC2007 test (in-domain);
- `exdark`: 500 images of the ExDark test split (low light), drawn with seed 0;
- `flir`: 500 images of the FLIR ADAS v2 thermal val split, drawn with seed 0.

`eval`: AUROC of VOC test against ExDark and against FLIR for the two scores
(`yolo.infer.ood`); score quintiles inside each set, mAP (VOC07, AP50, classes mapped to VOC
by `datasets.eval_view`) of each group, and Spearman correlation between the mean score and
the mAP of the 15 groups; MACs of the monitor against the detector.
"""

import argparse
import functools
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

from yolo.infer import ood  # noqa: E402

NET = "tiny-yolov2-voc"
LAYER = 13
OUT = ROOT / "build" / "m25" / "ood"
SETS = {  # name: (dataset, split, images, random draw)
    "fit": ("voc", "2007:trainval", 2000, True),
    "voc": ("voc", "2007:test", 500, False),
    "exdark": ("exdark", "test", 500, True),
    "flir": ("flir", "val", 500, True),
}
TEST_SETS = ("voc", "exdark", "flir")
GROUPS = 5
CONF, IOU = 0.005, 0.45

_W = {}


@functools.lru_cache(maxsize=None)
def _split(name):
    from yolo.data import datasets

    ds, split, _, _ = SETS[name]
    return tuple(datasets.load(ds, split=split))


def samples_of(name, n=None, seed=0):
    _, _, count, draw = SETS[name]
    s = list(_split(name))
    n = count if n is None else n
    if draw and n < len(s):
        idx = np.sort(np.random.default_rng(seed).choice(len(s), n, replace=False))
        s = [s[i] for i in idx]
    return s[:n]


# ------------------------------------------------------------------------- extraction
def _init(calib, layer):
    from calibrate import load_fused

    from yolo.quant.calibrate import load_scales
    from yolo.quant.int_model import IntNetwork, QuantModel, head_luts
    from yolo.models.tiny_yolo import PRETRAINED

    qm = QuantModel.from_fused(load_fused(NET, ROOT / "weights" / PRETRAINED[NET],
                                          dtype=np.float64), *load_scales(calib))
    _W.update(qm=qm, inet=IntNetwork(qm), luts=head_luts(qm), layer=layer)


def _extract(path):
    from PIL import Image

    from yolo.infer.nms import postprocess_int
    from yolo.infer.pipeline import preprocess
    from yolo.quant.quantize import quantize_input

    with Image.open(path) as img:
        x, _ = preprocess(img, 416, "stretch")
    outs = _W["inet"].forward(quantize_input(x[None]), all_outputs=True)
    heads = {i: outs[i] for i in _W["qm"].outputs}
    boxes, scores, labels = postprocess_int(heads, _W["qm"].net, _W["luts"], CONF, IOU)[0]
    return (outs[_W["layer"]][0], np.asarray(labels).tolist(),
            np.asarray(scores, np.float64).tolist(),
            np.asarray(boxes, np.float64).reshape(-1, 4).tolist())


def cmd_extract(args):
    from calibrate import net_tag

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    calib = ROOT / "build" / "quant" / net_tag(NET) / "calib.json"
    args.out.mkdir(parents=True, exist_ok=True)
    with mp.get_context("spawn").Pool(args.jobs, initializer=_init,
                                      initargs=(calib, args.layer)) as pool:
        for name in args.sets.split(","):
            dst = args.out / f"feat_{name}.npz"
            if dst.exists():
                print(f"{dst}: already there")
                continue
            t0 = time.time()
            s = samples_of(name, args.n)
            res = pool.map(_extract, [x["image"] for x in s], chunksize=4)
            dets = [{"labels": r[1], "scores": r[2], "boxes": r[3]} for r in res]
            np.savez(dst, maps=np.stack([r[0] for r in res]).astype(np.int8),
                     ids=np.array([x["id"] for x in s]),
                     width=np.array([x["width"] for x in s]),
                     height=np.array([x["height"] for x in s]))
            (args.out / f"dets_{name}.json").write_text(json.dumps(dets) + "\n")
            print(f"{name}: {len(s)} images  {time.time() - t0:.0f} s", flush=True)
    _init(calib, args.layer)
    (args.out / "scale.json").write_text(json.dumps(
        {"layer": args.layer, "scale": _W["qm"].scale_of(args.layer)}) + "\n")


def load_feats(out, name):
    f = np.load(out / f"feat_{name}.npz")
    scale = json.loads((out / "scale.json").read_text())["scale"]
    return f["maps"], scale, f


# ------------------------------------------------------------------------- fit
def standardizer(maps, scale, chunk=100):
    """Per-channel mean / std of the dequantized maps, by chunks (int8 maps stay int8)."""
    c = maps.shape[1]
    s1, s2, n = np.zeros(c), np.zeros(c), 0
    for k in range(0, len(maps), chunk):
        m = maps[k:k + chunk].astype(np.float64) * scale
        s1 += m.sum(axis=(0, 2, 3))
        s2 += (m ** 2).sum(axis=(0, 2, 3))
        n += m.shape[0] * m.shape[2] * m.shape[3]
    st = ood.Standardizer()
    st.mean = (s1 / n)[None, :, None, None]
    st.std = np.sqrt(np.maximum(s2 / n - (s1 / n) ** 2, 0))[None, :, None, None] + 1e-6
    return st


class _Batches:
    """Indexable view that standardizes int8 maps on access (for `ood.train_ae`)."""

    def __init__(self, maps, scale, st):
        self.maps, self.scale, self.st = maps, scale, st

    def __len__(self):
        return len(self.maps)

    def __getitem__(self, idx):
        return self.st(self.maps[idx].astype(np.float64) * self.scale)


def ae_scores(net, maps, scale, st, chunk=50):
    return np.concatenate([ood.reconstruction_error(net, st(maps[k:k + chunk] * scale))
                           for k in range(0, len(maps), chunk)])


METHODS = ("norm", "mahalanobis", "autoencoder")
HELD_OUT = 500  # images of the fit set kept out of the fit: reference of the typicality


def split_fit(n, seed=0):
    """(fit indices, held-out indices) of the fit set."""
    perm = np.random.default_rng(seed).permutation(n)
    return np.sort(perm[HELD_OUT:]), np.sort(perm[:HELD_OUT])


def raw_scores(mon, maps, scale):
    """{method: raw score} of int8 maps."""
    maha, st, net = mon
    pooled = ood.pool(maps * scale)
    return {"norm": ood.norm(pooled), "mahalanobis": maha.score(pooled),
            "autoencoder": ae_scores(net, maps, scale, st)}


def cmd_fit(args):
    maps, scale, _ = load_feats(args.out, "fit")
    tr, ho = split_fit(len(maps), args.seed)
    fit = maps[tr]
    t0 = time.time()
    maha = ood.Mahalanobis().fit(ood.pool(fit * scale))
    print(f"Mahalanobis: {len(fit)} images, shrinkage {maha.shrinkage:.3f}  "
          f"{time.time() - t0:.0f} s")
    st = standardizer(fit, scale)
    net = ood.make_ae(maps.shape[1], maps.shape[2:], tuple(args.hidden), seed=args.seed)
    losses = ood.train_ae(net, _Batches(fit, scale, st), args.iters, args.batch, args.lr,
                          args.seed, log_every=200)
    ref = {m: float(np.median(v)) for m, v in raw_scores((maha, st, net), maps[ho],
                                                          scale).items()}
    print("typicality references (median on the held-out images): "
          + ", ".join(f"{m} {v:.4g}" for m, v in ref.items()))
    params = {f"ae_{i}_{k}": v for i, p in enumerate(net.params) for k, v in p.items()}
    np.savez(args.out / "monitor.npz", maha_mean=maha.mean, maha_prec=maha.prec,
             maha_shrinkage=maha.shrinkage, st_mean=st.mean, st_std=st.std,
             hidden=np.array(args.hidden), losses=np.array(losses),
             ref=np.array([ref[m] for m in METHODS]), **params)
    print(f"monitor: {args.out / 'monitor.npz'}  ({time.time() - t0:.0f} s)")


def load_monitor(out):
    z = np.load(out / "monitor.npz")
    maha = ood.Mahalanobis()
    maha.mean, maha.prec, maha.shrinkage = z["maha_mean"], z["maha_prec"], float(
        z["maha_shrinkage"])
    st = ood.Standardizer()
    st.mean, st.std = z["st_mean"], z["st_std"]
    c = st.mean.shape[1]
    net = ood.make_ae(c, (13, 13), tuple(int(h) for h in z["hidden"]))
    for i, p in enumerate(net.params):
        for k in p:
            p[k][...] = z[f"ae_{i}_{k}"]
    return (maha, st, net), dict(zip(METHODS, z["ref"].tolist())), z["losses"]


# ------------------------------------------------------------------------- eval
def group_map(name, feats, dets, members):
    """mAP (VOC07, classes mapped to VOC) of the images `members` of a set."""
    from yolo.data import datasets
    from yolo.infer.metrics import evaluate, to_voc_pixels

    ds = SETS[name][0]
    view = datasets.eval_view(ds, 20)
    by_id = {s["id"]: s for s in _split(name)}
    samples = datasets.remap([by_id[str(feats["ids"][k])] for k in members], view.gt_lut)
    per = {c: ([], [], []) for c in range(20)}
    for k in members:
        d = dets[k]
        px = to_voc_pixels(d["boxes"], int(feats["width"][k]), int(feats["height"][k]))
        for c, s, b in zip(d["labels"], d["scores"], px):
            per[c][0].append(str(feats["ids"][k]))
            per[c][1].append(s)
            per[c][2].append(b)
    per = {c: (v[0], np.array(v[1]), np.array(v[2]).reshape(-1, 4)) for c, v in per.items()}
    per = datasets.remap_detections(per, view.det_lut)
    return float(evaluate(per, samples, len(view.names), use_07=True)[1])


def cmd_eval(args):
    from yolo.models.tiny_yolo import load_cfg

    mon, ref, losses = load_monitor(args.out)
    raw, typ, data = {}, {}, {}
    for name in TEST_SETS:
        maps, scale, f = load_feats(args.out, name)
        data[name] = (f, json.loads((args.out / f"dets_{name}.json").read_text()))
        raw[name] = raw_scores(mon, maps, scale)
        typ[name] = {m: ood.typicality(v, ref[m]) for m, v in raw[name].items()}
    res = {"auroc_raw": {}, "auroc": {}, "groups": [], "spearman": {}, "ref": ref,
           "images": {n: len(data[n][1]) for n in TEST_SETS},
           "median_raw": {n: {m: float(np.median(v)) for m, v in raw[n].items()}
                          for n in TEST_SETS}}
    for m in METHODS:
        res["auroc_raw"][m] = {o: ood.auroc(raw["voc"][m], raw[o][m]) for o in ("exdark", "flir")}
        res["auroc"][m] = {o: ood.auroc(typ["voc"][m], typ[o][m]) for o in ("exdark", "flir")}
    # Score quintiles inside each set; mAP of each group (typicality scores).
    for m in METHODS:
        means, maps_ = [], []
        for name in TEST_SETS:
            sc = typ[name][m]
            edges = np.quantile(sc, np.linspace(0, 1, GROUPS + 1))
            g = np.clip(np.searchsorted(edges, sc, side="right") - 1, 0, GROUPS - 1)
            for q in range(GROUPS):
                members = np.flatnonzero(g == q).tolist()
                mp = group_map(name, data[name][0], data[name][1], members)
                res["groups"].append({"method": m, "set": name, "group": q,
                                      "images": len(members),
                                      "score": float(sc[members].mean()), "map": mp})
                means.append(float(sc[members].mean()))
                maps_.append(mp)
        res["spearman"][m] = ood.spearman(means, maps_)
    res["map"] = {n: group_map(n, data[n][0], data[n][1], list(range(len(data[n][1]))))
                  for n in TEST_SETS}
    maha, _, net = mon
    d = len(maha.mean)
    pool_ops = 2 * net.net["input"][0] * net.net["input"][1] * net.net["input"][2]
    res["macs"] = {"detector": ood.macs(load_cfg(NET)), "autoencoder": ood.macs(net.net),
                   "mahalanobis": d * d + d + pool_ops, "norm": d + pool_ops}
    res["shrinkage"] = maha.shrinkage
    res["ae_loss"] = {"first": float(np.mean(losses[:50])), "last": float(np.mean(losses[-50:]))}
    (args.out / "eval.json").write_text(json.dumps(res, indent=1) + "\n")
    print(markdown(res))
    print(f"results: {args.out / 'eval.json'}")


def markdown(res):
    lines = ["| Score | AUROC ExDark (raw) | AUROC FLIR (raw) | AUROC ExDark (typicality) | "
             "AUROC FLIR (typicality) | Spearman score / mAP (15 groups) | MACs |",
             "|---|---|---|---|---|---|---|"]
    det = res["macs"]["detector"]
    for m in METHODS:
        a, r, mc = res["auroc"][m], res["auroc_raw"][m], res["macs"][m]
        lines.append(f"| {m} | {r['exdark']:.3f} | {r['flir']:.3f} | **{a['exdark']:.3f}** | "
                     f"**{a['flir']:.3f}** | {res['spearman'][m]:+.2f} | "
                     f"{mc / 1e6:.2f} M ({mc / det * 100:.2f} %) |")
    lines += ["", "| Score | Set | " + " | ".join(f"Q{q + 1}" for q in range(GROUPS)) + " |",
              "|---|---|" + "---|" * GROUPS]
    for m in METHODS:
        for s in TEST_SETS:
            g = [x for x in res["groups"] if x["method"] == m and x["set"] == s]
            lines.append(f"| {m} | {s} (mAP {res['map'][s] * 100:.1f}) | "
                         + " | ".join(f"{x['map'] * 100:.1f}" for x in g) + " |")
    lines += ["", f"MACs against the detector ({det / 1e6:.0f} M per image)."]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=OUT)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("extract")
    p.add_argument("--sets", default=",".join(SETS))
    p.add_argument("--n", type=int, default=None, help="images per set (default: SETS)")
    p.add_argument("--layer", type=int, default=LAYER)
    p.add_argument("--jobs", type=int, default=16)
    p = sub.add_parser("fit")
    p.add_argument("--hidden", type=int, nargs="+", default=[128, 32])
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    sub.add_parser("eval")
    args = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
