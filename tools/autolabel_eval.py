"""Quality and speed of the labellers on a project dataset (M22, T22.7 on VOC, T22.10 by
domain, §8.3).

    python tools/autolabel_eval.py                       # VOC 2007 test, first 200 images
    python tools/autolabel_eval.py --n 500 --models grounding-dino-tiny
    python tools/autolabel_eval.py --dataset visdrone --n 100 --models grounding-dino-tiny,owlv2-base

Each labeller labels `--n` images of the test split of `--dataset` (`DATASETS` of
`yolo.data.datasets`) with the dataset's class names as the class list (prompts through the
alias map of tools/autolabel_models.py), through `label_folder` of tools/autolabel.py. VOC
takes its first `--n` images (T22.7); the other datasets a random subset with a fixed
`--seed`, as VisDrone or AU-AIR are video sequences. The raw predictions (score ≥ 0.05) land
in `build/autolabel/eval/[<dataset>-<split>/]<model>/autolabel.json` and are reused by a
second run. They are scored as detections with `evaluate` (mAP@0.5, VOC07 11 points,
difficult objects and VisDrone ignored regions ignored); precision / recall over all classes
are given at several export thresholds, and the mAP of small objects (< 32² px) apart.
Open-vocabulary labellers are zero-shot here: their mAP is a labelling quality.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

import autolabel as al  # noqa: E402
import autolabel_io as lio  # noqa: E402
import autolabel_models as lm  # noqa: E402
from yolo.data import datasets  # noqa: E402
from yolo.infer.metrics import (class_gts, eval_class, evaluate, ignore_regions,  # noqa: E402
                                to_voc_pixels)

MODELS = ("grounding-dino-tiny", "owlv2-base", "yolo:tiny-yolov2-voc")
CONFS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.55, 0.6)
OUT = ROOT / "build" / "autolabel" / "eval"
SMALL = 32 * 32     # area (px²) under which an object is small (COCO convention)


def subset(samples, n, seed):
    """`n` samples: the first ones (`seed` < 0) or a random subset, in dataset order."""
    if seed < 0 or n >= len(samples):
        return samples[:n]
    keep = sorted(random.Random(seed).sample(range(len(samples)), n))
    return [samples[k] for k in keep]


def image_root(samples):
    """Common folder of the sample images: the keys of the provenance are relative to it."""
    return Path(os.path.commonpath([str(Path(a["image"]).parent) for a in samples]))


def _area(xyxy):
    b = np.asarray(xyxy, dtype=np.float64).reshape(-1, 4)
    return (b[:, 2] - b[:, 0] + 1) * (b[:, 3] - b[:, 1] + 1)


def small_only(samples):
    """Samples whose objects ≥ `SMALL` become difficult (neither TP nor FP when matched)."""
    return [dict(a, difficult=a["difficult"] | (_area(a["xyxy"]) >= SMALL)) for a in samples]


def detections(prov, samples, root, n_classes, conf=0.0, small=False):
    """Raw predictions of the provenance above `conf` → {c: (ids, scores, VOC corners)};
    `small`: only the boxes under `SMALL` px²."""
    per = {c: ([], [], []) for c in range(n_classes)}
    for a in samples:
        rec = prov["images"][Path(a["image"]).relative_to(root).as_posix()]
        boxes, scores, labels, _ = lio.labels_of(dict(rec, removed=[], manual=[]), conf)
        px = to_voc_pixels(boxes, a["width"], a["height"])
        for b, s, c, ar in zip(px, scores, labels, _area(px)):
            if small and ar >= SMALL:
                continue
            per[c][0].append(a["id"])
            per[c][1].append(s)
            per[c][2].append(b)
    return {c: (ids, np.array(s), np.array(b).reshape(-1, 4)) for c, (ids, s, b) in per.items()}


def precision_recall(dets, samples, n_classes):
    """Precision and recall over all classes (micro), difficult objects and ignored regions
    ignored."""
    tp = fp = npos = 0
    ignore = ignore_regions(samples)
    for c in range(n_classes):
        gts = class_gts(samples, c)
        n = sum(int((~np.asarray(d, bool)).sum()) for _, d in gts.values())
        ids, scores, boxes = dets[c]
        rec, prec, _ = eval_class(ids, scores, boxes, gts, ignore=ignore)
        t = rec[-1] * n if len(rec) else 0.0
        # detections not ignored = t / precision; with no TP, all of them are FP
        fp += (t / prec[-1] - t) if len(prec) and prec[-1] > 0 else len(ids)
        tp, npos = tp + t, npos + n
    return tp / max(tp + fp, 1e-9), tp / max(npos, 1)


def present(samples, n_classes):
    """Classes with at least one non-difficult object: the mAP is their mean."""
    return [c for c in range(n_classes)
            if any((~a["difficult"][a["labels"] == c]).any() for a in samples)]


def score(prov, samples, root, n_classes):
    """mAP, AP per class, small-object mAP and P / R by threshold of a provenance."""
    cls = present(samples, n_classes)
    aps, _ = evaluate(detections(prov, samples, root, n_classes), samples, n_classes)
    small = small_only(samples)
    cls_s = present(small, n_classes)
    aps_s, _ = evaluate(detections(prov, samples, root, n_classes, small=True), small,
                        n_classes)
    pr = {t: precision_recall(detections(prov, samples, root, n_classes, t), samples, n_classes)
          for t in CONFS}
    return {"map": float(aps[cls].mean()) if cls else float("nan"), "aps": aps,
            "map_small": float(aps_s[cls_s].mean()) if cls_s else float("nan"),
            "n_small": int(sum((~a["difficult"]).sum() for a in small)), "pr": pr}


def run(model, samples, names, out_dir, labeller=None):
    """Label `samples` (cached in `out_dir/<model>/`) and score them."""
    out = out_dir / model.replace(":", "_")
    root = image_root(samples)
    prompts = [lm.prompt_of(c) for c in names]
    timing = out / "timing.json"
    prov = lio.read_provenance(out)
    rels = [Path(a["image"]).relative_to(root).as_posix() for a in samples]
    todo = [r for r in rels if prov is None or r not in prov["images"]]
    if todo:
        lab = labeller or lm.get_labeller(model)
        t0 = time.time()
        prov = al.run_folder(lab, [a["image"] for a in samples], root, out, names, prompts,
                             conf=0.3, resume=False)
        spi = (time.time() - t0) / len(todo)
        timing.write_text(json.dumps({"s_per_image": spi, "n": len(todo),
                                      "device": lm.pick_device()}))
    spi = json.loads(timing.read_text())["s_per_image"] if timing.exists() else float("nan")
    return dict(score(prov, samples, root, len(names)), model=model,
                version=prov["model_version"], s_per_image=spi)


def table(results, samples, names, where):
    n_obj = int(sum((~a["difficult"]).sum() for a in samples))
    lines = [f"{where}, {len(samples)} images, {n_obj} objects ({results[0]['n_small']} small, "
             f"< 32² px), the {len(names)} class names as prompts (alias map: "
             + ", ".join(f"`{c}` → `{lm.prompt_of(c)}`" for c in names
                         if lm.prompt_of(c) != c.lower())
             + f"), raw predictions ≥ {lm.FLOOR}, device {lm.pick_device()}.", "",
             "| labeller | mAP@0.5 | mAP small | s / image | "
             + " | ".join(f"P / R @ {t}" for t in CONFS) + " |",
             "|---|---|---|---|" + "---|" * len(CONFS)]
    for r in results:
        lines.append(f"| `{r['model']}` | {r['map']:.3f} | {r['map_small']:.3f} | "
                     f"{r['s_per_image']:.2f} | "
                     + " | ".join(f"{p:.2f} / {q:.2f}" for p, q in r["pr"].values()) + " |")
    cls = present(samples, len(names))
    lines += ["", "AP per class (classes present in the subset):", "",
              "| class | objects | " + " | ".join(f"`{r['model']}`" for r in results) + " |",
              "|---|---|" + "---|" * len(results)]
    for c in cls:
        n = int(sum((~a["difficult"][a["labels"] == c]).sum() for a in samples))
        lines.append(f"| {names[c]} | {n} | "
                     + " | ".join(f"{r['aps'][c]:.3f}" for r in results) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    datasets.add_args(ap)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=None,
                    help="random subset; < 0: the first --n (default: VOC first, others 0)")
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="default results/autolabel_voc.md or results/autolabel_<tag>.md")
    args = ap.parse_args()

    split = datasets.split_of(args)
    seed = args.seed if args.seed is not None else (-1 if args.dataset == "voc" else 0)
    samples = subset(datasets.load_args(args), args.n, seed)
    names = list(datasets.DATASETS[args.dataset].classes)
    tag = datasets.tag(args.dataset, split)
    out_dir = OUT / tag if tag else OUT
    where = "VOC 2007 test" if args.dataset == "voc" else f"{args.dataset} {split}"
    if seed >= 0:
        where += f" (random subset, seed {seed})"
    results = []
    for model in args.models.split(","):
        print(f"{model} …", flush=True)
        results.append(run(model, samples, names, out_dir))
        print(f"  mAP {results[-1]['map']:.3f}, {results[-1]['s_per_image']:.2f} s / image",
              flush=True)
    text = table(results, samples, names, where)
    out = args.out or ROOT / "results" / f"autolabel_{tag or 'voc'}.md"
    title = "VOC (M22, T22.7)" if args.dataset == "voc" else f"{args.dataset} (M22, T22.10)"
    out.write_text(f"# Auto-labellers on {title}\n\n{text}")
    print(text)


if __name__ == "__main__":
    main()
