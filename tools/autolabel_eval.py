"""Quality and speed of the labellers on a VOC 2007 test subset (M22, T22.7, §8.3).

    python tools/autolabel_eval.py                       # 200 images, the three labellers
    python tools/autolabel_eval.py --n 500 --models grounding-dino-tiny

Each labeller labels the first `--n` images of VOC 2007 test with the 20 VOC names as the
class list (prompts through the alias map of tools/autolabel_models.py), through
`label_folder` of tools/autolabel.py: the raw predictions (score ≥ 0.05) land in
`build/autolabel/eval/<model>/autolabel.json` and are reused by a second run. They are
scored as detections with `evaluate` (mAP@0.5, VOC07 11 points, difficult objects ignored),
and precision / recall over all classes are given at several export thresholds.
Open-vocabulary labellers are zero-shot on VOC: their mAP is a labelling quality.
"""

import argparse
import json
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
from yolo.data.voc import VOC_CLASSES, load_split  # noqa: E402
from yolo.infer.metrics import class_gts, eval_class, evaluate, to_voc_pixels  # noqa: E402

MODELS = ("grounding-dino-tiny", "owlv2-base", "yolo:tiny-yolov2-voc")
CONFS = (0.2, 0.25, 0.3, 0.35, 0.4, 0.5)
OUT = ROOT / "build" / "autolabel" / "eval"


def detections(prov, samples, conf=0.0):
    """Raw predictions of the provenance above `conf` → {c: (ids, scores, VOC corners)}."""
    per = {c: ([], [], []) for c in range(len(VOC_CLASSES))}
    for a in samples:
        rec = prov["images"][Path(a["image"]).name]
        boxes, scores, labels, _ = lio.labels_of(dict(rec, removed=[], manual=[]), conf)
        px = to_voc_pixels(boxes, a["width"], a["height"])
        for b, s, c in zip(px, scores, labels):
            per[c][0].append(a["id"])
            per[c][1].append(s)
            per[c][2].append(b)
    return {c: (ids, np.array(s), np.array(b).reshape(-1, 4)) for c, (ids, s, b) in per.items()}


def precision_recall(dets, samples):
    """Precision and recall over all classes (micro), difficult objects ignored."""
    tp = fp = npos = 0
    for c in range(len(VOC_CLASSES)):
        gts = class_gts(samples, c)
        n = sum(int((~np.asarray(d, bool)).sum()) for _, d in gts.values())
        ids, scores, boxes = dets[c]
        rec, prec, _ = eval_class(ids, scores, boxes, gts)
        t = rec[-1] * n if len(rec) else 0.0
        # detections not ignored = t / precision; with no TP, all of them are FP
        fp += (t / prec[-1] - t) if len(prec) and prec[-1] > 0 else len(ids)
        tp, npos = tp + t, npos + n
    return tp / max(tp + fp, 1e-9), tp / max(npos, 1)


def run(model, samples, root):
    out = OUT / model.replace(":", "_")
    names = list(VOC_CLASSES)
    prompts = [lm.prompt_of(c) for c in names]
    timing = out / "timing.json"
    prov = lio.read_provenance(out)
    todo = [a for a in samples if prov is None or Path(a["image"]).name not in prov["images"]]
    if todo:
        lab = lm.get_labeller(model)
        t0 = time.time()
        prov = al.run_folder(lab, [a["image"] for a in samples], root, out, names, prompts,
                             conf=0.3, resume=False)
        spi = (time.time() - t0) / len(todo)
        timing.write_text(json.dumps({"s_per_image": spi, "n": len(todo),
                                      "device": lm.pick_device()}))
    spi = json.loads(timing.read_text())["s_per_image"] if timing.exists() else float("nan")
    aps, m = evaluate(detections(prov, samples), samples, len(VOC_CLASSES))
    pr = {t: precision_recall(detections(prov, samples, t), samples) for t in CONFS}
    return {"model": model, "version": prov["model_version"], "map": m, "aps": aps,
            "s_per_image": spi, "pr": pr}


def table(results, n):
    lines = [f"VOC 2007 test, first {n} images, 20 VOC classes as prompts (alias map), "
             f"raw predictions ≥ {lm.FLOOR}, device {lm.pick_device()}.", "",
             "| labeller | mAP@0.5 | s / image | "
             + " | ".join(f"P / R @ {t}" for t in CONFS) + " |",
             "|---|---|---|" + "---|" * len(CONFS)]
    for r in results:
        lines.append(f"| `{r['model']}` | {r['map']:.3f} | {r['s_per_image']:.2f} | "
                     + " | ".join(f"{p:.2f} / {q:.2f}" for p, q in r["pr"].values()) + " |")
    lines += ["", "AP per class:", "",
              "| class | " + " | ".join(f"`{r['model']}`" for r in results) + " |",
              "|---|" + "---|" * len(results)]
    for c, name in enumerate(VOC_CLASSES):
        lines.append(f"| {name} | " + " | ".join(f"{r['aps'][c]:.3f}" for r in results) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "results" / "autolabel_voc.md")
    args = ap.parse_args()

    samples = load_split(args.devkit, "2007", "test")[:args.n]
    root = args.devkit / "VOC2007" / "JPEGImages"
    results = []
    for model in args.models.split(","):
        print(f"{model} …", flush=True)
        results.append(run(model, samples, root))
        print(f"  mAP {results[-1]['map']:.3f}, {results[-1]['s_per_image']:.2f} s / image")
    text = table(results, len(samples))
    args.out.write_text(f"# Auto-labellers on VOC (M22, T22.7)\n\n{text}")
    print(text)


if __name__ == "__main__":
    main()
