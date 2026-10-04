"""mAP PASCAL VOC d'un réseau ou d'un jeu de détections (T3.3, T3.4, §8.3).

    # inférence + évaluation (détections écrites dans --out, format du devkit)
    python tools/eval_voc.py --net tiny-yolov2-voc --weights weights/yolov2-tiny-voc.weights
    # évaluation seule de fichiers comp4_det_test_<classe>.txt existants
    python tools/eval_voc.py --dets build/eval/tiny-yolov2-voc-letterbox

Seuil de confiance bas (0,005, comme `darknet detector valid`) : la mAP intègre toute la
courbe précision/rappel ; le 0,25 de la démo couperait les rappels élevés.
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from yolo.data.voc import VOC_CLASSES, load_split  # noqa: E402
from yolo.infer.metrics import (evaluate, read_detections, to_voc_pixels,  # noqa: E402
                                write_detections)
from yolo.infer.nms import IOU_THR  # noqa: E402
from yolo.infer.pipeline import INTERPS, MODES, detect, preprocess  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import build  # noqa: E402


def _load(task):
    from PIL import Image

    path, size, mode, interp = task
    with Image.open(path) as img:
        return preprocess(img, size, mode, interp)


def run_inference(net, samples, size, mode, interp, conf, iou, batch, workers):
    """Détections de tout le split : {c: (ids, scores, coins pixels VOC)}."""
    per_class = {c: ([], [], []) for c in range(len(VOC_CLASSES))}
    tasks = [(s["image"], size, mode, interp) for s in samples]
    t0 = time.time()
    with mp.Pool(workers) as pool:
        it = pool.imap(_load, tasks, chunksize=4)
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            items = [next(it) for _ in chunk]
            x = np.stack([x for x, _ in items])
            dets = detect(net, x, [wh for _, wh in items], mode, conf, iou)
            for s, (boxes, scores, labels) in zip(chunk, dets):
                px = to_voc_pixels(boxes, s["width"], s["height"])
                for b, sc, c in zip(px, scores, labels):
                    per_class[c][0].append(s["id"])
                    per_class[c][1].append(sc)
                    per_class[c][2].append(b)
            done = start + len(chunk)
            if done % (batch * 25) == 0 or done == len(samples):
                el = time.time() - t0
                print(f"{done}/{len(samples)} images  {el:6.0f} s  "
                      f"({el / done * 1000:.0f} ms/image)", flush=True)
    return {c: (v[0], np.array(v[1]), np.array(v[2]).reshape(-1, 4))
            for c, v in per_class.items()}


def table(aps, m, title):
    lines = [f"### {title}", "", "| Classe | AP |", "|---|---|"]
    lines += [f"| {name} | {ap * 100:.1f} |" for name, ap in zip(VOC_CLASSES, aps)]
    lines += [f"| **mAP** | **{m * 100:.2f}** |", ""]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--dets", type=Path, default=None,
                    help="dossier de détections existantes (pas d'inférence)")
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--split", default="2007:test")
    ap.add_argument("--subset", type=int, default=0, help="n premières images seulement")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--resize", choices=MODES, default="letterbox")
    ap.add_argument("--interp", choices=INTERPS, default="pil")
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--iou", type=float, default=IOU_THR, help="seuil de la NMS")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--area", action="store_true", help="AP par aire (VOC2010+) au lieu de 11 pts")
    ap.add_argument("--out", type=Path, default=None, help="dossier des détections écrites")
    ap.add_argument("--markdown", type=Path, default=None, help="ajoute la table à ce fichier")
    args = ap.parse_args()

    year, split = args.split.split(":")
    samples = load_split(args.devkit, int(year), split)
    if args.subset:
        samples = samples[:args.subset]

    if args.dets:
        dets = read_detections(VOC_CLASSES, args.dets)
        title = f"{args.dets.name} — VOC{year} {split}"
    else:
        if args.weights is None:
            ap.error("--weights ou --dets requis")
        net = build(args.net)
        load_darknet_weights(net, args.weights)
        dets = run_inference(net, samples, args.size, args.resize, args.interp, args.conf,
                             args.iou, args.batch, args.workers)
        run = f"{args.net}-{args.resize}" + ("-darknet" if args.interp == "darknet" else "")
        out = args.out or ROOT / "build" / "eval" / run
        write_detections(dets, VOC_CLASSES, out)
        print(f"détections : {out}")
        title = (f"{args.net} ({args.weights.name}), VOC{year} {split}, {len(samples)} images, "
                 f"{args.size}×{args.size} {args.resize} ({args.interp}), conf {args.conf}, "
                 f"NMS {args.iou}")

    aps, m = evaluate(dets, samples, len(VOC_CLASSES), use_07=not args.area)
    title += ", AP " + ("aire" if args.area else "11 points")
    text = table(aps, m, title)
    print(text)
    if args.markdown:
        with args.markdown.open("a") as f:
            f.write(text + "\n")


if __name__ == "__main__":
    main()
