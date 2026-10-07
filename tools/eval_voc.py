"""mAP PASCAL VOC d'un réseau ou d'un jeu de détections (T3.3, T3.4, §8.3).

    # inférence + évaluation (détections écrites dans --out, format du devkit)
    python tools/eval_voc.py --net tiny-yolov2-voc --weights weights/yolov2-tiny-voc.weights
    # évaluation seule de fichiers comp4_det_test_<classe>.txt existants
    python tools/eval_voc.py --dets build/eval/tiny-yolov2-voc-letterbox
    # autre jeu (T11.0) : COCO val2017 avec la métrique COCO, poids VOC sur KITTI (T11.2)
    python tools/eval_voc.py --net tiny-yolov3-coco --weights weights/yolov3-tiny.weights \
        --dataset coco
    python tools/eval_voc.py --net tiny-yolov2-voc --weights weights/yolov2-tiny-voc.weights \
        --dataset kitti

Hors VOC, seules les classes communes au jeu et au modèle sont évaluées
(`yolo.data.datasets.eval_view`) ; `--metric coco` donne AP@[.5:.95] (`yolo.infer.coco_eval`).

Seuil de confiance bas (0,005, comme `darknet detector valid`) : la mAP intègre toute la
courbe précision/rappel ; le 0,25 de la démo couperait les rappels élevés.

`--jobs N` (N > 1) : N processus, un thread BLAS chacun, qui traitent les mêmes lots que le
chemin série, dans le même ordre (machines à beaucoup de cœurs, runtime TPU de Colab).
"""

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from yolo.data import datasets  # noqa: E402
from yolo.data.letterbox import input_size, parse_size, size_label  # noqa: E402
from yolo.infer import coco_eval  # noqa: E402
from yolo.infer.metrics import (evaluate, read_detections, to_voc_pixels,  # noqa: E402
                                write_detections)
from yolo.infer.nms import IOU_THR  # noqa: E402
from yolo.infer.pipeline import INTERPS, MODES, detect, preprocess  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import build, load_cfg  # noqa: E402


def _load(task):
    from PIL import Image

    path, size, mode, interp, channels = task
    with Image.open(path) as img:
        return preprocess(img, size, mode, interp, channels)


_W = {}  # réseau de chaque processus de --jobs


def _init(name, weights, conf, iou):
    # Une exception dans l'initialiseur ferait relancer les fils sans fin : on la garde
    # pour la relever dans la première tâche (comme eval_quant.py).
    try:
        net = build(name)
        load_darknet_weights(net, weights)
        _W.update(net=net, conf=conf, iou=iou)
    except Exception as exc:  # noqa: BLE001
        _W["error"] = exc


def _batch(task):
    """Un lot dans un processus de --jobs : chargement, détection, boîtes en pixels VOC."""
    if "error" in _W:
        raise _W["error"]
    loads, mode, sizes = task
    items = [_load(t) for t in loads]
    x = np.stack([x for x, _ in items])
    dets = detect(_W["net"], x, [wh for _, wh in items], mode, _W["conf"], _W["iou"])
    return [(to_voc_pixels(boxes, w, h), scores, labels)
            for (boxes, scores, labels), (w, h) in zip(dets, sizes)]


def _serial(net, samples, tasks, mode, conf, iou, batch, workers):
    with mp.Pool(workers) as pool:
        it = pool.imap(_load, tasks, chunksize=4)
        for start in range(0, len(samples), batch):
            chunk = samples[start:start + batch]
            items = [next(it) for _ in chunk]
            x = np.stack([x for x, _ in items])
            dets = detect(net, x, [wh for _, wh in items], mode, conf, iou)
            yield chunk, [(to_voc_pixels(boxes, s["width"], s["height"]), scores, labels)
                          for s, (boxes, scores, labels) in zip(chunk, dets)]


def _parallel(name, weights, samples, tasks, mode, conf, iou, batch, jobs, blas_threads):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(blas_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(blas_threads))
    ctx = mp.get_context("spawn")  # les variables d'environnement BLAS valent pour les fils
    starts = range(0, len(samples), batch)
    work = [(tasks[i:i + batch], mode, [(s["width"], s["height"]) for s in samples[i:i + batch]])
            for i in starts]
    with ctx.Pool(jobs, initializer=_init, initargs=(name, weights, conf, iou)) as pool:
        for i, res in zip(starts, pool.imap(_batch, work)):
            yield samples[i:i + batch], res


def run_inference(net, samples, size, mode, interp, conf, iou, batch, workers, jobs=1,
                  name=None, weights=None, blas_threads=1):
    """Détections de tout le split : {c: (ids, scores, coins pixels VOC)}, c classe du modèle.
    `jobs` > 1 : lots répartis sur `jobs` processus (réseau `name` + `weights` rechargé dans
    chacun) ; mêmes lots, même ordre que le chemin série."""
    per_class = {c: ([], [], []) for c in range(net.net["classes"])}
    channels = net.net["input"][0]
    tasks = [(s["image"], size, mode, interp, channels) for s in samples]
    t0 = time.time()
    if jobs > 1:
        chunks = _parallel(name, weights, samples, tasks, mode, conf, iou, batch, jobs,
                           blas_threads)
    else:
        chunks = _serial(net, samples, tasks, mode, conf, iou, batch, workers)
    done = 0
    for chunk, res in chunks:
        for s, (px, scores, labels) in zip(chunk, res):
            for b, sc, c in zip(px, scores, labels):
                per_class[c][0].append(s["id"])
                per_class[c][1].append(sc)
                per_class[c][2].append(b)
        done += len(chunk)
        if done % (batch * 25) == 0 or done == len(samples):
            el = time.time() - t0
            print(f"{done}/{len(samples)} images  {el:6.0f} s  "
                  f"({el / done * 1000:.0f} ms/image)", flush=True)
    return {c: (v[0], np.array(v[1]), np.array(v[2]).reshape(-1, 4))
            for c, v in per_class.items()}


def table(aps, m, title, names):
    lines = [f"### {title}", "", "| Classe | AP |", "|---|---|"]
    lines += [f"| {name} | {ap * 100:.1f} |" for name, ap in zip(names, aps)]
    lines += [f"| **mAP** | **{m * 100:.2f}** |", ""]
    return "\n".join(lines)


def coco_table(res, title, names):
    """Table de la métrique COCO : résumé puis AP et AP50 par classe."""
    lines = [f"### {title}", "", "| " + " | ".join(coco_eval.STATS) + " |",
             "|---" * len(coco_eval.STATS) + "|",
             "| " + " | ".join(f"{res[k] * 100:.1f}" for k in coco_eval.STATS) + " |", "",
             "| Classe | AP | AP50 |", "|---|---|---|"]
    lines += [f"| {n} | {a * 100:.1f} | {b * 100:.1f} |"
              for n, a, b in zip(names, res["ap_class"], res["ap50_class"])]
    return "\n".join(lines + [""])


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--dets", type=Path, default=None,
                    help="dossier de détections existantes (pas d'inférence)")
    datasets.add_args(ap)
    ap.add_argument("--metric", choices=("voc", "coco"), default=None,
                    help="défaut : celle du jeu (coco pour coco et flir)")
    ap.add_argument("--subset", type=int, default=0, help="n premières images seulement")
    ap.add_argument("--size", type=parse_size, default=None, help="S ou LxH (entrée non carrée, ex. 640x192) ; défaut : entrée de la cfg")
    ap.add_argument("--resize", choices=MODES, default="letterbox")
    ap.add_argument("--interp", choices=INTERPS, default="pil")
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--iou", type=float, default=IOU_THR, help="seuil de la NMS")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--jobs", type=int, default=1,
                    help="processus d'inférence (1 : un seul, BLAS multithread)")
    ap.add_argument("--blas-threads", type=int, default=1, help="threads BLAS par processus de --jobs")
    ap.add_argument("--area", action="store_true", help="AP par aire (VOC2010+) au lieu de 11 pts")
    ap.add_argument("--out", type=Path, default=None, help="dossier des détections écrites")
    ap.add_argument("--markdown", type=Path, default=None, help="ajoute la table à ce fichier")
    ap.add_argument("--no-figures", action="store_true",
                    help="pas de courbes PR à côté des détections (M13 ; aussi YOLO_FIGURES=0)")
    args = ap.parse_args()

    split = datasets.split_of(args)
    metric = args.metric or datasets.DATASETS[args.dataset].metric
    samples = datasets.load_args(args)
    if args.subset:
        samples = samples[:args.subset]
    view = datasets.eval_view(args.dataset, load_cfg(args.net)["classes"])
    samples = datasets.remap(samples, view.gt_lut)
    where = (f"VOC{split.replace(':', ' ')}" if args.dataset == "voc"
             else f"{args.dataset} {split}")

    if args.dets:
        dets = read_detections(view.names, args.dets)
        title = f"{args.dets.name} — {where}"
    else:
        if args.weights is None:
            ap.error("--weights ou --dets requis")
        net = build(args.net)
        load_darknet_weights(net, args.weights)
        args.size = input_size(net.net, args.size)
        dets = run_inference(net, samples, args.size, args.resize, args.interp, args.conf,
                             args.iou, args.batch, args.workers, args.jobs, args.net,
                             args.weights, args.blas_threads)
        dets = datasets.remap_detections(dets, view.det_lut)
        run = f"{args.net}-{args.resize}" + ("-darknet" if args.interp == "darknet" else "")
        tag = datasets.tag(args.dataset, split)
        out = args.out or ROOT / "build" / "eval" / (f"{run}-{tag}" if tag else run)
        write_detections(dets, view.names, out)
        print(f"détections : {out}")
        title = (f"{args.net} ({args.weights.name}), {where}, {len(samples)} images, "
                 f"{size_label(args.size)} {args.resize} ({args.interp}), conf {args.conf}, "
                 f"NMS {args.iou}")

    if metric == "coco":
        res = coco_eval.evaluate(dets, samples, len(view.names))
        text = coco_table(res, title + ", métrique COCO (101 points)", view.names)
    else:
        aps, m = evaluate(dets, samples, len(view.names), use_07=not args.area)
        title += ", AP " + ("aire" if args.area else "11 points")
        text = table(aps, m, title, view.names)
    print(text)
    if args.markdown:
        with args.markdown.open("a") as f:
            f.write(text + "\n")
    if metric == "voc" and not args.no_figures:
        sys.path.insert(0, str(ROOT))
        from tools.figures.auto import after_run

        # --out donné (notebooks M14) : courbes à côté des détections, <out>/figures/.
        det_dir = args.dets or out
        fig_dir = (args.out / "figures" if args.out
                   else ROOT / "build" / "figures" / "eval" / det_dir.name)
        after_run("eval-voc", det_dir, out=fig_dir, dets=dets, samples=samples,
                  names=view.names, title=title)


if __name__ == "__main__":
    main()
