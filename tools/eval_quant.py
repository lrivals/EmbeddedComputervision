"""mAP du modèle entier et sensibilité par couche (T4.5, §10.5 étape 2).

    # mAP INT8 (modèle entier bit-exact) et flottante sur VOC2007 test complet
    python tools/eval_quant.py --variants float,int
    # détections entières par image (comparaison au stade FPGA, T8.2)
    python tools/eval_quant.py --variants int --resize stretch --save-dets build/m8/int.jsonl
    # sensibilité : une seule couche quantifiée à la fois (fake-quant), 1 000 images
    python tools/eval_quant.py --variants float,fq:all,fq:each --subset 1000

Variantes : `float` (réseau fusionné, float32), `int` (modèle entier, `IntNetwork`, moteur
f64 bit-exact), `fq:all` (simulation flottante de toutes les couches quantifiées),
`fq:<id>` (seule la conv <id> quantifiée), `fq:each` (toutes les `fq:<id>`).
Même prétraitement (`--resize` letterbox | stretch, PIL) et même seuil (0,005) que
`tools/eval_voc.py`. La calibration (T4.2) est faite en letterbox dans les deux cas.
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

from yolo.data.voc import VOC_CLASSES, load_split  # noqa: E402
from yolo.infer.metrics import evaluate, to_voc_pixels  # noqa: E402
from yolo.infer.nms import IOU_THR  # noqa: E402
from yolo.infer.pipeline import MODES  # noqa: E402
from yolo.models.tiny_yolo import PRETRAINED  # noqa: E402

_W = {}


def _init(*args):
    # Une exception dans l'initialiseur ferait relancer les fils sans fin : on la garde
    # pour la relever dans la première tâche.
    try:
        _setup(*args)
    except Exception as exc:  # noqa: BLE001
        _W["error"] = exc


def _setup(name, weights, calib, conf, iou):
    from calibrate import load_fused

    from yolo.quant.calibrate import load_scales
    from yolo.quant.int_model import IntNetwork, QuantModel, head_luts

    fused64 = load_fused(name, weights, dtype=np.float64)
    qm = QuantModel.from_fused(fused64, *load_scales(calib))
    _W.update(fused=load_fused(name, weights, dtype=np.float32), qm=qm,
              inet=IntNetwork(qm, "f64"), luts=head_luts(qm), conf=conf, iou=iou)


def _run(task):
    from PIL import Image

    from yolo.infer.pipeline import detect, detect_int, preprocess
    from yolo.quant.int_model import fake_quant_forward

    if "error" in _W:
        raise _W["error"]
    path, width, height, size, mode, variants = task
    with Image.open(path) as img:
        x, wh = preprocess(img, size, mode)
    x = x[None]
    fused, qm, conf, iou = _W["fused"], _W["qm"], _W["conf"], _W["iou"]
    out = {}
    for v in variants:
        if v == "float":
            dets = detect(fused, x, [wh], mode, conf, iou)
        elif v == "int":
            dets = detect_int(_W["inet"], _W["luts"], x, [wh], mode, conf, iou)
        else:
            layers = None if v == "fq:all" else [int(v[3:])]

            class _Sim:  # même interface que `Network` pour `detect`
                net = fused.net

                @staticmethod
                def forward(x, train=False):
                    return fake_quant_forward(fused, qm, x, layers)

            dets = detect(_Sim, x, [wh], mode, conf, iou)
        boxes, scores, labels = dets[0]
        out[v] = (labels, scores, to_voc_pixels(boxes, width, height))
        if v == "int":  # boîtes normalisées, pour --save-dets (comparaison bit à bit, T8.2)
            out["int:raw"] = (np.asarray(boxes, dtype=np.float64).reshape(-1, 4).tolist(),
                              [float(s) for s in scores], [int(c) for c in labels])
    return out


def expand(variants, name):
    from yolo.models.tiny_yolo import load_cfg

    out = []
    for v in variants:
        if v == "fq:each":
            out += [f"fq:{i}" for i, layer in enumerate(load_cfg(name)["layers"])
                    if layer["type"] == "conv"]
        else:
            out.append(v)
    return out


def run(args, samples, variants):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(args.blas_threads))
    os.environ.setdefault("OMP_NUM_THREADS", str(args.blas_threads))
    ctx = mp.get_context("spawn")  # les variables d'environnement BLAS valent pour les fils
    per = {v: {c: ([], [], []) for c in range(len(VOC_CLASSES))} for v in variants}
    tasks = [(s["image"], s["width"], s["height"], args.size, args.resize, variants)
             for s in samples]
    t0 = time.time()
    raw = open(args.save_dets, "w") if args.save_dets else None  # noqa: SIM115
    with ctx.Pool(args.jobs, initializer=_init,
                  initargs=(args.net, args.weights, args.calib, args.conf, args.iou)) as pool:
        for k, (s, res) in enumerate(zip(samples, pool.imap(_run, tasks, chunksize=2)), 1):
            if raw is not None:
                boxes, scores, labels = res["int:raw"]
                raw.write(json.dumps({"image": s["id"], "boxes": boxes, "scores": scores,
                                      "labels": labels}) + "\n")
            res.pop("int:raw", None)
            for v, (labels, scores, px) in res.items():
                for c, sc, b in zip(labels, scores, px):
                    per[v][c][0].append(s["id"])
                    per[v][c][1].append(sc)
                    per[v][c][2].append(b)
            if k % 200 == 0 or k == len(samples):
                el = time.time() - t0
                print(f"{k}/{len(samples)} images  {el:6.0f} s  ({el / k * 1000:.0f} ms/image)",
                      flush=True)
    if raw is not None:
        raw.close()
    return {v: {c: (d[0], np.array(d[1]), np.array(d[2]).reshape(-1, 4))
                for c, d in pc.items()} for v, pc in per.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc", choices=sorted(PRETRAINED))
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--calib", type=Path, default=None,
                    help="échelles (défaut build/quant/<net>/calib.json)")
    ap.add_argument("--variants", default="float,int")
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--split", default="2007:test")
    ap.add_argument("--subset", type=int, default=0, help="n premières images seulement")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--resize", choices=MODES, default="letterbox")
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--iou", type=float, default=IOU_THR)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--blas-threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None, help="table JSON des mAP")
    ap.add_argument("--save-dets", type=Path, default=None,
                    help="détections du modèle entier, une ligne JSON par image (T8.2)")
    args = ap.parse_args()
    if args.save_dets and "int" not in args.variants.split(","):
        ap.error("--save-dets demande la variante int")
    args.weights = args.weights or ROOT / "weights" / PRETRAINED[args.net]
    args.calib = args.calib or ROOT / "build" / "quant" / args.net / "calib.json"

    year, split = args.split.split(":")
    samples = load_split(args.devkit, int(year), split)
    if args.subset:
        samples = samples[:args.subset]
    variants = expand(args.variants.split(","), args.net)
    dets = run(args, samples, variants)

    res = {}
    for v in variants:
        aps, m = evaluate(dets[v], samples, len(VOC_CLASSES), use_07=True)
        res[v] = {"map": m, "aps": list(aps)}
        print(f"{v:8s} mAP {m * 100:.2f}")
    out = args.out or ROOT / "build" / "quant" / args.net / (
        f"eval_{split}_{len(samples)}_{args.resize}_"
        f"{'-'.join(v.replace(':', '') for v in variants)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"net": args.net, "images": len(samples), "resize": args.resize,
                               "calib": str(args.calib),
                               "results": res}, indent=1) + "\n")
    print(f"résultats : {out}")


if __name__ == "__main__":
    main()
