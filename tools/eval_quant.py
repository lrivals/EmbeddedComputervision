"""mAP du modèle entier et sensibilité par couche (T4.5, §10.5 étape 2).

    # mAP INT8 (modèle entier bit-exact) et flottante sur VOC2007 test complet
    python tools/eval_quant.py --variants float,int
    # détections entières par image (comparaison au stade FPGA, T8.2)
    python tools/eval_quant.py --variants int --resize stretch --save-dets build/m8/int.jsonl
    # sensibilité : une seule couche quantifiée à la fois (fake-quant), 1 000 images
    python tools/eval_quant.py --variants float,fq:all,fq:each --subset 1000

Variantes : `float` (réseau fusionné, float32), `int` (modèle entier, `IntNetwork`, moteur
f64 bit-exact), `fq:all` (simulation flottante de toutes les couches quantifiées),
`fq:<id>` (seule la conv <id> quantifiée), `fq:each` (toutes les `fq:<id>`), `int-hwpp`
(modèle entier + post-traitement matériel tout entier et NMS sans tri, T9.1 ; `--hw-cap`
emplacements de sélection, débordements comptés). La table `--out` donne aussi le nombre
moyen de détections par image (T12.2-b) et, pour `int-hwpp`, les candidates perdues. `--model-dir` : évalue un modèle entier
exporté (manifest + blobs, ex. 4 bits ou puissances de 2, M9.2-M9.3) au lieu de celui de
`--calib` ; seules les variantes `int` et `int-hwpp` s'appliquent alors.
Même prétraitement (`--resize` letterbox | stretch, PIL) et même seuil (0,005) que
`tools/eval_voc.py`. La calibration (T4.2) est faite en letterbox dans les deux cas.

Autres jeux (T11.0) : `--dataset coco|kitti|…` ; seules les classes communes au jeu et au
modèle sont évaluées (`yolo.data.datasets.eval_view`), avec la métrique du jeu (`--metric`) ;
`--calib build/quant/<net>/calib-<jeu>.json` pour une calibration faite sur ce jeu (T11.1,
T11.3). Les détections de `--save-dets` restent en classes du modèle.

    python tools/eval_quant.py --net tiny-yolov3-coco --dataset coco --variants float,int
"""

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from yolo.data import datasets  # noqa: E402
from yolo.infer import coco_eval  # noqa: E402
from yolo.infer.metrics import evaluate, to_voc_pixels  # noqa: E402
from yolo.infer.nms import IOU_THR  # noqa: E402
from yolo.infer.pipeline import MODES  # noqa: E402
from yolo.models.tiny_yolo import PRETRAINED, load_cfg  # noqa: E402

_W = {}


def _init(*args):
    # Une exception dans l'initialiseur ferait relancer les fils sans fin : on la garde
    # pour la relever dans la première tâche.
    try:
        _setup(*args)
    except Exception as exc:  # noqa: BLE001
        _W["error"] = exc


def _setup(name, weights, calib, conf, iou, hw_cap=256, model_dir=None):
    from calibrate import load_fused

    from yolo.io.export import load_model
    from yolo.quant.calibrate import load_scales
    from yolo.quant.int_model import IntNetwork, QuantModel, head_luts

    if model_dir:
        qm, fused = load_model(model_dir)[0], None
    else:
        fused64 = load_fused(name, weights, dtype=np.float64)
        qm = QuantModel.from_fused(fused64, *load_scales(calib))
        fused = load_fused(name, weights, dtype=np.float32)
    _W.update(fused=fused, qm=qm,
              inet=IntNetwork(qm, "f64"), luts=head_luts(qm), conf=conf, iou=iou,
              hw_cap=hw_cap)


class _Memo:
    """`IntNetwork` dont la passe avant est gardée pour l'image courante."""

    def __init__(self, inet):
        self.qm, self._inet, self._key, self._out = inet.qm, inet, None, None

    def forward(self, x):
        if self._key is None or not np.array_equal(self._key, x):
            self._key, self._out = x, self._inet.forward(x)
        return self._out


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
    inet = _Memo(_W["inet"])  # une seule passe entière pour int et int-hwpp
    out = {}
    for v in variants:
        if v == "float":
            dets = detect(fused, x, [wh], mode, conf, iou)
        elif v == "int":
            dets = detect_int(inet, _W["luts"], x, [wh], mode, conf, iou)
        elif v == "int-hwpp":
            from yolo.infer.hw_postproc import postprocess_hw_counted

            ov = []
            post = partial(postprocess_hw_counted, cap=_W["hw_cap"], overflow=ov)
            dets = detect_int(inet, _W["luts"], x, [wh], mode, conf, iou, post=post)
            out["hwpp:overflow"] = sum(ov)
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
        # Boîtes normalisées pour --save-dets (comparaison bit à bit, T8.2) : int, ou int-hwpp
        # évalué seul (comparaison à yolo_bench --hw-post, T12.9).
        if v == "int" or (v == "int-hwpp" and "int" not in variants):
            out["int:raw"] = (np.asarray(boxes, dtype=np.float64).reshape(-1, 4).tolist(),
                              [float(s) for s in scores], [int(c) for c in labels])
    return out


def dets_per_image(per_class, n_images):
    """Détections par image d'une variante ({classe: (ids, scores, boîtes)}, T12.2-b)."""
    return sum(len(d[0]) for d in per_class.values()) / max(n_images, 1)


def expand(variants, name):
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
    n_classes = load_cfg(args.net)["classes"]
    per = {v: {c: ([], [], []) for c in range(n_classes)} for v in variants}
    tasks = [(s["image"], s["width"], s["height"], args.size, args.resize, variants)
             for s in samples]
    t0 = time.time()
    raw = open(args.save_dets, "w") if args.save_dets else None  # noqa: SIM115
    overflow = 0
    with ctx.Pool(args.jobs, initializer=_init,
                  initargs=(args.net, args.weights, args.calib, args.conf, args.iou,
                            args.hw_cap, args.model_dir)) as pool:
        for k, (s, res) in enumerate(zip(samples, pool.imap(_run, tasks, chunksize=2)), 1):
            if raw is not None:
                boxes, scores, labels = res["int:raw"]
                raw.write(json.dumps({"image": s["id"], "boxes": boxes, "scores": scores,
                                      "labels": labels}) + "\n")
            res.pop("int:raw", None)
            overflow += res.pop("hwpp:overflow", 0)
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
    if "int-hwpp" in variants:
        print(f"int-hwpp : {overflow} candidates perdues (sélection pleine, {args.hw_cap})")
    dets = {v: {c: (d[0], np.array(d[1]), np.array(d[2]).reshape(-1, 4))
                for c, d in pc.items()} for v, pc in per.items()}
    return dets, overflow


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc", help="réseau pré-entraîné, ou chemin d'un .cfg (élagué, T10.11 ; --weights requis)")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--calib", type=Path, default=None,
                    help="échelles (défaut build/quant/<net>/calib.json)")
    ap.add_argument("--variants", default="float,int")
    datasets.add_args(ap)
    ap.add_argument("--metric", choices=("voc", "coco"), default=None,
                    help="défaut : celle du jeu (coco pour coco et flir)")
    ap.add_argument("--subset", type=int, default=0, help="n premières images seulement")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--resize", choices=MODES, default="letterbox")
    ap.add_argument("--conf", type=float, default=0.005)
    ap.add_argument("--iou", type=float, default=IOU_THR)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--blas-threads", type=int, default=2)
    ap.add_argument("--model-dir", type=Path, default=None,
                    help="modèle entier exporté à évaluer (variantes int, int-hwpp)")
    ap.add_argument("--hw-cap", type=int, default=256,
                    help="emplacements de sélection de la NMS sans tri (int-hwpp)")
    ap.add_argument("--out", type=Path, default=None, help="table JSON des mAP")
    ap.add_argument("--save-dets", type=Path, default=None,
                    help="détections du modèle entier, une ligne JSON par image (T8.2)")
    args = ap.parse_args()
    if args.save_dets and not {"int", "int-hwpp"} & set(args.variants.split(",")):
        ap.error("--save-dets demande la variante int ou int-hwpp")
    if args.model_dir and set(args.variants.split(",")) - {"int", "int-hwpp"}:
        ap.error("--model-dir : variantes int et int-hwpp seulement")
    from calibrate import net_tag

    if args.weights is None and args.net in PRETRAINED:
        args.weights = ROOT / "weights" / PRETRAINED[args.net]
    args.calib = args.calib or ROOT / "build" / "quant" / net_tag(args.net) / "calib.json"

    split = datasets.split_of(args)
    metric = args.metric or datasets.DATASETS[args.dataset].metric
    samples = datasets.load_args(args)
    if args.subset:
        samples = samples[:args.subset]
    view = datasets.eval_view(args.dataset, load_cfg(args.net)["classes"])
    samples = datasets.remap(samples, view.gt_lut)
    variants = expand(args.variants.split(","), args.net)
    dets, overflow = run(args, samples, variants)
    dets = {v: datasets.remap_detections(d, view.det_lut) for v, d in dets.items()}

    res = {}
    for v in variants:
        n_det = dets_per_image(dets[v], len(samples))
        if metric == "coco":
            stats = coco_eval.evaluate(dets[v], samples, len(view.names))
            res[v] = {"map": stats["AP"], "aps": list(stats["ap_class"]),
                      "coco": {k: stats[k] for k in coco_eval.STATS}}
            line = " ".join(f"{k} {stats[k] * 100:.2f}" for k in ("AP", "AP50", "AP75"))
        else:
            aps, m, rec = evaluate(dets[v], samples, len(view.names), use_07=True,
                                   with_recall=True)
            res[v] = {"map": m, "aps": list(aps), "recall": float(rec.mean())}
            line = f"mAP {m * 100:.2f}  rappel {rec.mean() * 100:.1f}"
        res[v]["dets_per_image"] = n_det
        if v == "int-hwpp":
            res[v]["overflow"] = overflow
        print(f"{v:8s} {line}  ({n_det:.1f} détections par image)")
    tag = datasets.tag(args.dataset, split)
    split_name = tag or split.split(":")[-1]
    out = args.out or ROOT / "build" / "quant" / net_tag(args.net) / (
        f"eval_{split_name}_{len(samples)}_{args.resize}_"
        f"{'-'.join(v.replace(':', '') for v in variants)}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"net": args.net, "dataset": args.dataset, "split": split,
                               "metric": metric, "classes": list(view.names),
                               "images": len(samples), "resize": args.resize,
                               "calib": str(args.calib), "conf": args.conf, "iou": args.iou,
                               "model_dir": str(args.model_dir) if args.model_dir else None,
                               "hw_cap": args.hw_cap if "int-hwpp" in variants else None,
                               "results": res}, indent=1) + "\n")
    print(f"résultats : {out}")


if __name__ == "__main__":
    main()
