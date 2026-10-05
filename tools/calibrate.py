"""Calibration des échelles d'activation du modèle entier (T4.2, §9.2).

    python tools/calibrate.py --net tiny-yolov2-voc     # → build/quant/<net>/calib.json
    python tools/calibrate.py --net tiny-yolov3-coco --images 500

Images de calibration : tirées au hasard (graine fixe) dans VOC2007 trainval, jamais dans
le split de test ; même prétraitement que l'évaluation (letterbox, PIL).
"""

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

import numpy as np  # noqa: E402

from yolo.data.voc import load_split  # noqa: E402
from yolo.infer.pipeline import preprocess  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import PRETRAINED, build  # noqa: E402
from yolo.quant.calibrate import (CHOICES, HEAD_SCALE, PERCENTILES, ActStats,  # noqa: E402
                                  calib_dict, choose_scales, save_calib)
from yolo.quant.fuse_bn import fuse_network  # noqa: E402


def _load(task):
    from PIL import Image

    path, size = task
    with Image.open(path) as img:
        return preprocess(img, size)[0]


def net_tag(name):
    """Nom des dossiers de sortie : le réseau, ou le nom de fichier d'un .cfg (T10.11)."""
    return name if name in PRETRAINED else Path(name).stem


def load_fused(name, weights=None, dtype=np.float32):
    """`name` : réseau de `PRETRAINED`, ou chemin d'un .cfg avec ses `weights` Darknet."""
    if weights is None and name not in PRETRAINED:
        raise ValueError(f"{name} : --weights requis hors des réseaux pré-entraînés")
    net = build(name)
    load_darknet_weights(net, weights or ROOT / "weights" / PRETRAINED[name])
    return fuse_network(net, dtype=dtype)


def calib_images(devkit, n, seed=0, split="2007:trainval"):
    year, s = split.split(":")
    samples = load_split(devkit, int(year), s)
    idx = np.random.default_rng(seed).choice(len(samples), size=min(n, len(samples)),
                                             replace=False)
    return [samples[i]["image"] for i in sorted(idx)]


def collect(fused, paths, size=416, batch=8, workers=4):
    stats = ActStats(fused)
    t0 = time.time()
    with mp.Pool(workers) as pool:
        it = pool.imap(_load, [(p, size) for p in paths], chunksize=4)
        for start in range(0, len(paths), batch):
            n = min(batch, len(paths) - start)
            x = np.stack([next(it) for _ in range(n)])
            stats.update(fused.forward(x, train=False, all_outputs=True))
            done = start + n
            if done % (batch * 16) == 0 or done == len(paths):
                print(f"{done}/{len(paths)} images  {time.time() - t0:5.0f} s", flush=True)
    return stats


def markdown(calib):
    keys = [f"p{p:g}" for p in PERCENTILES] + ["max"]
    head = calib["head_scale"]
    lines = [
        f"### {calib['network']} — {calib['images']} images VOC2007 trainval, "
        f"choix « {calib['choice']} »",
        "",
        "Erreur quadratique moyenne FP32 / INT8 de la sortie de chaque convolution, pour "
        "chaque seuil d'écrêtage c (percentile de |x|, s = c/127). **Gras** : seuil retenu.",
        "",
        "| Couche | act | " + " | ".join(f"MSE {k}" for k in keys)
        + " | c retenu | s_y final | écrêté |",
        "|---|---|" + "---|" * len(keys) + "---|---|---|",
    ]
    for r in calib["layers"]:
        cells = []
        for k in keys:
            v = f"{r['mse'][k]:.2e}"
            cells.append(f"**{v}**" if k == r["pick"] else v)
        pick = "fixe (tête)" if r["pick"] == "head" else (
            f"{r['pick']} = {r['candidates'][r['pick']]:.3g}")
        lines.append(f"| L{r['id']:02d} | {r['act']} | " + " | ".join(cells)
                     + f" | {pick} | {r['final_scale']:.4g} | "
                     f"{r['final_clip_rate'] * 100:.3f} % |")
    if calib["route_groups"]:
        groups = ", ".join("{" + ", ".join(f"L{g:02d}" for g in grp) + "}"
                           for grp in calib["route_groups"])
        lines += ["", f"Échelle commune imposée aux sources des routes : {groups} "
                  "(la plus grande des deux)."]
    if head is not None:
        lines += ["", f"Têtes linéaires : échelle fixe {head:g} (voir le docstring de "
                  "`yolo.quant.calibrate`) ; la "
                  "colonne « écrêté » donne la part des sorties hors de ±127·s."]
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov2-voc", help="réseau pré-entraîné, ou chemin d'un .cfg (élagué, T10.11 ; --weights requis)")
    ap.add_argument("--weights", type=Path, default=None)
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--images", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--choice", choices=CHOICES, default="mse")
    ap.add_argument("--head-scale", default=str(HEAD_SCALE),
                    help="échelle des têtes linéaires, ou « calib »")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--markdown", type=Path, default=None)
    args = ap.parse_args()

    head = None if args.head_scale == "calib" else float(args.head_scale)
    fused = load_fused(args.net, args.weights)
    paths = calib_images(args.devkit, args.images, args.seed)
    stats = collect(fused, paths, workers=args.workers)
    scales, rows, groups = choose_scales(fused, stats, args.choice, head)
    calib = calib_dict(fused, scales, rows, groups, stats.images, args.choice, head)
    out = args.out or ROOT / "build" / "quant" / net_tag(args.net) / "calib.json"
    save_calib(out, calib)
    text = markdown(calib)
    print(text)
    print(f"échelles : {out}")
    md = args.markdown or ROOT / "results" / f"calibration_{net_tag(args.net)}.md"
    md.write_text(f"# Calibration INT8 — {args.net} (T4.2)\n\n" + text)
    print(f"rapport : {md}")


if __name__ == "__main__":
    main()
