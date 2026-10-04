"""Entraînement / affinage d'un Tiny-YOLO sur VOC en NumPy (T2.7, T2.9, §7.1).

    # affinage de Tiny-YOLOv3 VOC depuis les poids COCO (têtes réinitialisées)
    python tools/train.py --net tiny-yolov3-voc --init coco --iters 20000 --out build/train/v3
    # reprise
    python tools/train.py --out build/train/v3 --resume --iters 30000

Sorties dans `--out` : `checkpoint.npz` (reprise exacte), `loss.csv` (une ligne par
itération), `final.weights` (format Darknet).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.data.loader import DataLoader, VOCDataset  # noqa: E402
from yolo.data.voc import load_split  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights, save_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import build  # noqa: E402
from yolo.train.optim import SGD  # noqa: E402
from yolo.train.schedule import StepSchedule  # noqa: E402
from yolo.train.trainer import Trainer, copy_matching  # noqa: E402

COCO_WEIGHTS = {"tiny-yolov3-voc": ("tiny-yolov3-coco", ROOT / "weights" / "yolov3-tiny.weights")}


def init_weights(net, name, init, dtype):
    if init == "he":
        return
    if init == "coco":
        src_name, path = COCO_WEIGHTS[name]
        src = build(src_name, dtype=dtype)
        load_darknet_weights(src, path)
        skipped = copy_matching(src, net)
        print(f"poids COCO {path.name} copiés ; couches réinitialisées : {skipped}")
        return
    load_darknet_weights(net, init)
    print(f"poids {init} chargés")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="tiny-yolov3-voc")
    ap.add_argument("--init", default="he", help="he | coco | chemin d'un .weights Darknet")
    ap.add_argument("--devkit", type=Path, default=ROOT / "data" / "VOCdevkit")
    ap.add_argument("--splits", default="2007:trainval,2012:trainval")
    ap.add_argument("--subset", type=int, default=0, help="n premières images seulement")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--burn-in", type=int, default=1000)
    ap.add_argument("--steps", default="", help="ex. 16000,18000")
    ap.add_argument("--scales", default="", help="ex. 0.1,0.1")
    ap.add_argument("--size", type=int, default=416)
    ap.add_argument("--multiscale", action="store_true", help="§2.2 : 320-608 tous les 10 lots")
    ap.add_argument("--ignore-thresh", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "train")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    dtype = np.float32
    net = build(args.net, dtype=dtype, rng=args.seed)
    steps = [int(s) for s in args.steps.split(",") if s]
    scales = [float(s) for s in args.scales.split(",") if s]
    trainer = Trainer(net, SGD(net.params), StepSchedule(args.lr, args.burn_in, steps=steps,
                                                         scales=scales),
                      size=None if args.multiscale else args.size, seed=args.seed,
                      log_path=args.out / "loss.csv", ignore_thresh=args.ignore_thresh)
    ckpt = args.out / "checkpoint.npz"
    if args.resume:
        trainer.load_checkpoint(ckpt)
        print(f"reprise à l'itération {trainer.it}")
    else:
        init_weights(net, args.net, args.init, dtype)

    samples = []
    for item in args.splits.split(","):
        year, split = item.split(":")
        samples += load_split(args.devkit, int(year), split)
    if args.subset:
        samples = samples[:args.subset]
    loader = DataLoader(VOCDataset(samples), args.batch, workers=args.workers, seed=args.seed)
    print(f"{len(samples)} images, {len(loader)} lots par époque")

    def report(tr, res, lr):
        n = args.batch
        p = {k: v / n for k, v in res.parts.items()}
        print(f"it {tr.it:6d}  lr {lr:.2e}  perte {res.total / n:8.3f}  "
              f"coord {p['coord']:.3f} obj {p['obj']:.3f} noobj {p['noobj']:.3f} "
              f"cls {p['cls']:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        trainer.fit(loader, args.iters, checkpoint=ckpt, save_every=args.save_every,
                    callback=report)
    finally:
        loader.close()
    save_darknet_weights(net, args.out / "final.weights")
    print(f"poids : {args.out / 'final.weights'}")


if __name__ == "__main__":
    main()
