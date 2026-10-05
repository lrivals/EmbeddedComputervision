"""Entraînement / affinage d'un Tiny-YOLO sur VOC en NumPy (T2.7, T2.9, §7.1).

    # affinage de Tiny-YOLOv3 VOC depuis les poids COCO (têtes réinitialisées)
    python tools/train.py --net tiny-yolov3-voc --init coco --iters 20000 --out build/train/v3
    # reprise
    python tools/train.py --out build/train/v3 --resume --iters 30000

    # QAT 4 bits de Tiny-YOLOv2 VOC depuis les poids Darknet (T9.3.3) ; pas initiaux de
    # tools/quant_lowbit.py --scheme w4a4
    python tools/train.py --net tiny-yolov2-voc --init weights/yolov2-tiny-voc.weights \
        --qat w4a4 --qat-steps build/m9/models/tiny-yolov2-voc-w4a4-ptq/steps.json \
        --lr 1e-4 --burn-in 100 --batch 8 --iters 4000 --out build/train/qat-w4a4

    # même entraînement sur le GPU (CuPy, T12.11) ; le CPU reste le défaut et la référence
    python tools/train.py … --device gpu

Sorties dans `--out` : `checkpoint.npz` (reprise exacte), `loss.csv` (une ligne par
itération), `final.weights` (format Darknet ; pas en QAT ni en ADMM : réseau à BN fusionnée,
exporté par tools/quant_lowbit.py --checkpoint). Les checkpoints sont en NumPy quel que
soit le backend ; la reprise exacte n'est garantie que sur le même backend.
"""

import json

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo import backend  # noqa: E402
from yolo.data.loader import DataLoader, VOCDataset  # noqa: E402
from yolo.data.voc import load_split  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights, save_darknet_weights  # noqa: E402
from yolo.models.tiny_yolo import build  # noqa: E402
from yolo.quant.fuse_bn import fuse_network  # noqa: E402
from yolo.train.optim import NO_DECAY, SGD  # noqa: E402
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
    ap.add_argument("--qat", default="", help="schéma wXaY : QAT sur le réseau fusionné (T9.3)")
    ap.add_argument("--qat-steps", type=Path, default=None,
                    help="steps.json de tools/quant_lowbit.py (pas initiaux)")
    ap.add_argument("--admm", type=Path, default=None,
                    help="plan de poids JSON {id conv: niveaux} (pow2.py ; {} : mixed6 partout) : "
                         "ADMM sur le réseau fusionné (T9.2)")
    ap.add_argument("--admm-rho", type=float, default=1e-3)
    ap.add_argument("--admm-every", type=int, default=100,
                    help="itérations entre deux pas Z / U de l'ADMM")
    ap.add_argument("--admm-growth", type=float, default=1.3,
                    help="facteur de ρ à chaque pas Z / U (plafond 1)")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu",
                    help="backend de l'entraînement : NumPy (défaut) ou CuPy (T12.11)")
    args = ap.parse_args()
    if args.device == "gpu":
        try:
            backend.use("gpu")
        except RuntimeError as exc:
            sys.exit(str(exc))

    dtype = np.float32
    net = build(args.net, dtype=dtype, rng=args.seed)
    grad_hook = None
    no_decay = NO_DECAY
    if args.qat or args.admm:
        # Réseau à BN fusionnée (§9.1) : initialisé ici, écrasé par le checkpoint en reprise.
        from yolo.quant.lowbit import qat_from_steps

        init_weights(net, args.net, args.init, dtype)
        fused = fuse_network(net, dtype=dtype)
        scheme = args.qat or "w8a8"
        steps = json.loads(args.qat_steps.read_text())["log2_steps"] if args.qat_steps else {}
        if not steps and args.qat:
            ap.error("--qat demande --qat-steps")
        net = qat_from_steps(fused, scheme, steps) if args.qat else fused
        no_decay = NO_DECAY + ("log2_s",)
    elif not args.resume:
        init_weights(net, args.net, args.init, dtype)
    if args.device == "gpu":
        net.to_device()  # avant l'ADMM et le SGD : Z, U et vitesses suivent les paramètres
    if args.admm:
        from yolo.train.admm import ADMM

        from yolo.quant.pow2 import load_plan

        admm = ADMM(net, load_plan(args.admm, net.net), args.admm_rho, args.admm_every,
                    growth=args.admm_growth, log_path=args.out / "admm.csv")
        grad_hook = admm.hook
    steps = [int(s) for s in args.steps.split(",") if s]
    scales = [float(s) for s in args.scales.split(",") if s]
    trainer = Trainer(net, SGD(net.params, no_decay=no_decay),
                      StepSchedule(args.lr, args.burn_in, steps=steps, scales=scales),
                      size=None if args.multiscale else args.size, seed=args.seed,
                      log_path=args.out / "loss.csv", grad_hook=grad_hook,
                      ignore_thresh=args.ignore_thresh)
    ckpt = args.out / "checkpoint.npz"
    if args.resume:
        trainer.load_checkpoint(ckpt)
        if args.admm:
            admm.load(args.out / "admm.npz")
        print(f"reprise à l'itération {trainer.it}")
    print(f"backend : {args.device}")

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
        if args.admm and args.save_every and tr.it % args.save_every == 0:
            admm.save(args.out / "admm.npz")

    args.out.mkdir(parents=True, exist_ok=True)
    try:
        trainer.fit(loader, args.iters, checkpoint=ckpt, save_every=args.save_every,
                    callback=report)
    finally:
        loader.close()
    if args.admm:
        admm.save(args.out / "admm.npz")
    if args.qat or args.admm:
        print(f"checkpoint : {ckpt} (export : tools/quant_lowbit.py --checkpoint)")
        return
    save_darknet_weights(net.to_numpy(), args.out / "final.weights")
    print(f"poids : {args.out / 'final.weights'}")


if __name__ == "__main__":
    main()
