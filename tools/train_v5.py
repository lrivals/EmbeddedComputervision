"""Training / fine-tuning of YOLOv5n or YOLOv7-tiny in NumPy with the YOLOv5 recipe
(T33.7, T34.4).

    # YOLOv5n on VOC from its COCO weights (heads reinitialized), 640, 30 epochs
    python tools/train_v5.py --net yolov5n-voc --init coco --epochs 30 \\
        --out build/train/yolov5n-voc
    # YOLOv7-tiny on VOC from its Darknet COCO weights, OTA loss (hyp.scratch.tiny.yaml)
    python tools/train_v5.py --net tiny-yolov7-voc --init coco --epochs 30 \\
        --out build/train/tiny-yolov7-voc
    # resume
    python tools/train_v5.py --net tiny-yolov7-voc --out build/train/tiny-yolov7-voc --resume

`--net`: an Ultralytics name `<arch>-<dataset>` (`yolov5n-voc`, `ul_net.description`) or a
YOLOv7-tiny one, `tiny-yolov7-<dataset>`, whose cfg is made from `yolov7-tiny.cfg` with the
classes of the dataset (`tools/make_cfg.py`, written to `<out>/net.cfg`). `--init coco`
copies the COCO weights except the layers of a different shape (the heads), whose biases
get the prior of the reference (`init_head_biases`).

Outputs in `--out`: `checkpoint.npz` (exact resume), `loss.csv`, and the EMA weights:
`final.npz` (torch-named state dict, read by `tools/eval_voc.py --net yolov5n-voc
--weights`) or `final.weights` (Darknet, with `--net <out>/net.cfg`).
"""

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from yolo import backend  # noqa: E402
from yolo.data import datasets  # noqa: E402
from yolo.data.loader import DataLoader, V5Dataset  # noqa: E402
from yolo.io.darknet_weights import load_darknet_weights, save_darknet_weights  # noqa: E402
from yolo.models import ul_net  # noqa: E402
from yolo.models.tiny_yolo import CFG_DIR, CFG_FILES, PRETRAINED, build  # noqa: E402
from yolo.train.loss_v5 import HYP_V5, HYP_V7_TINY  # noqa: E402
from yolo.train.optim import SGDGroups  # noqa: E402
from yolo.train.schedule import V5Schedule  # noqa: E402
from yolo.train.trainer import copy_matching  # noqa: E402
from yolo.train.trainer_v5 import TrainerV5, init_head_biases  # noqa: E402

V7 = "tiny-yolov7"


def build_net(name, out, dtype, seed):
    """(network, Darknet cfg path or None, COCO source builder)."""
    if name.startswith(V7 + "-"):
        from tools.make_cfg import make_cfg

        dataset = name[len(V7) + 1:]
        if dataset == "coco":
            cfg = CFG_DIR / CFG_FILES["tiny-yolov7-coco"]
        else:
            cfg = out / "net.cfg"
            n = len(datasets.DATASETS[dataset].classes)
            cfg.parent.mkdir(parents=True, exist_ok=True)
            cfg.write_text(f"# tools/train_v5.py: yolov7-tiny.cfg with {dataset} ({n} classes)\n"
                           + make_cfg((CFG_DIR / CFG_FILES["tiny-yolov7-coco"]).read_text(), n))

        def coco():
            src = build("tiny-yolov7-coco", dtype=dtype)
            load_darknet_weights(src, ROOT / "weights" / PRETRAINED["tiny-yolov7-coco"])
            return src

        return build(str(cfg), dtype=dtype, rng=seed), cfg, coco
    arch = name.partition("-")[0]
    net = ul_net.build(ul_net.description(name), dtype=dtype, rng=seed)
    return net, None, lambda: ul_net.build(f"{arch}-coco", pretrained=True, dtype=dtype)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", default="yolov5n-voc")
    ap.add_argument("--init", default="coco", help="coco | he")
    ap.add_argument("--dataset", choices=sorted(datasets.DATASETS), default=None,
                    help="default: the suffix of --net")
    ap.add_argument("--data-root", type=Path, default=None)
    ap.add_argument("--splits", default=None)
    ap.add_argument("--subset", type=int, default=0, help="first n images only")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--iters", type=int, default=0, help="stop after n batches (0: all epochs)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=0.01)
    ap.add_argument("--lrf", type=float, default=0.01)
    ap.add_argument("--cos", action="store_true", help="one-cycle cosine instead of linear")
    ap.add_argument("--warmup-epochs", type=float, default=3.0)
    ap.add_argument("--hyp", choices=("v5", "v7-tiny"), default=None,
                    help="loss hyperparameters; default: v7-tiny (OTA) for tiny-yolov7")
    ap.add_argument("--no-ota", action="store_true", help="v7-tiny without OTA")
    ap.add_argument("--no-mosaic", action="store_true")
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--out", type=Path, default=ROOT / "build" / "train" / "v5")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    args = ap.parse_args()
    if args.device == "gpu":
        try:
            backend.use("gpu")
        except RuntimeError as exc:
            sys.exit(str(exc))
    dataset = args.dataset or args.net.partition("-")[2].split("-")[-1]

    dtype = np.float32
    args.out.mkdir(parents=True, exist_ok=True)
    net, cfg, coco = build_net(args.net, args.out, dtype, args.seed)
    v7 = cfg is not None
    if not args.resume and args.init == "coco":
        skipped = copy_matching(coco(), net)
        init_head_biases(net, layers=skipped, img_size=args.size)
        print(f"COCO weights copied; reinitialized layers: {skipped}")
    elif not args.resume:
        init_head_biases(net, img_size=args.size)
    if args.device == "gpu":
        net.to_device()

    samples = datasets.load(dataset, args.data_root, args.splits, role="train")
    samples = datasets.remap(samples, datasets.class_lut(dataset, net.net["classes"]))
    if args.subset:
        samples = samples[:args.subset]
    loader = DataLoader(V5Dataset(samples, hyp={"mosaic": 0.0} if args.no_mosaic else None),
                        args.batch, workers=args.workers, seed=args.seed)
    print(f"{len(samples)} images, {len(loader)} batches per epoch")

    hyp = HYP_V7_TINY if (args.hyp or ("v7-tiny" if v7 else "v5")) == "v7-tiny" else HYP_V5
    if args.no_ota:
        hyp = {**hyp, "ota": False}
    optimizer = SGDGroups(net.params, momentum=0.937, weight_decay=5e-4)
    schedule = V5Schedule(args.lr0, args.epochs, len(loader), lrf=args.lrf, cos=args.cos,
                          warmup_epochs=args.warmup_epochs)
    trainer = TrainerV5(net, optimizer, schedule, size=args.size, seed=args.seed,
                        log_path=args.out / "loss.csv", hyp=hyp, ema=not args.no_ema,
                        batch=args.batch)
    ckpt = args.out / "checkpoint.npz"
    if args.resume:
        trainer.load_checkpoint(ckpt)
        print(f"resumed at iteration {trainer.it}")

    def report(tr, res, lr):
        n = args.batch
        p = {k: v / n for k, v in res.parts.items()}
        print(f"it {tr.it:6d}  lr {lr:.2e}  loss {res.total / n:8.4f}  box {p['box']:.4f} "
              f"obj {p['obj']:.4f} cls {p['cls']:.4f}", flush=True)

    iters = args.iters or args.epochs * len(loader)
    try:
        trainer.fit(loader, iters, checkpoint=ckpt, save_every=args.save_every, callback=report)
    finally:
        loader.close()
    final = trainer.ema.swapped(net) if trainer.ema is not None else _same(net)
    with final as n:
        n = n.to_numpy() if hasattr(n, "to_numpy") else n
        if v7:
            save_darknet_weights(n, args.out / "final.weights")
            print(f"weights: {args.out / 'final.weights'} (--net {cfg})")
        else:
            np.savez(args.out / "final.npz", **{k: backend.to_numpy(v)
                                                for k, v in ul_net.state_dict(n).items()})
            print(f"weights: {args.out / 'final.npz'} (--net {args.net})")


class _same:
    def __init__(self, net):
        self.net = net

    def __enter__(self):
        return self.net

    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
