"""Training / fine-tuning of YOLOv5n, YOLOv7-tiny or YOLOv8n in NumPy with the YOLOv5 recipe
(T33.7, T34.4) and its YOLOv8 settings (T35.6).

    # YOLOv5n on VOC from its COCO weights (heads reinitialized), 640, 30 epochs
    python tools/train_v5.py --net yolov5n-voc --init coco --epochs 30 \\
        --out build/train/yolov5n-voc
    # YOLOv7-tiny on VOC from its Darknet COCO weights, OTA loss (hyp.scratch.tiny.yaml)
    python tools/train_v5.py --net tiny-yolov7-voc --init coco --epochs 30 \\
        --out build/train/tiny-yolov7-voc
    # YOLOv8n on VOC from its COCO weights: v8 loss (TAL, BCE + CIoU + DFL), optimizer=auto,
    # mosaic off for the last 10 epochs
    python tools/train_v5.py --net yolov8n-voc --init coco --epochs 30 \\
        --out build/train/yolov8n-voc
    # resume
    python tools/train_v5.py --net tiny-yolov7-voc --out build/train/tiny-yolov7-voc --resume

`--net`: an Ultralytics name `<arch>-<dataset>` (`yolov5n-voc`, `yolov8n-voc`,
`ul_net.description`) or a
YOLOv7-tiny one, `tiny-yolov7-<dataset>`, whose cfg is made from `yolov7-tiny.cfg` with the
classes of the dataset (`tools/make_cfg.py`, written to `<out>/net.cfg`). `--init coco`
copies the COCO weights except the layers of a different shape (the heads), whose biases
get the prior of the reference (`init_head_biases`).

YOLOv8 settings (`--net yolov8n-*`): `--optimizer auto` by default, the rule of the reference
(`optim.auto_optimizer`: AdamW at lr0 0.002·5/(4 + C) for at most 10,000 optimizer steps,
else SGD at 0.01; no bias warmup), printed and written to `<out>/optimizer.txt`; mosaic
off for the last `--close-mosaic 10` epochs (a second loader from that iteration on).

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
from yolo.train.optim import AdamWGroups, SGDGroups, auto_optimizer  # noqa: E402
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
    ap.add_argument("--optimizer", choices=("sgd", "adamw", "auto"), default=None,
                    help="default: auto for yolov8n (T35.6), sgd otherwise")
    ap.add_argument("--close-mosaic", type=int, default=None,
                    help="last epochs without mosaic; default: 10 for yolov8n, 0 otherwise")
    ap.add_argument("--hyp", choices=("v5", "v7-tiny"), default=None,
                    help="loss hyperparameters; default: v7-tiny (OTA) for tiny-yolov7, "
                         "the v8 loss for yolov8n")
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
    v8 = net.net.get("head") == "dfl"
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
    def make_loader(mosaic):
        return DataLoader(V5Dataset(samples, hyp=None if mosaic else {"mosaic": 0.0}),
                          args.batch, workers=args.workers, seed=args.seed)

    loader = make_loader(not args.no_mosaic)
    print(f"{len(samples)} images, {len(loader)} batches per epoch")

    if v8 and args.hyp is None:
        hyp = None  # HYP_V8, set by TrainerV5 from the head
    else:
        hyp = HYP_V7_TINY if (args.hyp or ("v7-tiny" if v7 else "v5")) == "v7-tiny" else HYP_V5
        if args.no_ota:
            hyp = {**hyp, "ota": False}
    name = args.optimizer or ("auto" if v8 else "sgd")
    lr0, momentum, warmup_bias_lr = args.lr0, 0.937, 0.1
    if name == "auto":
        name, lr0, momentum = auto_optimizer(net.net["classes"], len(samples), args.epochs,
                                             args.batch)
        warmup_bias_lr = 0.0
    if name == "adamw":
        momentum = 0.9  # β1 of the reference AdamW
        optimizer = AdamWGroups(net.params, momentum=momentum, weight_decay=5e-4)
    else:
        optimizer = SGDGroups(net.params, momentum=momentum, weight_decay=5e-4)
    used = f"optimizer {name}, lr0 {lr0:g}, momentum {momentum:g}, warmup_bias_lr {warmup_bias_lr:g}"
    print(used)
    (args.out / "optimizer.txt").write_text(used + "\n")
    schedule = V5Schedule(lr0, args.epochs, len(loader), lrf=args.lrf, cos=args.cos,
                          warmup_epochs=args.warmup_epochs, warmup_bias_lr=warmup_bias_lr)
    trainer = TrainerV5(net, optimizer, schedule, size=args.size, seed=args.seed,
                        log_path=args.out / "loss.csv", hyp=hyp, ema=not args.no_ema,
                        batch=args.batch)
    ckpt = args.out / "checkpoint.npz"
    if args.resume:
        trainer.load_checkpoint(ckpt)
        print(f"resumed at iteration {trainer.it}")

    def report(tr, res, lr):
        n = args.batch
        parts = " ".join(f"{k} {res.parts[k] / n:.4f}" for k in tr.parts)
        print(f"it {tr.it:6d}  lr {lr:.2e}  loss {res.total / n:8.4f}  {parts}", flush=True)

    iters = args.iters or args.epochs * len(loader)
    close = args.close_mosaic if args.close_mosaic is not None else (10 if v8 else 0)
    phases = [(loader, iters)]
    if close and not args.no_mosaic:
        last = min(iters, max(args.epochs - close, 0) * len(loader))
        phases = [(loader, last), (None, iters)]
    try:
        for i, (ld, until) in enumerate(phases):
            if ld is None:
                loader.close()
                loader = ld = make_loader(False)
                print(f"mosaic off from iteration {trainer.it} (last {close} epochs)")
            if trainer.it < until:
                trainer.fit(ld, until, checkpoint=ckpt, save_every=args.save_every,
                            callback=report)
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
