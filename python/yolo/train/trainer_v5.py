"""YOLOv5 / YOLOv7 / YOLOv8 training loop (T33.7, T34.4, T35.6): the recipe of the reference
`train.py`.

    for each batch:  L, δ = yolo_v5_loss(forward(X));  grads += backward(δ)
        (YOLOv8: yolo_v8_loss, BCE + CIoU + DFL with the task-aligned assigner)
        every `accumulate` batches (1 → 64/N in the warmup, then 64/N):
            SGD Nesterov step (3 groups, lr and momentum of `V5Schedule`), EMA update

The loss is the total of the reference (already × N), its gradients are summed over the
accumulated batches, and the weight decay is scaled by N·accumulate/64 (nominal batch 64),
as the reference. The same loop trains the Ultralytics networks (`ul_net`) and the Darknet
YOLOv7-tiny graph: both are `graph.Network`, seen through `v5_levels` (`v8_levels` for the
anchor-free YOLOv8, whose loss parts are box / cls / dfl instead of box / obj / cls).

Checkpoints: those of `Trainer` (parameters, BN state, optimizer, iteration) plus the EMA
and the accumulation state; resuming reproduces the run on the same backend.
"""

import csv
import time
from pathlib import Path

import numpy as np

from yolo.backend import to_numpy
from yolo.data.targets import is_v8, v5_levels, v8_levels
from yolo.train.ema import ModelEMA
from yolo.train.loss_v5 import HYP_V5, yolo_v5_loss
from yolo.train.loss_v8 import HYP_V8, yolo_v8_loss
from yolo.train.trainer import Trainer

PARTS_V5, PARTS_V8 = ("box", "obj", "cls"), ("box", "cls", "dfl")
LOG_FIELDS_V5 = ("it", "lr", "momentum", "size", "loss", *PARTS_V5, "seconds")
NBS = 64


def head_conv(net, hid):
    """Layer id of the conv that produces head `hid` (the layer itself for `ul_net`, the
    conv before a Darknet [yolo] layer)."""
    return hid - 1 if net.layers[hid]["type"] == "yolo" else hid


def init_head_biases(net, layers=None, img_size=640):
    """Prior of the reference `_initialize_biases`: objectness + log(8 / (S/s)²), classes
    + log(0.6 / (C − 0.99)) on the head convs (all, or the ids of `layers`). YOLOv8
    (`Detect.bias_init`): box convs set to 1.0, class convs to log(5 / C / (S/s)²)."""
    c = net.net["classes"]
    if is_v8(net.net):
        for hid, stride in v8_levels(net.net):
            box, cls = net.layers[hid]["from"]
            if layers is None or box in layers:
                net.params[box]["b"][...] = 1.0
            if layers is None or cls in layers:
                net.params[cls]["b"][...] = np.log(5 / c / (img_size / stride) ** 2)
        return
    for hid, stride, anchors in v5_levels(net.net):
        i = head_conv(net, hid)
        if layers is not None and i not in layers:
            continue
        b = net.params[i]["b"].reshape(len(anchors), 5 + c)
        b[:, 4] += np.log(8 / (img_size / stride) ** 2)
        b[:, 5:] += np.log(0.6 / (c - 0.99))


class TrainerV5(Trainer):
    """Trains `net` (`graph.Network`, `is_v5`) with `optimizer` (`optim.SGDGroups`) and
    `schedule` (`schedule.V5Schedule`). `size`: square input; `hyp`: loss hyperparameters
    (`HYP_V5`, `HYP_V7_TINY`, `HYP_V8`; default: the one of the head); `ema`: keep a
    `ModelEMA`; `nbs`: nominal batch."""

    def __init__(self, net, optimizer, schedule, size=640, seed=0, log_path=None, hyp=None,
                 ema=True, nbs=NBS, batch=16):
        self.net = net
        self.optimizer = optimizer
        self.schedule = schedule
        self.size = size
        self.seed = seed
        self.it = 0
        self.v8 = is_v8(net.net)
        self.hyp = hyp if hyp is not None else (HYP_V8 if self.v8 else HYP_V5)
        self.levels = v8_levels(net.net) if self.v8 else v5_levels(net.net)
        self.parts = PARTS_V8 if self.v8 else PARTS_V5
        self.log_path = None if log_path is None else Path(log_path)
        self.grad_hook = None
        self.nbs = nbs
        self.batch = batch
        self.base_wd = optimizer.weight_decay
        self.ema = ModelEMA(net) if ema else None
        self._grads = None
        self._pending = 0
        self.steps = 0

    def step(self, images, boxes, labels):
        """One batch; an optimizer step when `accumulate` batches are summed."""
        t0 = time.perf_counter()
        n = images.shape[0]
        xp = self.array_module()
        x = images.astype(self.net.dtype, copy=False)
        outputs = self.net.forward(x if xp is np else xp.asarray(x), train=True)
        outputs = {h: to_numpy(o) for h, o in outputs.items()}
        if self.v8:
            res = yolo_v8_loss(outputs, boxes, labels, self.levels, self.net.net["classes"],
                               self.hyp)
        else:
            res = yolo_v5_loss(outputs, boxes, labels, self.levels, self.net.net["classes"],
                               self.hyp, img_size=self.size)
        douts = res.douts if xp is np else {h: xp.asarray(d) for h, d in res.douts.items()}
        _, grads = self.net.backward(douts)
        if self._grads is None:
            self._grads = grads
        else:
            for acc, g in zip(self._grads, grads):
                for k in acc:
                    acc[k] += g[k]
        self._pending += 1
        lrs, momentum = self.schedule(self.it)
        accumulate = self.schedule.accumulate(self.it, n, self.nbs)
        if self._pending >= accumulate:
            self.optimizer.weight_decay = self.base_wd * n * accumulate / self.nbs
            self.optimizer.step(self._grads, lrs, momentum)
            if self.ema is not None:
                self.ema.update(self.net)
            self._grads, self._pending = None, 0
            self.steps += 1
        self._log(lrs["weights"], momentum, n, res, time.perf_counter() - t0)
        self.it += 1
        return res, lrs["weights"]

    def _log(self, lr, momentum, n, res, seconds):
        if self.log_path is None:
            return
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(("it", "lr", "momentum", "size", "loss", *self.parts, "seconds"))
            p = res.parts
            w.writerow([self.it, f"{lr:.6g}", f"{momentum:.4g}", self.size,
                        f"{res.total / n:.6g}", *(f"{p[k] / n:.6g}" for k in self.parts),
                        f"{seconds:.3f}"])

    def size_for(self, it):
        return self.size

    # ------------------------------------------------------------- checkpoints
    def checkpoint_arrays(self):
        arrays = super().checkpoint_arrays()
        arrays.update({"steps": np.array(self.steps), "pending": np.array(self._pending)})
        if self.ema is not None:
            arrays.update({f"ema/{k}": np.asarray(to_numpy(v))
                           for k, v in self.ema.state_dict().items()})
        if self._grads is not None:
            arrays.update({f"acc/{i}/{k}": to_numpy(v)
                           for i, g in enumerate(self._grads) for k, v in g.items()})
        return arrays

    def load_checkpoint(self, path):
        super().load_checkpoint(path)  # ignores the keys below
        with np.load(path) as data:
            self.steps = int(data["steps"]) if "steps" in data.files else 0
            self._pending = int(data["pending"]) if "pending" in data.files else 0
            ema = {k[4:]: data[k] for k in data.files if k.startswith("ema/")}
            acc = {k[4:]: data[k] for k in data.files if k.startswith("acc/")}
        if self.ema is not None and ema:
            self.ema.load_state_dict(ema)
        self._grads = None
        if acc:
            xp = self.array_module()
            self._grads = [{k: xp.zeros_like(v) for k, v in p.items()} for p in self.net.params]
            for key, v in acc.items():
                i, k = key.split("/")
                self._grads[int(i)][k][...] = xp.asarray(v)
