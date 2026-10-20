"""Float forward of the networks converted from PyTorch (YOLOv5n, YOLOv8n; T26.1).

`tools/convert_ultralytics.py` writes, once and offline, `<net>.npz` (BN folded into the
convs) and `<net>.json`: a flat list of layers, each reading earlier layers by index. This
module runs that list with the NumPy layers of `yolo.layers` (ADR 0001: no torch here).

Layer types (field `from`: index or list of indices of earlier layers; −1 = the previous one;
the network input is index −1 of the first layer):

- `conv`: k, s, p, cin, cout, act ("silu" | "linear"), weights `w<i>` / `b<i>` in the npz;
- `add`: element-wise sum of two layers (bottleneck shortcut);
- `concat`: channel concatenation;
- `slice`: channels [start, stop) of a layer (C2f split);
- `maxpool`: k, s (stride 1, same size; SPPF);
- `upsample`: nearest ×s.

`outputs`: the indices of the raw head maps (YOLOv5: the 1×1 head convs, A·(5 + C)
channels; YOLOv8: box (4·16 DFL bins) and class maps concatenated, 64 + C channels).
"""

import json
from pathlib import Path

import numpy as np

from yolo.layers.activations import silu_forward
from yolo.layers.conv import conv_forward
from yolo.layers.pool import maxpool_forward
from yolo.layers.upsample import upsample_forward


def load(path):
    """(description, weights) of a converted network; `path` without extension."""
    p = Path(path)
    desc = json.loads(p.with_suffix(".json").read_text())
    w = dict(np.load(p.with_suffix(".npz")))
    return desc, w


def _src(i, f):
    return i - 1 if f == -1 else f


def forward(desc, weights, x, all_outputs=False):
    """Raw head maps (list, in `desc["outputs"]` order) of the float input x (N, 3, H, W)."""
    outs = []
    x = np.asarray(x, np.float64)

    def get(j):
        return x if j < 0 else outs[j]

    for i, layer in enumerate(desc["layers"]):
        t = layer["type"]
        f = layer.get("from", -1)
        if t == "conv":
            y, _ = conv_forward(get(_src(i, f)), weights[f"w{i}"].astype(np.float64),
                                weights[f"b{i}"].astype(np.float64), s=layer["s"], pad=layer["p"])
            if layer["act"] == "silu":
                y, _ = silu_forward(y)
        elif t == "add":
            a, b = f
            y = get(_src(i, a)) + get(_src(i, b))
        elif t == "concat":
            y = np.concatenate([get(_src(i, j)) for j in f], axis=1)
        elif t == "slice":
            y = get(_src(i, f))[:, layer["start"]:layer["stop"]]
        elif t == "maxpool":
            y, _ = maxpool_forward(get(_src(i, f)), k=layer["k"], s=layer["s"])
        elif t == "upsample":
            y, _ = upsample_forward(get(_src(i, f)), s=layer["s"])
        else:
            raise ValueError(f"layer {i}: unknown type {t!r}")
        outs.append(y)
    if all_outputs:
        return outs
    return [outs[j] for j in desc["outputs"]]
