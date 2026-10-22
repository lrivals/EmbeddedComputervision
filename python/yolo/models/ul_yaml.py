"""Network description from an Ultralytics model YAML (T33.1, T35.0).

`parse(cfg)` turns the YAML dict of a YOLOv5 (v6.0+) or YOLOv8 model (the `model.yaml` read
from a checkpoint by `yolo.io.torch_pickle`, or `YOLOV5N` / `YOLOV8N` below) into the flat
layer list run by `yolo.models.ul_graph` (T26.1 schema, see that module), BN **not** folded:
each conv carries `bn` (conv + BN + SiLU, or a plain conv with bias for the heads) and `name`,
the prefix of its parameters in the torch state dict (`model.2.cv1` → `model.2.cv1.conv.weight`,
`model.2.cv1.bn.running_mean`, ...; heads: `model.24.m.0` → `.weight`, `.bias`).

Scaling as `parse_model` of the reference repositories: repeats n' = max(round(n·gd), 1)
for n > 1, channels c' = ⌈c·gw / 8⌉·8; Conv padding k // 2 unless given.

The two YAMLs are kept as Python dicts (no pyyaml in `yolo/`, ADR 0001); they equal the
`model.yaml` of the official checkpoints (test). Ultralytics, AGPL-3.0.
"""

import math

YOLOV5N = {
    "nc": 80, "depth_multiple": 0.33, "width_multiple": 0.25,
    "anchors": [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326]],
    "backbone": [
        [-1, 1, "Conv", [64, 6, 2, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, "C3", [128]],
        [-1, 1, "Conv", [256, 3, 2]], [-1, 6, "C3", [256]], [-1, 1, "Conv", [512, 3, 2]],
        [-1, 9, "C3", [512]], [-1, 1, "Conv", [1024, 3, 2]], [-1, 3, "C3", [1024]],
        [-1, 1, "SPPF", [1024, 5]],
    ],
    "head": [
        [-1, 1, "Conv", [512, 1, 1]], [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 6], 1, "Concat", [1]], [-1, 3, "C3", [512, False]],
        [-1, 1, "Conv", [256, 1, 1]], [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]], [-1, 3, "C3", [256, False]],
        [-1, 1, "Conv", [256, 3, 2]], [[-1, 14], 1, "Concat", [1]],
        [-1, 3, "C3", [512, False]], [-1, 1, "Conv", [512, 3, 2]],
        [[-1, 10], 1, "Concat", [1]], [-1, 3, "C3", [1024, False]],
        [[17, 20, 23], 1, "Detect", ["nc", "anchors"]],
    ],
    "ch": 3,
}

YOLOV8N = {
    "nc": 80, "depth_multiple": 0.33, "width_multiple": 0.25,
    "backbone": [
        [-1, 1, "Conv", [64, 3, 2]], [-1, 1, "Conv", [128, 3, 2]], [-1, 3, "C2f", [128, True]],
        [-1, 1, "Conv", [256, 3, 2]], [-1, 6, "C2f", [256, True]], [-1, 1, "Conv", [512, 3, 2]],
        [-1, 6, "C2f", [512, True]], [-1, 1, "Conv", [1024, 3, 2]],
        [-1, 3, "C2f", [1024, True]], [-1, 1, "SPPF", [1024, 5]],
    ],
    "head": [
        [-1, 1, "nn.Upsample", ["None", 2, "nearest"]], [[-1, 6], 1, "Concat", [1]],
        [-1, 3, "C2f", [512]], [-1, 1, "nn.Upsample", ["None", 2, "nearest"]],
        [[-1, 4], 1, "Concat", [1]], [-1, 3, "C2f", [256]],
        [-1, 1, "Conv", [256, 3, 2]], [[-1, 12], 1, "Concat", [1]], [-1, 3, "C2f", [512]],
        [-1, 1, "Conv", [512, 3, 2]], [[-1, 9], 1, "Concat", [1]], [-1, 3, "C2f", [1024]],
        [[15, 18, 21], 1, "Detect", ["nc"]],
    ],
    "ch": 3,
}

CONFIGS = {"yolov5n": YOLOV5N, "yolov8n": YOLOV8N}
REG_MAX = 16  # DFL bins per box side (YOLOv8)


def make_divisible(x, divisor=8):
    return math.ceil(x / divisor) * divisor


def scaled_repeats(n, gd):
    return max(round(n * gd), 1) if n > 1 else n


class _Emitter:
    """Flat layer list; each method returns the index of the layer it emits last."""

    def __init__(self):
        self.layers = []

    def _add(self, layer):
        self.layers.append(layer)
        return len(self.layers) - 1

    def conv(self, src, cin, cout, k, s, name, p=None, bn=True):
        return self._add({"type": "conv", "from": src, "k": k, "s": s,
                          "p": k // 2 if p is None else p, "cin": cin, "cout": cout,
                          "act": "silu" if bn else "linear", "bn": bn, "name": name})

    def bottleneck(self, src, c, shortcut, k, name):
        y = self.conv(self.conv(src, c, c, k[0], 1, f"{name}.cv1"), c, c, k[1], 1, f"{name}.cv2")
        return self._add({"type": "add", "from": [src, y]}) if shortcut else y

    def c3(self, src, c1, c2, n, shortcut, name):
        ch = c2 // 2
        a = self.conv(src, c1, ch, 1, 1, f"{name}.cv1")
        for j in range(n):
            a = self.bottleneck(a, ch, shortcut, (1, 3), f"{name}.m.{j}")
        b = self.conv(src, c1, ch, 1, 1, f"{name}.cv2")
        return self.conv(self._add({"type": "concat", "from": [a, b]}), 2 * ch, c2, 1, 1,
                         f"{name}.cv3")

    def c2f(self, src, c1, c2, n, shortcut, name):
        c = c2 // 2
        y = self.conv(src, c1, 2 * c, 1, 1, f"{name}.cv1")
        parts = [self._add({"type": "slice", "from": y, "start": 0, "stop": c}),
                 self._add({"type": "slice", "from": y, "start": c, "stop": 2 * c})]
        for j in range(n):
            parts.append(self.bottleneck(parts[-1], c, shortcut, (3, 3), f"{name}.m.{j}"))
        return self.conv(self._add({"type": "concat", "from": parts}), (2 + n) * c, c2, 1, 1,
                         f"{name}.cv2")

    def sppf(self, src, c1, c2, k, name):
        ys = [self.conv(src, c1, c1 // 2, 1, 1, f"{name}.cv1")]
        for _ in range(3):
            ys.append(self._add({"type": "maxpool", "from": ys[-1], "k": k, "s": 1}))
        return self.conv(self._add({"type": "concat", "from": ys}), 4 * (c1 // 2), c2, 1, 1,
                         f"{name}.cv2")

    def detect_v8(self, srcs, chs, nc, name):
        c2, c3 = max(16, chs[0] // 4, REG_MAX * 4), max(chs[0], min(nc, 100))
        outs = []
        for lv, (s, c) in enumerate(zip(srcs, chs)):
            branches = []
            for br, (cm, cout) in (("cv2", (c2, 4 * REG_MAX)), ("cv3", (c3, nc))):
                y = self.conv(s, c, cm, 3, 1, f"{name}.{br}.{lv}.0")
                y = self.conv(y, cm, cm, 3, 1, f"{name}.{br}.{lv}.1")
                branches.append(self.conv(y, cm, cout, 1, 1, f"{name}.{br}.{lv}.2", bn=False))
            outs.append(self._add({"type": "concat", "from": branches}))
        return outs


def parse(cfg, nc=None, name=None, size=640):
    """`ul_graph` description (BN not folded) of the model YAML dict `cfg`.

    `nc`: number of classes (default: the YAML's). Fields: `name`, `classes`, `outputs`,
    `layers`, `head` ("anchor" | "dfl"), `anchors` (pixels per level) or `reg_max`,
    `input` (3, size, size), `stride`.
    """
    nc = cfg["nc"] if nc is None else nc
    gd, gw = cfg["depth_multiple"], cfg["width_multiple"]
    e = _Emitter()
    idx, ch = [], []  # output layer index and channels of each `model.<i>`
    rows = cfg["backbone"] + cfg["head"]
    for i, (f, n, m, args) in enumerate(rows):
        mname = f"model.{i}"
        if isinstance(f, list):
            src = [idx[-1] if j == -1 else idx[j] for j in f]
            cins = [ch[j] for j in f]
        else:
            src = (idx[-1] if idx else -1) if f == -1 else idx[f]
            cins = [ch[f] if idx else cfg.get("ch", 3)]
        n = scaled_repeats(n, gd)
        if m == "Detect":
            if "anchors" in cfg:
                no = (nc + 5) * (len(cfg["anchors"][0]) // 2)
                outputs = [e.conv(s, c, no, 1, 1, f"{mname}.m.{lv}", bn=False)
                           for lv, (s, c) in enumerate(zip(src, cins))]
            else:
                outputs = e.detect_v8(src, cins, nc, mname)
            break
        if m in ("Conv", "C3", "C2f", "SPPF"):
            c1, c2 = cins[0], make_divisible(args[0] * gw)
            if m == "Conv":
                k, s = (args[1:3] + [1, 1])[:2]
                p = args[3] if len(args) > 3 else None
                out = e.conv(src, c1, c2, k, s, mname, p=p)
            elif m == "C3":
                out = e.c3(src, c1, c2, n, args[1] if len(args) > 1 else True, mname)
            elif m == "C2f":
                out = e.c2f(src, c1, c2, n, args[1] if len(args) > 1 else False, mname)
            else:
                out = e.sppf(src, c1, c2, args[1] if len(args) > 1 else 5, mname)
        elif m == "nn.Upsample":
            c2 = cins[0]
            out = e._add({"type": "upsample", "from": src, "s": int(args[1])})
        elif m == "Concat":
            c2 = sum(cins)
            out = e._add({"type": "concat", "from": src})
        else:
            raise ValueError(f"model.{i}: unsupported module {m!r}")
        idx.append(out)
        ch.append(c2)
    desc = {"name": name or "ultralytics", "classes": nc, "outputs": outputs,
            "layers": e.layers, "head": "anchor" if "anchors" in cfg else "dfl",
            "input": [cfg.get("ch", 3), size, size], "stride": [8, 16, 32]}
    if "anchors" in cfg:
        desc["anchors"] = [[float(a) for a in lv] for lv in cfg["anchors"]]
    else:
        desc["reg_max"] = REG_MAX
    return desc


def num_params(desc, folded=False):
    """Trainable parameters: conv weights, plus γ, β per BN channel or a bias per channel;
    `folded`: one bias per channel everywhere (BN folded, the published count)."""
    total = 0
    for layer in desc["layers"]:
        if layer["type"] == "conv":
            total += layer["k"] ** 2 * layer["cin"] * layer["cout"]
            total += 2 * layer["cout"] if layer["bn"] and not folded else layer["cout"]
    return total
