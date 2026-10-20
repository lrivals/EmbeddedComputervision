"""Offline converter of YOLOv5n / YOLOv8n PyTorch weights to NumPy (T26.1).

    python tools/convert_ultralytics.py --net yolov8n --weights yolov8n.pt --out weights/
    python tools/convert_ultralytics.py --net yolov5n --random --out build/m26/   # test

torch is imported **only here**, once per network (ADR 0001: `python/yolo/` stays NumPy
only); extra `convert` of python/pyproject.toml. Output: `<out>/<net>.npz` (BN folded into
the convs, float32) and `<out>/<net>.json` (flat layer list, run by
`python/yolo/models/ul_graph.py`).

The torch modules below rebuild the two architectures with the **parameter names of the
reference implementations**, so that their state dict loads with `strict=True`:

- YOLOv5n (ultralytics/yolov5 v6+, `yolov5n.yaml`: depth 0.33, width 0.25; anchor-based
  `Detect`, 3 heads of 3 anchors);
- YOLOv8n (ultralytics 8.x, `yolov8n.yaml`, anchor-free `Detect` with DFL, reg_max 16).

`--weights` takes either a state dict saved with `torch.save(model.state_dict())`, or an
official checkpoint (`{"model": DetectionModel}`), which needs the `ultralytics` (v8) or the
yolov5 repository (v5) importable to unpickle.

**Licence**: the Ultralytics weights and code are AGPL-3.0. The converted arrays are a
derived work of the weights: check the licence before storing them on Drive or showing them
in the article (docs/tasks/M26-recent-models.md).
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
BN_EPS = 1e-3  # ultralytics BatchNorm2d(eps=0.001)


# ------------------------------------------------------------------------- torch modules
def autopad(k, p=None):
    return k // 2 if p is None else p


class Conv(nn.Module):
    """conv (no bias) + BN + SiLU, attribute names of ultralytics `Conv`."""

    def __init__(self, c1, c2, k=1, s=1, p=None):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), bias=False)
        self.bn = nn.BatchNorm2d(c2, eps=BN_EPS, momentum=0.03)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, c1, c2, shortcut=True, k=(1, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        y = self.cv2(self.cv1(x))
        return x + y if self.add else y


class C3(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, (1, 3), 1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C2f(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, (3, 3), 1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class SPPF(nn.Module):
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Concat(nn.Module):
    def __init__(self, d=1):
        super().__init__()
        self.d = d

    def forward(self, x):
        return torch.cat(x, self.d)


class DetectV5(nn.Module):
    """YOLOv5 `Detect`: one 1×1 conv per level, na·(5 + nc) channels (raw maps)."""

    def __init__(self, nc, anchors, ch):
        super().__init__()
        self.nc, self.no, self.nl, self.na = nc, nc + 5, len(anchors), len(anchors[0]) // 2
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        return [self.m[i](x[i]) for i in range(self.nl)]


class DFL(nn.Module):
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        self.conv.weight.data[:] = torch.arange(c1, dtype=torch.float).view(1, c1, 1, 1)
        self.c1 = c1


class DetectV8(nn.Module):
    """YOLOv8 `Detect`: per level, box branch (4·reg_max DFL bins) and class branch; raw maps
    = their concatenation (64 + nc channels)."""

    def __init__(self, nc, ch, reg_max=16):
        super().__init__()
        self.nc, self.nl, self.reg_max = nc, len(ch), reg_max
        c2, c3 = max((16, ch[0] // 4, reg_max * 4)), max(ch[0], min(nc, 100))
        self.cv2 = nn.ModuleList(nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3),
                                               nn.Conv2d(c2, 4 * reg_max, 1)) for x in ch)
        self.cv3 = nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3),
                                               nn.Conv2d(c3, nc, 1)) for x in ch)
        self.dfl = DFL(reg_max)

    def forward(self, x):
        return [torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])), 1) for i in range(self.nl)]


ANCHORS_V5 = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]


def _ch(c, width=0.25, max_c=1024):
    return math.ceil(min(c, max_c) * width / 8) * 8


def _n(n, depth=0.33):
    return max(round(n * depth), 1) if n > 1 else n


# (from, module) rows of the yaml files, scaled to "n" (width 0.25, depth 0.33).
def arch_v5(nc=80):
    c = _ch
    return [
        (-1, Conv(3, c(64), 6, 2, 2)), (-1, Conv(c(64), c(128), 3, 2)),
        (-1, C3(c(128), c(128), _n(3))), (-1, Conv(c(128), c(256), 3, 2)),
        (-1, C3(c(256), c(256), _n(6))), (-1, Conv(c(256), c(512), 3, 2)),
        (-1, C3(c(512), c(512), _n(9))), (-1, Conv(c(512), c(1024), 3, 2)),
        (-1, C3(c(1024), c(1024), _n(3))), (-1, SPPF(c(1024), c(1024), 5)),
        (-1, Conv(c(1024), c(512), 1, 1)), (-1, nn.Upsample(None, 2, "nearest")),
        ([-1, 6], Concat(1)), (-1, C3(c(1024), c(512), _n(3), False)),
        (-1, Conv(c(512), c(256), 1, 1)), (-1, nn.Upsample(None, 2, "nearest")),
        ([-1, 4], Concat(1)), (-1, C3(c(512), c(256), _n(3), False)),
        (-1, Conv(c(256), c(256), 3, 2)), ([-1, 14], Concat(1)),
        (-1, C3(c(512), c(512), _n(3), False)), (-1, Conv(c(512), c(512), 3, 2)),
        ([-1, 10], Concat(1)), (-1, C3(c(1024), c(1024), _n(3), False)),
        ([17, 20, 23], DetectV5(nc, ANCHORS_V5, (c(256), c(512), c(1024)))),
    ]


def arch_v8(nc=80):
    c = _ch
    return [
        (-1, Conv(3, c(64), 3, 2)), (-1, Conv(c(64), c(128), 3, 2)),
        (-1, C2f(c(128), c(128), _n(3), True)), (-1, Conv(c(128), c(256), 3, 2)),
        (-1, C2f(c(256), c(256), _n(6), True)), (-1, Conv(c(256), c(512), 3, 2)),
        (-1, C2f(c(512), c(512), _n(6), True)), (-1, Conv(c(512), c(1024), 3, 2)),
        (-1, C2f(c(1024), c(1024), _n(3), True)), (-1, SPPF(c(1024), c(1024), 5)),
        (-1, nn.Upsample(None, 2, "nearest")), ([-1, 6], Concat(1)),
        (-1, C2f(c(1024) + c(512), c(512), _n(3))), (-1, nn.Upsample(None, 2, "nearest")),
        ([-1, 4], Concat(1)), (-1, C2f(c(512) + c(256), c(256), _n(3))),
        (-1, Conv(c(256), c(256), 3, 2)), ([-1, 12], Concat(1)),
        (-1, C2f(c(256) + c(512), c(512), _n(3))), (-1, Conv(c(512), c(512), 3, 2)),
        ([-1, 9], Concat(1)), (-1, C2f(c(512) + c(1024), c(1024), _n(3))),
        ([15, 18, 21], DetectV8(nc, (c(256), c(512), c(1024)))),
    ]


ARCHS = {"yolov5n": arch_v5, "yolov8n": arch_v8}


class Model(nn.Module):
    """`model.<i>` modules as in ultralytics `DetectionModel` (state dict keys model.<i>.…)."""

    def __init__(self, net, nc=80):
        super().__init__()
        rows = ARCHS[net](nc)
        self.froms = [f for f, _ in rows]
        self.model = nn.Sequential(*(m for _, m in rows))

    def forward(self, x):
        outs = []
        for f, m in zip(self.froms, self.model):
            if isinstance(f, list):
                x = m([outs[-1] if j == -1 else outs[j] for j in f])
            elif f != -1:
                x = m(outs[f])
            else:
                x = m(x)
            outs.append(x)
        return x


# ------------------------------------------------------------------------- conversion
def fold(conv, bn=None):
    """(W, b) float32 of a conv, BN folded: W·γ/σ, β − μ·γ/σ."""
    w = conv.weight.detach().double()
    b = conv.bias.detach().double() if conv.bias is not None else torch.zeros(w.shape[0],
                                                                                dtype=torch.double)
    if bn is not None:
        g = bn.weight.detach().double() / torch.sqrt(bn.running_var.detach().double() + bn.eps)
        w = w * g[:, None, None, None]
        b = (b - bn.running_mean.detach().double()) * g + bn.bias.detach().double()
    return w.float().numpy(), b.float().numpy()


class Emitter:
    """Flat layer list of `ul_graph`, built by walking the torch modules."""

    def __init__(self):
        self.layers, self.w = [], {}

    def _add(self, layer):
        self.layers.append(layer)
        return len(self.layers) - 1

    def conv(self, src, m, act="silu"):
        conv, bn = (m.conv, m.bn) if isinstance(m, Conv) else (m, None)
        w, b = fold(conv, bn)
        i = self._add({"type": "conv", "from": src, "k": conv.kernel_size[0],
                       "s": conv.stride[0], "p": conv.padding[0], "cin": conv.in_channels,
                       "cout": conv.out_channels, "act": act if bn is not None else "linear"})
        self.w[f"w{i}"], self.w[f"b{i}"] = w, b
        return i

    def module(self, src, m):
        """Emits module `m` reading layer(s) `src`; returns the index of its output."""
        if isinstance(m, Conv):
            return self.conv(src, m)
        if isinstance(m, Bottleneck):
            y = self.conv(self.conv(src, m.cv1), m.cv2)
            return self._add({"type": "add", "from": [src, y]}) if m.add else y
        if isinstance(m, C3):
            a = self.conv(src, m.cv1)
            for b in m.m:
                a = self.module(a, b)
            c = self.conv(src, m.cv2)
            return self.conv(self._add({"type": "concat", "from": [a, c]}), m.cv3)
        if isinstance(m, C2f):
            y = self.conv(src, m.cv1)
            parts = [self._add({"type": "slice", "from": y, "start": 0, "stop": m.c}),
                     self._add({"type": "slice", "from": y, "start": m.c, "stop": 2 * m.c})]
            for b in m.m:
                parts.append(self.module(parts[-1], b))
            return self.conv(self._add({"type": "concat", "from": parts}), m.cv2)
        if isinstance(m, SPPF):
            x = self.conv(src, m.cv1)
            ys = [x]
            for _ in range(3):
                ys.append(self._add({"type": "maxpool", "from": ys[-1], "k": m.m.kernel_size,
                                     "s": 1}))
            return self.conv(self._add({"type": "concat", "from": ys}), m.cv2)
        if isinstance(m, nn.Upsample):
            return self._add({"type": "upsample", "from": src, "s": int(m.scale_factor)})
        if isinstance(m, Concat):
            return self._add({"type": "concat", "from": list(src)})
        raise TypeError(type(m).__name__)

    def heads(self, srcs, m):
        if isinstance(m, DetectV5):
            return [self.conv(s, c, act="linear") for s, c in zip(srcs, m.m)]
        out = []
        for s, b2, b3 in zip(srcs, m.cv2, m.cv3):
            box = self.conv(self.conv(self.conv(s, b2[0]), b2[1]), b2[2], act="linear")
            cls = self.conv(self.conv(self.conv(s, b3[0]), b3[1]), b3[2], act="linear")
            out.append(self._add({"type": "concat", "from": [box, cls]}))
        return out


def convert(model, net):
    """(description, weights) of a torch `Model` in eval mode."""
    e = Emitter()
    idx = []  # output index of each model.<i>
    for f, m in zip(model.froms, model.model):
        if isinstance(m, (DetectV5, DetectV8)):
            outputs = e.heads([idx[j] for j in f], m)
            break
        if isinstance(f, list):
            src = [idx[-1] if j == -1 else idx[j] for j in f]
        else:
            src = (idx[-1] if idx else -1) if f == -1 else idx[f]
        idx.append(e.module(src, m))
    det = model.model[-1]
    desc = {"name": net, "classes": det.nc, "outputs": outputs, "layers": e.layers,
            "head": "anchor" if isinstance(det, DetectV5) else "dfl"}
    if isinstance(det, DetectV5):
        desc["anchors"] = det.anchors.view(det.nl, -1).tolist()  # pixels, per level
    else:
        desc["reg_max"] = det.reg_max
    return desc, e.w


def load_weights(model, path):
    """State dict or official checkpoint into `model` (strict)."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, dict) and "model" in obj and hasattr(obj["model"], "state_dict"):
        obj = obj["model"].float().state_dict()
    elif hasattr(obj, "state_dict"):
        obj = obj.float().state_dict()
    model.load_state_dict(obj, strict=True)


def randomize_bn(model, seed=0):
    """Non-trivial BN statistics, so that folding is tested (random init has μ = 0, σ = 1)."""
    g = torch.Generator().manual_seed(seed)
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            n = m.num_features
            m.running_mean.copy_(torch.randn(n, generator=g) * 0.1)
            m.running_var.copy_(torch.rand(n, generator=g) + 0.5)
            m.weight.data.copy_(torch.rand(n, generator=g) + 0.5)
            m.bias.data.copy_(torch.randn(n, generator=g) * 0.1)


def build(net, weights=None, seed=0, nc=80):
    torch.manual_seed(seed)
    model = Model(net, nc)
    if weights:
        load_weights(model, weights)
    else:
        randomize_bn(model, seed)
    return model.eval()


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--net", choices=sorted(ARCHS), required=True)
    ap.add_argument("--weights", type=Path, help="state dict or official checkpoint (.pt)")
    ap.add_argument("--random", action="store_true", help="random weights (tests)")
    ap.add_argument("--classes", type=int, default=80)
    ap.add_argument("--out", type=Path, default=ROOT / "weights")
    ap.add_argument("--check", type=int, default=1, help="images to compare NumPy vs torch")
    args = ap.parse_args()
    if not args.weights and not args.random:
        ap.error("--weights or --random")
    model = build(args.net, args.weights, nc=args.classes)
    desc, w = convert(model, args.net)
    args.out.mkdir(parents=True, exist_ok=True)
    base = args.out / args.net
    np.savez(base.with_suffix(".npz"), **w)
    base.with_suffix(".json").write_text(json.dumps(desc, indent=1) + "\n")
    print(f"{base}.npz / .json: {len(desc['layers'])} layers, "
          f"{sum(v.size for k, v in w.items() if k.startswith('w')):,} weights")
    if args.check:
        sys.path.insert(0, str(ROOT / "python"))
        from yolo.models import ul_graph

        x = torch.rand(args.check, 3, 640, 640, generator=torch.Generator().manual_seed(1))
        with torch.no_grad():
            want = [t.double().numpy() for t in model(x)]
        got = ul_graph.forward(desc, w, x.double().numpy())
        err = max(float(np.abs(a - b).max()) for a, b in zip(got, want))
        print(f"NumPy vs torch, {args.check} image(s) 640×640: max abs error {err:.2e}")


if __name__ == "__main__":
    main()
