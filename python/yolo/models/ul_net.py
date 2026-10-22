"""Trainable Ultralytics networks (YOLOv5n, T33.3), BN kept unfused, no torch.

The description of `yolo.models.ul_yaml` runs in the generic `yolo.models.graph.Network`
(forward, backward with gradient accumulation, train / eval BN). Ultralytics BN conventions:
ε = 10⁻³, momentum 0.03 (μ̂ ← 0.97·μ̂ + 0.03·μ_B, i.e. `bn_momentum = 0.97` here), running
variance from the unbiased batch variance.

Parameters move to and from torch state dicts (`yolo.io.torch_pickle`) by the `name` of each
conv. Inference, quantization and export work on the folded description (`fold`,
`quant/fuse_bn.py` formula), run by `yolo.models.ul_graph`.
"""

from pathlib import Path

import numpy as np

from yolo.io.torch_pickle import load_checkpoint
from yolo.models import ul_yaml
from yolo.models.graph import Network
from yolo.quant.fuse_bn import fuse_bn

BN_EPS = 1e-3
BN_MOMENTUM = 0.97  # keep factor: PyTorch momentum 0.03
WEIGHTS = Path(__file__).resolve().parents[3] / "weights" / "ultralytics"
PRETRAINED = {"yolov5n-coco": "yolov5n.pt"}
UL_FIELDS = ("type", "from", "k", "s", "p", "cin", "cout", "act", "start", "stop")


def description(name):
    """Description (`specs` / `ul_yaml` format) of a network name such as `yolov5n-coco`;
    `<arch>-<dataset>` (e.g. `yolov5n-voc`, T33.7) is the `ul_yaml` architecture with the
    classes of the dataset."""
    from yolo.models.specs import NETWORKS

    if name in NETWORKS:
        return NETWORKS[name]
    arch, _, dataset = name.partition("-")
    if arch in ul_yaml.CONFIGS and dataset:
        from yolo.data.datasets import DATASETS

        return ul_yaml.parse(ul_yaml.CONFIGS[arch], nc=len(DATASETS[dataset].classes),
                             name=name)
    raise KeyError(name)


def is_ul_name(name):
    """True for the names of `description` (Ultralytics networks)."""
    try:
        return description(str(name)).get("head") is not None
    except KeyError:
        return False


def _keys(layer):
    n = layer["name"]
    if layer["bn"]:
        return {"W": f"{n}.conv.weight", "gamma": f"{n}.bn.weight", "beta": f"{n}.bn.bias",
                "mean": f"{n}.bn.running_mean", "var": f"{n}.bn.running_var"}
    return {"W": f"{n}.weight", "b": f"{n}.bias"}


def load_state_dict(net, sd):
    """Copies a torch state dict (`name -> array`) into the parameters and BN state of `net`."""
    for i, layer in enumerate(net.layers):
        if layer["type"] != "conv":
            continue
        for k, key in _keys(layer).items():
            v = np.asarray(sd[key], dtype=net.dtype)
            dst = net.state[i] if k in ("mean", "var") else net.params[i]
            if dst[k].shape != v.shape:
                raise ValueError(f"{key}: shape {v.shape}, expected {dst[k].shape}")
            dst[k] = v.copy()
    return net


def state_dict(net):
    """Torch-named state dict of `net` (parameters and running statistics)."""
    sd = {}
    for i, layer in enumerate(net.layers):
        if layer["type"] != "conv":
            continue
        for k, key in _keys(layer).items():
            sd[key] = (net.state[i] if k in ("mean", "var") else net.params[i])[k]
    return sd


def random_state_dict(desc, rng=0):
    """State dict of random weights with non-trivial BN statistics (tests, `--random`)."""
    rng = np.random.default_rng(rng)
    sd = {}
    for layer in desc["layers"]:
        if layer["type"] != "conv":
            continue
        k, cin, cout = layer["k"], layer["cin"], layer["cout"]
        keys = _keys(layer)
        sd[keys["W"]] = (rng.standard_normal((cout, cin, k, k))
                         * np.sqrt(2.0 / (k * k * cin))).astype(np.float32)
        if layer["bn"]:
            sd[keys["gamma"]] = (rng.random(cout) + 0.5).astype(np.float32)
            sd[keys["beta"]] = (rng.standard_normal(cout) * 0.1).astype(np.float32)
            sd[keys["mean"]] = (rng.standard_normal(cout) * 0.1).astype(np.float32)
            sd[keys["var"]] = (rng.random(cout) + 0.5).astype(np.float32)
        else:
            sd[keys["b"]] = (rng.standard_normal(cout) * 0.1).astype(np.float32)
    return sd


def build(name_or_desc, state_dict=None, pretrained=False, dtype=np.float64, rng=None):
    """Network of a name (`yolov5n-coco`) or description; weights from `state_dict`, or from
    `weights/ultralytics/<file>.pt` with `pretrained`, else He initialization (§7.1)."""
    desc = description(name_or_desc) if isinstance(name_or_desc, str) else name_or_desc
    net = Network(desc, dtype=dtype, rng=rng, bn_momentum=BN_MOMENTUM, bn_eps=BN_EPS,
                  bn_unbiased=True)
    if pretrained:
        state_dict, _ = load_checkpoint(WEIGHTS / PRETRAINED[desc["name"]])
    if state_dict is not None:
        load_state_dict(net, state_dict)
    return net


def ul_description(desc):
    """`desc` in the T26.1 `ul_graph` schema: BN folded, no `bn` / `name` fields."""
    out = {k: desc[k] for k in ("name", "classes", "outputs") if k in desc}
    out["layers"] = [{k: v for k, v in layer.items() if k in UL_FIELDS}
                     for layer in desc["layers"]]
    for k in ("head", "anchors", "reg_max"):
        if k in desc:
            out[k] = desc[k]
    return out


def fold(desc, sd, eps=BN_EPS):
    """(`ul_graph` description, weights `w<i>` / `b<i>` float32) with BN folded (§9.1)."""
    w = {}
    for i, layer in enumerate(desc["layers"]):
        if layer["type"] != "conv":
            continue
        keys = _keys(layer)
        W = np.asarray(sd[keys["W"]], np.float64)
        if layer["bn"]:
            W, b = fuse_bn(W, *(np.asarray(sd[keys[k]], np.float64)
                                for k in ("gamma", "beta", "mean", "var")), eps=eps)
        else:
            b = np.asarray(sd[keys["b"]], np.float64)
        w[f"w{i}"], w[f"b{i}"] = W.astype(np.float32), b.astype(np.float32)
    return ul_description(desc), w


def fold_bn(net):
    """Folded `ul_graph` description and weights of `net` (its running statistics)."""
    return fold(net.net, state_dict(net), eps=net.bn_eps)


def heads(net, x, train=False):
    """Raw head maps (list, `outputs` order) of the input x (N, 3, H, W)."""
    out = net.forward(x, train=train)
    return [out[i] for i in net.outputs]


def num_params(desc, folded=False):
    return ul_yaml.num_params(desc, folded)
