"""T33.1, T35.0: network description from the Ultralytics YAML."""

import copy
import json
from pathlib import Path

import pytest

from yolo.models import ul_yaml
from yolo.models.specs import NETWORKS, infer_shapes, layer_cost

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "weights" / "m33"
CHECKPOINTS = ROOT / "weights" / "ultralytics"


def macs(net, size):
    net = copy.deepcopy(net)
    net["input"] = [3, size, size]
    return sum(layer_cost(layer, i, o)[1] for layer, (i, o) in zip(net["layers"],
                                                                   infer_shapes(net)))


def test_scaling():
    assert [ul_yaml.make_divisible(c * 0.25) for c in (64, 128, 256, 512, 1024)] == \
        [16, 32, 64, 128, 256]
    assert [ul_yaml.scaled_repeats(n, 0.33) for n in (1, 3, 6, 9)] == [1, 1, 2, 3]


def test_yolov5n_architecture():
    """Table of M33: widths, C3 repeats, the 6×6 / 2 first conv, three 255-channel heads."""
    net = NETWORKS["yolov5n-coco"]
    layers = net["layers"]
    assert layers[0] == {"type": "conv", "from": -1, "k": 6, "s": 2, "p": 2, "cin": 3,
                         "cout": 16, "act": "silu", "bn": True, "name": "model.0"}
    assert sum(1 for layer in layers if layer["type"] == "add") == 1 + 2 + 3 + 1
    assert sum(1 for layer in layers if layer["type"] == "maxpool") == 3
    shapes = infer_shapes(net)
    assert [shapes[j][1] for j in net["outputs"]] == [(255, 80, 80), (255, 40, 40),
                                                      (255, 20, 20)]
    assert all(layers[j]["bn"] is False and layers[j]["name"] == f"model.24.m.{lv}"
               for lv, j in enumerate(net["outputs"]))
    assert net["anchors"][2] == [116, 90, 156, 198, 373, 326]


@pytest.mark.parametrize("net, params", [("yolov5n", 1_867_405), ("yolov8n", 3_151_888)])
def test_param_counts(net, params):
    """BN folded, C = 80: the T26.1 counts (published 1.87 M and 3.15 M)."""
    desc = ul_yaml.parse(ul_yaml.CONFIGS[net])
    assert ul_yaml.num_params(desc, folded=True) == params
    assert ul_yaml.num_params(desc) - params == sum(
        layer["cout"] for layer in desc["layers"] if layer["type"] == "conv" and layer["bn"])


def test_macs():
    """4.5 GFLOPs at 640 (published) = 2.25 GMAC ± 1 %; 416 value recorded."""
    net = NETWORKS["yolov5n-coco"]
    assert macs(net, 640) == pytest.approx(2.25e9, rel=0.01)
    assert macs(net, 640) == 2_234_060_800
    assert macs(net, 416) == 943_890_688


def test_yolov8n_architecture():
    """Table of M35: C2f repeats 1-2-2-1 (shortcut) then 1 ×4 (none), decoupled heads of
    64 + 80 channels, widths 64 / 80 of the box / class branches."""
    net = NETWORKS["yolov8n-coco"]
    layers = net["layers"]
    assert net["head"] == "dfl" and net["reg_max"] == 16
    assert sum(1 for layer in layers if layer["type"] == "add") == 1 + 2 + 2 + 1
    assert sum(1 for layer in layers if layer["type"] == "slice") == 2 * 8
    shapes = infer_shapes(net)
    assert [shapes[j][1] for j in net["outputs"]] == [(144, 80, 80), (144, 40, 40),
                                                      (144, 20, 20)]
    by_name = {layer.get("name"): layer for layer in layers}
    assert by_name["model.22.cv2.0.0"]["cout"] == 64 and by_name["model.22.cv3.0.0"]["cout"] == 80
    assert by_name["model.22.cv2.2.2"] == {
        "type": "conv", "from": by_name_index(layers, "model.22.cv2.2.1"), "k": 1, "s": 1,
        "p": 0, "cin": 64, "cout": 64, "act": "linear", "bn": False, "name": "model.22.cv2.2.2"}
    assert by_name["model.21.cv2"]["cin"] == 3 * 128 and by_name["model.21.cv2"]["cout"] == 256


def by_name_index(layers, name):
    return next(i for i, layer in enumerate(layers) if layer.get("name") == name)


def test_yolov8n_macs():
    """8.7 GFLOPs at 640 (published) = 4.35 GMAC ± 1 %; 416 value recorded."""
    net = NETWORKS["yolov8n-coco"]
    assert macs(net, 640) == pytest.approx(4.35e9, rel=0.01)
    assert macs(net, 640) == 4_371_456_000
    assert macs(net, 416) == 1_846_940_160


def test_yolov8n_checkpoint_meta():
    """Strides and the 80 COCO class names come from the checkpoint (T35.0)."""
    pt = CHECKPOINTS / "yolov8n.pt"
    if not pt.exists():
        pytest.skip("official weights absent (M33 step 0)")
    from yolo.io.torch_pickle import load_checkpoint

    sd, meta = load_checkpoint(pt)
    assert [float(s) for s in meta["stride"]] == [8.0, 16.0, 32.0]
    names = meta["names"]
    assert len(names) == 80 and names[0] == "person" and names[79] == "toothbrush"
    assert "model.22.dfl.conv.weight" in sd


def test_classes():
    desc = ul_yaml.parse(ul_yaml.YOLOV5N, nc=20)
    assert {desc["layers"][j]["cout"] for j in desc["outputs"]} == {3 * 25}


@pytest.mark.parametrize("net", ["yolov5n", "yolov8n"])
def test_equals_t261_description(net):
    path = FIXTURES / f"{net}_t261.json"
    if not path.exists():
        pytest.skip("T26.1 fixtures absent (M33 step 0)")
    ref = json.loads(path.read_text())
    desc = ul_yaml.parse(ul_yaml.CONFIGS[net], name=net)
    assert [{k: v for k, v in layer.items() if k not in ("bn", "name")}
            for layer in desc["layers"]] == ref["layers"]
    assert desc["outputs"] == ref["outputs"] and desc["head"] == ref["head"]


@pytest.mark.parametrize("net", ["yolov5n", "yolov8n"])
def test_yaml_equals_checkpoint(net):
    pt = CHECKPOINTS / f"{net}.pt"
    if not pt.exists():
        pytest.skip("official weights absent (M33 step 0)")
    from yolo.io.torch_pickle import load_checkpoint

    assert load_checkpoint(pt)[1]["yaml"] == ul_yaml.CONFIGS[net]
