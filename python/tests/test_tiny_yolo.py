"""T1.8 : Tiny-YOLOv2 et Tiny-YOLOv3 construits depuis les .cfg == tableaux du §3."""

import numpy as np
import pytest

from yolo.models.specs import NETWORKS, infer_shapes, layer_cost
from yolo.models.tiny_yolo import build, load_cfg

NAMES = ["tiny-yolov2-voc", "tiny-yolov3-voc"]


@pytest.mark.parametrize("name", NAMES)
def test_cfg_matches_specs(name):
    spec, cfg = NETWORKS[name], load_cfg(name)
    assert cfg["input"] == spec["input"]
    assert cfg["classes"] == spec["classes"]
    layers = cfg["layers"]
    if name == "tiny-yolov2-voc":
        # La sortie [region] (couche 15) n'est pas dans le tableau du §3.1.
        assert layers[-1] == {"type": "region", "num": 5}
        assert len(cfg["anchors"]) == 5
        layers = layers[:-1]
    else:
        assert cfg["anchors"] == spec["anchors"]
    assert layers == spec["layers"]


@pytest.fixture(scope="module", params=NAMES)
def net(request):
    return build(request.param, rng=0)


def test_params_and_shapes_match_specs(net):
    spec = NETWORKS[net.net["name"]]
    shapes = infer_shapes(spec)
    for i, layer in enumerate(spec["layers"]):
        assert net.shapes[i] == shapes[i], i
        assert net.num_params(i) == layer_cost(layer, *shapes[i])[0], i


def test_forward_416(net):
    x = np.random.default_rng(1).standard_normal((1, 3, 416, 416)).astype(np.float32)
    out = net.forward(x, train=False)
    expected = {i: (1,) + out_shape for i, (_, out_shape) in enumerate(net.shapes)
                if i in net.outputs}
    assert {i: y.shape for i, y in out.items()} == expected
    assert all(y.dtype == np.float32 and np.all(np.isfinite(y)) for y in out.values())
    if net.net["name"] == "tiny-yolov3-voc":
        assert expected == {16: (1, 75, 13, 13), 23: (1, 75, 26, 26)}
    else:
        assert expected == {15: (1, 125, 13, 13)}


def test_he_init(net):
    # §7.1 : W ~ N(0, 2/(k²·C_in)), γ = 1, β = 0, biais de tête 0.
    for i, layer in enumerate(net.layers):
        if layer["type"] != "conv":
            continue
        p = net.params[i]
        std = np.sqrt(2.0 / (layer["k"] ** 2 * net.shapes[i][0][0]))
        assert abs(p["W"].std() / std - 1) < 0.1, i
        assert abs(p["W"].mean()) < 0.1 * std, i
        if layer["bn"]:
            assert np.all(p["gamma"] == 1) and np.all(p["beta"] == 0)
        else:
            assert np.all(p["b"] == 0)


def test_coco_cfg():
    cfg = load_cfg("tiny-yolov3-coco")
    assert cfg["classes"] == 80
    assert [cfg["layers"][i]["cout"] for i in (15, 22)] == [255, 255]
    assert [cfg["layers"][i]["mask"] for i in (16, 23)] == [[3, 4, 5], [1, 2, 3]]
