"""T1.8 : Tiny-YOLOv2 et Tiny-YOLOv3 construits depuis les .cfg == tableaux du §3 ;
T24.0-T24.1 : YOLOv4-tiny (stride 2, routes groupées, scale_x_y)."""

import numpy as np
import pytest

from yolo.layers.route import route_backward, route_forward
from yolo.models.cfg import parse_cfg
from yolo.models.graph import Network
from yolo.models.specs import NETWORKS, infer_shapes, layer_cost
from yolo.models.tiny_yolo import build, load_cfg
from yolo.testing.gradcheck import check_grad

NAMES = ["tiny-yolov2-voc", "tiny-yolov3-voc", "tiny-yolov4-coco"]


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
    elif net.net["name"] == "tiny-yolov4-coco":
        assert expected == {30: (1, 255, 13, 13), 37: (1, 255, 26, 26)}
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


# T24.0 : sortie de Darknet (AlexeyAB) pour yolov4-tiny.cfg, « BF » = 2·MACs/1e9 par conv.
V4_DARKNET = {
    0: ((32, 208, 208), 0.075), 1: ((64, 104, 104), 0.399), 2: ((64, 104, 104), 0.797),
    3: ((32, 104, 104), 0), 4: ((32, 104, 104), 0.199), 6: ((64, 104, 104), 0),
    7: ((64, 104, 104), 0.089), 8: ((128, 104, 104), 0), 9: ((128, 52, 52), 0),
    10: ((128, 52, 52), 0.797), 11: ((64, 52, 52), 0), 16: ((256, 52, 52), 0),
    17: ((256, 26, 26), 0), 24: ((512, 26, 26), 0), 25: ((512, 13, 13), 0),
    26: ((512, 13, 13), 0.797), 28: ((512, 13, 13), 0.399), 29: ((255, 13, 13), 0.044),
    32: ((128, 13, 13), 0.011), 33: ((128, 26, 26), 0), 34: ((384, 26, 26), 0),
    35: ((256, 26, 26), 1.196), 36: ((255, 26, 26), 0.088),
}


def test_v4_layer_table():
    net = NETWORKS["tiny-yolov4-coco"]
    shapes = infer_shapes(net)
    total = 0
    for i, layer in enumerate(net["layers"]):
        macs = layer_cost(layer, *shapes[i])[1]
        total += macs
        if i in V4_DARKNET:
            shape, bf = V4_DARKNET[i]
            assert shapes[i][1] == shape, i
            assert round(2 * macs / 1e9, 3) == bf, i
    # Darknet : « Total BFLOPS 6.910 », dont 0,002 pour les trois maxpools.
    assert total == 3_453_938_176
    assert abs(2 * total / 1e9 - 6.908) < 1e-3


def test_v4_cfg_options():
    cfg = load_cfg("tiny-yolov4-coco")
    assert [cfg["layers"][i]["s"] for i in (0, 1, 2)] == [2, 2, 1]
    assert cfg["layers"][3] == {"type": "route", "from": [2], "groups": 2, "group_id": 1}
    assert cfg["layers"][34] == {"type": "route", "from": [33, 23]}
    assert [cfg["layers"][i].get("scale_x_y") for i in (30, 37)] == [1.05, 1.05]


CFG_HEAD = "[net]\nheight=8\nwidth=8\nchannels=3\n[convolutional]\nfilters=4\nsize=1\n" \
    "activation=linear\n[convolutional]\nfilters=2\nsize=1\nactivation=linear\n"


def test_grouped_route_multi_source_rejected():
    with pytest.raises(ValueError, match="plusieurs sources"):
        parse_cfg(CFG_HEAD + "[route]\nlayers=-1,-2\ngroups=2\ngroup_id=1\n")
    with pytest.raises(ValueError, match="group_id"):
        parse_cfg(CFG_HEAD + "[route]\nlayers=-1\ngroups=2\ngroup_id=2\n")


def test_grouped_route_gradcheck():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, 6, 3, 4))
    w = rng.standard_normal((2, 2, 3, 4))
    y, cache = route_forward([x], groups=3, group_id=1)
    np.testing.assert_array_equal(y, x[:, 2:4])
    (dx,), _ = route_backward(w, cache)
    assert check_grad(lambda: float(np.sum(route_forward([x], 3, 1)[0] * w)), x, dx) <= 1e-7


def test_grouped_route_in_graph_gradcheck():
    # Conv → route groupée → conv, plus une concat qui relit la conv (bloc CSP réduit) :
    # le gradient de la première conv cumule la moitié groupée et la concat.
    cfg = CFG_HEAD.replace("activation=linear\n[convolutional]", "activation=linear\n"
                           "[route]\nlayers=-1\ngroups=2\ngroup_id=1\n[convolutional]", 1)
    cfg = cfg.replace("[convolutional]\nfilters=4", "[convolutional]\nfilters=4", 1)
    cfg += "[route]\nlayers=-1,-3\n"
    net = Network(parse_cfg(cfg), dtype=np.float64, rng=0)
    assert [s[1] for s in net.shapes] == [(4, 8, 8), (2, 8, 8), (2, 8, 8), (6, 8, 8)]
    x = np.random.default_rng(1).standard_normal((1, 3, 8, 8))
    w = np.random.default_rng(2).standard_normal((1, 6, 8, 8))

    def loss():
        return float(np.sum(net.forward(x, train=False)[3] * w))

    loss()
    dx, grads = net.backward({3: w})
    assert check_grad(loss, x, dx) <= 1e-7
    assert check_grad(loss, net.params[0]["W"], grads[0]["W"]) <= 1e-7
