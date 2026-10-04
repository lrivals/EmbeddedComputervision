"""T0.5 : chaque ligne des tableaux du §3 est reproduite par le comptage."""

import pytest

from yolo.models.specs import TINY_YOLOV2_VOC, TINY_YOLOV3_VOC, infer_shapes, layer_cost

# id : (sortie (C, H, W), paramètres, MACs en M arrondis à 0,1) — tableaux §3.1 et §3.2.
V2_TABLE = {
    0: ((16, 416, 416), 464, 74.8),
    1: ((16, 208, 208), 0, 0),
    2: ((32, 208, 208), 4_672, 199.4),
    3: ((32, 104, 104), 0, 0),
    4: ((64, 104, 104), 18_560, 199.4),
    5: ((64, 52, 52), 0, 0),
    6: ((128, 52, 52), 73_984, 199.4),
    7: ((128, 26, 26), 0, 0),
    8: ((256, 26, 26), 295_424, 199.4),
    9: ((256, 13, 13), 0, 0),
    10: ((512, 13, 13), 1_180_672, 199.4),
    11: ((512, 13, 13), 0, 0),
    12: ((1024, 13, 13), 4_720_640, 797.4),
    13: ((1024, 13, 13), 9_439_232, 1594.9),
    14: ((125, 13, 13), 128_125, 21.6),
}

V3_TABLE = {
    12: ((1024, 13, 13), 4_720_640, 797.4),
    13: ((256, 13, 13), 262_656, 44.3),
    14: ((512, 13, 13), 1_180_672, 199.4),
    15: ((75, 13, 13), 38_475, 6.5),
    16: ((75, 13, 13), 0, 0),
    17: ((256, 13, 13), 0, 0),
    18: ((128, 13, 13), 33_024, 5.5),
    19: ((128, 26, 26), 0, 0),
    20: ((384, 26, 26), 0, 0),
    21: ((256, 26, 26), 885_248, 598.1),
    22: ((75, 26, 26), 19_275, 13.0),
    23: ((75, 26, 26), 0, 0),
}


def costs(net):
    return [
        (outs, *layer_cost(layer, ins, outs))
        for layer, (ins, outs) in zip(net["layers"], infer_shapes(net))
    ]


@pytest.mark.parametrize("net, expected", [(TINY_YOLOV2_VOC, V2_TABLE), (TINY_YOLOV3_VOC, V3_TABLE)])
def test_lines(net, expected):
    rows = costs(net)
    for i, (shape, params, macs_m) in expected.items():
        out, p, m = rows[i]
        assert out == shape, f"couche {i}"
        assert p == params, f"couche {i}"
        assert round(m / 1e6, 1) == macs_m, f"couche {i}"


def test_layer_counts():
    assert len(TINY_YOLOV2_VOC["layers"]) == 15
    assert len(TINY_YOLOV3_VOC["layers"]) == 24


def test_v3_backbone_equals_v2():
    rows = costs(TINY_YOLOV3_VOC)[:12]
    assert rows == costs(TINY_YOLOV2_VOC)[:12]
    assert sum(r[1] for r in rows) == 1_573_776
    # Le §3.2 affiche 1 071,8 M : somme des lignes déjà arrondies (74,8 + 5 × 199,4).
    # La valeur exacte est 1 071 562 752 MACs.
    assert sum(r[2] for r in rows) == 1_071_562_752


@pytest.mark.parametrize("net, params_m, macs_g", [
    (TINY_YOLOV2_VOC, 15.86, 3.49),
    (TINY_YOLOV3_VOC, 8.71, 2.74),
])
def test_totals(net, params_m, macs_g):
    rows = costs(net)
    assert round(sum(r[1] for r in rows) / 1e6, 2) == params_m
    assert round(sum(r[2] for r in rows) / 1e9, 2) == macs_g
