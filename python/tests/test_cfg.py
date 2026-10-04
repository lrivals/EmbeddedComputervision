"""T1.7 : parser des .cfg Darknet."""

import pytest

from yolo.models.cfg import parse_cfg

TEXT = """
[net]
# commentaire
batch=64
width=32
height=24
channels=3

[convolutional]
batch_normalize=1
filters=8
size=3
stride=1
pad=1
activation=leaky

[maxpool]
size=2
stride=2

[convolutional]
filters=4
size=1
stride=1
pad=1
activation=linear

[route]
layers = -1, 0

[upsample]
stride=2

[yolo]
mask = 0,1
anchors = 10,14,  23,27,  37,58
classes=3
num=3
"""


def test_parse():
    net = parse_cfg(TEXT, name="t")
    assert net["name"] == "t"
    assert net["input"] == (3, 24, 32)
    assert net["classes"] == 3
    assert net["anchors"] == [[10, 14], [23, 27], [37, 58]]
    assert net["layers"] == [
        {"type": "conv", "k": 3, "s": 1, "cout": 8, "act": "leaky", "bn": True},
        {"type": "maxpool", "k": 2, "s": 2},
        {"type": "conv", "k": 1, "s": 1, "cout": 4, "act": "linear", "bn": False},
        {"type": "route", "from": [2, 0]},
        {"type": "upsample", "s": 2},
        {"type": "yolo", "mask": [0, 1]},
    ]


def test_region_anchors_in_pixels():
    net = parse_cfg("[net]\nwidth=416\nheight=416\n[region]\nanchors=1.0,2.5\nclasses=20\nnum=1\n")
    assert net["anchors"] == [[32.0, 80.0]]
    assert net["layers"] == [{"type": "region", "num": 1}]


def test_unknown_section():
    with pytest.raises(ValueError):
        parse_cfg("[net]\nwidth=8\nheight=8\n[shortcut]\nfrom=-3\n")
