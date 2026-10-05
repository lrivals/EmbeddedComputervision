"""T10.11 : élagage structuré par norme de filtre."""

import numpy as np

from yolo.models.cfg import parse_cfg
from yolo.models.tiny_yolo import CFG_DIR, CFG_FILES, build
from yolo.prune import keep_count, prunable, prune, select, write_cfg


def _net():
    net = build("tiny-yolov2-voc", dtype=np.float64, rng=0)
    rng = np.random.default_rng(1)
    for i, layer in enumerate(net.layers):
        if layer["type"] == "conv" and layer["bn"]:
            c = layer["cout"]
            net.params[i]["gamma"] = rng.uniform(0.5, 1.5, c)
            net.params[i]["beta"] = rng.normal(0, 0.1, c)
            net.state[i] = {"mean": rng.normal(0, 0.1, c), "var": rng.uniform(0.5, 2, c)}
    return net


def test_keep_count_multiples():
    assert keep_count(1024, 0.5) == 512
    assert keep_count(64, 0.7) == 32  # au moins un multiple
    assert keep_count(64, 0.0) == 64
    assert keep_count(256, 0.3) % 32 == 0


def test_prunable_chain_v2():
    net = _net()
    # L00 et L02 (≤ 32 filtres) et la tête sont exclues ; L13 alimente la tête.
    assert prunable(net) == {4: 6, 6: 8, 8: 10, 10: 12, 12: 13, 13: 14}


def test_rate_zero_is_identity():
    net = _net()
    x = np.random.default_rng(2).uniform(0, 1, (1, 3, 64, 64))
    a = net.forward(x, train=False)
    b = prune(net, select(net, 0.0)).forward(x, train=False)
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_pruned_equals_masked_next_conv():
    """Élaguer L08 == annuler, dans L10, les poids des canaux d'entrée retirés."""
    net = _net()
    keep = {8: select(net, 0.5)[8]}
    p = prune(net, keep)
    assert p.params[8]["W"].shape[0] == 128 and p.params[10]["W"].shape[1] == 128
    drop = np.setdiff1d(np.arange(256), keep[8])
    net.params[10]["W"][:, drop] = 0
    x = np.random.default_rng(3).uniform(0, 1, (1, 3, 64, 64))
    a, b = net.forward(x, train=False), p.forward(x, train=False)
    assert all(np.allclose(a[k], b[k]) for k in a)


def test_keeps_largest_filters():
    net = _net()
    net.params[6]["W"][5] *= 100.0
    assert 5 in select(net, 0.7)[6]


def test_write_cfg_roundtrip():
    text = (CFG_DIR / CFG_FILES["tiny-yolov2-voc"]).read_text()
    cfg = parse_cfg(write_cfg(text, {8: 128, 13: 512}))
    assert cfg["layers"][8]["cout"] == 128 and cfg["layers"][13]["cout"] == 512
    assert cfg["layers"][10]["cout"] == 512
