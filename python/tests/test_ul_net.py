"""T33.3: trainable YOLOv5n (BN unfused), forward against torch, folding, gradcheck."""

from pathlib import Path

import numpy as np
import pytest

from yolo.layers import blocks
from yolo.models import ul_graph, ul_net, ul_yaml
from yolo.models.specs import NETWORKS
from yolo.testing.gradcheck import check_grad

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "weights" / "m33"
CHECKPOINTS = ROOT / "weights" / "ultralytics"

# Conv 3×3 / 2, C3 (n = 2, shortcut), SPPF, one anchor head: small enough to check every
# parameter against the blocks of T33.2.
MINI = {
    "nc": 2, "depth_multiple": 1.0, "width_multiple": 1.0, "anchors": [[10, 13, 16, 30]],
    "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 2, "C3", [8]], [-1, 1, "SPPF", [8, 5]]],
    "head": [[[2], 1, "Detect", ["nc", "anchors"]]],
    "ch": 3,
}


def _loss(net, x, rs):
    def f():
        outs = ul_net.heads(net, x, train=True)
        return np.concatenate([(o * r).ravel() for o, r in zip(outs, rs)])
    return f


def _grads(net, x, rng):
    outs = ul_net.heads(net, x, train=True)
    rs = [rng.standard_normal(o.shape) for o in outs]
    ul_net.heads(net, x, train=True)
    dx, grads = net.backward(dict(zip(net.outputs, rs)))
    return rs, dx, grads


@pytest.fixture(scope="module")
def mini():
    desc = ul_yaml.parse(MINI, name="mini", size=16)
    net = ul_net.build(desc, ul_net.random_state_dict(desc, rng=1))
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 16, 16))
    rs, dx, grads = _grads(net, x, rng)
    return net, x, rs, dx, grads


def test_mini_gradcheck_all_parameters(mini):
    net, x, rs, dx, grads = mini
    f = _loss(net, x, rs)
    assert check_grad(f, x, dx) <= 1e-7
    for i, p in enumerate(net.params):
        for k, v in p.items():
            assert check_grad(f, v, grads[i][k]) <= 1e-7, (i, k)


def _block_params(sd, prefix):
    """Torch-named state dict entries below `prefix` → the flat names of `blocks`."""
    rename = {"conv.weight": "W", "bn.weight": "gamma", "bn.bias": "beta"}
    out = {}
    for k, v in sd.items():
        if k.startswith(prefix):
            head, _, tail = k[len(prefix):].rpartition(".")
            head, _, mod = head.rpartition(".")
            name = rename.get(f"{mod}.{tail}")
            if name:
                out[f"{head}.{name}" if head else name] = v
    return out


def test_flat_list_equals_blocks(mini):
    """The graph running the flat list of C3 and SPPF == `blocks.c3` / `blocks.sppf`."""
    net, x, *_ = mini
    sd = ul_net.state_dict(net)
    outs = net.forward(x, train=True, all_outputs=True)
    at = {layer.get("name"): i for i, layer in enumerate(net.layers)}
    y1, _ = blocks.c3_forward(outs[0], _block_params(sd, "model.1."), {}, n=2)
    assert np.abs(y1 - outs[at["model.1.cv3"]]).max() <= 1e-12
    y2, _ = blocks.sppf_forward(y1, _block_params(sd, "model.2."), {})
    assert np.abs(y2 - outs[at["model.2.cv2"]]).max() <= 1e-12


def test_bn_running_statistics():
    """PyTorch convention: μ̂ ← 0.97 μ̂ + 0.03 μ_B, σ̂² from the unbiased batch variance."""
    desc = ul_yaml.parse(MINI, size=16)
    net = ul_net.build(desc, rng=0)
    x = np.random.default_rng(3).standard_normal((2, 3, 16, 16))
    from yolo.layers.conv import conv_forward

    y = conv_forward(x, net.params[0]["W"], s=2, pad=1)[0]
    net.forward(x, train=True)
    m = y.size // y.shape[1]
    assert np.allclose(net.state[0]["mean"], 0.03 * y.mean(axis=(0, 2, 3)))
    assert np.allclose(net.state[0]["var"], 0.97 + 0.03 * y.var(axis=(0, 2, 3)) * m / (m - 1))


def test_state_dict_round_trip():
    desc = NETWORKS["yolov5n-coco"]
    sd = ul_net.random_state_dict(desc, rng=4)
    back = ul_net.state_dict(ul_net.build(desc, sd))
    assert sorted(back) == sorted(sd)
    assert all(np.array_equal(back[k], sd[k]) for k in sd)


def test_fold_matches_unfolded():
    """`ul_graph.forward(fold_bn(net))` == eval forward of the unfolded network (≤ 1e-5)."""
    desc = NETWORKS["yolov5n-coco"]
    net = ul_net.build(desc, ul_net.random_state_dict(desc, rng=5))
    x = np.random.default_rng(6).random((1, 3, 128, 128))
    want = ul_net.heads(net, x)
    fdesc, w = ul_net.fold_bn(net)
    got = ul_graph.forward(fdesc, w, x)
    assert max(float(np.abs(a - b).max()) for a, b in zip(got, want)) <= 1e-5


def test_whole_network_gradcheck():
    """YOLOv5n, 2 images 64×64, training mode: sampled entries of x and of parameters of the
    stem, a C3, the SPPF and the heads."""
    desc = NETWORKS["yolov5n-coco"]
    net = ul_net.build(desc, ul_net.random_state_dict(desc, rng=7))
    rng = np.random.default_rng(8)
    x = rng.random((2, 3, 64, 64))
    rs, dx, grads = _grads(net, x, rng)
    f = _loss(net, x, rs)
    assert check_grad(f, x, dx, n_samples=6, rng=0) <= 1e-7
    names = {layer.get("name"): i for i, layer in enumerate(net.layers)}
    picks = [("model.0", "W"), ("model.0", "gamma"), ("model.4.m.1.cv2", "W"),
             ("model.4.cv3", "beta"), ("model.9.cv2", "W"), ("model.23.cv3", "gamma"),
             ("model.24.m.0", "W"), ("model.24.m.2", "b")]
    for name, k in picks:
        i = names[name]
        assert check_grad(f, net.params[i][k], grads[i][k], n_samples=4, rng=1) <= 1e-7, name


def test_official_weights_equal_torch():
    """Eval forward with the official weights == the torch heads on 3 COCO images (≤ 1e-4)."""
    ref_path = FIXTURES / "yolov5n_heads.npz"
    if not (ref_path.exists() and (CHECKPOINTS / "yolov5n.pt").exists()):
        pytest.skip("official weights or fixtures absent (M33 step 0)")
    ref = np.load(ref_path)
    net = ul_net.build("yolov5n-coco", pretrained=True)
    got = ul_net.heads(net, ref["x"].astype(np.float64))
    for g, k in zip(got, ("p3", "p4", "p5")):
        assert g.shape == ref[k].shape
        assert np.abs(g - ref[k]).max() <= 1e-4
    fdesc, w = ul_net.fold_bn(net)
    folded = ul_graph.forward(fdesc, w, ref["x"][:1].astype(np.float64))
    assert max(float(np.abs(a - b[:1]).max()) for a, b in zip(folded, got)) <= 1e-5
