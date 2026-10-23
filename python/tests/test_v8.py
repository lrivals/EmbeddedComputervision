"""T35.2, T35.3: YOLOv8 DFL decoding, decoupled head and float network against torch."""

import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from yolo.infer.decode import anchor_points, decode, dfl_expect, dfl_expect_backward, dist2bbox
from yolo.layers import blocks
from yolo.models import ul_graph, ul_net, ul_yaml
from yolo.models.specs import NETWORKS
from yolo.testing.gradcheck import check_grad

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "weights" / "m33"
CHECKPOINTS = ROOT / "weights" / "ultralytics"
HEADS = FIXTURES / "yolov8n_heads.npz"
needs_torch_refs = pytest.mark.skipif(
    not (HEADS.exists() and (CHECKPOINTS / "yolov8n.pt").exists()),
    reason="official weights or torch fixtures absent (local only, M35)")

# Conv 3×3 / 2, C2f (n = 2, shortcut), SPPF, Conv 3×3 / 2, then the v8 head on two levels.
MINI = {
    "nc": 3, "depth_multiple": 1.0, "width_multiple": 1.0,
    "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 2, "C2f", [8, True]],
                 [-1, 1, "SPPF", [8, 5]], [-1, 1, "Conv", [16, 3, 2]]],
    "head": [[[2, 3], 1, "Detect", ["nc"]]],
    "ch": 3,
}


# --------------------------------------------------------------------- DFL decoding
def test_anchor_points():
    pts, st = anchor_points([(80, 80), (40, 40), (20, 20)], [8, 16, 32])
    assert len(pts) == 8400 and st.tolist() == [8.0] * 6400 + [16.0] * 1600 + [32.0] * 400
    assert pts[:3].tolist() == [[0.5, 0.5], [1.5, 0.5], [2.5, 0.5]]
    assert pts[80].tolist() == [0.5, 1.5] and pts[6400].tolist() == [0.5, 0.5]
    assert len(anchor_points([(52, 52), (26, 26), (13, 13)], [8, 16, 32])[0]) == 3549


def test_dfl_expectation():
    z = np.full((1, 16), -1e3)
    z[0, 5] = 0
    assert dfl_expect(z)[0][0] == pytest.approx(5.0)
    assert dfl_expect(np.zeros((1, 16)))[0][0] == pytest.approx(7.5)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((3, 4, 16)) * 2
    r = rng.standard_normal((3, 4))
    d, p = dfl_expect(z)
    assert check_grad(lambda: dfl_expect(z)[0] * r, z, dfl_expect_backward(r, p)) <= 1e-7


def test_dist2bbox():
    a = np.array([[2.5, 3.5]])
    assert dist2bbox(np.array([[1.0, 2.0, 3.0, 4.0]]), a).tolist() == [[1.5, 1.5, 5.5, 7.5]]


@needs_torch_refs
@pytest.mark.parametrize("size, keys, y", [(640, ("p3", "p4", "p5"), "y"),
                                           (416, ("q3", "q4", "q5"), "y416")])
def test_decoded_boxes_equal_torch(size, keys, y):
    """Boxes (xywh pixels) and scores of the Ultralytics `Detect` (inference output) on the
    3 images, from the stored torch head maps: ≤ 1e-4."""
    f = np.load(HEADS)
    net = copy.deepcopy(NETWORKS["yolov8n-coco"])
    net["input"] = [3, size, size]
    boxes, obj, scores = decode(dict(zip(net["outputs"], (f[k] for k in keys))), net)
    ref = f[y].transpose(0, 2, 1)
    assert np.abs(boxes * size - ref[..., :4]).max() <= 1e-4
    assert np.abs(scores - ref[..., 4:]).max() <= 1e-4
    assert np.array_equal(obj, scores.max(axis=-1))


# --------------------------------------------------------------------- network
def _grads(net, x, rng):
    outs = ul_net.heads(net, x, train=True)
    rs = [rng.standard_normal(o.shape) for o in outs]
    ul_net.heads(net, x, train=True)
    dx, grads = net.backward(dict(zip(net.outputs, rs)))
    return rs, dx, grads


def _loss(net, x, rs):
    def f():
        outs = ul_net.heads(net, x, train=True)
        return np.concatenate([(o * r).ravel() for o, r in zip(outs, rs)])
    return f


@pytest.fixture(scope="module")
def mini():
    desc = ul_yaml.parse(MINI, name="mini", size=16)
    net = ul_net.build(desc, ul_net.random_state_dict(desc, rng=1))
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, 3, 16, 16))
    rs, dx, grads = _grads(net, x, rng)
    return net, x, rs, dx, grads


def test_mini_gradcheck(mini):
    """Every tensor of the C2f (the split, the two bottlenecks, cv2) and of one level of each
    head branch, 4 sampled entries each; the blocks themselves are checked on every entry in
    `test_blocks.py`."""
    net, x, rs, dx, grads = mini
    f = _loss(net, x, rs)
    assert check_grad(f, x, dx, n_samples=8, rng=0) <= 1e-7
    keep = ("model.1.", "model.4.cv2.1.", "model.4.cv3.0.")
    checked = 0
    for i, layer in enumerate(net.layers):
        if not str(layer.get("name", "")).startswith(keep):
            continue
        for k, v in net.params[i].items():
            assert check_grad(f, v, grads[i][k], n_samples=4, rng=i) <= 1e-7, (layer["name"], k)
            checked += 1
    assert checked == 6 * 3 + 2 * (2 * 3 + 2)


def _block_params(sd, prefix):
    """Torch-named state dict entries below `prefix` → the flat names of `blocks`."""
    rename = {"conv.weight": "W", "bn.weight": "gamma", "bn.bias": "beta"}
    out = {}
    for k, v in sd.items():
        if not k.startswith(prefix):
            continue
        head, _, tail = k[len(prefix):].rpartition(".")
        stem, _, mod = head.rpartition(".")
        name = rename.get(f"{mod}.{tail}")
        if name:
            out[f"{stem}.{name}" if stem else name] = v
        elif tail in ("weight", "bias") and "bn" not in head:
            out[f"{head}.{'W' if tail == 'weight' else 'b'}"] = v
    return out


def test_flat_list_equals_blocks(mini):
    """The graph running the flat list of C2f and Detect == `blocks.c2f` / `detect_v8`."""
    net, x, *_ = mini
    sd = ul_net.state_dict(net)
    outs = net.forward(x, train=True, all_outputs=True)
    at = {layer.get("name"): i for i, layer in enumerate(net.layers)}
    y1, _ = blocks.c2f_forward(outs[0], _block_params(sd, "model.1."), {}, n=2, shortcut=True)
    assert np.abs(y1 - outs[at["model.1.cv2"]]).max() <= 1e-12
    heads, _ = blocks.detect_v8_forward([outs[at["model.2.cv2"]], outs[at["model.3"]]],
                                        _block_params(sd, "model.4."), {})
    for h, j in zip(heads, net.outputs):
        assert np.abs(h - outs[j]).max() <= 1e-12


def test_dfl_conv_not_read():
    """`model.22.dfl.conv` (fixed weights 0…15) is rebuilt by `dfl_expect`, never read."""
    desc = NETWORKS["yolov8n-coco"]
    sd = ul_net.random_state_dict(desc, rng=3)
    assert not any(".dfl." in k for k in sd)
    back = ul_net.state_dict(ul_net.build(desc, sd))
    assert sorted(back) == sorted(sd)


def test_fold_matches_unfolded():
    desc = NETWORKS["yolov8n-coco"]
    net = ul_net.build(desc, ul_net.random_state_dict(desc, rng=4))
    x = np.random.default_rng(5).random((1, 3, 96, 96))
    want = ul_net.heads(net, x)
    fdesc, w = ul_net.fold_bn(net)
    got = ul_graph.forward(fdesc, w, x)
    assert max(float(np.abs(a - b).max()) for a, b in zip(got, want)) <= 1e-5


def test_whole_network_gradcheck():
    """YOLOv8n, 2 images 64×64, training mode: sampled entries of x and of parameters of the
    stem, a backbone C2f bottleneck, the SPPF, a neck C2f and the box / class head convs."""
    desc = NETWORKS["yolov8n-coco"]
    net = ul_net.build(desc, ul_net.random_state_dict(desc, rng=6))
    rng = np.random.default_rng(7)
    x = rng.random((2, 3, 64, 64))
    rs, dx, grads = _grads(net, x, rng)
    f = _loss(net, x, rs)
    assert check_grad(f, x, dx, n_samples=6, rng=0) <= 1e-7
    names = {layer.get("name"): i for i, layer in enumerate(net.layers)}
    picks = [("model.0", "W"), ("model.0", "gamma"), ("model.4.m.1.cv2", "W"),
             ("model.4.cv1", "beta"), ("model.9.cv2", "W"), ("model.15.cv2", "gamma"),
             ("model.22.cv2.0.2", "W"), ("model.22.cv2.2.2", "b"), ("model.22.cv3.1.1", "W"),
             ("model.22.cv3.0.2", "b")]
    for name, k in picks:
        i = names[name]
        assert check_grad(f, net.params[i][k], grads[i][k], n_samples=4, rng=1) <= 1e-7, name


@needs_torch_refs
def test_official_weights_equal_torch():
    """Eval forward with the official weights == the torch head maps on 3 COCO images at
    640 (≤ 1e-4); the folded network (float32 weights) within 10⁻⁶ of the largest logit of
    each map from the unfolded one (1.6·10⁻⁵ absolute on logits up to ≈ 20)."""
    ref = np.load(HEADS)
    net = ul_net.build("yolov8n-coco", pretrained=True)
    got = ul_net.heads(net, ref["x"].astype(np.float64))
    for g, k in zip(got, ("p3", "p4", "p5")):
        assert g.shape == ref[k].shape
        assert np.abs(g - ref[k]).max() <= 1e-4
    fdesc, w = ul_net.fold_bn(net)
    folded = ul_graph.forward(fdesc, w, ref["x"][:1].astype(np.float64))
    for a, b in zip(folded, got):
        assert np.abs(a - b[:1]).max() <= 1e-6 * np.abs(b[:1]).max()


def test_runs_without_torch():
    """Decoding, assigner and loss import and run with `torch` blocked."""
    code = ("import sys; sys.modules['torch'] = None\n"
            "import numpy as np\n"
            "from yolo.train.loss_v8 import yolo_v8_loss\n"
            "outs = {0: np.zeros((1, 67, 4, 4)), 1: np.zeros((1, 67, 2, 2))}\n"
            "r = yolo_v8_loss(outs, [np.array([[0.5, 0.5, 0.4, 0.4]])], [np.array([1])],\n"
            "                 [(0, 8), (1, 16)], 3)\n"
            "assert 'torch' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
            "print(int(r.masks['fg'].sum()))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT / "python", check=True)
    assert int(out.stdout) > 0
