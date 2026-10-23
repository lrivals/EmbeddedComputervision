"""YOLOv5 training recipe (T33.7) and its YOLOv8 settings (T35.6): optimizer groups, warmup,
EMA, augmentation, trainer."""

import numpy as np
import pytest

from yolo.data.augment_v5 import box_candidates, mosaic4, random_perspective
from yolo.data.loader import DataLoader, V5Dataset
from yolo.data.targets import v5_levels, v8_levels
from yolo.models import ul_net, ul_yaml
from yolo.train.ema import ModelEMA
from yolo.train.optim import AdamWGroups, SGDGroups, auto_optimizer, param_group
from yolo.train.schedule import V5Schedule
from yolo.train.trainer_v5 import TrainerV5, head_conv, init_head_biases

MINI = {
    "nc": 2, "depth_multiple": 1.0, "width_multiple": 1.0, "anchors": [[8, 8, 16, 12]],
    "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "Conv", [8, 3, 2]],
                 [-1, 1, "C3", [8]]],
    "head": [[[2], 1, "Detect", ["nc", "anchors"]]],
    "ch": 3,
}


# The YOLOv8 head (T35.6): Conv, C2f, two levels of the decoupled DFL head.
MINI_V8 = {
    "nc": 2, "depth_multiple": 1.0, "width_multiple": 1.0,
    "backbone": [[-1, 1, "Conv", [8, 3, 2]], [-1, 1, "C2f", [8, True]],
                 [-1, 1, "Conv", [16, 3, 2]]],
    "head": [[[1, 2], 1, "Detect", ["nc"]]],
    "ch": 3,
}


def mini(size=32, rng=0, yaml=MINI):
    return ul_net.build(ul_yaml.parse(yaml, name="mini", size=size),
                        state_dict=ul_net.random_state_dict(
                            ul_yaml.parse(yaml, size=size), rng=rng))


def test_param_groups():
    assert [param_group(k) for k in ("W", "gamma", "beta", "b")] == \
        ["weights", "bn", "bias", "bias"]


def test_sgd_nesterov_matches_torch_formula():
    rng = np.random.default_rng(0)
    p = [{"W": rng.standard_normal(3), "b": rng.standard_normal(3)}]
    ref = {k: v.copy() for k, v in p[0].items()}
    buf = {k: np.zeros(3) for k in ref}
    opt = SGDGroups(p, momentum=0.9, weight_decay=0.01)
    lrs = {"weights": 0.1, "bn": 0.05, "bias": 0.2}
    for _ in range(3):
        g = {k: rng.standard_normal(3) for k in ref}
        opt.step([g], lrs)
        for k in ref:  # torch.optim.SGD(nesterov=True), dampening 0
            d = g[k] + (0.01 * ref[k] if k == "W" else 0)
            buf[k] = 0.9 * buf[k] + d
            ref[k] = ref[k] - lrs[param_group(k)] * (d + 0.9 * buf[k])
    for k in ref:
        assert np.allclose(p[0][k], ref[k])


def test_schedule_warmup_and_linear_decay():
    s = V5Schedule(0.01, epochs=10, iters_per_epoch=50, lrf=0.01)
    assert s.n_warmup == 150
    lrs, mom = s(0)
    assert lrs["weights"] == 0 and lrs["bias"] == 0.1 and mom == 0.8
    lrs, mom = s(75)  # half the warmup, epoch 1
    lf = (1 - 1 / 10) * 0.99 + 0.01
    assert np.isclose(lrs["weights"], 0.5 * 0.01 * lf)
    assert np.isclose(lrs["bias"], 0.1 + 0.5 * (0.01 * lf - 0.1))
    assert np.isclose(mom, 0.8 + 0.5 * 0.137)
    lrs, mom = s(9 * 50)
    assert np.isclose(lrs["bn"], 0.01 * ((1 - 0.9) * 0.99 + 0.01)) and mom == 0.937
    assert s.accumulate(0, 16) == 1 and s.accumulate(150, 16) == 4
    assert 1 < s.accumulate(100, 16) < 4
    # Short runs: warmup of at least 100 iterations.
    assert V5Schedule(0.01, 1, 10).n_warmup == 100


def test_ema_formula():
    net = mini()
    ema = ModelEMA(net, decay=0.9, tau=2)
    w0 = net.params[0]["W"].copy()
    net.params[0]["W"] += 1.0
    ema.update(net)
    d = 0.9 * (1 - np.exp(-1 / 2))
    assert np.allclose(ema.params[0]["W"], d * w0 + (1 - d) * (w0 + 1))
    with ema.swapped(net):
        assert np.allclose(net.params[0]["W"], d * w0 + (1 - d) * (w0 + 1))
    assert np.allclose(net.params[0]["W"], w0 + 1)


def test_head_bias_init():
    net = mini()
    hid, stride, anchors = v5_levels(net.net)[0]
    assert stride == 4
    i = head_conv(net, hid)
    b0 = net.params[i]["b"].copy()
    init_head_biases(net, img_size=32)
    d = (net.params[i]["b"] - b0).reshape(2, 7)
    assert np.allclose(d[:, 4], np.log(8 / (32 / 4) ** 2))
    assert np.allclose(d[:, 5:], np.log(0.6 / (2 - 0.99)))
    assert np.allclose(d[:, :4], 0)


def test_head_bias_init_v8():
    """`Detect.bias_init` of YOLOv8: box convs 1.0, class convs log(5 / C / (S/s)²); with
    `layers`, only those convs (the reinitialized class convs of a fine-tuning)."""
    net = mini(yaml=MINI_V8)
    init_head_biases(net, img_size=32)
    levels = v8_levels(net.net)
    assert [s for _, s in levels] == [2, 4]
    for hid, stride in levels:
        box, cls = net.layers[hid]["from"]
        assert np.all(net.params[box]["b"] == 1.0)
        assert np.allclose(net.params[cls]["b"], np.log(5 / 2 / (32 / stride) ** 2))
    hid, stride = levels[0]
    box, cls = net.layers[hid]["from"]
    net.params[box]["b"][...] = 0.3
    init_head_biases(net, layers=[cls], img_size=640)
    assert np.all(net.params[box]["b"] == 0.3)
    assert np.allclose(net.params[cls]["b"], np.log(5 / 2 / (640 / stride) ** 2))


def test_auto_optimizer():
    """YOLOv8 `optimizer=auto`: AdamW for ≤ 10,000 steps (lr 0.002·5/(4 + C)), else SGD."""
    assert auto_optimizer(20, 16_551, 30, 16) == ("adamw", round(0.002 * 5 / 24, 6), 0.9)
    assert auto_optimizer(20, 16_551, 100, 16) == ("sgd", 0.01, 0.9)    # 259 · 100 steps
    assert auto_optimizer(80, 1, 1, 1)[1] == 0.000119


def test_adamw_groups_matches_torch_formula():
    """`torch.optim.AdamW` per group: decay p·(1 − lr·wd) on the weights group only."""
    rng = np.random.default_rng(0)
    p = [{"W": rng.standard_normal(3), "gamma": rng.standard_normal(3),
          "b": rng.standard_normal(3)}]
    ref = {k: v.copy() for k, v in p[0].items()}
    m = {k: np.zeros(3) for k in ref}
    v = {k: np.zeros(3) for k in ref}
    opt = AdamWGroups(p, momentum=0.9, weight_decay=0.05)
    lrs = {"weights": 0.01, "bn": 0.02, "bias": 0.03}
    for t in range(1, 4):
        g = {k: rng.standard_normal(3) for k in ref}
        opt.step([g], lrs, momentum=0.5)                 # ignored, as the reference
        for k in ref:
            lr = lrs[param_group(k)]
            if k == "W":
                ref[k] *= 1 - lr * 0.05
            m[k] = 0.9 * m[k] + 0.1 * g[k]
            v[k] = 0.999 * v[k] + 0.001 * g[k] ** 2
            mh, vh = m[k] / (1 - 0.9 ** t), v[k] / (1 - 0.999 ** t)
            ref[k] -= lr * mh / (np.sqrt(vh) + 1e-8)
    for k in ref:
        assert np.allclose(p[0][k], ref[k], rtol=0, atol=1e-14)


# --------------------------------------------------------------------- augmentation
def _square_image(color, box, size=(120, 90)):
    img = np.full((size[1], size[0], 3), 30, dtype=np.uint8)
    x1, y1, x2, y2 = box
    img[y1:y2, x1:x2] = color
    return img


def test_mosaic_and_perspective_keep_boxes_on_pixels():
    rng = np.random.default_rng(0)
    colors = [(250, 0, 0), (0, 250, 0), (0, 0, 250), (250, 250, 0)]
    items = [(_square_image(c, (20, 15, 70, 60)), np.array([[20.0, 15, 70, 60]]), [k])
             for k, c in enumerate(colors)]
    for trial in range(5):
        canvas, boxes, labels = mosaic4(items, 128, rng)
        img, b, lab = random_perspective(canvas, boxes, labels, 128, rng)
        for (x1, y1, x2, y2), k in zip(b, lab):
            inner = img[int(y1) + 2:int(y2) - 2, int(x1) + 2:int(x2) - 2].reshape(-1, 3)
            if len(inner) > 20:
                assert np.abs(inner.mean(0) - colors[k]).max() < 30


def test_box_candidates():
    b1 = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 10, 10], [0, 0, 100, 1000]], float)
    b2 = np.array([[0, 0, 8, 8], [0, 0, 1.5, 8], [0, 0, 2.5, 2.5], [0, 0, 3, 400]], float)
    assert box_candidates(b1, b2).tolist() == [True, False, False, False]


def _samples(tmp_path, n=4, texture=False):
    from PIL import Image

    out = []
    for k in range(n):
        img = _square_image((200, 60, 60), (30, 20, 90, 70), size=(128, 96))
        if texture:  # distinct features in every cell (a flat box looks the same everywhere)
            img = (img * 0.6 + np.random.default_rng(k).integers(0, 100, img.shape)).astype(
                np.uint8)
        path = tmp_path / f"{k}.png"
        Image.fromarray(img).save(path)
        box = np.array([[60 / 128, 45 / 96, 60 / 128, 50 / 96]])
        out.append({"image": str(path), "boxes": box, "labels": np.array([1]),
                    "difficult": np.array([False]), "id": str(k), "width": 128, "height": 96})
    return out


def test_v5_dataset(tmp_path):
    samples = _samples(tmp_path)
    x, b, lab = V5Dataset(samples, train=False).load(0, 64)
    assert x.shape == (3, 64, 64) and np.isclose(x[:, 0, 0], 114 / 255).all()
    # 128 × 96 → 64 × 48, bands of 8 px at the top and bottom.
    assert np.allclose(b, [[60 / 128, (45 / 96 * 48 + 8) / 64, 60 / 128, 50 / 96 * 48 / 64]])
    ds = V5Dataset(samples)
    rng = np.random.default_rng(1)
    for _ in range(4):
        x, b, lab = ds.load(0, 64, rng)
        assert x.shape == (3, 64, 64) and x.dtype == np.float32
        assert len(b) == len(lab) and np.all(b[:, 2:] > 0) and np.all(b[:, :2] <= 1)


# --------------------------------------------------------------------- trainer
def _trainer(net, tmp_path, name):
    opt = SGDGroups(net.params)
    sched = V5Schedule(0.01, epochs=3, iters_per_epoch=2)
    return TrainerV5(net, opt, sched, size=32, log_path=tmp_path / f"{name}.csv", batch=2)


def test_trainer_resume_is_exact(tmp_path):
    samples = _samples(tmp_path)
    loader = DataLoader(V5Dataset(samples), 2, seed=0)
    a = _trainer(mini(), tmp_path, "a")
    a.fit(loader, 3)
    a.save_checkpoint(tmp_path / "ck.npz")
    b = _trainer(mini(rng=5), tmp_path, "b")
    b.load_checkpoint(tmp_path / "ck.npz")
    assert b.it == 3 and b.steps == a.steps and b._pending == a._pending
    a.fit(loader, 5)
    b.fit(loader, 5)
    for pa, pb in zip(a.net.params, b.net.params):
        for k in pa:
            assert np.allclose(pa[k], pb[k])
    for ea, eb in zip(a.ema.params, b.ema.params):
        for k in ea:
            assert np.allclose(ea[k], eb[k])
    assert (tmp_path / "a.csv").read_text().splitlines()[0].startswith("it,lr,momentum")


def test_trainer_v8_step(tmp_path):
    """A YOLOv8 network trains with the v8 loss and logs box / cls / dfl."""
    samples = _samples(tmp_path)
    loader = DataLoader(V5Dataset(samples), 2, seed=0)
    net = mini(yaml=MINI_V8)
    init_head_biases(net, img_size=32)
    tr = TrainerV5(net, SGDGroups(net.params), V5Schedule(0.01, epochs=3, iters_per_epoch=2),
                   size=32, log_path=tmp_path / "v8.csv", batch=2)
    assert tr.parts == ("box", "cls", "dfl")
    w0 = net.params[0]["W"].copy()
    tr.fit(loader, 2)
    head = (tmp_path / "v8.csv").read_text().splitlines()
    assert head[0] == "it,lr,momentum,size,loss,box,cls,dfl,seconds" and len(head) == 3
    assert not np.array_equal(net.params[0]["W"], w0)


@pytest.mark.slow
def test_overfit_one_image_v8(tmp_path):
    """T2.8 criterion on the v8 recipe: one image, no augmentation, box and DFL losses go
    down and the decoded box is back on the ground truth."""
    from yolo.infer.boxes import iou
    from yolo.infer.nms import postprocess

    samples = _samples(tmp_path, 1, texture=True)
    x, boxes, labels = V5Dataset(samples, train=False).load(0, 64)
    net = mini(size=64, yaml=MINI_V8)
    init_head_biases(net, img_size=64)
    opt = AdamWGroups(net.params, weight_decay=0)
    sched = V5Schedule(0.002, epochs=1500, iters_per_epoch=1, warmup_epochs=0)
    tr = TrainerV5(net, opt, sched, size=64, ema=False, nbs=1, batch=1)
    first = None
    for _ in range(1500):
        res, _ = tr.step(x[None], [boxes], [labels])
        first = first or res.parts["box"]
    assert res.parts["box"] < 0.01 * first
    b, s, lab = postprocess(net.forward(x[None], train=False), net.net, 0.25, 0.45)[0]
    assert lab[0] == 1 and iou(b[:1], boxes)[0, 0] > 0.95


@pytest.mark.slow
def test_overfit_one_image(tmp_path):
    """T2.8 criterion on the v5 recipe: one image, no augmentation, the box loss goes down
    and the decoded boxes are back on the ground truth."""
    from yolo.infer.boxes import iou
    from yolo.infer.nms import postprocess

    samples = _samples(tmp_path, 1, texture=True)
    ds = V5Dataset(samples, train=False)
    x, boxes, labels = ds.load(0, 64)
    net = mini(size=64)
    init_head_biases(net, img_size=64)
    opt = SGDGroups(net.params, weight_decay=0)
    sched = V5Schedule(0.05, epochs=1500, iters_per_epoch=1, warmup_epochs=0)
    tr = TrainerV5(net, opt, sched, size=64, ema=False, nbs=1, batch=1)
    first = None
    for _ in range(1500):
        res, _ = tr.step(x[None], [boxes], [labels])
        first = first or res.parts["box"]
    assert res.parts["box"] < 0.01 * first
    b, s, lab = postprocess(net.forward(x[None], train=False), net.net, 0.25, 0.45)[0]
    assert len(b) == 1 and lab[0] == 1 and iou(b, boxes)[0, 0] > 0.95
