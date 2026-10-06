"""T11.4, T11.7 : entrée non carrée et entrée à un canal, du Python au noyau HLS en C-sim.

Réseaux synthétiques : cfg Tiny-YOLOv2 / v3 de tools/make_cfg.py à entrée 128 × 64 (largeur
× hauteur) et un canal, poids aléatoires, exportés avec leurs dumps. Le golden C++
(`golden_run`), le post-traitement matériel (Python, golden, noyau `yolo_post` : `tb_post`)
et le moteur conv derrière le driver (`tb_net`) doivent être égaux au Python à l'octet.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import jsonschema
import numpy as np
import pytest

from yolo.data.augment import IDENTITY, affine, augment
from yolo.data.letterbox import (as_hw, boxes_from_letterbox, boxes_to_letterbox, input_size,
                                 letterbox_image, letterbox_params, parse_size, size_label)
from yolo.data.targets import anchor_ref, anchors_frac, build_targets, heads
from yolo.infer import hw_postproc as hp
from yolo.infer.decode import decode, decode_head_int
from yolo.infer.pipeline import INTERPS, MODES, preprocess
from yolo.io.export import export_model, load_model
from yolo.models.tiny_yolo import CFG_DIR, CFG_FILES, build
from yolo.quant.calibrate import ActStats, choose_scales
from yolo.quant.fuse_bn import fuse_network
from yolo.quant.int_model import QuantModel
from yolo.quant.lut import HeadLuts
from yolo.quant.quantize import INPUT_SCALE
from yolo.train.loss import yolo_loss
from yolo.train.optim import SGD
from yolo.train.schedule import StepSchedule
from yolo.train.trainer import Trainer, scaled_input

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
GOLDEN = ROOT / "build" / "golden" / "golden_run"
HLS = ROOT / "build" / "hls"
SCHEMA = json.loads((ROOT / "docs" / "manifest.schema.json").read_text())
H, W, C = 64, 128, 1
CLASSES = 3
IMAGES = ["img0", "img1"]


def _cfg(tmp, base, channels=C):
    import make_cfg

    path = tmp / f"{base}-{W}x{H}-c{channels}.cfg"
    path.write_text(make_cfg.make_cfg((CFG_DIR / CFG_FILES[base]).read_text(), CLASSES,
                                      size=(H, W), channels=channels))
    return path


def _logit(x):
    x = np.clip(x, 1e-15, 1 - 1e-15)
    return np.log(x / (1 - x))


# ------------------------------------------------------------------- tailles et letterbox
def test_parse_size():
    assert parse_size("416") == 416 and parse_size("416x416") == 416
    assert parse_size("640x192") == (192, 640) and parse_size("640×192") == (192, 640)
    assert as_hw(416) == (416, 416) and as_hw((192, 640)) == (192, 640)
    assert size_label((192, 640)) == "640×192" and size_label(416) == "416×416"
    net = {"input": (3, 192, 640)}
    assert input_size(net) == (192, 640) and input_size(net, 320) == 320
    assert input_size({"input": (3, 416, 416)}) == 416


@pytest.mark.parametrize("w,h", [(1242, 375), (300, 400), (500, 500)])
def test_letterbox_nonsquare_reversible(w, h):
    rng = np.random.default_rng(0)
    boxes = np.concatenate([rng.uniform(0.1, 0.9, (20, 2)), rng.uniform(0.01, 0.5, (20, 2))], 1)
    for size in ((192, 640), (64, 128)):
        lb = boxes_to_letterbox(boxes, w, h, size)
        assert np.all(lb[:, :2] >= 0) and np.all(lb[:, :2] <= 1)
        np.testing.assert_allclose(boxes_from_letterbox(lb, w, h, size), boxes, atol=1e-6)


def test_letterbox_image_nonsquare_gray():
    from PIL import Image

    img = Image.fromarray(np.full((100, 100, 3), 200, np.uint8))
    out, (nw, nh, dx, dy) = letterbox_image(img, (64, 128), channels=1)
    assert out.shape == (64, 128, 1)
    assert (nw, nh, dx, dy) == letterbox_params(100, 100, (64, 128)) == (64, 64, 32, 0)
    assert np.all(out[:, :dx] == 0.5) and np.all(out[:, dx + nw:] == 0.5)
    np.testing.assert_allclose(out[:, dx:dx + nw], 200 / 255, atol=1e-6)


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("interp", INTERPS)
@pytest.mark.parametrize("channels", [1, 3])
def test_preprocess_shapes(mode, interp, channels):
    from PIL import Image

    img = Image.fromarray(np.random.default_rng(0).integers(0, 256, (75, 248, 3), np.uint8))
    x, wh = preprocess(img, (64, 128), mode, interp, channels)
    assert x.shape == (channels, 64, 128) and x.dtype == np.float32 and wh == (248, 75)
    # Un canal : luminance PIL (« L »), comme Image.convert.
    if channels == 1 and mode == "stretch" and interp == "pil":
        ref = np.asarray(img.convert("L").resize((128, 64), Image.BILINEAR)) / 255.0
        np.testing.assert_allclose(x[0], ref, atol=1e-6)


def test_identity_augment_nonsquare():
    from PIL import Image

    img = Image.fromarray(np.zeros((200, 300, 3), np.uint8))
    box = np.array([[0.4, 0.6, 0.3, 0.2]])
    m = affine(300, 200, (64, 128), IDENTITY)
    nw, nh, dx, dy = letterbox_params(300, 200, (64, 128))
    np.testing.assert_allclose(m, [[nw / 300, 0, dx], [0, nh / 200, dy]])
    out, b, _ = augment(img, box, [1], (64, 128), IDENTITY, channels=1)
    assert out.shape == (64, 128, 1)
    np.testing.assert_allclose(b, boxes_to_letterbox(box, 300, 200, (64, 128)), atol=1e-12)


# ------------------------------------------------------------ cibles, décodage, perte
def test_anchor_ref():
    assert anchor_ref({"input": (3, 416, 416)}) == (416, 416)
    assert anchor_ref({"input": (3, 320, 320)}) == (416, 416)  # carré : 416 à toute taille
    assert anchor_ref({"input": (1, 192, 640)}) == (640, 192)
    np.testing.assert_allclose(anchors_frac([[64, 48]], (640, 192)), [[0.1, 0.25]])


def test_decode_inverts_encode_nonsquare(tmp_path):
    spec = build(_cfg(tmp_path, "tiny-yolov3-voc")).net
    ref = anchor_ref(spec)
    assert ref == (W, H)
    rng = np.random.default_rng(0)
    boxes = np.concatenate([rng.uniform(0.05, 0.95, (6, 2)), rng.uniform(0.05, 0.6, (6, 2))], 1)
    hl = heads(spec)
    outs = build(_cfg(tmp_path, "tiny-yolov3-voc")).forward(np.zeros((1, C, H, W), np.float32))
    grids = {hid: outs[hid].shape[-2:] for hid, _ in hl}
    assert sorted(grids.values()) == [(2, 4), (4, 8)]
    t = build_targets([boxes], [np.arange(6) % CLASSES], spec["anchors"], hl, grids, ref)
    outputs = {}
    for hid, mask in hl:
        tt, (gh, gw) = t[hid], grids[hid]
        p = np.zeros((1, len(mask), 5 + CLASSES, gh, gw))
        p[:, :, 0], p[:, :, 1] = _logit(tt["x"]), _logit(tt["y"])
        p[:, :, 2], p[:, :, 3] = tt["tw"], tt["th"]
        p[:, :, 4] = np.where(tt["obj"], 10.0, -10.0)
        p[:, :, 5:] = -10.0
        for n, a, i, j in zip(*np.nonzero(tt["obj"])):
            p[n, a, 5 + tt["cls"][n, a, i, j], i, j] = 10.0
        outputs[hid] = p.reshape(1, -1, gh, gw)
    dboxes, obj, _ = decode(outputs, spec)
    got = dboxes[0, obj[0] > 0.5]
    # Chaque vérité retrouvée (deux vérités peuvent tomber sur la même cellule et ancre).
    d = np.abs(got[None] - boxes[:, None]).max(axis=2).min(axis=1)
    assert np.sum(d < 1e-9) >= len(got) >= 1


def test_loss_gradcheck_nonsquare(tmp_path):
    spec = build(_cfg(tmp_path, "tiny-yolov3-voc")).net
    hl = heads(spec)
    rng = np.random.default_rng(1)
    outs = {hid: rng.standard_normal((1, 3 * (5 + CLASSES), *g)) * 0.5
            for (hid, _), g in zip(hl, [(2, 4), (4, 8)])}
    gt = [np.array([[0.3, 0.6, 0.2, 0.5], [0.8, 0.3, 0.1, 0.2]])]
    lab = [np.array([0, 2])]
    kw = dict(anchor_ref=anchor_ref(spec))
    res = yolo_loss(outs, gt, lab, spec["anchors"], hl, CLASSES, **kw)
    eps = 1e-6
    for hid, _ in hl:
        for idx in [tuple(rng.integers(0, s) for s in outs[hid].shape) for _ in range(8)]:
            up = {k: v.copy() for k, v in outs.items()}
            dn = {k: v.copy() for k, v in outs.items()}
            up[hid][idx] += eps
            dn[hid][idx] -= eps
            num = (yolo_loss(up, gt, lab, spec["anchors"], hl, CLASSES, **kw).total
                   - yolo_loss(dn, gt, lab, spec["anchors"], hl, CLASSES, **kw).total) / (2 * eps)
            assert abs(num - res.douts[hid][idx]) < 1e-5 * max(1.0, abs(num)), (hid, idx)


def test_trainer_nonsquare_gray(tmp_path):
    net = build(_cfg(tmp_path, "tiny-yolov2-voc"), dtype=np.float64, rng=0)
    assert scaled_input(net.net, 416) == (H, W)  # échelle de référence : l'entrée de la cfg
    assert scaled_input(net.net, 608) == (96, 192)
    assert scaled_input({"input": (3, 416, 416)}, 352) == 352
    tr = Trainer(net, SGD(net.params), StepSchedule(1e-3, burn_in=2), size=(H, W))
    rng = np.random.default_rng(0)
    res, _ = tr.step(rng.uniform(0, 1, (2, C, H, W)), [np.array([[0.5, 0.5, 0.3, 0.6]])] * 2,
                     [np.array([1])] * 2)
    assert np.isfinite(res.total)


# --------------------------------------------------------- post-traitement matériel
def test_hw_postproc_single_cell_nonsquare(tmp_path):
    """Une seule cellule au-dessus du seuil, à une position et des t aléatoires : la boîte
    entière du noyau (repère W × H) == décodage entier flottant à moins d'un pixel Q4."""
    spec = build(_cfg(tmp_path, "tiny-yolov3-voc")).net
    ref = anchor_ref(spec)
    lut = HeadLuts(0.125)
    rng = np.random.default_rng(3)
    anchors = anchors_frac(spec["anchors"], ref)
    for (hid, mask), (gh, gw) in zip(heads(spec), [(2, 4), (4, 8)]):
        for _ in range(20):
            q = np.full((len(mask), 5 + CLASSES, gh, gw), -100, np.int8)
            a, i, j = rng.integers(len(mask)), rng.integers(gh), rng.integers(gw)
            q[a, :4, i, j] = rng.integers(-20, 20, 4)
            q[a, 4, i, j] = 40
            q[a, 5 + rng.integers(CLASSES), i, j] = 40
            q = q.reshape(-1, gh, gw)
            h = hp.HwHead(q, hp.anchors_q8(np.asarray(spec["anchors"])[mask]), CLASSES,
                          False, lut.scale, lut.exp_frac, np.asarray(lut.sigmoid, np.int64),
                          np.asarray(lut.exp, np.int64), np.asarray(lut.softmax_exp, np.int64),
                          ref)
            assert (1 << h.stride_log2) * gw == W and (1 << h.stride_log2) * gh == H
            boxes, _ = hp.run([h], 0.25, 0.45)
            b, _, _ = hp.to_detections(boxes, ref)
            want, _, _ = decode_head_int(q, anchors[mask], CLASSES, "v3", lut, 0.25)
            assert len(b) == 1 and len(want) == 1
            np.testing.assert_allclose(b[0], want[0], atol=1 / (16 * H))


def test_hw_stride_must_match():
    lut = HeadLuts(0.125)
    q = np.zeros((3 * 8, 2, 5), np.int8)  # 5 colonnes : 128 / 5 n'est pas entier
    h = hp.HwHead(q, np.ones((3, 2), np.int64), 3, False, lut.scale, lut.exp_frac,
                  lut.sigmoid, lut.exp, lut.softmax_exp, (W, H))
    with pytest.raises(ValueError):
        h.stride_log2


# ------------------------------------------------- export, golden C++, HLS (C-sim)
@pytest.fixture(scope="module", params=["tiny-yolov2-voc", "tiny-yolov3-voc"])
def exported(request, tmp_path_factory):
    from export_model import write_dumps

    tmp = tmp_path_factory.mktemp(request.param)
    cfg = _cfg(tmp, request.param)
    fused = fuse_network(build(cfg, dtype=np.float64, rng=1))
    rng = np.random.default_rng(0)
    xs = rng.uniform(0, 1, (len(IMAGES), C, H, W))
    stats = ActStats(fused, per_image=1024, rng=0)
    stats.update(fused.forward(xs, train=False, all_outputs=True))
    scales, _, _ = choose_scales(fused, stats)
    qm = QuantModel.from_fused(fused, INPUT_SCALE, scales)
    out = tmp / "model" / cfg.stem
    m = export_model(qm, out)
    reloaded, _ = load_model(out)
    write_dumps(reloaded, [{"id": k, "x": x} for k, x in zip(IMAGES, xs)], out)
    return cfg.stem, out, m, reloaded


def test_export_nonsquare_gray(exported):
    _, out, m, qm = exported
    jsonschema.Draft202012Validator(SCHEMA).validate(m)
    assert m["input"]["shape"] == [1, C, H, W]
    assert qm.net["input"] == (C, H, W)
    x = np.load(out / "dumps" / IMAGES[0] / "input.npy")
    assert x.shape == (1, C, H, W) and x.dtype == np.int8
    l0 = next(e for e in m["layers"] if e["type"] == "conv")
    assert l0["cin"] == C


def test_golden_run_equals_dumps(exported, tmp_path):
    if not GOLDEN.exists():
        pytest.skip("golden_run absent (make test-cpp)")
    _, out, _, _ = exported
    for image in IMAGES:
        d = out / "dumps" / image
        res = tmp_path / image
        subprocess.run([str(GOLDEN), "run", str(out), str(d / "input.npy"), str(res)],
                       check=True, capture_output=True)
        layers = sorted(d.glob("L*.npy"))
        assert layers
        for f in layers:
            assert np.array_equal(np.load(res / f.name), np.load(f)), (image, f.name)
        want = json.loads((d / "detections.json").read_text())
        got = json.loads((res / "detections.json").read_text())
        assert got["labels"] == want["labels"]
        np.testing.assert_allclose(np.array(got["boxes"]).reshape(-1, 4),
                                   np.array(want["boxes"]).reshape(-1, 4), atol=1e-12)


def test_hwpp_python_equals_golden_nonsquare(exported):
    if not GOLDEN.exists():
        pytest.skip("golden_run absent (make test-cpp)")
    from tests.test_hw_postproc import _golden

    _, out, m, qm = exported
    luts = {e["id"]: HeadLuts(e["scale"]) for e in m["layers"] if "lut_offset" in e}
    ids = [hid for hid, _ in hp.heads(qm.net)]
    rng = np.random.default_rng(1)
    cases = [({h: np.load(out / "dumps" / im / f"L{h:02d}.npy")[0] for h in ids}, 0.005)
             for im in IMAGES]
    for _ in range(4):
        shapes = {h: cases[0][0][h].shape for h in ids}
        cases.append(({h: rng.integers(-128, 128, shapes[h]).astype(np.int8) for h in ids},
                      float(rng.choice([0.005, 0.25]))))
    with tempfile.TemporaryDirectory() as tmp:
        for n, (hd, conf) in enumerate(cases):
            files = []
            for h in ids:
                f = Path(tmp) / f"{n}_{h}.npy"
                np.save(f, hd[h])
                files.append(f)
            want = hp.run(hp.make_heads({h: hd[h][None] for h in ids}, qm.net, luts), conf, 0.45)
            assert _golden(out, conf, 0.45, files) == (want[0], want[1]), (n, conf)


@pytest.mark.parametrize("tb", ["tb_post", "tb_net"])
def test_hls_csim_nonsquare_gray(exported, tb):
    """Noyaux `yolo_post` et `yolo_conv` (C-sim gcc, make csim-gcc) == golden à l'octet."""
    exe = HLS / tb
    if not exe.exists():
        pytest.skip(f"{exe} absent (make csim-gcc)")
    name, out, _, _ = exported
    args = [str(exe), "--model", str(out.parent), "--net", name]
    for image in IMAGES:
        args += ["--image", image]
    r = subprocess.run(args, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]


def test_perf_model_equals_csim_nonsquare_gray(exported, tmp_path):
    """Modèle de cycles (tools/perf_model.py, noyau actuel) == compteurs C-sim de `tb_net`,
    couche par couche, à cin = 1 et entrée non carrée (T11.7 : cycles de L00 à cin = 1)."""
    import importlib.util

    exe = HLS / "tb_net"
    if not exe.exists():
        pytest.skip(f"{exe} absent (make csim-gcc)")
    spec = importlib.util.spec_from_file_location("perf_model", ROOT / "tools" / "perf_model.py")
    pm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pm)
    name, out, _, _ = exported
    csv_path = tmp_path / "cycles.csv"
    subprocess.run([str(exe), "--model", str(out.parent), "--net", name, "--image", IMAGES[0],
                    "--csv", str(csv_path)], check=True, capture_output=True)
    rows = [line.split(",") for line in csv_path.read_text().splitlines()[1:]]
    layers = {d["layer"]: d for d in pm.conv_layers(out / "manifest.json")}
    assert layers[0]["cin"] == C and (layers[0]["h"], layers[0]["w"]) == (H, W)
    assert len(rows) == len(layers)
    for _, _, layer, overlapped, sequential, compute in rows:
        got = pm.kernel_cycles(layers[int(layer)])
        assert (got["overlapped"], got["sequential"], got["compute"]) == (
            int(overlapped), int(sequential), int(compute)), layer


def test_yolo_bench_hw_post_nonsquare_gray(exported, tmp_path):
    """Chaîne ARM (driver + noyaux C-sim, make sw-sim) : `yolo_bench --hw-post` sur l'entrée
    du dump == post-traitement matériel Python (repère W × H) ; colonnes de cycles de la
    NMS et débordements dans --times (T11.6)."""
    exe = ROOT / "build" / "sw" / "yolo_bench"
    if not exe.exists():
        pytest.skip(f"{exe} absent (make sw-sim)")
    name, out, m, qm = exported
    image = IMAGES[0]
    (tmp_path / "ids.txt").write_text(image + "\n")
    subprocess.run([str(exe), "--model", str(out), "--inputs",
                    str(out / "dumps" / image / "input.npy"), "--ids", str(tmp_path / "ids.txt"),
                    "--warmup", "0", "--hw-post", "--times", str(tmp_path / "t.csv"),
                    "--dets", str(tmp_path / "d.jsonl")], check=True, capture_output=True)
    head, row = (line.split(",") for line in (tmp_path / "t.csv").read_text().splitlines())
    stats = dict(zip(head, row))
    luts = {e["id"]: HeadLuts(e["scale"]) for e in m["layers"] if "lut_offset" in e}
    ids = [hid for hid, _ in hp.heads(qm.net)]
    outs = {h: np.load(out / "dumps" / image / f"L{h:02d}.npy") for h in ids}
    boxes, overflow = hp.run(hp.make_heads(outs, qm.net, luts), 0.005, 0.45)
    assert int(stats["overflow"]) == overflow
    assert int(stats["nms_cycles"]) > 0 and int(stats["post_cycles"]) >= int(stats["nms_cycles"])
    want_b, want_s, want_l = hp.to_detections(boxes, anchor_ref(qm.net))
    got = json.loads((tmp_path / "d.jsonl").read_text().splitlines()[0])
    assert got["labels"] == want_l.tolist()
    np.testing.assert_allclose(np.array(got["boxes"]).reshape(-1, 4), want_b, atol=1e-12)
    np.testing.assert_allclose(got["scores"], want_s, atol=1e-12)
