"""T5.1 : le golden C++ relit le modèle exporté comme Python (couches, échelles, décalages,
blobs). Lance `build/golden/golden_run inspect` (construit par `make test-cpp`)."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from yolo.io.export import load_luts, load_model

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "build" / "golden" / "golden_run"
NETS = ["tiny-yolov2-voc", "tiny-yolov3-coco"]


def inspect(model_dir):
    out = subprocess.run([str(GOLDEN), "inspect", str(model_dir)], check=True,
                         capture_output=True, text=True).stdout
    head, anchors, buffers, layers = None, [], {}, {}
    for line in out.splitlines():
        tok = line.split()
        if tok[0] == "network":
            head = tok
        elif tok[0] == "anchor":
            anchors.append([float(tok[1]), float(tok[2])])
        elif tok[0] == "buffer":
            buffers[tok[1]] = int(tok[2])
        elif tok[0] == "layer":
            # layer <id> <type> out <C> <H> <W> puis des paires clé valeur
            fields = {"type": tok[2], "out": [int(t) for t in tok[4:7]],
                      **dict(zip(tok[7::2], tok[8::2]))}
            layers[int(tok[1])] = fields
    return head, anchors, buffers, layers


@pytest.mark.parametrize("net", NETS)
def test_inspect_matches_python(net):
    model_dir = ROOT / "model" / net
    if not GOLDEN.exists() or not (model_dir / "manifest.json").exists():
        pytest.skip("golden_run ou export absent (make test-cpp, make export)")
    check_inspect(model_dir)


def test_inspect_wbits4(tmp_path):
    """T10.10 : poids 4 bits paquetés dans weights.bin, dépaquetés par le golden C++."""
    from tests.test_export import _qm

    from yolo.io.export import export_model

    if not GOLDEN.exists():
        pytest.skip("golden_run absent (make test-cpp)")
    qm, _ = _qm("tiny-yolov2-voc", np.random.default_rng(0))
    for i in sorted(qm.convs)[1:3]:
        qm.convs[i].qW = np.clip(qm.convs[i].qW, -8, 7).astype(np.int8)
        qm.convs[i].wbits = 4
    export_model(qm, tmp_path)
    check_inspect(tmp_path)


def check_inspect(model_dir):
    head, anchors, buffers, layers = inspect(model_dir)
    qm, m = load_model(model_dir)
    luts = load_luts(model_dir, m)

    assert head[1] == m["network"] and int(head[3]) == m["classes"]
    assert [int(t) for t in head[5:8]] == m["input"]["shape"][1:]
    assert float(head[9]) == m["input"]["scale"]
    assert anchors == [[float(a) for a in pair] for pair in m["anchors"]]
    assert buffers == m["buffers"]
    assert sorted(layers) == [e["id"] for e in m["layers"]]
    for e in m["layers"]:
        got = layers[e["id"]]
        assert got["type"] == e["type"]
        assert got["out"] == e["out_shape"]
        if e["type"] == "conv":
            c = qm.convs[e["id"]]
            fp = e["fused_pool"] or {"k": 0, "s": 0}
            assert [int(got[k]) for k in ("k", "pad", "cin", "cout")] == \
                [e["k"], e["pad"], e["cin"], e["cout"]]
            assert int(got["leaky"]) == (e["act"] == "leaky")
            assert (int(got["pool_k"]), int(got["pool_s"])) == (fp["k"], fp["s"])
            assert int(got["shift"]) == e["shift"] == c.n
            assert float(got["in_scale"]) == e["in_scale"] == c.sx
            assert float(got["out_scale"]) == e["out_scale"] == c.sy
            w = c.qW.astype(np.int64).ravel()
            assert int(got["w_sum"]) == w.sum()
            assert int(got["w_first"]) == w[0] and int(got["w_last"]) == w[-1]
            assert int(got["b_sum"]) == c.qb.astype(np.int64).sum()
            assert int(got["b_first"]) == c.qb[0]
            assert int(got["m0_sum"]) == c.M0.astype(np.int64).sum()
            assert int(got["m0_first"]) == c.M0[0]
        elif e["type"] in ("yolo", "region"):
            assert float(got["scale"]) == e["scale"]
            assert int(got["exp_frac"]) == e["exp_frac"]
            assert int(got["lut_sum"]) == sum(int(t.astype(np.int64).sum())
                                              for t in luts[e["id"]].values())
        elif e["type"] in ("route", "upsample"):
            assert float(got["scale"]) == e["scale"]

