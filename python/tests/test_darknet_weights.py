"""T1.9 : chargeur de poids Darknet."""

from pathlib import Path

import numpy as np
import pytest

from yolo.io.darknet_weights import load_darknet_weights, save_darknet_weights
from yolo.models.tiny_yolo import build

WEIGHTS = Path(__file__).resolve().parents[2] / "weights"


@pytest.mark.parametrize("version", [(0, 1, 0), (0, 2, 0)])  # en-tête de 4 ou 5 mots
def test_roundtrip(tmp_path, version):
    src = build("tiny-yolov3-voc", rng=0)
    for st in src.state:
        if st:
            st["mean"][:] = np.random.default_rng(1).standard_normal(st["mean"].shape)
            st["var"][:] = 1.5
    path = tmp_path / "net.weights"
    save_darknet_weights(src, path, version=version, seen=1234)
    dst = build("tiny-yolov3-voc", rng=2)
    header = load_darknet_weights(dst, path)
    assert header == {"version": version, "seen": 1234}
    for i in range(len(src.layers)):
        for k, v in src.params[i].items():
            np.testing.assert_array_equal(dst.params[i][k], v)
        for k, v in src.state[i].items():
            np.testing.assert_array_equal(dst.state[i][k], v)


def test_size_mismatch(tmp_path):
    path = tmp_path / "net.weights"
    save_darknet_weights(build("tiny-yolov3-voc", rng=0), path)
    with pytest.raises(ValueError, match="trop court"):
        load_darknet_weights(build("tiny-yolov3-coco", rng=0), path)
    with path.open("ab") as f:
        f.write(b"\0" * 8)
    with pytest.raises(ValueError, match="non lus"):
        load_darknet_weights(build("tiny-yolov3-voc", rng=0), path)


@pytest.mark.parametrize("name,file", [("tiny-yolov3-coco", "yolov3-tiny.weights"),
                                       ("tiny-yolov2-voc", "yolov2-tiny-voc.weights")])
def test_pretrained(name, file):
    path = WEIGHTS / file
    if not path.exists():
        pytest.skip(f"{path} absent (tools/get_weights.sh)")
    net = build(name)
    load_darknet_weights(net, path)  # lève une erreur si le fichier n'est pas lu exactement
    assert all(np.all(st["var"] > 0) for st in net.state if st)
    x = np.random.default_rng(0).uniform(0, 1, (1, 3, 416, 416)).astype(np.float32)
    out = net.forward(x, train=False)
    assert all(np.all(np.isfinite(y)) for y in out.values())
