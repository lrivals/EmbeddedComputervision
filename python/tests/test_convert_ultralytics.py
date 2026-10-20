"""T26.1: converted NumPy forward == PyTorch forward (≤ 1e-4) for YOLOv5n and YOLOv8n,
BN folded; skipped without torch. Random weights with non-trivial BN statistics: the official
weights are not stored in the repo (AGPL-3.0, see tools/convert_ultralytics.py)."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("convert_ultralytics",
                                               ROOT / "tools" / "convert_ultralytics.py")
cu = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cu)

from yolo.models import ul_graph  # noqa: E402


@pytest.mark.parametrize("net, params", [("yolov5n", 1.87e6), ("yolov8n", 3.15e6)])
def test_numpy_equals_torch(net, params):
    model = cu.build(net, seed=3)
    desc, w = cu.convert(model, net)
    n_params = sum(v.size for v in w.values())
    assert n_params == pytest.approx(params, rel=0.01)  # published sizes of the "n" models
    x = torch.rand(3, 3, 320, 320, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        want = [t.double().numpy() for t in model(x)]
    got = ul_graph.forward(desc, w, x.double().numpy())
    assert [g.shape for g in got] == [t.shape for t in want]
    for g, t in zip(got, want):
        assert np.abs(g - t).max() < 1e-4
    no = 3 * 85 if net == "yolov5n" else 64 + 80
    assert [g.shape[1] for g in got] == [no] * 3


def test_state_dict_round_trip(tmp_path):
    """A saved state dict loads back strictly (the names are those of the reference)."""
    model = cu.build("yolov8n", seed=1)
    path = tmp_path / "sd.pt"
    torch.save(model.state_dict(), path)
    other = cu.Model("yolov8n")
    cu.load_weights(other, path)
    assert all(torch.equal(a, b) for a, b in zip(model.state_dict().values(),
                                                  other.state_dict().values()))
    keys = list(model.state_dict())
    assert "model.0.conv.weight" in keys and "model.22.dfl.conv.weight" in keys
