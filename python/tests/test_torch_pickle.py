"""T33.0: torch-free `.pt` reader.

The synthetic files are assembled opcode by opcode (protocol 2, as torch writes them), so the
tests need neither torch nor a file made by it. The official checkpoints are compared with
the state dicts read by torch once (`weights/m33/<net>_sd.npz`, made by the one-off fixture
script of M33); those tests are skipped when the files are absent (AGPL weights, not in git).
"""

import pickle
import struct
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from yolo.io import torch_pickle as tp

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "weights" / "m33"
CHECKPOINTS = ROOT / "weights" / "ultralytics"


class Pickle:
    """Minimal protocol-2 pickle writer (the opcodes torch's pickles use)."""

    def __init__(self):
        self.b = bytearray(b"\x80\x02")

    def glob(self, module, name):
        self.b += f"c{module}\n{name}\n".encode()
        return self

    def str(self, s):
        e = s.encode()
        self.b += b"X" + struct.pack("<I", len(e)) + e
        return self

    def int(self, v):
        self.b += b"J" + struct.pack("<i", v)
        return self

    def op(self, code):
        self.b += {"mark": b"(", "tuple": b"t", "reduce": b"R", "persid": b"Q", "dict": b"}",
                   "setitems": b"u", "false": b"\x89", "etuple": b")", "newobj": b"\x81",
                   "build": b"b", "mlist": b"l", "tuple1": b"\x85"}[code]
        return self

    def ints(self, vals):
        self.op("mark")
        for v in vals:
            self.int(v)
        return self.op("tuple")

    def odict(self):
        return self.glob("collections", "OrderedDict").op("etuple").op("reduce")

    def tensor(self, storage, key, numel, offset, size, stride):
        self.glob("torch._utils", "_rebuild_tensor_v2").op("mark")
        self.op("mark").str("storage").glob("torch", storage).str(key).str("cpu").int(numel)
        self.op("tuple").op("persid")
        self.int(offset).ints(size).ints(stride).op("false").odict()
        return self.op("tuple").op("reduce")

    def bytes(self):
        return bytes(self.b + b".")


def write_pt(path, pkl, storages, root="archive"):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{root}/data.pkl", pkl)
        for key, arr in storages.items():
            z.writestr(f"{root}/data/{key}", np.ascontiguousarray(arr).astype(
                arr.dtype.newbyteorder("<")).tobytes())
        z.writestr(f"{root}/version", "3\n")
    return path


S0 = np.arange(6, dtype=np.float32) * 1.5 - 2
S1 = (np.arange(10) / 7).astype(np.float16)
S2 = np.array([3, -1, 2**40], dtype=np.int64)


def state_dict_pickle():
    """{"a": S0 as 2×3, "t": its transpose view, "h": S1[4:7], "n": S2} as a state dict."""
    p = Pickle().odict().op("mark")
    p.str("a").tensor("FloatStorage", "0", 6, 0, (2, 3), (3, 1))
    p.str("t").tensor("FloatStorage", "0", 6, 0, (3, 2), (1, 3))
    p.str("h").tensor("HalfStorage", "1", 10, 4, (3,), (1,))
    p.str("n").tensor("LongStorage", "2", 3, 0, (3,), (1,))
    return p.op("setitems").bytes()


@pytest.fixture
def sd_file(tmp_path):
    return write_pt(tmp_path / "sd.pt", state_dict_pickle(), {"0": S0, "1": S1, "2": S2},
                    root="sd")


def test_state_dict_tensors(sd_file):
    sd = tp.load_pt(sd_file, cast=None)
    assert list(sd) == ["a", "t", "h", "n"]
    assert np.array_equal(sd["a"], S0.reshape(2, 3)) and sd["a"].dtype == np.float32
    assert np.array_equal(sd["t"], S0.reshape(2, 3).T) and sd["t"].flags["C_CONTIGUOUS"]
    assert sd["h"].dtype == np.float16 and np.array_equal(sd["h"], S1[4:7])
    assert sd["n"].dtype == np.int64 and sd["n"][2] == 2**40


def test_float16_cast_is_exact(sd_file):
    sd = tp.load_pt(sd_file)
    assert sd["h"].dtype == np.float32 and sd["n"].dtype == np.int64
    assert np.array_equal(sd["h"].astype(np.float16), S1[4:7])  # round trip: no rounding


def test_module_tree(tmp_path):
    """A checkpoint {"model": Model} whose module tree is walked into torch names."""
    p = Pickle().op("dict").op("mark").str("model")
    p.glob("models.yolo", "Model").op("etuple").op("newobj")
    p.op("dict").op("mark")
    p.str("_parameters").odict().op("mark").str("w")
    p.tensor("FloatStorage", "0", 6, 0, (2, 3), (3, 1)).op("setitems")
    p.str("_buffers").odict().op("mark").str("running_mean")
    p.tensor("FloatStorage", "0", 6, 1, (2,), (1,))
    p.str("tmp").tensor("FloatStorage", "0", 6, 0, (1,), (1,)).op("setitems")
    p.str("_non_persistent_buffers_set").glob("__builtin__", "set")
    p.op("mark").str("tmp").op("mlist").op("tuple1").op("reduce")  # set(["tmp"])
    p.str("_modules").odict().op("mark").str("0")
    p.glob("torch.nn.modules.conv", "Conv2d").op("etuple").op("newobj")
    p.op("dict").op("mark").str("_parameters").odict().op("mark").str("weight")
    p.tensor("HalfStorage", "1", 10, 0, (2, 1, 1, 1), (1, 1, 1, 1)).op("setitems")
    p.op("setitems").op("build").op("setitems")
    p.str("stride").tensor("FloatStorage", "3", 3, 0, (3,), (1,))
    p.op("setitems").op("build")
    p.str("epoch").int(-1).op("setitems")
    path = write_pt(tmp_path / "ck.pt", p.bytes(),
                    {"0": S0, "1": S1, "3": np.array([8, 16, 32], np.float32)})
    sd, meta = tp.load_checkpoint(path)
    assert list(sd) == ["w", "running_mean", "0.weight"]  # "tmp" is not persistent
    assert np.array_equal(sd["running_mean"], S0[1:3])
    assert sd["0.weight"].dtype == np.float32
    assert meta["stride"] == [8.0, 16.0, 32.0]


def test_refuses_unlisted_global(tmp_path):
    marker = tmp_path / "pwned"
    p = Pickle().glob("os", "system").str(f"touch {marker}").op("tuple1").op("reduce")
    path = write_pt(tmp_path / "evil.pt", p.bytes(), {})
    with pytest.raises(pickle.UnpicklingError, match="os.system"):
        tp.unpickle(path)
    assert not marker.exists()


@pytest.mark.parametrize("module,name", [("builtins", "eval"), ("torch", "load"),
                                         ("numpy", "load"), ("torch._utils", "_rebuild_qtensor")])
def test_refuses_other_globals(tmp_path, module, name):
    path = write_pt(tmp_path / "x.pt", Pickle().glob(module, name).bytes(), {})
    with pytest.raises(pickle.UnpicklingError):
        tp.unpickle(path)


def test_refuses_bad_persistent_id(tmp_path):
    p = Pickle().op("mark").str("file").str("/etc/passwd").op("tuple").op("persid")
    with pytest.raises(pickle.UnpicklingError, match="persistent id"):
        tp.unpickle(write_pt(tmp_path / "x.pt", p.bytes(), {}))


def test_runs_without_torch(sd_file):
    code = ("import sys; sys.modules['torch'] = None\n"
            "from yolo.io.torch_pickle import load_pt\n"
            f"sd = load_pt({str(sd_file)!r})\n"
            "assert 'torch' not in [m for m in sys.modules if sys.modules[m] is not None]\n"
            "print(sorted(sd))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=ROOT / "python", check=True)
    assert out.stdout.strip() == "['a', 'h', 'n', 't']"


@pytest.mark.parametrize("net", ["yolov5n", "yolov8n"])
def test_official_checkpoint_bit_equal(net):
    ref_path, pt = FIXTURES / f"{net}_sd.npz", CHECKPOINTS / f"{net}.pt"
    if not (ref_path.exists() and pt.exists()):
        pytest.skip("official weights or fixtures absent (M33 step 0)")
    ref = np.load(ref_path)
    raw = tp.load_pt(pt, cast=None)
    assert list(raw) == ref.files
    for k in ref.files:
        assert raw[k].dtype == ref[k].dtype and raw[k].shape == ref[k].shape, k
        assert raw[k].tobytes() == ref[k].tobytes(), k
    f32 = tp.load_pt(pt)
    for k in ref.files:  # the float16 → float32 cast is the only rounding (none)
        want = ref[k].astype(np.float32) if ref[k].dtype.kind == "f" else ref[k]
        assert np.array_equal(f32[k], want), k
    _, meta = tp.load_checkpoint(pt)
    assert meta["stride"] == [8.0, 16.0, 32.0] and len(meta["names"]) == 80
    if net == "yolov5n":
        assert meta["anchors"][0] == [10, 13, 16, 30, 33, 23]
