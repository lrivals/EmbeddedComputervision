"""Torch-free reader of PyTorch `.pt` files (T33.0, ADR 0001: no torch in `yolo/`).

A `.pt` file (torch ≥ 1.6) is a zip: `<root>/data.pkl` (the pickled object),
`<root>/data/<key>` (raw storages), `<root>/version`, sometimes `<root>/byteorder`. The root
folder name varies (`archive`, the file stem, ...).

The pickle is read by an `Unpickler` whose `find_class` is a whitelist:

- tensors: `torch._utils._rebuild_tensor_v2` / `_rebuild_parameter` → NumPy arrays, the
  storages coming from `persistent_load` (`("storage", type, key, location, numel)`);
- plain containers and small torch objects: `OrderedDict`, `set`, `torch.Size` (tuple),
  `torch.device` and dtype objects (strings);
- modules (`torch.nn.modules.*`, `models.*` of the yolov5 repo, `ultralytics.nn.*`) →
  `_Module` stubs that only keep their state: the real classes are never imported.

Anything else raises `pickle.UnpicklingError` before being called: a `.pt` downloaded from the
web is untrusted data, the whitelist is the protection.

Ultralytics checkpoints are `{"model": Module, "ema": Module | None, "epoch", ...}`, saved
in float16; `load_checkpoint` walks the module tree into a state dict with the torch names
(`model.2.cv1.conv.weight`, ...) and casts floating arrays to float32 (exact).
"""

import collections
import pickle
import zipfile
from pathlib import Path

import numpy as np

# Storage classes of torch → NumPy dtype (`UntypedStorage`: raw bytes).
STORAGE_DTYPES = {
    "DoubleStorage": np.float64, "FloatStorage": np.float32, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
    "UntypedStorage": np.uint8,
}
TORCH_DTYPES = ("float64", "float32", "float16", "int64", "int32", "int16", "int8", "uint8",
                "bool", "double", "float", "half", "long", "int")
MODULE_PREFIXES = ("torch.nn.modules.", "models.", "ultralytics.nn.")


class _StorageType:
    def __init__(self, name):
        self.name, self.dtype = name, np.dtype(STORAGE_DTYPES[name])

    def __repr__(self):
        return f"torch.{self.name}"


class _Module:
    """Stand-in for a pickled `nn.Module`: NEWOBJ + BUILD only fill `__dict__`."""

    _torch_class = "?"

    def __repr__(self):
        return f"<{self._torch_class}>"


_STUBS = {}


def _stub(module, name):
    key = f"{module}.{name}"
    if key not in _STUBS:
        _STUBS[key] = type(name, (_Module,), {"_torch_class": key})
    return _STUBS[key]


def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad=False, hooks=None,
                       metadata=None):
    a = storage
    if not size:
        return a[offset:offset + 1].reshape(()).copy()
    item = a.itemsize
    view = np.lib.stride_tricks.as_strided(a[offset:], tuple(size),
                                           tuple(s * item for s in stride), writeable=False)
    return view.copy()


def _rebuild_parameter(data, requires_grad=False, hooks=None, state=None):
    return data


def _torch_size(*args):
    return tuple(*args) if args else ()


def _torch_device(*args):
    return ":".join(str(a) for a in args)


SAFE = {
    ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
    ("torch._utils", "_rebuild_parameter"): _rebuild_parameter,
    ("torch._utils", "_rebuild_parameter_with_state"): _rebuild_parameter,
    ("collections", "OrderedDict"): collections.OrderedDict,
    ("builtins", "set"): set,
    ("__builtin__", "set"): set,
    ("torch", "Size"): _torch_size,
    ("torch", "device"): _torch_device,
}


class _Unpickler(pickle.Unpickler):
    def __init__(self, file, zf, root, byteorder="little"):
        super().__init__(file)
        self.zf, self.root, self.order = zf, root, "<" if byteorder == "little" else ">"
        self.storages = {}

    def find_class(self, module, name):
        if (module, name) in SAFE:
            return SAFE[(module, name)]
        if module == "torch" and name in STORAGE_DTYPES:
            return _StorageType(name)
        if module == "torch" and name in TORCH_DTYPES:
            return f"torch.{name}"
        if module.startswith(MODULE_PREFIXES):
            return _stub(module, name)
        raise pickle.UnpicklingError(f"global {module}.{name} refused (not whitelisted)")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage"
                and isinstance(pid[1], _StorageType)):
            raise pickle.UnpicklingError(f"unsupported persistent id {pid!r}")
        _, stype, key, _location, numel = pid
        if key not in self.storages:
            raw = self.zf.read(f"{self.root}/data/{key}")
            a = np.frombuffer(raw, dtype=stype.dtype.newbyteorder(self.order))
            if a.size != numel and stype.name != "UntypedStorage":
                raise pickle.UnpicklingError(f"storage {key}: {a.size} items, {numel} expected")
            self.storages[key] = a.astype(stype.dtype, copy=False)
        return self.storages[key]


def unpickle(path):
    """The object pickled in the `.pt` file `path` (modules as `_Module` stubs)."""
    with zipfile.ZipFile(path) as zf:
        pkl = [n for n in zf.namelist() if n.endswith("/data.pkl") and n.count("/") == 1]
        if len(pkl) != 1:
            raise pickle.UnpicklingError(f"{path}: no <root>/data.pkl (legacy format?)")
        root = pkl[0].split("/")[0]
        order = "little"
        if f"{root}/byteorder" in zf.namelist():
            order = zf.read(f"{root}/byteorder").decode().strip()
        with zf.open(pkl[0]) as f:
            return _Unpickler(f, zf, root, order).load()


def module_state_dict(m, prefix=""):
    """State dict of a `_Module` tree: `_parameters` and persistent `_buffers`, dotted names."""
    out = collections.OrderedDict()
    d = m.__dict__
    for name, p in d.get("_parameters", {}).items():
        if p is not None:
            out[prefix + name] = p
    skip = d.get("_non_persistent_buffers_set", set())
    for name, b in d.get("_buffers", {}).items():
        if b is not None and name not in skip:
            out[prefix + name] = b
    for name, child in d.get("_modules", {}).items():
        if child is not None:
            out.update(module_state_dict(child, f"{prefix}{name}."))
    return out


def _cast(sd, cast):
    if cast is None:
        return sd
    return collections.OrderedDict(
        (k, v.astype(cast) if np.issubdtype(v.dtype, np.floating) else v) for k, v in sd.items())


def load_checkpoint(path, cast=np.float32):
    """(state dict, meta) of an Ultralytics checkpoint, or of a pickled state dict.

    Uses `ema` when present, else `model`. `meta`: `yaml` (model YAML dict), `stride`,
    `names`, and for anchor heads `anchors` in pixels per level (the checkpoint stores them
    in stride units). `cast`: dtype of the floating arrays (None keeps float16).
    """
    obj = unpickle(path)
    meta = {}
    if isinstance(obj, dict) and ("model" in obj or "ema" in obj):
        obj = obj.get("ema") or obj["model"]
    if isinstance(obj, _Module):
        d = obj.__dict__
        sd = module_state_dict(obj)
        meta = {"yaml": d.get("yaml"), "names": d.get("names")}
        stride = d.get("stride")
        if stride is not None:
            meta["stride"] = [float(s) for s in np.asarray(stride)]
        anchors = [k for k in sd if k.endswith(".anchors")]
        if anchors and "stride" in meta:
            a = sd[anchors[0]].astype(np.float64)  # (nl, na, 2), stride units
            px = a * np.asarray(meta["stride"])[:, None, None]
            meta["anchors"] = px.reshape(len(px), -1).tolist()
    elif isinstance(obj, dict):
        sd = collections.OrderedDict((k, v) for k, v in obj.items() if isinstance(v, np.ndarray))
    else:
        raise pickle.UnpicklingError(f"{path}: neither a checkpoint nor a state dict")
    return _cast(sd, cast), meta


def load_pt(path, cast=np.float32):
    """State dict (`name -> np.ndarray`, torch names) of a checkpoint or state dict file."""
    return load_checkpoint(Path(path), cast)[0]
