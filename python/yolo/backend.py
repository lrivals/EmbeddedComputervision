"""Backend de calcul de l'entraînement : NumPy (CPU, défaut et référence) ou CuPy (GPU) — T12.11.

Le CPU reste le défaut. CuPy n'est importé que par `use("gpu")` (tools/train.py
--device gpu) ; sans cet appel, `python/yolo/` ne dépend que de NumPy. Pas de repli
silencieux : un GPU demandé mais indisponible arrête l'exécution avec un message clair.

Les couches choisissent leur module de tableaux d'après leurs entrées (`get_xp`) : un
tableau NumPy est toujours calculé par NumPy, à l'octet près comme avant. Seuls les
paramètres et le lot sont sur le GPU ; données, augmentation, cibles, perte et tout ce qui
est entier ou bit-exact restent sur le CPU.
"""

import os
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as _np_swv

_xp = np  # module des tableaux de l'entraînement, choisi une fois par `use`


def use(device):
    """Choisit le backend ("cpu" ou "gpu") ; rend le module de tableaux."""
    global _xp
    if device == "cpu":
        _xp = np
        return _xp
    if device != "gpu":
        raise ValueError(f"backend inconnu : {device!r} (cpu ou gpu)")
    os.environ.setdefault("CUPY_TF32", "0")  # float32 strict (comparaison CPU / GPU)
    try:
        import cupy
    except ImportError as exc:
        raise RuntimeError("--device gpu : CuPy absent (pip install cupy-cuda12x)") from exc
    try:
        n = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:  # noqa: BLE001  (pilote ou CUDA absent)
        raise RuntimeError(f"--device gpu : CUDA indisponible ({exc})") from exc
    if n < 1:
        raise RuntimeError("--device gpu : aucun GPU CUDA visible")
    _xp = cupy
    return _xp


def device():
    return "cpu" if _xp is np else "gpu"


def _cupy():
    return sys.modules.get("cupy")


def is_device_array(a):
    cp = _cupy()
    return cp is not None and isinstance(a, cp.ndarray)


def get_xp(*arrays):
    """cupy si l'un des tableaux est un `cupy.ndarray`, numpy sinon."""
    cp = _cupy()
    if cp is not None and any(isinstance(a, cp.ndarray) for a in arrays):
        return cp
    return np


def to_device(a):
    """Tableau sur le backend choisi par `use` (copie vers le GPU au besoin)."""
    return _xp.asarray(a)


def to_numpy(a):
    """Tableau NumPy (copie depuis le GPU au besoin)."""
    return a.get() if is_device_array(a) else np.asarray(a)


def copy_into(dst, src):
    """dst[...] = src, quel que soit le backend de chacun."""
    dst[...] = get_xp(dst).asarray(src)


def sliding_window_view(x, shape, axis):
    """`numpy.lib.stride_tricks.sliding_window_view` pour les deux backends (vue sans copie)."""
    xp = get_xp(x)
    if xp is np:
        return _np_swv(x, shape, axis=axis)
    swv = getattr(getattr(xp.lib, "stride_tricks", None), "sliding_window_view", None)
    if swv is not None:
        return swv(x, shape, axis=axis)
    # Repli : même vue par as_strided, fenêtres ajoutées en fin d'axes.
    axis = tuple(a % x.ndim for a in axis)
    out_shape = list(x.shape)
    for a, k in zip(axis, shape):
        out_shape[a] -= k - 1
    strides = x.strides + tuple(x.strides[a] for a in axis)
    return xp.lib.stride_tricks.as_strided(x, tuple(out_shape) + tuple(shape), strides)


def add_at(a, idx, v):
    """a[idx] += v en cumulant les indices répétés (`np.add.at`)."""
    if get_xp(a) is np:
        np.add.at(a, idx, v)
    else:
        import cupyx

        cupyx.scatter_add(a, idx, v)


__all__ = ["use", "device", "get_xp", "to_device", "to_numpy", "copy_into",
           "sliding_window_view", "add_at", "is_device_array"]
