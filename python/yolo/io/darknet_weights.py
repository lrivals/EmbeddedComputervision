"""Chargeur des poids Darknet `.weights` (format hors base) — §7.1 (pré-entraînement).

Format : en-tête major, minor, revision (int32) puis `seen` (int64 si version ≥ 0.2, int32
sinon), soit 4 ou 5 mots de 32 bits. Puis, pour chaque convolution dans l'ordre du `.cfg` :
[β, γ, μ_run, σ²_run] si BN, sinon [biais], puis les poids (F, C, k, k) ; tout en float32.
"""

import numpy as np


def read_header(f):
    major, minor, revision = np.fromfile(f, dtype="<i4", count=3)
    seen_dtype = "<i8" if (major * 10 + minor) >= 2 and major < 1000 and minor < 1000 else "<i4"
    (seen,) = np.fromfile(f, dtype=seen_dtype, count=1)
    return {"version": (int(major), int(minor), int(revision)), "seen": int(seen)}


def load_darknet_weights(net, path):
    """Remplit `net.params` et `net.state` (moyennes glissantes BN) ; renvoie l'en-tête.

    Lève une erreur si le fichier est trop court ou s'il reste des octets non lus.
    """
    with open(path, "rb") as f:
        header = read_header(f)

        def read(shape):
            n = int(np.prod(shape))
            data = np.fromfile(f, dtype="<f4", count=n)
            if data.size != n:
                raise ValueError(f"{path} : fichier trop court ({data.size}/{n} valeurs)")
            return data.reshape(shape).astype(net.dtype)

        for i, layer in enumerate(net.layers):
            if layer["type"] != "conv":
                continue
            p = net.params[i]
            cout = layer["cout"]
            if layer["bn"]:
                p["beta"][:] = read(cout)
                p["gamma"][:] = read(cout)
                net.state[i]["mean"][:] = read(cout)
                net.state[i]["var"][:] = read(cout)
            else:
                p["b"][:] = read(cout)
            p["W"][:] = read(p["W"].shape)
        rest = len(f.read())
    if rest:
        raise ValueError(f"{path} : {rest} octets non lus après la dernière couche")
    return header


def save_darknet_weights(net, path, version=(0, 2, 0), seen=0):
    """Écriture au même format (tests, export vers Darknet)."""
    major, minor, revision = version
    with open(path, "wb") as f:
        np.array(version, dtype="<i4").tofile(f)
        seen_dtype = "<i8" if (major * 10 + minor) >= 2 else "<i4"
        np.array([seen], dtype=seen_dtype).tofile(f)
        for i, layer in enumerate(net.layers):
            if layer["type"] != "conv":
                continue
            p = net.params[i]
            if layer["bn"]:
                arrays = [p["beta"], p["gamma"], net.state[i]["mean"], net.state[i]["var"]]
            else:
                arrays = [p["b"]]
            for a in arrays + [p["W"]]:
                np.asarray(a, dtype="<f4").tofile(f)
