"""Max pooling k×k, stride 2 ou 1 — §4.4.

Stride 1 (couche 11) : complétion de k − 1 lignes/colonnes à droite et en bas par
réplication du bord, ce qui conserve la taille et donne le même maximum qu'un −∞.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def maxpool_forward(x, k=2, s=2):
    """Renvoie (y, cache) ; le cache mémorise l'argmax de chaque fenêtre (§4.4)."""
    n, c, h, w = x.shape
    extra = k - 1 if s == 1 else 0
    xp = np.pad(x, ((0, 0), (0, 0), (0, extra), (0, extra)), mode="edge") if extra else x
    win = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]  # (N,C,Ho,Wo,k,k)
    ho, wo = win.shape[2:4]
    flat = win.reshape(n, c, ho, wo, k * k)
    # À égalité, argmax prend la première position : l'élément d'origine avant sa réplique.
    arg = flat.argmax(axis=-1)
    y = np.take_along_axis(flat, arg[..., None], axis=-1)[..., 0]
    return y, (arg, x.shape, xp.shape, k, s)


def maxpool_backward(dy, cache):
    """§4.4 : δx[argmax] += δy[i, j], 0 ailleurs. Renvoie (dx, {})."""
    arg, x_shape, xp_shape, k, s = cache
    n, c, ho, wo = dy.shape
    u, v = np.divmod(arg, k)
    rows = np.arange(ho)[None, None, :, None] * s + u
    cols = np.arange(wo)[None, None, None, :] * s + v
    nn = np.arange(n)[:, None, None, None]
    cc = np.arange(c)[None, :, None, None]
    dxp = np.zeros(xp_shape, dtype=dy.dtype)
    # Fenêtres disjointes en stride 2, chevauchantes en stride 1 : add.at cumule les doublons.
    np.add.at(dxp, (nn, cc, rows, cols), dy)
    h, w = x_shape[2:]
    if xp_shape[2] > h:
        # La complétion réplique le bord : son gradient revient au bord.
        dxp[:, :, h - 1, :] += dxp[:, :, h:, :].sum(axis=2)
        dxp[:, :, :, w - 1] += dxp[:, :, :, w:].sum(axis=3)
        dxp = dxp[:, :, :h, :w]
    return dxp, {}
