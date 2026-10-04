"""Convolution 2D, NCHW, poids (F, C, k, k) — §4.1.

Deux implémentations aux mêmes conventions :
- `conv_forward_naive` / `conv_backward_naive` : boucles du §4.1, référence de test ;
- `conv_forward` / `conv_backward` : im2col + produit matriciel, utilisée en entraînement.

`b=None` : convolution sans biais (suivie d'une BN, §4.2).
"""

import itertools

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _out_size(n, k, s, pad):
    # §4.1 : H_o = ⌊(H + 2P − k)/s⌋ + 1
    return (n + 2 * pad - k) // s + 1


def _pad(x, pad):
    if pad == 0:
        return x
    return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)))


def _unpad(xp, pad):
    if pad == 0:
        return xp
    return xp[:, :, pad:-pad, pad:-pad]


def conv_forward_naive(x, W, b=None, s=1, pad=None):
    """§4.1, boucles explicites. Renvoie (y, cache)."""
    n_, c_, h, w = x.shape
    f_, _, k, _ = W.shape
    pad = k // 2 if pad is None else pad
    xp = _pad(x, pad)
    ho, wo = _out_size(h, k, s, pad), _out_size(w, k, s, pad)
    y = np.zeros((n_, f_, ho, wo), dtype=np.result_type(x, W))
    for n, f, i, j in itertools.product(range(n_), range(f_), range(ho), range(wo)):
        acc = 0.0 if b is None else b[f]
        acc += np.sum(W[f] * xp[n, :, i * s:i * s + k, j * s:j * s + k])
        y[n, f, i, j] = acc
    return y, (xp, W, b is not None, s, pad)


def conv_backward_naive(dy, cache):
    """§4.1, boucles explicites. Renvoie (dx, {"W", "b"?})."""
    xp, W, has_b, s, pad = cache
    n_, f_, ho, wo = dy.shape
    k = W.shape[2]
    dW = np.zeros_like(W)
    dxp = np.zeros_like(xp)
    for n, f, i, j in itertools.product(range(n_), range(f_), range(ho), range(wo)):
        g = dy[n, f, i, j]
        dW[f] += g * xp[n, :, i * s:i * s + k, j * s:j * s + k]
        dxp[n, :, i * s:i * s + k, j * s:j * s + k] += g * W[f]
    grads = {"W": dW}
    if has_b:
        grads["b"] = dy.sum(axis=(0, 2, 3))
    return _unpad(dxp, pad), grads


def conv_forward(x, W, b=None, s=1, pad=None):
    """§4.1 par im2col : Y = W[F×Ck²] · X_col[Ck²×HoWo]. Renvoie (y, cache)."""
    k = W.shape[2]
    pad = k // 2 if pad is None else pad
    xp = _pad(x, pad)
    # Fenêtres (N, C, Ho, Wo, k, k) : vue sans copie, sous-échantillonnée par le stride.
    cols = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    y = np.tensordot(cols, W, axes=([1, 4, 5], [1, 2, 3]))  # (N, Ho, Wo, F)
    y = np.ascontiguousarray(y.transpose(0, 3, 1, 2))
    if b is not None:
        y += b[None, :, None, None]
    return y, (xp, W, b is not None, s, pad)


def conv_backward(dy, cache):
    """§4.1 : dW = corrélation de δY avec X_p, δX_p = Σ_f δY·W (col2im). Renvoie (dx, grads)."""
    xp, W, has_b, s, pad = cache
    _, _, ho, wo = dy.shape
    k = W.shape[2]
    cols = sliding_window_view(xp, (k, k), axis=(2, 3))[:, :, ::s, ::s]
    dW = np.tensordot(dy, cols, axes=([0, 2, 3], [0, 2, 3]))  # (F, C, k, k)
    # δX_p[n,c,is+u,js+v] += Σ_f δY[n,f,i,j] W[f,c,u,v] : une tranche par (u, v).
    dxp = np.zeros_like(xp, dtype=np.result_type(dy, W))
    dcols = np.tensordot(dy, W, axes=([1], [0]))  # (N, Ho, Wo, C, k, k)
    dcols = dcols.transpose(0, 3, 1, 2, 4, 5)  # (N, C, Ho, Wo, k, k)
    for u in range(k):
        for v in range(k):
            dxp[:, :, u:u + s * ho:s, v:v + s * wo:s] += dcols[..., u, v]
    grads = {"W": dW}
    if has_b:
        grads["b"] = dy.sum(axis=(0, 2, 3))
    return _unpad(dxp, pad), grads
