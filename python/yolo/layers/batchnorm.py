"""Batch normalization par canal sur (N, H, W) — §4.2.

`state` = {"mean", "var"} : moyennes glissantes, mises à jour en place en entraînement et
utilisées en inférence.
"""

import numpy as np

from yolo.backend import get_xp


def bn_init_state(c, dtype=np.float64):
    return {"mean": np.zeros(c, dtype=dtype), "var": np.ones(c, dtype=dtype)}


def bn_forward(x, gamma, beta, state, train=True, momentum=0.9, eps=1e-5):
    """Renvoie (y, cache). En entraînement, statistiques du lot ; sinon moyennes glissantes."""
    axes = (0, 2, 3)
    if train:
        # §4.2 : μ_c, σ²_c sur M = N·H·W valeurs
        mu = x.mean(axis=axes)
        var = x.var(axis=axes)
        # §4.2 : μ_run ← m·μ_run + (1 − m)·μ_c (de même pour σ²)
        state["mean"] *= momentum
        state["mean"] += (1 - momentum) * mu
        state["var"] *= momentum
        state["var"] += (1 - momentum) * var
    else:
        mu, var = state["mean"], state["var"]
    inv_std = 1.0 / get_xp(var).sqrt(var + eps)
    x_hat = (x - mu[None, :, None, None]) * inv_std[None, :, None, None]
    y = gamma[None, :, None, None] * x_hat + beta[None, :, None, None]
    return y, (x_hat, gamma, inv_std, train)


def bn_backward(dy, cache):
    """§4.2 : dγ = Σ δy·x̂, dβ = Σ δy, δx = γ/(M√(σ²+ε)) (M δy − Σ δy − x̂ Σ δy·x̂)."""
    x_hat, gamma, inv_std, train = cache
    axes = (0, 2, 3)
    dgamma = (dy * x_hat).sum(axis=axes)
    dbeta = dy.sum(axis=axes)
    g = (gamma * inv_std)[None, :, None, None]
    if train:
        m = dy.size // dy.shape[1]
        dx = g / m * (m * dy - dbeta[None, :, None, None]
                      - x_hat * dgamma[None, :, None, None])
    else:
        # Inférence : transformation affine, μ et σ² fixes.
        dx = g * dy
    return dx, {"gamma": dgamma, "beta": dbeta}
