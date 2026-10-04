"""Route : concaténation sur l'axe des canaux — §4.5.

Le cumul des gradients d'une carte lue par plusieurs couches est fait par le graphe
(`yolo.models.graph`).
"""

import numpy as np


def route_forward(xs):
    """y = [x_a; x_b; …] sur les canaux. Renvoie (y, cache)."""
    return np.concatenate(xs, axis=1), [x.shape[1] for x in xs]


def route_backward(dy, cache):
    """§4.5 : découpe δy en δx_a, δx_b, … Renvoie (liste des dx, {})."""
    splits = np.cumsum(cache)[:-1]
    return np.split(dy, splits, axis=1), {}
