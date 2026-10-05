"""Route : concaténation sur l'axe des canaux — §4.5.

Le cumul des gradients d'une carte lue par plusieurs couches est fait par le graphe
(`yolo.models.graph`).
"""

import numpy as np

from yolo.backend import get_xp


def route_forward(xs):
    """y = [x_a; x_b; …] sur les canaux. Renvoie (y, cache)."""
    return get_xp(*xs).concatenate(xs, axis=1), [x.shape[1] for x in xs]


def route_backward(dy, cache):
    """§4.5 : découpe δy en δx_a, δx_b, … Renvoie (liste des dx, {})."""
    splits = np.cumsum(cache)[:-1].tolist()
    return get_xp(dy).split(dy, splits, axis=1), {}
