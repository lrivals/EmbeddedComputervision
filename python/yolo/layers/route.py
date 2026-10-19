"""Route : concaténation sur l'axe des canaux — §4.5.

Le cumul des gradients d'une carte lue par plusieurs couches est fait par le graphe
(`yolo.models.graph`).
"""

import numpy as np

from yolo.backend import get_xp


def route_forward(xs, groups=1, group_id=0):
    """y = [x_a; x_b; …] sur les canaux. Renvoie (y, cache).

    `groups` > 1 (CSP de YOLOv4-tiny) : une seule source, dont y garde le groupe de canaux
    `group_id` sur `groups`.
    """
    if groups > 1:
        (x,) = xs
        c = x.shape[1] // groups
        return x[:, group_id * c:(group_id + 1) * c], ("group", x.shape, group_id * c, c)
    return get_xp(*xs).concatenate(xs, axis=1), [x.shape[1] for x in xs]


def route_backward(dy, cache):
    """§4.5 : découpe δy en δx_a, δx_b, … ; route groupée : δy dans un δx nul hors du
    groupe. Renvoie (liste des dx, {})."""
    if isinstance(cache, tuple) and cache[0] == "group":
        _, shape, off, c = cache
        dx = get_xp(dy).zeros(shape, dtype=dy.dtype)
        dx[:, off:off + c] = dy
        return [dx], {}
    splits = np.cumsum(cache)[:-1].tolist()
    return get_xp(dy).split(dy, splits, axis=1), {}
