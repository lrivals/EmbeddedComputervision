"""Taux d'apprentissage : montée progressive puis paliers (§7.1).

Montée (*burn-in*) polynomiale lr·(it/burn_in)^power, comme Darknet (hors base) ; la spec
demande seulement une montée progressive, sans quoi le modèle diverge [1506.02640#005.1].
Paliers : le taux est multiplié par `scales[k]` à partir de l'itération `steps[k]`
(YOLOv2 : divisé par 10 aux époques 60 et 90 [1612.08242#005.3]).
"""


def lr_at(it, base_lr, burn_in=0, power=4.0, steps=(), scales=()):
    """Taux de l'itération `it` (0-indexée)."""
    if it < burn_in:
        return base_lr * ((it + 1) / burn_in) ** power
    lr = base_lr
    for step, scale in zip(steps, scales):
        if it >= step:
            lr *= scale
    return lr


class StepSchedule:
    """`lr_at` figé dans un objet appelable `schedule(it)`."""

    def __init__(self, base_lr, burn_in=0, power=4.0, steps=(), scales=()):
        if len(steps) != len(scales):
            raise ValueError("steps et scales doivent avoir la même longueur")
        self.kw = dict(base_lr=base_lr, burn_in=burn_in, power=power, steps=tuple(steps),
                       scales=tuple(scales))

    def __call__(self, it):
        return lr_at(it, **self.kw)
