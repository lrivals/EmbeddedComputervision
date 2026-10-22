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


class V5Schedule:
    """YOLOv5 / YOLOv7 learning rate (T33.7, `train.py` of the reference).

    Per epoch e: lr = lr0·lf(e), lf linear (1 − e/E)(1 − lrf) + lrf, or with `cos` the
    one-cycle (1 − cos(πe/E))/2·(lrf − 1) + 1. Warmup over the first
    n_w = max(round(warmup_epochs·iters_per_epoch), 100) iterations: each group's lr goes
    linearly from 0 (from `warmup_bias_lr` for the biases) to its scheduled value, the
    momentum from `warmup_momentum` to `momentum`. `__call__(it)` → ({group: lr}, momentum).
    """

    def __init__(self, lr0, epochs, iters_per_epoch, lrf=0.01, cos=False, warmup_epochs=3.0,
                 warmup_bias_lr=0.1, warmup_momentum=0.8, momentum=0.937):
        import math

        self.lr0, self.epochs, self.ipe = lr0, epochs, iters_per_epoch
        self.lrf, self.cos = lrf, cos
        self.n_warmup = max(round(warmup_epochs * iters_per_epoch), 100)
        self.warmup_bias_lr, self.warmup_momentum = warmup_bias_lr, warmup_momentum
        self.momentum = momentum
        self._math = math

    def lf(self, epoch):
        x = epoch / self.epochs
        if self.cos:
            return (1 - self._math.cos(x * self._math.pi)) / 2 * (self.lrf - 1) + 1
        return (1 - x) * (1 - self.lrf) + self.lrf

    def __call__(self, it):
        lr = self.lr0 * self.lf(it // self.ipe)
        if it >= self.n_warmup:
            return {g: lr for g in ("weights", "bn", "bias")}, self.momentum
        f = it / self.n_warmup
        lrs = {"weights": f * lr, "bn": f * lr,
               "bias": self.warmup_bias_lr + f * (lr - self.warmup_bias_lr)}
        return lrs, self.warmup_momentum + f * (self.momentum - self.warmup_momentum)

    def accumulate(self, it, batch, nbs=64):
        """Batches accumulated per optimizer step: ramps from 1 to nbs / batch in the warmup."""
        top = max(round(nbs / batch), 1)
        if it >= self.n_warmup:
            return top
        return max(1, round(1 + it / self.n_warmup * (top - 1)))
