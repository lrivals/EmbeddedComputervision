"""Exponential moving average of the weights (T33.7, `ModelEMA` of YOLOv5).

    d(t) = decay·(1 − e^{−t/τ}),   θ_ema ← d θ_ema + (1 − d) θ      (t: optimizer steps)

on every parameter and BN running statistic. The EMA weights are the ones evaluated and
saved; `swapped(net)` puts them in the network for the duration of a `with` block.
"""

import math
from contextlib import contextmanager

from yolo.backend import get_xp


class ModelEMA:
    def __init__(self, net, decay=0.9999, tau=2000, updates=0):
        self.decay, self.tau, self.updates = decay, tau, updates
        self.params = [{k: get_xp(v).array(v, copy=True) for k, v in p.items()}
                       for p in net.params]
        self.state = [{k: get_xp(v).array(v, copy=True) for k, v in s.items()}
                      for s in net.state]

    def d(self, t=None):
        t = self.updates if t is None else t
        return self.decay * (1 - math.exp(-t / self.tau))

    def update(self, net):
        self.updates += 1
        d = self.d()
        for ema, cur in ((self.params, net.params), (self.state, net.state)):
            for e, c in zip(ema, cur):
                for k in e:
                    e[k] *= d
                    e[k] += (1 - d) * c[k]

    @contextmanager
    def swapped(self, net):
        """The EMA weights in `net` inside the block, the trained ones back after."""
        net.params, self.params = self.params, net.params
        net.state, self.state = self.state, net.state
        try:
            yield net
        finally:
            net.params, self.params = self.params, net.params
            net.state, self.state = self.state, net.state

    def state_dict(self):
        out = {"updates": self.updates}
        for name, part in (("param", self.params), ("state", self.state)):
            out.update({f"{name}/{i}/{k}": v for i, d in enumerate(part) for k, v in d.items()})
        return out

    def load_state_dict(self, sd):
        from yolo.backend import copy_into

        for key, value in sd.items():
            if key == "updates":
                self.updates = int(value)
                continue
            name, i, k = key.split("/")
            copy_into((self.params if name == "param" else self.state)[int(i)][k], value)
