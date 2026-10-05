"""SGD avec momentum et weight decay (§7.1).

    v ← μ v − lr (g + wd·p) ;  p ← p + v          (μ = 0,9, wd = 5e-4)

Pas de weight decay sur γ, β et les biais (convention usuelle, hors base).
"""

from yolo.backend import copy_into, get_xp

NO_DECAY = ("gamma", "beta", "b")


class SGD:
    """`params` : liste de dicts de tableaux (`Network.params`), mis à jour en place."""

    def __init__(self, params, momentum=0.9, weight_decay=5e-4, no_decay=NO_DECAY):
        self.params = params
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.no_decay = set(no_decay)
        self.velocity = [{k: get_xp(v).zeros_like(v) for k, v in p.items()} for p in params]

    def step(self, grads, lr):
        mu = self.momentum
        for p, g, v in zip(self.params, grads, self.velocity):
            for k in p:
                d = g[k]
                if self.weight_decay and k not in self.no_decay:
                    d = d + self.weight_decay * p[k]
                # §7.1 : v = μ v − lr (g + wd p) ; p += v
                v[k] *= mu
                v[k] -= lr * d
                p[k] += v[k]

    def state_dict(self):
        return {f"{i}/{k}": v for i, vel in enumerate(self.velocity) for k, v in vel.items()}

    def load_state_dict(self, state):
        for key, value in state.items():
            i, k = key.split("/")
            copy_into(self.velocity[int(i)][k], value)
