"""Optimiseurs (§7.1) : SGD avec momentum et weight decay, AdamW.

SGD (référence Darknet) :

    v ← μ v − lr (g + wd·p) ;  p ← p + v          (μ = 0,9, wd = 5e-4)

AdamW (Loshchilov et Hutter, 2019 ; mêmes conventions que `torch.optim.AdamW`) :

    m ← β₁ m + (1 − β₁) g ;  v ← β₂ v + (1 − β₂) g²
    p ← p − lr·wd·p − lr · m̂ / (√v̂ + ε)          (m̂ = m / (1 − β₁ᵗ), v̂ = v / (1 − β₂ᵗ))

Le weight decay d'AdamW est découplé du gradient : il n'est pas normalisé par √v̂, et le
même `wd` vaut beaucoup moins qu'en SGD (défaut 1e-2). Pas de weight decay sur γ, β et les
biais (convention usuelle, hors base).
"""

import numpy as np

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
            i, _, k = key.partition("/")
            if not i.isdigit() or "/" in k:
                raise ValueError(f"état d'optimiseur inattendu pour SGD : {key!r}")
            copy_into(self.velocity[int(i)][k], value)


class AdamW:
    """`params` : liste de dicts de tableaux (`Network.params`), mis à jour en place."""

    def __init__(self, params, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2,
                 no_decay=NO_DECAY):
        self.params = params
        self.betas = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.no_decay = set(no_decay)
        self.t = 0
        self.m = [{k: get_xp(v).zeros_like(v) for k, v in p.items()} for p in params]
        self.v = [{k: get_xp(v).zeros_like(v) for k, v in p.items()} for p in params]

    def step(self, grads, lr):
        b1, b2 = self.betas
        self.t += 1
        step = lr / (1 - b1 ** self.t)
        sqrt_bc2 = (1 - b2 ** self.t) ** 0.5
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            for k in p:
                xp = get_xp(p[k])
                m[k] *= b1
                m[k] += (1 - b1) * g[k]
                v[k] *= b2
                v[k] += (1 - b2) * g[k] * g[k]
                if self.weight_decay and k not in self.no_decay:
                    p[k] *= 1 - lr * self.weight_decay
                p[k] -= step * m[k] / (xp.sqrt(v[k]) / sqrt_bc2 + self.eps)

    def state_dict(self):
        state = {"t": np.array(self.t)}
        for name, moments in (("m", self.m), ("v", self.v)):
            state.update({f"{name}/{i}/{k}": a
                          for i, mom in enumerate(moments) for k, a in mom.items()})
        return state

    def load_state_dict(self, state):
        for key, value in state.items():
            name, _, rest = key.partition("/")
            if key == "t":
                self.t = int(value)
            elif name in ("m", "v") and rest.count("/") == 1:
                i, k = rest.split("/")
                copy_into(getattr(self, name)[int(i)][k], value)
            else:
                raise ValueError(f"état d'optimiseur inattendu pour AdamW : {key!r}")


OPTIMIZERS = {"sgd": SGD, "adamw": AdamW}
