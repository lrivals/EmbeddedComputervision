"""ADMM pour des poids sur des niveaux imposés — REQ-YOLO, T9.2.3, §9.2 [2019-ding#009.1].

Problème : min_W L(W) sous W ∈ S (niveaux équidistants ou puissances de 2, `pow2.py`).
Avec Z ∈ S, U la variable duale réduite et ρ > 0, on itère :

- pas W : SGD sur L(W) + ρ/2 ‖W − Z + U‖², soit un gradient ajouté ρ (W − Z + U)
  (`hook`, appelé par le `Trainer` avant chaque pas) ;
- tous les `every` pas : Z ← Π_S(W + U) (projection, α par canal compris), puis
  U ← U + W − Z, et ρ ← min(ρ·`growth`, `rho_max`) (ρ croissant, hors base).

Le résidu primal ‖W − Z‖ / ‖W‖ de chaque couche doit tendre vers 0 ; on finit par la
projection dure W → Π_S(W) (tools/quant_lowbit.py --weights pow2 --checkpoint).
Les convs travaillent sur les poids **fusionnés** (BN comprise, §9.1).

Backend (T12.11) : Z et U suivent les poids (GPU possible) ; la projection Π_S, tous les
`every` pas seulement, est calculée en NumPy sur le CPU.
"""

import csv

import numpy as np

from yolo.backend import get_xp, to_numpy
from yolo.quant.pow2 import dequantize, project


class ADMM:
    def __init__(self, net, plan, rho=1e-3, every=100, growth=1.3, rho_max=1.0, log_path=None):
        self.net = net
        self.plan = {int(k): v for k, v in plan.items()}
        self.rho = rho
        self.every = every
        self.growth = growth
        self.rho_max = rho_max
        self.log_path = log_path
        self.Z, self.U = {}, {}
        for i, kind in self.plan.items():
            W = net.params[i]["W"]
            self.Z[i] = self._project(W, kind)
            self.U[i] = get_xp(W).zeros_like(W)
        self.updates = 0

    @staticmethod
    def _project(W, kind):
        """Π_S(W) déquantifié, au type et sur le backend de W."""
        z = dequantize(*project(to_numpy(W), kind)).astype(W.dtype)
        return get_xp(W).asarray(z)

    def hook(self, trainer, grads):
        for i in self.plan:
            W = self.net.params[i]["W"]
            grads[i]["W"] = grads[i]["W"] + self.rho * (W - self.Z[i] + self.U[i])
        if (trainer.it + 1) % self.every == 0:
            self.update(trainer.it + 1)

    def residuals(self):
        """{id: ‖W − Z‖ / ‖W‖}."""
        out = {}
        for i in self.plan:
            W = self.net.params[i]["W"].astype(np.float64)
            norm = get_xp(W).linalg.norm
            out[i] = float(norm(W - self.Z[i]) / max(float(norm(W)), 1e-12))
        return out

    def update(self, it=None):
        for i, kind in self.plan.items():
            W = self.net.params[i]["W"]
            self.Z[i] = self._project(W + self.U[i], kind)
            self.U[i] += W - self.Z[i]
        self.rho = min(self.rho * self.growth, self.rho_max)
        self.updates += 1
        res = self.residuals()
        if self.log_path:
            new = not self.log_path.exists()
            with open(self.log_path, "a", newline="") as f:
                w = csv.writer(f)
                if new:
                    w.writerow(["it", "rho", *(f"res_{i}" for i in self.plan)])
                w.writerow([it, f"{self.rho:.6g}", *(f"{res[i]:.6g}" for i in self.plan)])
        return res

    def save(self, path):
        np.savez(path, rho=np.array(self.rho), updates=np.array(self.updates),
                 **{f"Z/{i}": to_numpy(z) for i, z in self.Z.items()},
                 **{f"U/{i}": to_numpy(u) for i, u in self.U.items()})

    def load(self, path):
        with np.load(path) as d:
            self.rho = float(d["rho"])
            self.updates = int(d["updates"])
            for k in d.files:
                kind, _, i = k.partition("/")
                if kind in ("Z", "U"):
                    xnp = get_xp(self.net.params[int(i)]["W"])
                    getattr(self, kind)[int(i)] = xnp.asarray(d[k])
