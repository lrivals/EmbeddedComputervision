"""Boucle d'entraînement (§7.1) : passe avant, perte (§6.2), passe arrière, SGD.

    for X, cibles in lots:
        si multi-échelle et it % 10 == 0 : nouvelle taille parmi 320, 352, …, 608   (§2.2)
        sorties = forward(X) ; L, δ = perte(sorties) ; grads = backward(δ / N)
        p ← SGD(p, grads, lr(it))

La perte de `yolo_loss` est une somme sur le lot ; le trainer la divise par N (perte
moyenne par image), ce qui rend le taux d'apprentissage indépendant de la taille du lot.

La taille multi-échelle de l'itération `it` est une fonction pure de (graine, it // 10), et
l'ordre des images une fonction de (graine, époque) : la reprise d'un checkpoint reproduit
exactement la suite de l'entraînement (sur le même backend).

Backend (T12.11) : si les paramètres sont sur le GPU (`Network.to_device`), le lot y est
copié ; la perte et ses cibles restent calculées en NumPy sur les sorties des têtes
(petites), dont les gradients repartent vers le GPU. Les checkpoints sont toujours en NumPy.
"""

import csv
import time
from pathlib import Path

import numpy as np

from yolo.backend import copy_into, get_xp, to_numpy
from yolo.data.targets import heads
from yolo.train.loss import yolo_loss

MULTISCALE_SIZES = tuple(range(320, 609, 32))  # §2.2 : multiples de 32 de 320 à 608
LOG_FIELDS = ("it", "lr", "size", "loss", "coord", "obj", "noobj", "cls", "seconds")


def multiscale_size(it, seed=0, sizes=MULTISCALE_SIZES, every=10):
    """§2.2 : une taille tirée tous les `every` lots, déterministe."""
    rng = np.random.default_rng([seed, it // every])
    return int(sizes[rng.integers(len(sizes))])


class Trainer:
    """Entraîne `net` (`yolo.models.graph.Network`).

    `size` : taille d'entrée fixe, ou `None` pour le multi-échelle (`multiscale_size`).
    `grad_hook(trainer, grads)` : appelé avant le pas SGD (terme de pénalité de l'ADMM, T9.2).
    Les autres options sont passées à `yolo_loss` (`ignore_thresh`, `lambda_coord`).
    """

    def __init__(self, net, optimizer, schedule, size=416, seed=0, log_path=None,
                 grad_hook=None, **loss_kwargs):
        self.net = net
        self.optimizer = optimizer
        self.schedule = schedule
        self.size = size
        self.seed = seed
        self.it = 0
        self.head_list = heads(net.net)
        region = any(net.layers[h]["type"] == "region" for h, _ in self.head_list)
        self.loss_kwargs = {"class_mode": "softmax" if region else "sigmoid", **loss_kwargs}
        self.log_path = Path(log_path) if log_path else None
        self.grad_hook = grad_hook

    def array_module(self):
        """numpy ou cupy, d'après les paramètres du réseau."""
        return get_xp(*(v for p in self.net.params for v in p.values()))

    def size_for(self, it):
        return self.size if self.size else multiscale_size(it, self.seed)

    def step(self, images, boxes, labels):
        """Une itération ; rend le `LossResult` (somme sur le lot) et le taux utilisé."""
        t0 = time.perf_counter()
        n = images.shape[0]
        xnp = self.array_module()
        x = images.astype(self.net.dtype, copy=False)
        outputs = self.net.forward(x if xnp is np else xnp.asarray(x), train=True)
        outputs = {h: to_numpy(o) for h, o in outputs.items()}
        res = yolo_loss(outputs, boxes, labels, self.net.net["anchors"], self.head_list,
                        self.net.net["classes"], **self.loss_kwargs)
        douts = {h: d / n for h, d in res.douts.items()}
        if xnp is not np:
            douts = {h: xnp.asarray(d) for h, d in douts.items()}
        _, grads = self.net.backward(douts)
        lr = self.schedule(self.it)
        if self.grad_hook:
            self.grad_hook(self, grads)
        self.optimizer.step(grads, lr)
        self._log(lr, images.shape[-1], n, res, time.perf_counter() - t0)
        self.it += 1
        return res, lr

    def fit(self, loader, iters, checkpoint=None, save_every=0, callback=None):
        """Jusqu'à l'itération `iters` ; reprend là où `self.it` en est."""
        per_epoch = len(loader)
        while self.it < iters:
            epoch, skip = divmod(self.it, per_epoch)
            batches = loader.epoch(epoch, self.size_for, first_iter=self.it, skip=skip)
            for images, boxes, labels in batches:
                res, lr = self.step(images, boxes, labels)
                if callback:
                    callback(self, res, lr)
                if checkpoint and save_every and self.it % save_every == 0:
                    self.save_checkpoint(checkpoint)
                if self.it >= iters:
                    break
        if checkpoint:
            self.save_checkpoint(checkpoint)

    def _log(self, lr, size, n, res, seconds):
        if self.log_path is None:
            return
        new = not self.log_path.exists()
        with open(self.log_path, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(LOG_FIELDS)
            p = res.parts
            w.writerow([self.it, f"{lr:.6g}", size, f"{res.total / n:.6g}",
                        *(f"{p[k] / n:.6g}" for k in ("coord", "obj", "noobj", "cls")),
                        f"{seconds:.3f}"])

    # ---------------------------------------------------------------- checkpoints
    def save_checkpoint(self, path):
        """Paramètres, état BN, vitesses SGD et itération dans un `.npz`."""
        arrays = {"it": np.array(self.it), "seed": np.array(self.seed)}
        for i, (p, s) in enumerate(zip(self.net.params, self.net.state)):
            arrays.update({f"param/{i}/{k}": to_numpy(v) for k, v in p.items()})
            arrays.update({f"state/{i}/{k}": to_numpy(v) for k, v in s.items()})
        arrays.update({f"velocity/{k}": to_numpy(v)
                       for k, v in self.optimizer.state_dict().items()})
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        tmp.replace(path)

    def load_checkpoint(self, path):
        with np.load(path) as data:
            self.it = int(data["it"])
            self.seed = int(data["seed"])
            velocity = {}
            for key in data.files:
                kind, _, rest = key.partition("/")
                if kind == "param":
                    i, k = rest.split("/")
                    copy_into(self.net.params[int(i)][k], data[key])
                elif kind == "state":
                    i, k = rest.split("/")
                    copy_into(self.net.state[int(i)][k], data[key])
                elif kind == "velocity":
                    velocity[rest] = data[key]
            self.optimizer.load_state_dict(velocity)


def copy_matching(src, dst):
    """Copie les paramètres et l'état BN de `src` vers `dst` couche par couche quand les formes
    concordent (transfert COCO → VOC : tout sauf les têtes). Rend les couches non copiées.
    """
    skipped = []
    for i, (ps, pd) in enumerate(zip(src.params, dst.params)):
        if not pd:
            continue
        if ps.keys() != pd.keys() or any(ps[k].shape != pd[k].shape for k in pd):
            skipped.append(i)
            continue
        for k in pd:
            pd[k][...] = ps[k]
        for k in dst.state[i]:
            dst.state[i][k][...] = src.state[i][k]
    return skipped
