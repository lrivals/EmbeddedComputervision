"""Graphe d'exécution d'un réseau décrit au format de `yolo.models.specs` (§3, §4.5, §7.1).

- Passe avant couche par couche, caches conservés pour la passe arrière.
- Passe arrière en ordre inverse ; le gradient d'une carte lue par plusieurs couches est la
  **somme** des contributions (§4.5 : dans Tiny-YOLOv3, la couche 13 alimente 14 et la
  route 17).
- Une couche « conv » = convolution, puis BN et leaky si `bn`, sinon convolution linéaire avec
  biais (§3).
- Les couches `yolo` / `region` sont l'identité : le décodage et la perte sont hors du graphe.
"""

import numpy as np

from yolo.backend import get_xp, to_device, to_numpy
from yolo.layers.activations import leaky_backward, leaky_forward
from yolo.layers.batchnorm import bn_backward, bn_forward, bn_init_state
from yolo.layers.conv import conv_backward, conv_forward
from yolo.layers.pool import maxpool_backward, maxpool_forward
from yolo.layers.route import route_backward, route_forward
from yolo.layers.upsample import upsample_backward, upsample_forward
from yolo.models.specs import infer_shapes

OUTPUT_TYPES = ("yolo", "region")


class Network:
    """Paramètres `params[i]`, état BN `state[i]` et exécution d'un réseau `net`."""

    def __init__(self, net, dtype=np.float32, rng=None, bn_momentum=0.9, bn_eps=1e-5):
        self.net = net
        self.layers = net["layers"]
        self.dtype = np.dtype(dtype)
        self.bn_momentum = bn_momentum
        self.bn_eps = bn_eps
        self.shapes = infer_shapes(net)
        self.inputs = [self._sources(i, layer) for i, layer in enumerate(self.layers)]
        outputs = [i for i, layer in enumerate(self.layers) if layer["type"] in OUTPUT_TYPES]
        self.outputs = outputs or [len(self.layers) - 1]
        self.params = [{} for _ in self.layers]
        self.state = [{} for _ in self.layers]
        self.init_he(rng)
        self._caches = None

    def _sources(self, i, layer):
        # -1 désigne l'entrée du réseau.
        return list(layer["from"]) if layer["type"] == "route" else [i - 1]

    def init_he(self, rng=None):
        """§7.1 : W ~ N(0, 2/(k²·C_in)), γ = 1, β = 0, biais de tête 0 ; μ_run = 0, σ²_run = 1."""
        rng = np.random.default_rng(rng)
        for i, layer in enumerate(self.layers):
            if layer["type"] != "conv":
                continue
            k, cout, cin = layer["k"], layer["cout"], self.shapes[i][0][0]
            std = np.sqrt(2.0 / (k * k * cin))
            p = {"W": (rng.standard_normal((cout, cin, k, k)) * std).astype(self.dtype)}
            if layer["bn"]:
                p["gamma"] = np.ones(cout, dtype=self.dtype)
                p["beta"] = np.zeros(cout, dtype=self.dtype)
                self.state[i] = bn_init_state(cout, self.dtype)
            else:
                p["b"] = np.zeros(cout, dtype=self.dtype)
            self.params[i] = p

    def to_device(self):
        """Paramètres et état BN sur le backend de `yolo.backend.use` (T12.11)."""
        self._move(to_device)
        return self

    def to_numpy(self):
        """Paramètres et état BN ramenés en NumPy (sauvegarde Darknet, export)."""
        self._move(to_numpy)
        return self

    def _move(self, fn):
        for group in (self.params, self.state):
            for d in group:
                for k in d:
                    d[k] = fn(d[k])
        self._caches = None

    def num_params(self, i):
        """Paramètres entraînables de la couche `i` (comptage du §3)."""
        return sum(p.size for p in self.params[i].values())

    # ------------------------------------------------------------------ passe avant
    def _conv_forward(self, i, x, train):
        layer, p = self.layers[i], self.params[i]
        y, c_conv = conv_forward(x, p["W"], p.get("b"), s=layer["s"])
        caches = [c_conv]
        if layer["bn"]:
            y, c_bn = bn_forward(y, p["gamma"], p["beta"], self.state[i], train=train,
                                 momentum=self.bn_momentum, eps=self.bn_eps)
            caches.append(c_bn)
        if layer["act"] == "leaky":
            y, c_act = leaky_forward(y)
            caches.append(c_act)
        return y, caches

    def forward(self, x, train=True, all_outputs=False):
        """Renvoie {id: sortie} des couches `yolo`/`region` (ou de la dernière couche) ;
        avec `all_outputs`, la liste des sorties de toutes les couches.
        """
        outs, caches = [], []
        for i, layer in enumerate(self.layers):
            t = layer["type"]
            src = [x if j < 0 else outs[j] for j in self.inputs[i]]
            if t == "conv":
                y, cache = self._conv_forward(i, src[0], train)
            elif t == "maxpool":
                y, cache = maxpool_forward(src[0], k=layer["k"], s=layer["s"])
            elif t == "upsample":
                y, cache = upsample_forward(src[0], s=layer["s"])
            elif t == "route":
                y, cache = route_forward(src)
            elif t in OUTPUT_TYPES:
                y, cache = src[0], None
            else:
                raise ValueError(f"couche {i} : type inconnu {t!r}")
            outs.append(y)
            caches.append(cache)
        self._caches = caches
        if all_outputs:
            return outs
        return {i: outs[i] for i in self.outputs}

    # ---------------------------------------------------------------- passe arrière
    def _conv_backward(self, i, dy, caches):
        layer = self.layers[i]
        grads = {}
        caches = list(caches)
        if layer["act"] == "leaky":
            dy, _ = leaky_backward(dy, caches.pop())
        if layer["bn"]:
            dy, g = bn_backward(dy, caches.pop())
            grads.update(g)
        dx, g = conv_backward(dy, caches.pop())
        grads.update(g)
        return [dx], grads

    def backward(self, douts):
        """`douts` : {id: δy} pour les couches de sortie. Renvoie (δx, grads[i] par couche).

        Les couches qui ne reçoivent aucun gradient ont des gradients de paramètres nuls.
        """
        if self._caches is None:
            raise RuntimeError("backward() appelé sans forward()")
        dout = {i: get_xp(d).array(d, copy=True) for i, d in douts.items()}
        grads = [{k: get_xp(v).zeros_like(v) for k, v in p.items()} for p in self.params]
        for i in reversed(range(len(self.layers))):
            dy = dout.pop(i, None)
            if dy is None:
                continue
            t, cache = self.layers[i]["type"], self._caches[i]
            if t == "conv":
                dxs, g = self._conv_backward(i, dy, cache)
                grads[i] = g
            elif t == "maxpool":
                dxs = [maxpool_backward(dy, cache)[0]]
            elif t == "upsample":
                dxs = [upsample_backward(dy, cache)[0]]
            elif t == "route":
                dxs = route_backward(dy, cache)[0]
            else:
                dxs = [dy]
            # §4.5 : une carte lue par plusieurs couches cumule ses gradients.
            for j, dx in zip(self.inputs[i], dxs):
                if j in dout:
                    dout[j] = dout[j] + dx
                else:
                    dout[j] = dx
        return dout.get(-1), grads
