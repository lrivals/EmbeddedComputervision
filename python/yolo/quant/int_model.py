"""Modèle entier bit-exact (§9.3, §10.5 étape 2) et simulation par fake-quant.

- `QuantModel` : paramètres entiers par convolution (qW int8, q_b int32, M0 int32, n) et
  échelles, construits depuis le réseau fusionné (T4.1) et les échelles calibrées (T4.2).
- `IntNetwork.forward(qx)` : passe avant **sans flottant** sur l'entrée int8 ; rend les têtes
  int8, ou les sorties int8 de toutes les couches (`all_outputs`, dumps `L00`…).
- `fake_quant_forward` : passe flottante où une partie des couches est quantifiée
  (poids et sortie fake-quantifiés), pour la sensibilité par couche (T4.5).
"""

from dataclasses import dataclass, field

import numpy as np

from yolo.layers.conv import conv_forward
from yolo.layers.activations import leaky_forward
from yolo.models.graph import OUTPUT_TYPES
from yolo.quant import int_layers as il
from yolo.quant.calibrate import scale_owners
from yolo.quant.quantize import (fake_quant, quantize_bias, quantize_weights_per_channel,
                                 requant_params)


@dataclass
class QConv:
    """Convolution quantifiée : s_x, s_w (F,), s_y ; qW, q_b, M0 (F,), n (§9.3)."""
    qW: np.ndarray
    qb: np.ndarray
    M0: np.ndarray
    n: int
    sx: float
    sw: np.ndarray
    sy: float
    act: str
    s: int = 1
    qmax: int = 127   # saturation de la sortie (2^{b−1} − 1 pour b bits, T9.3)
    wqmax: int = 127  # poids dans [−wqmax, wqmax]


@dataclass
class QuantModel:
    net: dict  # description au format `specs` (couches, ancres, classes)
    input_scale: float
    convs: dict = field(default_factory=dict)  # {id: QConv}

    @property
    def layers(self):
        return self.net["layers"]

    @property
    def outputs(self):
        outs = [i for i, layer in enumerate(self.layers) if layer["type"] in OUTPUT_TYPES]
        return outs or [len(self.layers) - 1]

    def scale_of(self, i):
        """Échelle de la sortie de la couche i (−1 : entrée)."""
        o = scale_owners(self.layers)[i] if i >= 0 else -1
        return self.input_scale if o < 0 else self.convs[o].sy

    @classmethod
    def from_fused(cls, fused, input_scale, act_scales, w_qmax=None, a_qmax=None,
                   weights=None):
        """`fused` : `Network` à BN fusionnée ; `act_scales` : {id conv: s_y}.

        `w_qmax`, `a_qmax` : {id conv: qmax} des poids et des sorties (127 par défaut) ;
        `weights` : {id conv: (qW, s_w)} déjà quantifiés (puissances de 2, T9.2).
        """
        w_qmax, a_qmax, weights = w_qmax or {}, a_qmax or {}, weights or {}
        layers = fused.layers
        owner = scale_owners(layers)
        qm = cls(net=fused.net, input_scale=float(input_scale))
        for i, layer in enumerate(layers):
            if layer["type"] != "conv":
                continue
            j = fused.inputs[i][0]
            o = -1 if j < 0 else owner[j]
            sx = input_scale if o < 0 else act_scales[o]
            sy = act_scales[i]
            p = fused.params[i]
            wq = w_qmax.get(i, 127)
            qW, sw = weights[i] if i in weights else quantize_weights_per_channel(p["W"], wq)
            qb = quantize_bias(p["b"], sx, sw)
            M0, n = requant_params(sx, sw, sy)
            qm.convs[i] = QConv(qW, qb, M0, n, float(sx), sw, float(sy), layer["act"],
                                layer["s"], a_qmax.get(i, 127), wq)
        return qm

    def acc_bound(self, i):
        """Borne du pire cas |acc_f| ≤ Σ|q_w,f|·127 + |q_b,f| (max sur f)."""
        c = self.convs[i]
        return int((np.abs(c.qW.astype(np.int64)).reshape(len(c.qW), -1).sum(axis=1) * 127
                    + np.abs(c.qb.astype(np.int64))).max())


def _sources(layers, i):
    layer = layers[i]
    return list(layer["from"]) if layer["type"] == "route" else [i - 1]


class IntNetwork:
    def __init__(self, qm, engine="f64"):
        if engine not in il.ENGINES:
            raise ValueError(f"moteur inconnu : {engine!r}")
        self.qm = qm
        self.engine = engine
        self.max_acc = {}  # |acc| max observé par conv (contrôle du débordement)

    def forward(self, qx, all_outputs=False):
        """`qx` : (N, 3, H, W) int8. Rend {id: tête int8} ou la liste des sorties int8."""
        qx = np.asarray(qx)
        if qx.dtype != np.int8:
            raise TypeError("entrée int8 attendue (quantize_input)")
        layers = self.qm.layers
        x = qx.astype(np.int64)
        outs = []
        for i, layer in enumerate(layers):
            t = layer["type"]
            src = [x if j < 0 else outs[j] for j in _sources(layers, i)]
            if t == "conv":
                c = self.qm.convs[i]
                y, acc = il.conv_int(src[0], c.qW, c.qb, c.M0, c.n, c.act, self.engine,
                                     c.qmax)
                self.max_acc[i] = max(self.max_acc.get(i, 0), int(np.abs(acc).max()))
            elif t == "maxpool":
                y = il.maxpool_int(src[0], k=layer["k"], s=layer["s"])
            elif t == "upsample":
                y = il.upsample_int(src[0], s=layer["s"])
            elif t == "route":
                y = il.route_int(src)
            elif t in OUTPUT_TYPES:
                y = src[0]
            else:
                raise ValueError(f"couche {i} : type inconnu {t!r}")
            assert y.dtype == np.int64, (i, y.dtype)  # aucun flottant dans la passe avant
            outs.append(y)
        outs = [o.astype(np.int8) for o in outs]
        if all_outputs:
            return outs
        return {i: outs[i] for i in self.qm.outputs}


def fake_quant_forward(fused, qm, x, layers=None, all_outputs=False):
    """Passe flottante du réseau fusionné où les convolutions de `layers` (toutes si None)
    ont des poids fake-quantifiés par canal, la leaky 13/128 et une sortie fake-quantifiée à
    s_y ; l'entrée
    est fake-quantifiée si `layers` est None.
    """
    quant = set(qm.convs) if layers is None else set(layers)
    if layers is None:
        x = fake_quant(x, qm.input_scale)
    outs = []
    for i, layer in enumerate(fused.layers):
        t = layer["type"]
        src = [x if j < 0 else outs[j] for j in fused.inputs[i]]
        if t == "conv":
            p = fused.params[i]
            if i in quant:
                c = qm.convs[i]
                W = (c.qW * c.sw[:, None, None, None]).astype(p["W"].dtype)
            else:
                W = p["W"]
            y, _ = conv_forward(src[0], W, p["b"], s=layer["s"])
            if layer["act"] == "leaky":
                # Pente 13/128 du matériel pour une couche quantifiée (§9.3), 0,1 sinon.
                slope = il.LEAKY_MUL / 2**il.LEAKY_SHIFT if i in quant else 0.1
                y, _ = leaky_forward(y, slope)
            if i in quant:
                y = fake_quant(y, qm.convs[i].sy, qm.convs[i].qmax)
        elif t == "maxpool":
            y = il.maxpool_forward(src[0], k=layer["k"], s=layer["s"])[0]
        elif t == "upsample":
            y = il.upsample_forward(src[0], s=layer["s"])[0]
        elif t == "route":
            y = il.route_forward(src)[0]
        else:
            y = src[0]
        outs.append(y)
    if all_outputs:
        return outs
    return {i: outs[i] for i in qm.outputs}


def head_luts(qm):
    """{id de tête: HeadLuts} à l'échelle de chaque tête (§9.4)."""
    from yolo.quant.lut import HeadLuts

    return {i: HeadLuts(qm.scale_of(i)) for i in qm.outputs}
