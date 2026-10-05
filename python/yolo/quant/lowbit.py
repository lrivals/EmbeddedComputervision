"""Schémas basse précision (T9.2, T9.3) : largeurs de bits par couche, réseau QAT initial et
modèle entier exportable.

Schéma « wXaY » : poids de **toutes** les convs sur X bits (par canal de sortie), sorties des
convs sur Y bits sauf celles des têtes, qui restent sur 8 bits au pas fixe des LUT
(`HEAD_SCALE` = 1/8, §9.4). L'entrée reste sur 8 bits (pixels) : la première couche est
décomposée en deux convolutions 4 bits dans le matériel (T9.3.4, `int_layers.conv_acc_split4`).

Les pas d'activation sont des puissances de 2 : initialisés en PTQ par `pow2_act_steps` sur
les valeurs de calibration, puis appris en QAT (`fake_quant.QATNetwork`).
"""

import re

import numpy as np

from yolo.models.graph import OUTPUT_TYPES
from yolo.quant.calibrate import HEAD_SCALE
from yolo.quant.fake_quant import STEP_PARAM, QATNetwork, pow2_act_steps, qmax_of_bits
from yolo.quant.int_model import QuantModel
from yolo.quant.quantize import INPUT_SCALE


def parse_scheme(scheme):
    """« w4a4 » → (4, 4)."""
    m = re.fullmatch(r"w(\d+)a(\d+)", scheme)
    if not m:
        raise ValueError(f"schéma inconnu : {scheme!r} (attendu wXaY)")
    wb, ab = int(m.group(1)), int(m.group(2))
    if not (2 <= wb <= 8 and 2 <= ab <= 8):
        raise ValueError("largeurs de bits dans [2, 8]")
    return wb, ab


def head_convs(net):
    """Convs qui alimentent une couche `yolo`/`region`."""
    layers = net["layers"]
    return [i - 1 for i, layer in enumerate(layers) if layer["type"] in OUTPUT_TYPES]


def layer_qmax(net, wbits, abits, int8_layers=()):
    """({id conv: qmax des poids}, {id conv: qmax de la sortie}) ; têtes à 127, ainsi que
    poids et sortie des convs de `int8_layers` (1re et dernière couche en INT8, T12.4)."""

    layers = net["layers"]
    for layer in layers:
        if layer["type"] == "route" and len(layer["from"]) > 1:
            raise NotImplementedError("route à plusieurs sources : échelles à unifier (v3)")
    convs = [i for i, layer in enumerate(layers) if layer["type"] == "conv"]
    heads = set(head_convs(net))
    keep = set(int8_layers)
    unknown = keep - set(convs)
    if unknown:
        raise ValueError(f"int8_layers : pas des convs : {sorted(unknown)}")
    w_qmax = {i: 127 if i in keep else qmax_of_bits(wbits) for i in convs}
    a_qmax = {i: 127 if i in heads or i in keep else qmax_of_bits(abits) for i in convs}
    return w_qmax, a_qmax


def build_qat(fused, scheme, values_of, head_scale=HEAD_SCALE, int8_layers=()):
    """Réseau QAT initialisé en PTQ : pas 2^j minimisant l'erreur du fake-quant sur les
    valeurs de calibration `values_of(id conv)` (sorties flottantes du réseau fusionné)."""
    wb, ab = parse_scheme(scheme)
    w_qmax, a_qmax = layer_qmax(fused.net, wb, ab, int8_layers)
    heads = set(head_convs(fused.net))
    learned = [i for i in a_qmax if i not in heads]
    steps = pow2_act_steps(values_of, learned, a_qmax.get)
    return QATNetwork(fused, w_qmax, a_qmax, {i: float(j) for i, j in steps.items()},
                      {i: head_scale for i in heads})


def qat_from_steps(fused, scheme, log2_steps, head_scale=HEAD_SCALE, lsq_scale=False):
    """Réseau QAT à pas initiaux donnés {id conv: log2 s} (`steps.json` de
    tools/quant_lowbit.py). Sans le facteur de LSQ par défaut : avec lr = 1e-4, le pas de
    k serait ~10³ fois trop petit pour franchir un demi-exposant en quelques milliers
    d'itérations (mesuré : |δk| ~ 1e-8 par itération)."""
    wb, ab = parse_scheme(scheme)
    w_qmax, a_qmax = layer_qmax(fused.net, wb, ab)
    heads = set(head_convs(fused.net))
    return QATNetwork(fused, w_qmax, a_qmax, {int(i): float(k) for i, k in log2_steps.items()},
                      {i: head_scale for i in heads}, lsq_scale=lsq_scale)


def to_quant_model(qat, weights=None):
    """Modèle entier (`QuantModel`) des paramètres courants de `qat` ; `weights` : poids
    entiers déjà calculés {id: (qW, s_w)} (puissances de 2, T9.2)."""
    return QuantModel.from_fused(qat, INPUT_SCALE, qat.act_scales(), w_qmax=qat.w_qmax,
                                 a_qmax=qat.a_qmax, weights=weights)


def load_params(net, checkpoint):
    """Paramètres d'un checkpoint du `Trainer` (`param/<i>/<k>`) dans `net`."""
    with np.load(checkpoint) as data:
        for key in data.files:
            kind, _, rest = key.partition("/")
            if kind == "param":
                i, k = rest.split("/")
                net.params[int(i)][k] = np.array(data[key], dtype=net.dtype)


def steps_table(qat):
    """[(id, qmax, log2 du pas)] des sorties, pour les rapports."""
    out = []
    for i, s in sorted(qat.act_scales().items()):
        learned = STEP_PARAM in qat.params[i]
        out.append((i, qat.a_qmax.get(i, 127), float(np.log2(s)), learned))
    return out
