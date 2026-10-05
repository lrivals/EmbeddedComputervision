"""Élagage structuré de canaux par norme de filtre (T10.11, §10.4).

Pour chaque conv élagable, les filtres sont classés par norme L1 du filtre **après fusion de la
BN** (γ_f/σ_f · W_f : c'est ce poids qui porte la sortie) et l'on garde les k premiers, k
arrondi au multiple de `multiple` (Tm = 32 : sinon la dernière tuile de sortie du moteur reste
partielle et la perte de tuilage mange le gain). La conv suivante perd les canaux d'entrée
correspondants.

Une conv est élagable si elle a une BN, n'est pas une tête, a plus de `multiple` filtres, et si
sa sortie n'atteint qu'une seule conv, éventuellement à travers des maxpool (cas de toute la
chaîne de Tiny-YOLOv2 ; les routes et upsample de Tiny-YOLOv3 sont laissés intacts).
"""

import copy
import re

import numpy as np

from yolo.models.graph import Network


def fused_l1(net, i):
    """Norme L1 de chaque filtre de la conv i, BN fusionnée."""
    W = np.asarray(net.params[i]["W"], dtype=np.float64)
    l1 = np.abs(W).reshape(len(W), -1).sum(axis=1)
    if net.layers[i]["bn"]:
        g = np.asarray(net.params[i]["gamma"], dtype=np.float64)
        var = np.asarray(net.state[i]["var"], dtype=np.float64)
        l1 = l1 * np.abs(g) / np.sqrt(var + net.bn_eps)
    return l1


def consumers(net, i):
    """Couches qui lisent la sortie de la couche i."""
    return [j for j, src in enumerate(net.inputs) if i in src]


def next_conv(net, i):
    """Unique conv lue (à travers des maxpool) par la sortie de la conv i, sinon None."""
    j = i
    while True:
        c = consumers(net, j)
        if len(c) != 1:
            return None
        j = c[0]
        t = net.layers[j]["type"]
        if t == "conv":
            return j
        if t != "maxpool":
            return None


def prunable(net, multiple=32):
    """{id conv: id de la conv suivante} des convs élagables."""
    out = {}
    for i, layer in enumerate(net.layers):
        if layer["type"] != "conv" or not layer["bn"] or layer["cout"] <= multiple:
            continue
        j = next_conv(net, i)
        if j is not None:
            out[i] = j
    return out


def keep_count(cout, rate, multiple=32):
    """Filtres gardés pour un taux d'élagage `rate`, au multiple le plus proche (≥ multiple)."""
    k = int(round(cout * (1.0 - rate) / multiple)) * multiple
    return min(cout, max(multiple, k))


def select(net, rate, multiple=32, layers=None):
    """{id conv: indices gardés (croissants)} pour un taux uniforme ; `layers` : seulement
    ces convs (parmi les élagables)."""
    keep = {}
    for i in prunable(net, multiple):
        if layers is not None and i not in layers:
            continue
        k = keep_count(net.layers[i]["cout"], rate, multiple)
        keep[i] = np.sort(np.argsort(-fused_l1(net, i), kind="stable")[:k])
    return keep


def prune(net, keep):
    """Nouveau `Network` dont les convs de `keep` n'ont que les filtres gardés ; la conv
    suivante perd les canaux d'entrée correspondants. Paramètres et état BN copiés."""
    nxt = prunable(net, multiple=1)
    desc = copy.deepcopy(net.net)
    for i, idx in keep.items():
        desc["layers"][i]["cout"] = len(idx)
    out = Network(desc, dtype=net.dtype, rng=0, bn_momentum=net.bn_momentum, bn_eps=net.bn_eps)
    in_keep = {nxt[i]: idx for i, idx in keep.items()}
    for i, layer in enumerate(net.layers):
        if layer["type"] != "conv":
            continue
        fo = keep.get(i, slice(None))
        fi = in_keep.get(i, slice(None))
        p = {k: np.array(v[fo]) for k, v in net.params[i].items() if k != "W"}
        p["W"] = np.array(net.params[i]["W"][fo][:, fi])
        out.params[i] = p
        out.state[i] = {k: np.array(v[fo]) for k, v in net.state[i].items()}
    return out


def write_cfg(text, couts):
    """Texte .cfg dont les `filters=` des convs de `couts` ({id couche: cout}) sont changés.

    Les ids sont ceux du parser : rang de la section, [net] exclue.
    """
    lines, idx = [], -2  # [net] → −1, première couche → 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            idx += 1
        m = re.match(r"\s*filters\s*=\s*\d+", raw)
        if m and idx in couts:
            raw = f"filters={couts[idx]}"
        lines.append(raw)
    return "\n".join(lines) + "\n"
