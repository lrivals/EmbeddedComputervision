"""Sigmoïde et exponentielle du décodage par tables de 256 entrées — §9.4.

Les têtes sortent en int8 à l'échelle s (1/8 par défaut, `calibrate.HEAD_SCALE`). Une table
est indexée par l'entier lui-même (`q + 128`) et rend une valeur en virgule fixe Q16
(valeur réelle = entrée / 2¹⁶) :

- `sigmoid_lut(s)[q + 128] = round(σ(q s) · 2¹⁶)` (uint32 ; < 2¹⁶) ;
- `exp_lut(s)[q + 128]     = round(e^{q s} · 2^f)` (uint32), pour t_w et t_h, avec
  f = `exp_frac_bits(s)` ≤ 16 le plus grand tel que la table tienne sur 32 bits (16 pour
  s = 1/16 ; e^{127 s} croît vite avec s) ;
- `softmax_exp_lut(s)[d + 255] = round(e^{d s} · 2¹⁶)`, d = q_c − q_max ∈ [−255, 0], pour le
  softmax en trois étages de YOLOv2 (max, exponentielles et somme, division) [2024-zhang#014.1].

**Précision.** Aux points de la grille, l'erreur de la table est l'arrondi Q16 (≤ 2⁻¹⁷).
De bout en bout, la quantification de t au pas s ajoute au plus σ'(0)·s/2 = s/8, soit
1/128 ≈ 0,0078 pour s = 1/16 (le §9.4 annonce 0,0075) et 1/64 ≈ 0,016 pour s = 1/8, le pas
retenu : la saturation des logits au pas 1/16 coûte bien plus en mAP que cette erreur.

**Seuil sans sigmoïde (§9.4).** σ est croissante, donc σ(q s) > θ ⟺ q > logit(θ)/s : on
compare l'entier t_o à `logit_threshold_q(θ, s)` avant toute table.
"""

import math

import numpy as np

FRAC_BITS = 16
ONE = 1 << FRAC_BITS
LUT_SIZE = 256
OFFSET = 128  # index = q + 128, q ∈ [−128, 127]
SOFTMAX_OFFSET = 255  # index = d + 255, d ∈ [−255, 0]


def _fixed(v, frac=FRAC_BITS):
    out = np.floor(np.asarray(v, dtype=np.float64) * 2.0**frac + 0.5)
    if out.max() >= 2**32:
        raise OverflowError("valeur de table hors de l'uint32")
    return out.astype(np.uint32)


def lut_inputs(scale):
    """Valeurs réelles t des 256 entrées : (q = −128 … 127) · s."""
    return np.arange(-OFFSET, LUT_SIZE - OFFSET, dtype=np.float64) * scale


def sigmoid_lut(scale):
    t = lut_inputs(scale)
    return _fixed(1.0 / (1.0 + np.exp(-t)))


def exp_frac_bits(scale):
    """Bits fractionnaires de la table exponentielle : ≤ 16, max(e^{q s}) · 2^f < 2³²."""
    top = math.exp((LUT_SIZE - OFFSET - 1) * scale)
    f = min(FRAC_BITS, 31 - math.floor(math.log2(top)))
    if f < 0:
        raise OverflowError("échelle de tête trop grande pour la table exponentielle")
    return f


def exp_lut(scale):
    return _fixed(np.exp(lut_inputs(scale)), exp_frac_bits(scale))


def softmax_exp_lut(scale):
    d = np.arange(-SOFTMAX_OFFSET, LUT_SIZE - SOFTMAX_OFFSET, dtype=np.float64)
    return _fixed(np.exp(d * scale))


def lookup(lut, q, offset=OFFSET):
    """Entrées de `lut` pour les entiers `q` (int8 ou int64)."""
    return lut[np.asarray(q, dtype=np.int64) + offset]


def logit_threshold_q(theta, scale):
    """Plus petit entier q tel que σ(q s) > θ, soit ⌊logit(θ)/s⌋ + 1."""
    if not 0.0 < theta < 1.0:
        raise ValueError("θ doit être dans ]0, 1[")
    return math.floor(math.log(theta / (1.0 - theta)) / scale) + 1


class HeadLuts:
    """Les tables d'une tête d'échelle `scale`."""

    def __init__(self, scale):
        self.scale = scale
        self.sigmoid = sigmoid_lut(scale)
        self.exp = exp_lut(scale)
        self.exp_frac = exp_frac_bits(scale)
        self.softmax_exp = softmax_exp_lut(scale)
