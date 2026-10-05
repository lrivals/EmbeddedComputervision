"""Quantification symétrique et paramètres de requantification — §9.2, §9.3.

- Arrondi **hors ligne** (poids, biais, entrée) : `round_half_up(v) = ⌊v + ½⌋`, la même
  règle que le décalage arrondi de la passe avant entière (`int_layers.rshift_round`).
- Poids : int8 symétrique **par canal de sortie**, s_w,f = max|W_f| / 127 (§9.2).
- Biais : int32, q_b,f = round(b'_f / (s_x s_w,f)) (§9.3).
- Requantification : M0_f = round(s_x s_w,f / s_y · 2ⁿ) < 2³¹, avec n ≤ 31 le plus grand
  possible pour la couche (§9.3).
"""

import numpy as np

QMAX = 127
INPUT_SCALE = 1.0 / QMAX  # pixels [0, 1] → [0, 127]
SHIFT_MAX = 31
M0_LIMIT = 1 << 31


def round_half_up(v):
    """⌊v + ½⌋ en int64 : demis vers +∞, y compris pour les négatifs (−2,5 → −2)."""
    return np.floor(np.asarray(v, dtype=np.float64) + 0.5).astype(np.int64)


def quantize(x, scale, qmax=QMAX):
    """q = clip(round(x / s), −qmax, qmax) en int64 (§9.2, Z = 0)."""
    return np.clip(round_half_up(np.asarray(x, dtype=np.float64) / scale), -qmax, qmax)


def fake_quant(x, scale, qmax=QMAX):
    """Quantifié puis déquantifié, dans le dtype de `x`."""
    x = np.asarray(x)
    return (quantize(x, scale, qmax) * scale).astype(x.dtype)


def quantize_input(x):
    """Image float [0, 1] → int8 à l'échelle `INPUT_SCALE` (fait par l'hôte, hors réseau)."""
    return quantize(x, INPUT_SCALE).astype(np.int8)


def weight_scales(W):
    """s_w,f = max|W_f| / 127 ; un filtre nul reçoit l'échelle 1/127 (poids tous nuls)."""
    m = np.abs(np.asarray(W, dtype=np.float64)).reshape(len(W), -1).max(axis=1)
    return np.where(m > 0, m, 1.0) / QMAX


def quantize_weights_per_channel(W):
    """§9.2 : (qW int8 (F, C, k, k), s_w (F,))."""
    sw = weight_scales(W)
    return quantize(W, sw[:, None, None, None]).astype(np.int8), sw


def quantize_bias(b, sx, sw):
    """§9.3 : q_b = round(b' / (s_x s_w,f)) en int32 ; erreur si hors de l'int32."""
    qb = round_half_up(np.asarray(b, dtype=np.float64) / (sx * sw))
    if np.abs(qb).max(initial=0) >= M0_LIMIT:
        raise OverflowError("biais quantifié hors de l'int32")
    return qb.astype(np.int32)


def requant_params(sx, sw, sy, n_max=SHIFT_MAX):
    """§9.3 : (M0 (F,) int32, n), n le plus grand ≤ n_max tel que max M0 < 2³¹."""
    ratio = sx * np.asarray(sw, dtype=np.float64) / sy
    for n in range(n_max, 0, -1):
        m0 = round_half_up(ratio * 2.0**n)
        if m0.max() < M0_LIMIT:
            if m0.min() <= 0:
                raise ValueError("M0 nul : rapport d'échelles trop petit")
            return m0.astype(np.int32), n
    raise ValueError("rapport d'échelles trop grand pour M0 < 2³¹")
