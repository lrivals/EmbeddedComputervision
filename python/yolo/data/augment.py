"""Augmentation de données (§7.1) : échelle et translation aléatoires jusqu'à 20 %, exposition
et saturation jusqu'à un facteur 1,5 en HSV ; retournement horizontal en option (hors base).

La géométrie est une seule affinité, des pixels de l'image d'origine vers le carré d'entrée
`size` : letterbox (`yolo.data.letterbox`), puis échelle `r` et translation autour du centre,
puis retournement. Les mêmes coefficients transforment l'image (PIL `Image.transform`) et les
coins des boîtes, ce qui garantit que boîtes et pixels restent alignés.
"""

import numpy as np

from yolo.data.letterbox import GRAY, letterbox_params


def random_params(rng, jitter=0.2, hsv=1.5, flip=True):
    """Tirage des paramètres d'une augmentation (§7.1)."""
    return {
        "scale": rng.uniform(1 - jitter, 1 + jitter),
        "shift": rng.uniform(-jitter, jitter, 2),          # en fraction de `size`
        "flip": bool(flip and rng.random() < 0.5),
        "sat": _rand_factor(rng, hsv),
        "val": _rand_factor(rng, hsv),
    }


IDENTITY = {"scale": 1.0, "shift": np.zeros(2), "flip": False, "sat": 1.0, "val": 1.0}


def _rand_factor(rng, f):
    # Facteur dans [1/f, f], symétrique en échelle logarithmique.
    return float(np.exp(rng.uniform(-np.log(f), np.log(f))))


def affine(width, height, size, params):
    """Matrice 2×3 `M` : (x, y) pixels d'origine → (u, v) pixels du carré `size`."""
    nw, nh, dx, dy = letterbox_params(width, height, size)
    r = params["scale"]
    c = size / 2
    tx, ty = np.asarray(params["shift"]) * size
    # u = c + r·((nw/w)·x + dx − c) + t_x
    m = np.array([[r * nw / width, 0.0, c + r * (dx - c) + tx],
                  [0.0, r * nh / height, c + r * (dy - c) + ty]])
    if params["flip"]:
        m[0] = [-m[0, 0], 0.0, size - m[0, 2]]
    return m


def transform_boxes(boxes, width, height, size, m, min_size=2.0):
    """Boîtes normalisées (origine) → normalisées dans le carré `size`, rognées au carré.

    Rend (boîtes, masque des boîtes gardées) ; une boîte dont un côté rogné fait moins de
    `min_size` pixels est retirée.
    """
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    x1 = (b[:, 0] - b[:, 2] / 2) * width
    x2 = (b[:, 0] + b[:, 2] / 2) * width
    y1 = (b[:, 1] - b[:, 3] / 2) * height
    y2 = (b[:, 1] + b[:, 3] / 2) * height
    u = m[0, 0] * np.stack([x1, x2]) + m[0, 2]
    v = m[1, 1] * np.stack([y1, y2]) + m[1, 2]
    u1, u2 = np.clip(u.min(axis=0), 0, size), np.clip(u.max(axis=0), 0, size)
    v1, v2 = np.clip(v.min(axis=0), 0, size), np.clip(v.max(axis=0), 0, size)
    keep = (u2 - u1 >= min_size) & (v2 - v1 >= min_size)
    out = np.stack([(u1 + u2) / 2, (v1 + v2) / 2, u2 - u1, v2 - v1], axis=1) / size
    return out[keep], keep


def warp_image(img, size, m):
    """Image PIL → (size, size, 3) float32 dans [0, 1], fond gris (letterbox)."""
    from PIL import Image

    # PIL attend l'affinité inverse : (u, v) → (x, y).
    a, c = m[0, 0], m[0, 2]
    e, f = m[1, 1], m[1, 2]
    inv = (1 / a, 0.0, -c / a, 0.0, 1 / e, -f / e)
    fill = (int(round(GRAY * 255)),) * 3
    out = img.convert("RGB").transform((size, size), Image.AFFINE, inv,
                                       resample=Image.BILINEAR, fillcolor=fill)
    return np.asarray(out, dtype=np.float32) / 255.0


def rgb_to_hsv(rgb):
    """(…, 3) dans [0, 1] → HSV dans [0, 1]."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    mx, mn = rgb.max(axis=-1), rgb.min(axis=-1)
    d = mx - mn
    safe = np.where(d > 0, d, 1)
    h = np.where(mx == r, (g - b) / safe % 6, np.where(mx == g, (b - r) / safe + 2,
                                                       (r - g) / safe + 4)) / 6
    h = np.where(d > 0, h, 0)
    s = np.where(mx > 0, d / np.where(mx > 0, mx, 1), 0)
    return np.stack([h, s, mx], axis=-1)


def hsv_to_rgb(hsv):
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    choices = [(v, t, p), (q, v, p), (p, v, t), (p, q, v), (t, p, v), (v, p, q)]
    out = np.zeros(hsv.shape, dtype=hsv.dtype)
    for k, (r, g, b) in enumerate(choices):
        sel = i == k
        out[..., 0][sel], out[..., 1][sel], out[..., 2][sel] = r[sel], g[sel], b[sel]
    return out


def adjust_hsv(img, sat, val):
    """§7.1 : saturation et exposition multipliées par `sat`, `val` (bornées à [0, 1])."""
    if sat == 1.0 and val == 1.0:
        return img
    hsv = rgb_to_hsv(img)
    hsv[..., 1] = np.clip(hsv[..., 1] * sat, 0, 1)
    hsv[..., 2] = np.clip(hsv[..., 2] * val, 0, 1)
    return hsv_to_rgb(hsv).astype(np.float32)


def augment(img, boxes, labels, size, params):
    """Image PIL et boîtes normalisées (origine) → (HWC float32, boîtes, labels) au carré `size`."""
    m = affine(img.width, img.height, size, params)
    out = warp_image(img, size, m)
    out = adjust_hsv(out, params["sat"], params["val"])
    b, keep = transform_boxes(boxes, img.width, img.height, size, m)
    return out, b, np.asarray(labels)[keep]
