"""Letterbox : redimensionnement à rapport d'aspect conservé, complété en gris (§1, hors base).

Convention Darknet : l'image est réduite d'un facteur `min(W/w, H/h)`, centrée, et les
bandes sont remplies de gris 0,5. Les boîtes `(cx, cy, w, h)` normalisées passent du repère
de l'image d'origine à celui de l'image letterbox (et inversement) par une affinité exacte.

YOLOv5 / YOLOv7 variant (`mode="v5"`, T33.4): same factor, gray 114/255, bands of
`round(d − 0.1)` at the top / left and `round(d + 0.1)` at the bottom / right (Ultralytics
`letterbox`, `auto=False`, square input multiple of 32).

Taille d'entrée `size` : un entier S (carré S × S) ou un couple `(H, W)` (entrée non carrée,
T11.4). Canaux : 3 (RGB) ou 1 (niveaux de gris, entrée thermique, T11.7).
"""

import numpy as np

GRAY = 0.5
GRAY_V5 = 114 / 255
RESIZE_MODES = ("letterbox", "stretch", "v5")


def as_hw(size):
    """`size` entier ou (H, W) → (H, W)."""
    if isinstance(size, (tuple, list, np.ndarray)):
        h, w = (int(v) for v in size)
        return h, w
    return int(size), int(size)


def parse_size(text):
    """Option `--size` : « 416 » → 416, « 640x192 » (largeur × hauteur, comme une résolution
    d'image) → (H, W) = (192, 640) ; un carré est rendu en entier."""
    parts = str(text).lower().replace("×", "x").split("x")
    if len(parts) == 1:
        return int(parts[0])
    w, h = (int(v) for v in parts)
    return h if h == w else (h, w)


def size_label(size):
    """« 416×416 », « 640×192 » (largeur × hauteur)."""
    h, w = as_hw(size)
    return f"{w}×{h}"


def input_size(net, size=None):
    """Taille d'entrée : `size` donné (entier ou (H, W)), sinon celle de la cfg de `net`."""
    if size is not None:
        return size
    _, h, w = net["input"]
    return h if h == w else (h, w)


def image_mode(channels):
    """Mode PIL de l'entrée à `channels` canaux."""
    if channels not in (1, 3):
        raise ValueError(f"entrée à {channels} canaux : 1 ou 3 attendus")
    return "RGB" if channels == 3 else "L"


def to_array(img, channels=3):
    """Image PIL → (H, W, C) float32 dans [0, 1]."""
    x = np.asarray(img.convert(image_mode(channels)), dtype=np.float32) / 255.0
    return x if x.ndim == 3 else x[:, :, None]


def letterbox_params(width, height, size):
    """(nw, nh, dx, dy) : taille de l'image redimensionnée et décalage dans l'entrée `size`."""
    sh, sw = as_hw(size)
    scale = min(sw / width, sh / height)
    nw = max(1, int(round(width * scale)))
    nh = max(1, int(round(height * scale)))
    return nw, nh, (sw - nw) // 2, (sh - nh) // 2


def letterbox_params_v5(width, height, size):
    """(nw, nh, dx, dy) of the Ultralytics `letterbox` (`auto=False`): top / left band
    `round(d − 0.1)`, d the half margin (the odd pixel goes to the bottom / right)."""
    sh, sw = as_hw(size)
    scale = min(sw / width, sh / height)
    nw = max(1, int(round(width * scale)))
    nh = max(1, int(round(height * scale)))
    return nw, nh, int(round((sw - nw) / 2 - 0.1)), int(round((sh - nh) / 2 - 0.1))


def resize_params(width, height, size, mode="letterbox"):
    """`letterbox_params`, `letterbox_params_v5` (`v5`), or in `stretch` the image stretched
    to the whole input (no bands)."""
    if mode == "stretch":
        sh, sw = as_hw(size)
        return sw, sh, 0, 0
    if mode == "v5":
        return letterbox_params_v5(width, height, size)
    if mode != "letterbox":
        raise ValueError(f"prétraitement inconnu : {mode!r}")
    return letterbox_params(width, height, size)


def letterbox_image(img, size, channels=3, mode="letterbox"):
    """`img`: PIL image or (H, W, 3) uint8 array. Returns (H, W, C) float32 in [0, 1] and the
    `resize_params` parameters (`mode`: `letterbox` or `v5`).
    """
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    img = img.convert(image_mode(channels))
    sh, sw = as_hw(size)
    nw, nh, dx, dy = resize_params(img.width, img.height, size, mode)
    resized = to_array(img.resize((nw, nh), Image.BILINEAR), channels)
    out = np.full((sh, sw, channels), GRAY_V5 if mode == "v5" else GRAY, dtype=np.float32)
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, (nw, nh, dx, dy)


def boxes_to_letterbox(boxes, width, height, size, mode="letterbox"):
    """Boîtes normalisées dans l'image d'origine → normalisées dans l'image letterbox (ou
    étirée, `mode="stretch"`)."""
    sh, sw = as_hw(size)
    nw, nh, dx, dy = resize_params(width, height, size, mode)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, 0] = (b[:, 0] * nw + dx) / sw
    b[:, 1] = (b[:, 1] * nh + dy) / sh
    b[:, 2] = b[:, 2] * nw / sw
    b[:, 3] = b[:, 3] * nh / sh
    return b


def boxes_from_letterbox(boxes, width, height, size, mode="letterbox"):
    """Inverse de `boxes_to_letterbox` (reprojection des détections, §8.1)."""
    sh, sw = as_hw(size)
    nw, nh, dx, dy = resize_params(width, height, size, mode)
    b = np.asarray(boxes, dtype=np.float64).copy()
    b[..., 0] = (b[..., 0] * sw - dx) / nw
    b[..., 1] = (b[..., 1] * sh - dy) / nh
    b[..., 2] = b[..., 2] * sw / nw
    b[..., 3] = b[..., 3] * sh / nh
    return b
