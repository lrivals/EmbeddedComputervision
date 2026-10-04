"""Letterbox : redimensionnement à rapport d'aspect conservé, complété en gris (§1, hors base).

Convention Darknet : l'image est réduite d'un facteur `min(size/w, size/h)`, centrée, et les
bandes sont remplies de gris 0,5. Les boîtes `(cx, cy, w, h)` normalisées passent du repère
de l'image d'origine à celui de l'image letterbox (et inversement) par une affinité exacte.
"""

import numpy as np

GRAY = 0.5


def letterbox_params(width, height, size):
    """(nw, nh, dx, dy) : taille de l'image redimensionnée et décalage dans le carré `size`."""
    scale = min(size / width, size / height)
    nw = max(1, int(round(width * scale)))
    nh = max(1, int(round(height * scale)))
    return nw, nh, (size - nw) // 2, (size - nh) // 2


def letterbox_image(img, size):
    """`img` : image PIL ou tableau (H, W, 3) uint8. Rend (size, size, 3) float32 dans [0, 1]
    et les paramètres de `letterbox_params`.
    """
    from PIL import Image

    if not isinstance(img, Image.Image):
        img = Image.fromarray(np.asarray(img, dtype=np.uint8))
    img = img.convert("RGB")
    nw, nh, dx, dy = letterbox_params(img.width, img.height, size)
    resized = np.asarray(img.resize((nw, nh), Image.BILINEAR), dtype=np.float32) / 255.0
    out = np.full((size, size, 3), GRAY, dtype=np.float32)
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, (nw, nh, dx, dy)


def boxes_to_letterbox(boxes, width, height, size):
    """Boîtes normalisées dans l'image d'origine → normalisées dans l'image letterbox."""
    nw, nh, dx, dy = letterbox_params(width, height, size)
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4).copy()
    b[:, 0] = (b[:, 0] * nw + dx) / size
    b[:, 1] = (b[:, 1] * nh + dy) / size
    b[:, 2] = b[:, 2] * nw / size
    b[:, 3] = b[:, 3] * nh / size
    return b


def boxes_from_letterbox(boxes, width, height, size):
    """Inverse de `boxes_to_letterbox` (reprojection des détections, §8.1)."""
    nw, nh, dx, dy = letterbox_params(width, height, size)
    b = np.asarray(boxes, dtype=np.float64).copy()
    b[..., 0] = (b[..., 0] * size - dx) / nw
    b[..., 1] = (b[..., 1] * size - dy) / nh
    b[..., 2] = b[..., 2] * size / nw
    b[..., 3] = b[..., 3] * size / nh
    return b
