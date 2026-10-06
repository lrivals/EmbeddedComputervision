"""Image → détections dans l'image d'origine : prétraitement, passe avant, §8.1-§8.2.

Deux prétraitements : `letterbox` (rapport d'aspect conservé, celui de l'entraînement) et
`stretch` (redimensionnement direct à la taille d'entrée, celui du `darknet detector valid` de
l'époque YOLOv2, hors base). Deux interpolations : `pil` (bilinéaire de Pillow, avec
anti-crénelage en réduction, celle de l'entraînement) et `darknet` (`resize_image` de
Darknet : bilinéaire à coins alignés, sans anti-crénelage), pour comparer à Darknet.
"""

import numpy as np

from yolo.data.letterbox import (GRAY, as_hw, image_mode, letterbox_image, letterbox_params,
                                 to_array)
from yolo.infer.decode import to_original
from yolo.infer.nms import CONF_THR, IOU_THR, postprocess, postprocess_int

MODES = ("letterbox", "stretch")
INTERPS = ("pil", "darknet")


def resize_darknet(x, w, h):
    """(H, W, C) float → (h, w, C), comme `resize_image` de Darknet (image.c, hors base) :
    échelle (W−1)/(w−1), interpolation linéaire en x puis en y, sans filtrage préalable.
    """
    H, W = x.shape[:2]

    def coords(n_out, n_in):
        s = np.arange(n_out) * ((n_in - 1) / (n_out - 1) if n_out > 1 else 0.0)
        i = np.minimum(s.astype(np.int64), n_in - 1)
        return i, (s - i)[:, None], np.minimum(i + 1, n_in - 1)

    ix, dx, ix1 = coords(w, W)
    part = x[:, ix] * (1 - dx)[None] + x[:, ix1] * dx[None]
    iy, dy, iy1 = coords(h, H)
    return (part[iy] * (1 - dy)[:, :, None] + part[iy1] * dy[:, :, None]).astype(np.float32)


def preprocess(img, size, mode="letterbox", interp="pil", channels=3):
    """Image PIL → (C, H, W) float32 dans [0, 1] et (largeur, hauteur) d'origine.

    `size` : S (carré) ou (H, W) ; `channels` : 3 (RGB) ou 1 (niveaux de gris).
    """
    from PIL import Image

    img = img.convert(image_mode(channels))
    sh, sw = as_hw(size)
    if mode not in MODES:
        raise ValueError(f"prétraitement inconnu : {mode!r}")
    if interp == "pil":
        if mode == "letterbox":
            x, _ = letterbox_image(img, size, channels)
        else:
            x = to_array(img.resize((sw, sh), Image.BILINEAR), channels)
    elif interp == "darknet":
        src = to_array(img, channels)
        if mode == "letterbox":
            nw, nh, dx, dy = letterbox_params(img.width, img.height, size)
            x = np.full((sh, sw, channels), GRAY, dtype=np.float32)
            x[dy:dy + nh, dx:dx + nw] = resize_darknet(src, nw, nh)
        else:
            x = resize_darknet(src, sw, sh)
    else:
        raise ValueError(f"interpolation inconnue : {interp!r}")
    return np.ascontiguousarray(x.transpose(2, 0, 1)), img.size


def detect(net, x, orig_sizes, mode="letterbox", conf_thr=CONF_THR, iou_thr=IOU_THR):
    """`x` (N, C, H, W) prétraité ; `orig_sizes` [(w, h)]. Rend par image `(boxes, scores,
    labels)`, boîtes `(cx, cy, w, h)` normalisées dans l'image d'origine.
    """
    size = x.shape[-2:]
    dets = postprocess(net.forward(x, train=False), net.net, conf_thr, iou_thr)
    out = []
    for (b, s, lab), (w, h) in zip(dets, orig_sizes):
        # §8.1 : letterbox inverse ; en `stretch`, le repère normalisé est déjà celui d'origine
        out.append((to_original(b, w, h, size) if mode == "letterbox" else b, s, lab))
    return out


def detect_int(inet, luts, x, orig_sizes, mode="letterbox", conf_thr=CONF_THR, iou_thr=IOU_THR,
               post=None):
    """Comme `detect`, avec le modèle entier : `x` float est quantifié en int8 (fait par
    l'hôte), passe avant entière, décodage par tables (§9.4). `luts` : {id de tête: HeadLuts}.
    `post` remplace `postprocess_int` (même signature), par ex. le post-traitement matériel.
    """
    from yolo.quant.quantize import quantize_input

    size = x.shape[-2:]
    dets = (post or postprocess_int)(inet.forward(quantize_input(x)), inet.qm.net, luts,
                                     conf_thr, iou_thr)
    return [(to_original(b, w, h, size) if mode == "letterbox" else b, s, lab)
            for (b, s, lab), (w, h) in zip(dets, orig_sizes)]
