"""Image → détections dans l'image d'origine : prétraitement, passe avant, §8.1-§8.2.

Deux prétraitements : `letterbox` (rapport d'aspect conservé, celui de l'entraînement) et
`stretch` (redimensionnement direct au carré, celui du `darknet detector valid` de
l'époque YOLOv2, hors base). Deux interpolations : `pil` (bilinéaire de Pillow, avec
anti-crénelage en réduction, celle de l'entraînement) et `darknet` (`resize_image` de
Darknet : bilinéaire à coins alignés, sans anti-crénelage), pour comparer à Darknet.
"""

import numpy as np

from yolo.data.letterbox import GRAY, letterbox_image, letterbox_params
from yolo.infer.decode import to_original
from yolo.infer.nms import CONF_THR, IOU_THR, postprocess

MODES = ("letterbox", "stretch")
INTERPS = ("pil", "darknet")


def resize_darknet(x, w, h):
    """(H, W, 3) float → (h, w, 3), comme `resize_image` de Darknet (image.c, hors base) :
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


def preprocess(img, size, mode="letterbox", interp="pil"):
    """Image PIL → (3, size, size) float32 dans [0, 1] et (largeur, hauteur) d'origine."""
    from PIL import Image

    img = img.convert("RGB")
    if mode not in MODES:
        raise ValueError(f"prétraitement inconnu : {mode!r}")
    if interp == "pil":
        if mode == "letterbox":
            x, _ = letterbox_image(img, size)
        else:
            x = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float32) / 255.0
    elif interp == "darknet":
        src = np.asarray(img, dtype=np.float32) / 255.0
        if mode == "letterbox":
            nw, nh, dx, dy = letterbox_params(img.width, img.height, size)
            x = np.full((size, size, 3), GRAY, dtype=np.float32)
            x[dy:dy + nh, dx:dx + nw] = resize_darknet(src, nw, nh)
        else:
            x = resize_darknet(src, size, size)
    else:
        raise ValueError(f"interpolation inconnue : {interp!r}")
    return np.ascontiguousarray(x.transpose(2, 0, 1)), img.size


def detect(net, x, orig_sizes, mode="letterbox", conf_thr=CONF_THR, iou_thr=IOU_THR):
    """`x` (N, 3, S, S) prétraité ; `orig_sizes` [(w, h)]. Rend par image `(boxes, scores,
    labels)`, boîtes `(cx, cy, w, h)` normalisées dans l'image d'origine.
    """
    size = x.shape[-1]
    dets = postprocess(net.forward(x, train=False), net.net, conf_thr, iou_thr)
    out = []
    for (b, s, lab), (w, h) in zip(dets, orig_sizes):
        # §8.1 : letterbox inverse ; en `stretch`, le repère normalisé est déjà celui d'origine
        out.append((to_original(b, w, h, size) if mode == "letterbox" else b, s, lab))
    return out

