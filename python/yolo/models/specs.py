"""Description des réseaux Tiny-YOLO couche par couche (§3), numérotée comme les .cfg Darknet.

Données pures : chaque couche est un dict avec au moins `type`. Les formes de sortie se
déduisent avec `infer_shapes`, les paramètres et les MACs avec `layer_cost`.
"""

from yolo.models import ul_yaml

VOC_CLASSES = 20


def _conv(cout, k=3, act="leaky", s=1):
    # §3 : « conv » = convolution + BN + leaky 0,1 ; la conv de sortie est linéaire avec biais.
    return {"type": "conv", "k": k, "s": s, "cout": cout, "act": act, "bn": act == "leaky"}


def _pool(s=2):
    return {"type": "maxpool", "k": 2, "s": s}


def _head(num_anchors, classes):
    return _conv(num_anchors * (5 + classes), k=1, act="linear")


# Couches 0-11 communes aux deux réseaux (§3.1, §3.2).
_BACKBONE = [
    _conv(16), _pool(),
    _conv(32), _pool(),
    _conv(64), _pool(),
    _conv(128), _pool(),
    _conv(256), _pool(),
    _conv(512), _pool(s=1),  # couche 11 : maxpool 2×2 stride 1 (§4.4)
]

# §3.1 — 5 ancres × (5 + 20) = 125 filtres.
TINY_YOLOV2_VOC = {
    "name": "tiny-yolov2-voc",
    "input": (3, 416, 416),
    "classes": VOC_CLASSES,
    "layers": _BACKBONE + [
        _conv(1024),
        _conv(1024),
        _head(5, VOC_CLASSES),
    ],
}

# §3.2 — 3 ancres × (5 + 20) = 75 filtres par tête ; ancres de yolov3-tiny.cfg (hors base).
TINY_YOLOV3_VOC = {
    "name": "tiny-yolov3-voc",
    "input": (3, 416, 416),
    "classes": VOC_CLASSES,
    "anchors": [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
    "layers": _BACKBONE + [
        _conv(1024),                                             # 12
        _conv(256, k=1),                                         # 13
        _conv(512),                                              # 14
        _head(3, VOC_CLASSES),                                   # 15
        {"type": "yolo", "mask": [3, 4, 5]},                     # 16 : grandes ancres, 13×13
        {"type": "route", "from": [13]},                         # 17
        _conv(128, k=1),                                         # 18
        {"type": "upsample", "s": 2},                            # 19
        {"type": "route", "from": [19, 8]},                      # 20 : concaténation (§10.3)
        _conv(256),                                              # 21
        _head(3, VOC_CLASSES),                                   # 22
        {"type": "yolo", "mask": [0, 1, 2]},                     # 23 : petites ancres, 26×26
    ],
}



def _csp(c, base):
    """CSP block of YOLOv4-tiny starting at layer `base` (a conv of c channels): grouped
    route (second half), two convs of c/2, concat, 1×1 conv, concat with `base`, maxpool."""
    return [
        _conv(c),                                                          # base
        {"type": "route", "from": [base], "groups": 2, "group_id": 1},     # base + 1
        _conv(c // 2),                                                     # base + 2
        _conv(c // 2),                                                     # base + 3
        {"type": "route", "from": [base + 3, base + 2]},                   # base + 4
        _conv(c, k=1),                                                     # base + 5
        {"type": "route", "from": [base, base + 5]},                       # base + 6
        _pool(),                                                           # base + 7
    ]


# YOLOv4-tiny COCO (AlexeyAB yolov4-tiny.cfg, outside the base): two stride-2 convs, three
# CSP blocks, two heads with scale_x_y = 1.05. The second head uses mask 1,2,3, as in the cfg.
COCO_CLASSES = 80
TINY_YOLOV4_COCO = {
    "name": "tiny-yolov4-coco",
    "input": (3, 416, 416),
    "classes": COCO_CLASSES,
    "anchors": [[10, 14], [23, 27], [37, 58], [81, 82], [135, 169], [344, 319]],
    "layers": [_conv(32, s=2), _conv(64, s=2)] + _csp(64, 2) + _csp(128, 10) + _csp(256, 18) + [
        _conv(512),                                                        # 26
        _conv(256, k=1),                                                   # 27
        _conv(512),                                                        # 28
        _head(3, COCO_CLASSES),                                            # 29
        {"type": "yolo", "mask": [3, 4, 5], "scale_x_y": 1.05},            # 30 : 13×13
        {"type": "route", "from": [27]},                                   # 31
        _conv(128, k=1),                                                   # 32
        {"type": "upsample", "s": 2},                                      # 33
        {"type": "route", "from": [33, 23]},                               # 34
        _conv(256),                                                        # 35
        _head(3, COCO_CLASSES),                                            # 36
        {"type": "yolo", "mask": [1, 2, 3], "scale_x_y": 1.05},            # 37 : 26×26
    ],
}



def _route(*src):
    return {"type": "route", "from": list(src)}


def _elan(c, base):
    """ELAN block of YOLOv7-tiny at layer `base` (input: layer base − 1): two 1×1 convs of c,
    two 3×3 convs of c chained on the second, 4-source concat, 1×1 conv of 2c."""
    return [
        _conv(c, k=1),                                                     # base
        _route(base - 1),                                                  # base + 1
        _conv(c, k=1),                                                     # base + 2
        _conv(c),                                                          # base + 3
        _conv(c),                                                          # base + 4
        _route(base, base + 2, base + 3, base + 4),                        # base + 5
        _conv(2 * c, k=1),                                                 # base + 6
    ]


def _v7_head(src, c, mask):
    return [_route(src), _conv(c), _head(3, COCO_CLASSES),
            {"type": "yolo", "mask": mask, "scale_x_y": 2.0, "new_coords": 1}]


# YOLOv7-tiny COCO (AlexeyAB yolov7-tiny.cfg, outside the base, T26.0): leaky, ELAN blocks
# with 4-source concats, SPP with stride-1 maxpools 5 / 9 / 13 (layers 36-41), three heads
# (52, 26, 13 at 416) with `new_coords` (sigmoid in the decoding, wh = (2σ)²·p) and
# scale_x_y = 2.
TINY_YOLOV7_COCO = {
    "name": "tiny-yolov7-coco",
    "input": (3, 416, 416),
    "classes": COCO_CLASSES,
    "anchors": [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90],
                [156, 198], [373, 326]],
    "layers": [_conv(32, s=2), _conv(64, s=2)] + _elan(32, 2) + [_pool()] + _elan(64, 10)
    + [_pool()] + _elan(128, 18) + [_pool()] + _elan(256, 26) + [
        _conv(256, k=1),                                                   # 33
        _route(32),                                                        # 34
        _conv(256, k=1),                                                   # 35
        {"type": "maxpool", "k": 5, "s": 1},                               # 36
        _route(35),                                                        # 37
        {"type": "maxpool", "k": 9, "s": 1},                               # 38
        _route(35),                                                        # 39
        {"type": "maxpool", "k": 13, "s": 1},                              # 40
        _route(40, 38, 36, 35),                                            # 41 : SPP
        _conv(256, k=1),                                                   # 42
        _route(33, 42),                                                    # 43
        _conv(256, k=1),                                                   # 44
        _conv(128, k=1),                                                   # 45
        {"type": "upsample", "s": 2},                                      # 46
        _route(24),                                                        # 47
        _conv(128, k=1),                                                   # 48
        _route(48, 46),                                                    # 49
    ] + _elan(64, 50) + [
        _conv(64, k=1),                                                    # 57
        {"type": "upsample", "s": 2},                                      # 58
        _route(16),                                                        # 59
        _conv(64, k=1),                                                    # 60
        _route(60, 58),                                                    # 61
    ] + _elan(32, 62) + [
        _conv(128, s=2),                                                   # 69
        _route(69, 56),                                                    # 70
    ] + _elan(64, 71) + [
        _conv(256, s=2),                                                   # 78
        _route(78, 44),                                                    # 79
    ] + _elan(128, 80)
    + _v7_head(68, 128, [0, 1, 2])                                         # 87-90 : 52×52
    + _v7_head(77, 256, [3, 4, 5])                                         # 91-94 : 26×26
    + _v7_head(86, 512, [6, 7, 8]),                                        # 95-98 : 13×13
}

# YOLOv5n COCO (Ultralytics v6.0+ YAML, T33.1): `ul_graph` schema, every layer reads `from`;
# conv + BN + SiLU, C3 / SPPF blocks as `add` / `concat` / stride-1 `maxpool` layers, three
# anchor heads (80, 40, 20 at 640).
YOLOV5N_COCO = ul_yaml.parse(ul_yaml.YOLOV5N, name="yolov5n-coco")

NETWORKS = {net["name"]: net for net in (TINY_YOLOV2_VOC, TINY_YOLOV3_VOC, TINY_YOLOV4_COCO,
                                         TINY_YOLOV7_COCO, YOLOV5N_COCO)}


def layer_sources(i, layer):
    """Indices of the maps read by layer `i` (−1: network input).

    Darknet layers read the previous one, a route its absolute `from` list; the Ultralytics
    layers (T33.1) carry `from` on every type, an index or a list, −1 meaning the previous.
    """
    f = layer.get("from")
    if f is None:
        return [i - 1]
    if layer["type"] == "route":
        return list(f)
    return [i - 1 if j == -1 else j for j in (f if isinstance(f, list) else [f])]


def infer_shapes(net):
    """Formes (C, H, W) d'entrée et de sortie de chaque couche, dans l'ordre de `layers`."""
    shapes = []  # (entrée, sortie)
    prev = tuple(net["input"])
    for i, layer in enumerate(net["layers"]):
        t = layer["type"]
        if "from" in layer and t != "route":
            srcs = [tuple(net["input"]) if j < 0 else shapes[j][1]
                    for j in layer_sources(i, layer)]
            prev = srcs[0]
        c, h, w = prev
        if t == "conv":
            pad = layer.get("p", layer["k"] // 2)
            s = layer["s"]
            out = (layer["cout"], (h + 2 * pad - layer["k"]) // s + 1,
                   (w + 2 * pad - layer["k"]) // s + 1)
        elif t == "maxpool":
            # §4.4 : stride 1 complété par réplication (centré pour k impair), taille conservée.
            out = (c, h, w) if layer["s"] == 1 else (c, h // 2, w // 2)
        elif t == "upsample":
            out = (c, h * layer["s"], w * layer["s"])
        elif t == "route":
            srcs = [shapes[j][1] for j in layer["from"]]
            if any(sh[1:] != srcs[0][1:] for sh in srcs):
                raise ValueError(f"couche {i} : route de cartes de tailles différentes {srcs}")
            out = (sum(sh[0] for sh in srcs),) + srcs[0][1:]
            groups = layer.get("groups", 1)
            if groups > 1:  # CSP: one channel group of a single source
                if len(srcs) != 1 or out[0] % groups:
                    raise ValueError(f"couche {i} : route groupée invalide")
                out = (out[0] // groups,) + out[1:]
            prev = out  # une route n'a pas d'entrée propre : elle lit ses sources
        elif t == "concat":
            if any(sh[1:] != srcs[0][1:] for sh in srcs):
                raise ValueError(f"layer {i}: concat of maps of different sizes {srcs}")
            out = (sum(sh[0] for sh in srcs),) + srcs[0][1:]
        elif t == "add":
            if any(sh != srcs[0] for sh in srcs):
                raise ValueError(f"layer {i}: add of maps of different shapes {srcs}")
            out = srcs[0]
        elif t == "slice":
            out = (layer["stop"] - layer["start"], h, w)
        elif t in ("yolo", "region"):  # sorties : identité, décodées hors du réseau (§8.1)
            out = (c, h, w)
        else:
            raise ValueError(f"couche {i} : type inconnu {t!r}")
        shapes.append((prev, out))
        prev = out
    return shapes


def layer_cost(layer, in_shape, out_shape):
    """(paramètres, MACs) d'une couche ; nuls hors convolution.

    §4.2 : conv + BN = k²·Cin·Cout poids + γ, β (2 par canal), sans biais de convolution ;
    conv linéaire = k²·Cin·Cout poids + Cout biais.
    """
    if layer["type"] != "conv":
        return 0, 0
    cin = in_shape[0]
    cout, h, w = out_shape
    weights = layer["k"] ** 2 * cin * cout
    params = weights + (2 * cout if layer["bn"] else cout)
    return params, h * w * weights
