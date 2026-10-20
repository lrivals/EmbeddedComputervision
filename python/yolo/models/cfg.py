"""Parser des fichiers `.cfg` Darknet (format hors base) vers le format de `yolo.models.specs`.

Sections reconnues : [net], [convolutional], [maxpool], [route], [upsample], [yolo], [region].
Les indices de route négatifs sont rendus absolus. Une route `groups`/`group_id` (CSP de
YOLOv4-tiny) n'a qu'une source ; `scale_x_y` d'une [yolo] est gardé s'il diffère de 1. Les ancres sont ramenées en pixels pour
une entrée de 416 (conventions.md) : celles de [region] (YOLOv2) sont en cellules de 32 px.

YOLOv7-tiny (T26.0): a [yolo] with `new_coords=1` follows a conv with `activation=logistic`;
the conv is kept **linear** and the head gets `new_coords: 1`, the sigmoid moves into the
decoding (the engine heads stay linear int8). A stride-1 [maxpool] of odd size (SPP, 5 / 9 /
13) is centred, as in Darknet (padding size − 1, offset −padding/2).
"""

from pathlib import Path

REGION_STRIDE = 32  # une cellule de la grille 13×13 de YOLOv2 = 32 px à 416


def _sections(text):
    sections = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].split(";", 1)[0].strip()
        if not line:
            continue
        if line.startswith("["):
            sections.append((line.strip("[]").strip(), {}))
        else:
            if not sections:
                raise ValueError(f"option hors section : {raw!r}")
            key, _, value = line.partition("=")
            sections[-1][1][key.strip()] = value.strip()
    return sections


def _ints(value):
    return [int(v) for v in value.split(",") if v.strip()]


def _floats(value):
    return [float(v) for v in value.split(",") if v.strip()]


def _pairs(values):
    return [[values[i], values[i + 1]] for i in range(0, len(values), 2)]


def _conv(opts, i):
    k = int(opts.get("size", 1))
    if "padding" in opts:
        pad = int(opts["padding"])
    else:
        pad = k // 2 if int(opts.get("pad", 0)) else 0
    if pad != k // 2:
        raise ValueError(f"couche {i} : padding {pad} non supporté (attendu k//2)")
    act = opts.get("activation", "logistic")
    act = "silu" if act == "swish" else act  # Darknet's name of the SiLU (T26.2)
    if act not in ("leaky", "linear", "logistic", "silu"):
        raise ValueError(f"couche {i} : activation {act!r} non supportée")
    return {"type": "conv", "k": k, "s": int(opts.get("stride", 1)), "cout": int(opts["filters"]),
            "act": act, "bn": bool(int(opts.get("batch_normalize", 0)))}


def parse_cfg(source, name=None):
    """`source` : chemin d'un `.cfg` ou son texte. Renvoie un dict au format de `specs`."""
    path = Path(source) if "\n" not in str(source) else None
    text = path.read_text() if path is not None else source
    sections = _sections(text)
    if not sections or sections[0][0] not in ("net", "network"):
        raise ValueError("le .cfg doit commencer par [net]")
    net_opts = sections[0][1]
    net = {
        "name": name or (path.stem if path is not None else "cfg"),
        "input": (int(net_opts.get("channels", 3)), int(net_opts["height"]),
                  int(net_opts["width"])),
        "layers": [],
    }
    layers = net["layers"]
    for kind, opts in sections[1:]:
        i = len(layers)
        if kind in ("convolutional", "conv"):
            layers.append(_conv(opts, i))
        elif kind in ("maxpool", "max"):
            k = int(opts.get("size", opts.get("stride", 1)))
            s = int(opts.get("stride", 1))
            if int(opts.get("padding", k - 1)) != k - 1 or (s != 1 and k != s):
                raise ValueError(f"couche {i} : maxpool {k}/{s} non supporté")
            layers.append({"type": "maxpool", "k": k, "s": s})
        elif kind == "route":
            src = [j if j >= 0 else i + j for j in _ints(opts["layers"])]
            if any(not 0 <= j < i for j in src):
                raise ValueError(f"couche {i} : route vers {src} invalide")
            route = {"type": "route", "from": src}
            groups = int(opts.get("groups", 1))
            if groups > 1:
                if len(src) != 1:
                    raise ValueError(f"couche {i} : route groupée à plusieurs sources {src}")
                group_id = int(opts.get("group_id", 0))
                if not 0 <= group_id < groups:
                    raise ValueError(f"couche {i} : group_id {group_id} hors de [0, {groups})")
                route.update(groups=groups, group_id=group_id)
            layers.append(route)
        elif kind == "upsample":
            layers.append({"type": "upsample", "s": int(opts.get("stride", 2))})
        elif kind == "yolo":
            net["classes"] = int(opts["classes"])
            net["anchors"] = _pairs(_ints(opts["anchors"]))
            yolo = {"type": "yolo", "mask": _ints(opts["mask"])}
            scale_xy = float(opts.get("scale_x_y", 1))
            if scale_xy != 1:
                yolo["scale_x_y"] = scale_xy
            if int(opts.get("new_coords", 0)):
                # T26.0: the sigmoid of the head conv is applied by the decoding.
                if not layers or layers[-1].get("act") != "logistic":
                    raise ValueError(f"couche {i} : new_coords sans conv logistic avant")
                layers[-1]["act"] = "linear"
                yolo["new_coords"] = 1
            layers.append(yolo)
        elif kind == "region":
            net["classes"] = int(opts["classes"])
            net["anchors"] = [[a * REGION_STRIDE for a in pair]
                              for pair in _pairs(_floats(opts["anchors"]))]
            layers.append({"type": "region", "num": int(opts["num"])})
        else:
            raise ValueError(f"couche {i} : section [{kind}] non supportée")
    for i, layer in enumerate(layers):
        if layer.get("act") == "logistic":
            raise ValueError(f"couche {i} : activation 'logistic' hors d'une tête new_coords")
    return net
