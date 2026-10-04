"""Parser des fichiers `.cfg` Darknet (format hors base) vers le format de `yolo.models.specs`.

Sections reconnues : [net], [convolutional], [maxpool], [route], [upsample], [yolo], [region].
Les indices de route négatifs sont rendus absolus. Les ancres sont ramenées en pixels pour
une entrée de 416 (conventions.md) : celles de [region] (YOLOv2) sont en cellules de 32 px.
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
    if act not in ("leaky", "linear"):
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
            layers.append({"type": "maxpool", "k": k, "s": int(opts.get("stride", 1))})
        elif kind == "route":
            src = [j if j >= 0 else i + j for j in _ints(opts["layers"])]
            if any(not 0 <= j < i for j in src):
                raise ValueError(f"couche {i} : route vers {src} invalide")
            layers.append({"type": "route", "from": src})
        elif kind == "upsample":
            layers.append({"type": "upsample", "s": int(opts.get("stride", 2))})
        elif kind == "yolo":
            net["classes"] = int(opts["classes"])
            net["anchors"] = _pairs(_ints(opts["anchors"]))
            layers.append({"type": "yolo", "mask": _ints(opts["mask"])})
        elif kind == "region":
            net["classes"] = int(opts["classes"])
            net["anchors"] = [[a * REGION_STRIDE for a in pair]
                              for pair in _pairs(_floats(opts["anchors"]))]
            layers.append({"type": "region", "num": int(opts["num"])})
        else:
            raise ValueError(f"couche {i} : section [{kind}] non supportée")
    return net
