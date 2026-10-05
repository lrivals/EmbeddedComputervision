"""Export du modèle entier : manifest + blobs + LUT (T4.7, conventions.md « Format d'échange »).

- `layout(net)` : allocation des tampons d'activations (§10.3), commune au manifest
  d'exemple (`tools/make_example_manifest.py`) et à l'export réel :
  - chaîne principale en ping-pong A/B (bascule à chaque sortie allouée sur la chaîne) ;
  - une carte lue par plusieurs couches est gardée dans un tampon propre (`S`, `S1`…) ;
  - les sources d'une route multiple sont écrites bout à bout dans `R` (`R1`…), la première
    au début : la concaténation est gratuite ; une conv à maxpool fusionné y écrit sa carte
    avant pooling (`prepool_out`) ;
  - les têtes vont dans `H<grille>`.
- `build_manifest(net, scales, …)` : manifest au format du schéma ; `export_model(qm, dir)`
  écrit `manifest.json`, `weights.bin`, `bias.bin`, `requant.bin` et `luts.bin` ;
  `load_model(dir)` relit le tout en `QuantModel` (lecteur indépendant de l'export).
"""

import json
from pathlib import Path

import numpy as np

from yolo.models.graph import OUTPUT_TYPES
from yolo.models.specs import infer_shapes

ALIGN = 64
INPUT_BUF = "A"
FORMAT_VERSION = 1


def align(n):
    return -(-n // ALIGN) * ALIGN


def nbytes(shape):
    c, h, w = shape
    return c * h * w  # int8


def _consumers(layers):
    """{i: [couches qui lisent la sortie logique de i]}."""
    cons = {i: [] for i in range(-1, len(layers))}
    for i, layer in enumerate(layers):
        srcs = layer["from"] if layer["type"] == "route" else [i - 1]
        for j in srcs:
            cons[j].append(i)
    return cons


def layout(net):
    """Placement de chaque tenseur. Rend (entries, buffers) :
    entries[i] = dict des champs de tampon de la couche i (`in`, `out`, `prepool_out`,
    `fused_pool`, `fused_into`) ; buffers = {nom: taille en octets}.
    """
    layers = net["layers"]
    shapes = infer_shapes(net)
    cons = _consumers(layers)

    # Maxpool → conv qui le précède (fusion dans l'étage de sortie, §10.3) ; si une route
    # lit aussi la conv, celle-ci écrit en plus sa carte avant pooling (`prepool_out`).
    fused = {i: i - 1 for i, layer in enumerate(layers)
             if layer["type"] == "maxpool" and i > 0 and layers[i - 1]["type"] == "conv"}
    for i, layer in enumerate(layers):
        if layer["type"] == "maxpool" and i not in fused:
            raise ValueError(f"maxpool {i} non fusionnable (le matériel le fusionne à la conv)")
    pool_of = {conv: pool for pool, conv in fused.items()}

    # Routes multiples : sources bout à bout dans un tampon dédié.
    route_slot = {}  # source -> (tampon, offset)
    n_routes = 0
    for layer in layers:
        if layer["type"] == "route" and len(layer["from"]) > 1:
            buf = "R" if n_routes == 0 else f"R{n_routes}"
            n_routes += 1
            off = 0
            for j in layer["from"]:
                if j in route_slot or j < 0 or layers[j]["type"] in ("route", "maxpool"):
                    raise ValueError(f"source de route {j} non prise en charge")
                route_slot[j] = (buf, off)
                off += nbytes(shapes[j][1])

    head_of = {}  # conv de tête -> grille
    for i, layer in enumerate(layers):
        if layer["type"] in OUTPUT_TYPES:
            head_of[i - 1] = shapes[i][1][1]

    sizes = {INPUT_BUF: nbytes(net["input"])}

    def reserve(buf, off, shape):
        sizes[buf] = max(sizes.get(buf, 0), off + nbytes(shape))
        return {"buf": buf, "offset": off}

    entries = [{} for _ in layers]
    where = {-1: {"buf": INPUT_BUF, "offset": 0}}
    chain = INPUT_BUF
    n_keep = 0
    for i, layer in enumerate(layers):
        t = layer["type"]
        ins, outs = shapes[i]
        e = entries[i]
        if t in ("conv", "upsample"):
            e["in"] = where[i - 1]
            pool = pool_of.get(i)
            final = shapes[pool][1] if pool is not None else outs
            readers = cons[pool] if pool is not None else cons[i]
            if t == "conv":
                e["fused_pool"] = (None if pool is None else
                                   {"layer": pool, "k": layers[pool]["k"],
                                    "s": layers[pool]["s"]})
            if i in route_slot and pool is not None:
                e["prepool_out"] = reserve(*route_slot[i], outs)
            if i in route_slot and pool is None:
                out = reserve(*route_slot[i], final)
            elif i in head_of:
                out = reserve(f"H{head_of[i]}", 0, final)
            elif len(readers) > 1:
                out = reserve("S" if n_keep == 0 else f"S{n_keep}", 0, final)
                n_keep += 1
            else:
                chain = "B" if chain == "A" else "A"
                out = reserve(chain, 0, final)
            e["out"] = out
            where[i] = e.get("prepool_out", out)
            if pool is not None:
                where[pool] = out
        elif t == "maxpool":
            e["fused_into"] = fused[i]
        elif t == "route":
            e["out"] = where[layer["from"][0]]
            where[i] = e["out"]
        elif t in OUTPUT_TYPES:
            e["in"] = where[i - 1]
            where[i] = e["in"]
        else:
            raise ValueError(f"couche {i} : type inconnu {t!r}")
    return entries, dict(sorted(sizes.items()))


def build_manifest(net, input_scale, scale_of, shift_of, lut_offsets=None, exp_frac=None,
                   qmax_of=None):
    """Manifest de `net` (format `specs`).

    `scale_of(i)` : échelle de la sortie de la couche i (−1 : entrée) ; `shift_of(i)` : n de
    la conv i ; `lut_offsets` : {id de tête: offset dans luts.bin} et `exp_frac` : {id de
    tête: bits fractionnaires de la table exponentielle} (optionnels). `qmax_of(i)` :
    saturation de la sortie de la conv i, écrite seulement si ≠ 127 (T9.3).
    """
    layers = net["layers"]
    shapes = infer_shapes(net)
    places, buffers = layout(net)
    w_off = b_off = 0
    entries = []
    for i, layer in enumerate(layers):
        t = layer["type"]
        ins, outs = shapes[i]
        p = places[i]
        e = {"id": i, "type": t}
        if t == "conv":
            k, cout = layer["k"], layer["cout"]
            pool = p["fused_pool"]
            e.update(k=k, s=layer["s"], pad=k // 2, cin=ins[0], cout=cout, act=layer["act"],
                     fused_pool=pool)
            if "prepool_out" in p:
                e["prepool_out"] = p["prepool_out"]
            src = i - 1
            e.update(w_offset=w_off, b_offset=b_off, m0_offset=b_off, shift=shift_of(i),
                     in_scale=scale_of(src), out_scale=scale_of(i), **{"in": p["in"]},
                     out=p["out"],
                     out_shape=list(shapes[pool["layer"]][1] if pool else outs))
            if qmax_of is not None and qmax_of(i) != 127:
                e["qmax"] = int(qmax_of(i))
            w_off = align(w_off + k * k * ins[0] * cout)
            b_off = align(b_off + 4 * cout)
        elif t == "maxpool":
            e.update(k=layer["k"], s=layer["s"], fused_into=p["fused_into"],
                     out_shape=list(outs))
        elif t == "upsample":
            e.update(s=layer["s"], scale=scale_of(i), **{"in": p["in"]}, out=p["out"],
                     out_shape=list(outs))
        elif t == "route":
            e.update(**{"from": list(layer["from"])}, layout="contiguous", scale=scale_of(i),
                     out=p["out"], out_shape=list(outs))
        elif t == "yolo":
            e.update(mask=list(layer["mask"]), scale=scale_of(i), **{"in": p["in"]},
                     out_shape=list(outs))
        elif t == "region":
            e.update(num=layer["num"], scale=scale_of(i), **{"in": p["in"]},
                     out_shape=list(outs))
        if t in OUTPUT_TYPES and lut_offsets is not None:
            e["lut_offset"] = lut_offsets[i]
            e["exp_frac"] = exp_frac[i]
        entries.append(e)

    blobs = {"weights.bin": w_off, "bias.bin": b_off, "requant.bin": b_off}
    m = {
        "format_version": FORMAT_VERSION,
        "network": net["name"],
        "classes": net["classes"],
        "input": {"shape": [1, *net["input"]], "scale": input_scale,
                  "buf": {"buf": INPUT_BUF, "offset": 0}},
        "anchors": [[_num(a) for a in pair] for pair in net["anchors"]],
        "blobs": blobs,
        "buffers": buffers,
        "layers": entries,
    }
    return m


def _num(v):
    """Ancre : entier si elle l'est (Tiny-YOLOv3), sinon flottant (34,56 px en v2)."""
    return int(v) if float(v).is_integer() else float(v)


# ---------------------------------------------------------------------- export réel
LUT_NAMES = ("sigmoid", "exp", "softmax_exp")


def export_model(qm, out_dir):
    """Écrit le modèle entier `qm` dans `out_dir` ; rend le manifest."""
    from yolo.quant.int_model import head_luts

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    luts = head_luts(qm)
    lut_offsets, off = {}, 0
    for hid in sorted(luts):
        lut_offsets[hid] = off
        off = align(off + 4 * 256 * len(LUT_NAMES))
    m = build_manifest(qm.net, qm.input_scale, qm.scale_of, lambda i: qm.convs[i].n,
                       lut_offsets, {i: h.exp_frac for i, h in luts.items()},
                       lambda i: qm.convs[i].qmax)
    m["blobs"]["luts.bin"] = off

    w = np.zeros(m["blobs"]["weights.bin"], np.int8)
    b = np.zeros(m["blobs"]["bias.bin"] // 4, "<i4")
    m0 = np.zeros(m["blobs"]["requant.bin"] // 4, "<i4")
    for e in m["layers"]:
        if e["type"] != "conv":
            continue
        c = qm.convs[e["id"]]
        w[e["w_offset"]:e["w_offset"] + c.qW.size] = c.qW.ravel()
        b[e["b_offset"] // 4:e["b_offset"] // 4 + len(c.qb)] = c.qb
        m0[e["m0_offset"] // 4:e["m0_offset"] // 4 + len(c.M0)] = c.M0
    lut = np.zeros(off // 4, "<u4")
    for hid, o in lut_offsets.items():
        for k, name in enumerate(LUT_NAMES):
            lut[o // 4 + 256 * k:o // 4 + 256 * (k + 1)] = getattr(luts[hid], name)
    w.tofile(out_dir / "weights.bin")
    b.tofile(out_dir / "bias.bin")
    m0.tofile(out_dir / "requant.bin")
    lut.tofile(out_dir / "luts.bin")
    (out_dir / "manifest.json").write_text(json.dumps(m, indent=1, ensure_ascii=False) + "\n")
    return m


def load_model(model_dir):
    """Relit un export en `QuantModel` (les échelles des poids s_w ne sont pas exportées)."""
    from yolo.quant.int_model import QConv, QuantModel

    d = Path(model_dir)
    m = json.loads((d / "manifest.json").read_text())
    blobs = {k: (d / k).read_bytes() for k in m["blobs"]}
    for k, size in m["blobs"].items():
        if len(blobs[k]) != size:
            raise ValueError(f"{k} : {len(blobs[k])} octets, {size} attendus")
    w = np.frombuffer(blobs["weights.bin"], np.int8)
    b = np.frombuffer(blobs["bias.bin"], "<i4")
    m0 = np.frombuffer(blobs["requant.bin"], "<i4")
    layers, convs = [], {}
    for e in m["layers"]:
        t = e["type"]
        if t == "conv":
            k, cin, cout = e["k"], e["cin"], e["cout"]
            layers.append({"type": "conv", "k": k, "s": e["s"], "cout": cout,
                           "act": e["act"], "bn": False})
            qW = w[e["w_offset"]:e["w_offset"] + cout * cin * k * k].reshape(cout, cin, k, k)
            qb = b[e["b_offset"] // 4:e["b_offset"] // 4 + cout]
            M0 = m0[e["m0_offset"] // 4:e["m0_offset"] // 4 + cout]
            convs[e["id"]] = QConv(qW.copy(), qb.astype(np.int32), M0.astype(np.int32),
                                   e["shift"], e["in_scale"], None, e["out_scale"], e["act"],
                                   e["s"], e.get("qmax", 127))
        elif t == "maxpool":
            layers.append({"type": "maxpool", "k": e["k"], "s": e["s"]})
        elif t == "upsample":
            layers.append({"type": "upsample", "s": e["s"]})
        elif t == "route":
            layers.append({"type": "route", "from": list(e["from"])})
        elif t == "yolo":
            layers.append({"type": "yolo", "mask": list(e["mask"])})
        elif t == "region":
            layers.append({"type": "region", "num": e["num"]})
    net = {"name": m["network"], "input": tuple(m["input"]["shape"][1:]),
           "classes": m["classes"], "anchors": m["anchors"], "layers": layers}
    return QuantModel(net=net, input_scale=m["input"]["scale"], convs=convs), m


def load_luts(model_dir, manifest):
    """{id de tête: {nom: table uint32}} depuis `luts.bin`."""
    raw = np.fromfile(Path(model_dir) / "luts.bin", "<u4")
    out = {}
    for e in manifest["layers"]:
        if "lut_offset" in e:
            o = e["lut_offset"] // 4
            out[e["id"]] = {n: raw[o + 256 * k:o + 256 * (k + 1)]
                            for k, n in enumerate(LUT_NAMES)}
    return out
