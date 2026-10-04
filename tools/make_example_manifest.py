"""Génère model/example_manifest.json : Tiny-YOLOv3 VOC complet, valeurs fictives (T0.4).

Les formes viennent de yolo.models.specs ; les offsets de blobs et l'allocation des tampons
sont calculés ici. Les échelles sont fictives (puissances de 2), en attendant l'export réel (T4.7).

    python tools/make_example_manifest.py           # écrit le fichier
    python tools/make_example_manifest.py --check   # échoue si le fichier versionné diffère
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from yolo.models.specs import TINY_YOLOV3_VOC, infer_shapes  # noqa: E402

OUTPUT = ROOT / "model" / "example_manifest.json"
ALIGN = 64
SHIFT = 31
IN_SCALE = 2.0**-7    # pixels [0, 1] en int8 symétrique
LEAKY_SCALE = 2.0**-4
LINEAR_SCALE = 2.0**-3

# Tampon de chaque sortie de couche (§10.3) : ping-pong A/B sur la chaîne principale ;
# S garde la couche 13 (lue par 14 et par la route 17) ; R reçoit 19 puis 8 bout à bout
# pour que la route 20 soit gratuite ; H13/H26 reçoivent les têtes.
INPUT_BUF = "A"
OUT_BUF = {
    0: "B", 2: "A", 4: "B", 6: "A", 8: "B", 10: "A",
    12: "B", 13: "S", 14: "A", 15: "H13",
    18: "B", 19: "R", 21: "A", 22: "H26",
}
PREPOOL_OUT = {8: "R"}  # sortie 26×26×256 de la couche 8, avant le maxpool 9


def align(n):
    return -(-n // ALIGN) * ALIGN


def nbytes(shape):
    c, h, w = shape
    return c * h * w  # int8


def build():
    net = TINY_YOLOV3_VOC
    layers = net["layers"]
    shapes = infer_shapes(net)

    # Maxpool → conv qui le précède (fusion dans l'étage de sortie).
    fused = {i: i - 1 for i, layer in enumerate(layers)
             if layer["type"] == "maxpool" and layers[i - 1]["type"] == "conv"}
    pool_of = {conv: pool for pool, conv in fused.items()}

    # Position de la sortie de chaque couche : (tampon, offset).
    where = {}
    route_src = {}
    for i, layer in enumerate(layers):
        if layer["type"] == "route" and len(layer["from"]) > 1:
            # §10.3 : la première source au début du tampon, les suivantes juste après.
            first = layer["from"][0]
            off = nbytes(shapes[first][1])
            for j in layer["from"][1:]:
                route_src[j] = off
                off += nbytes(shapes[j][1])  # carte lue par la route : avant pooling

    w_off = b_off = m_off = 0
    buf_size = {INPUT_BUF: nbytes(net["input"])}
    entries = []
    scale = IN_SCALE
    scale_of = {}
    cur = {"buf": INPUT_BUF, "offset": 0}

    def reserve(buf, offset, shape):
        buf_size[buf] = max(buf_size.get(buf, 0), offset + nbytes(shape))
        return {"buf": buf, "offset": offset}

    for i, layer in enumerate(layers):
        t = layer["type"]
        ins, outs = shapes[i]
        e = {"id": i, "type": t}
        if t == "conv":
            k, cout = layer["k"], layer["cout"]
            out_scale = LEAKY_SCALE if layer["act"] == "leaky" else LINEAR_SCALE
            pool = pool_of.get(i)
            final = shapes[pool][1] if pool is not None else outs
            e.update(
                k=k, s=layer["s"], pad=k // 2, cin=ins[0], cout=cout, act=layer["act"],
                fused_pool=(None if pool is None else
                            {"layer": pool, "k": layers[pool]["k"], "s": layers[pool]["s"]}),
            )
            if i in PREPOOL_OUT:
                e["prepool_out"] = reserve(PREPOOL_OUT[i], route_src[i], outs)
            e.update(
                w_offset=w_off, b_offset=b_off, m0_offset=m_off, shift=SHIFT,
                in_scale=scale, out_scale=out_scale, **{"in": cur},
                out=reserve(OUT_BUF[i], 0 if i in PREPOOL_OUT else route_src.get(i, 0), final),
                out_shape=list(final),
            )
            w_off = align(w_off + k * k * ins[0] * cout)
            b_off = align(b_off + 4 * cout)
            m_off = align(m_off + 4 * cout)
            scale = out_scale
            cur = e["out"]
        elif t == "maxpool":
            e.update(k=layer["k"], s=layer["s"], fused_into=fused[i], out_shape=list(outs))
        elif t == "upsample":
            out = reserve(OUT_BUF[i], 0, outs)
            e.update(s=layer["s"], scale=scale, **{"in": cur}, out=out, out_shape=list(outs))
            cur = out
        elif t == "route":
            srcs = layer["from"]
            first = where[srcs[0]]
            scale = scale_of[srcs[0]]
            e.update(**{"from": srcs}, layout="contiguous", scale=scale, out=first,
                     out_shape=list(outs))
            cur = first
        elif t == "yolo":
            e.update(mask=layer["mask"], scale=scale, **{"in": cur}, out_shape=list(outs))
        entries.append(e)
        where[i] = e.get("prepool_out", e.get("out", cur))
        scale_of[i] = scale

    return {
        "format_version": 1,
        "network": net["name"],
        "classes": net["classes"],
        "input": {"shape": [1, *net["input"]], "scale": IN_SCALE,
                  "buf": {"buf": INPUT_BUF, "offset": 0}},
        "anchors": net["anchors"],
        "blobs": {"weights.bin": w_off, "bias.bin": b_off, "requant.bin": m_off},
        "buffers": dict(sorted(buf_size.items())),
        "layers": entries,
    }


def dumps(manifest):
    return json.dumps(manifest, indent=1, ensure_ascii=False) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text = dumps(build())
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            sys.exit(f"{OUTPUT.relative_to(ROOT)} n'est pas à jour : relancer sans --check")
        print("à jour")
    else:
        OUTPUT.write_text(text)
        print(f"écrit {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
