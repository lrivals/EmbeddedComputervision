"""ROM des poids sur la puce de `yolo_stream` (T10.8) : stream_rom.hpp et stream_rom.cpp.

    python tools/gen_stream_rom.py --out build/hls/stream_rom
    python tools/gen_stream_rom.py --model model/tiny-yolov2-voc --out …

Pour chaque étage dont `stream_model.plan` (KV260, 8 bits) garde les poids sur la puce, un
tableau `const int8_t stream_rom::S<k>[]` rangé [og][ig][i][j][pe·SIMD + s] : la boucle
PIPELINE de `conv_group` lit un mot de PE·SIMD octets par cycle, que `STREAM_ROM_PRAGMAS`
(ARRAY_RESHAPE cyclique de facteur PE·SIMD) met dans une seule ligne de ROM. Le script vérifie
que le plan (repliements, étages en DDR) est celui codé dans hls/stream/yolo_stream.hpp.
"""

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "tools"))

import numpy as np  # noqa: E402

from perf_model import conv_layers  # noqa: E402
from stream_model import load_board, plan  # noqa: E402
from yolo.io.export import unpack4, weight_bytes  # noqa: E402

HPP = ROOT / "hls" / "stream" / "yolo_stream.hpp"
DESC = ROOT / "hls" / "stream" / "stream_desc.hpp"


def hpp_table(name, text=None):
    """Valeurs d'une table constexpr de yolo_stream.hpp ou stream_desc.hpp (entiers ou
    booléens)."""
    text = text or HPP.read_text() + DESC.read_text()
    m = re.search(rf"{name}\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}", text)
    vals = [v.strip() for v in m.group(1).split(",") if v.strip()]
    return [v == "true" if v in ("true", "false") else int(v) for v in vals]


def check_plan(p):
    """Le plan du modèle == tables de yolo_stream.hpp ; rend les étages sur la puce."""
    want = [(s["layer"], s["pe"], s["simd"], s["weights"] == "ddr") for s in p["stages"]]
    got = list(zip(hpp_table("STAGE_LAYER"), hpp_table("STAGE_PE"), hpp_table("STAGE_SIMD"),
                   hpp_table("STAGE_FRAME")))
    if got != want:
        sys.exit(f"yolo_stream.hpp ≠ stream_model.plan :\n  hpp  {got}\n  plan {want}")
    return [k for k, s in enumerate(p["stages"]) if s["weights"] == "puce"]


def layer_weights(model, entry):
    """Poids int8 (C_out, C_in, K, K) d'une conv du manifest (dépaquetés si wbits = 4)."""
    k, cin, cout = entry["k"], entry["cin"], entry["cout"]
    n, wbits = cout * cin * k * k, entry.get("wbits", 8)
    raw = np.fromfile(model / "weights.bin", np.int8, count=weight_bytes(n, wbits),
                      offset=entry["w_offset"])
    return (unpack4(raw, n) if wbits == 4 else raw).reshape(cout, cin, k, k)


def rom_layout(W, pe, simd):
    """(C_out, C_in, K, K) → [og][ig][i][j][pe·SIMD + s], à plat."""
    cout, cin, k, _ = W.shape
    r = W.reshape(cout // pe, pe, cin // simd, simd, k, k)
    return r.transpose(0, 2, 4, 5, 1, 3).reshape(-1)


def c_array(name, values, per_line=32):
    rows = [", ".join(str(int(v)) for v in values[i:i + per_line])
            for i in range(0, len(values), per_line)]
    return f"const int8_t {name}[{len(values)}] = {{\n" + ",\n".join(rows) + "\n};\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", type=Path, default=ROOT / "model" / "tiny-yolov2-voc")
    ap.add_argument("--board", default="kv260")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    man = json.loads((args.model / "manifest.json").read_text())
    entries = {e["id"]: e for e in man["layers"]}
    p = plan(conv_layers(args.model / "manifest.json"),
             load_board(ROOT / "hw" / "boards" / f"{args.board}.yaml"))
    onchip = check_plan(p)

    decls, defs, pragmas = [], [], []
    for k in onchip:
        s = p["stages"][k]
        rom = rom_layout(layer_weights(args.model, entries[s["layer"]]), s["pe"], s["simd"])
        decls.append(f"extern const int8_t S{k}[{rom.size}];  // L{s['layer']:02d}, "
                     f"PE {s['pe']} × SIMD {s['simd']}")
        defs.append(c_array(f"S{k}", rom))
        pragmas.append(f'_Pragma("HLS ARRAY_RESHAPE variable=stream_rom::S{k} cyclic '
                       f'factor={s["pe"] * s["simd"]} dim=1")')
    args.out.mkdir(parents=True, exist_ok=True)
    src = f"{args.model.resolve()} ({man['network']})"
    (args.out / "stream_rom.hpp").write_text(
        f"// Généré par tools/gen_stream_rom.py depuis {src} : ne pas éditer.\n"
        "#pragma once\n\n#include <cstdint>\n\nnamespace stream_rom {\n\n"
        + "\n".join(decls) + "\n\n}  // namespace stream_rom\n\n"
        "#define STREAM_ROM_PRAGMAS \\\n  " + " \\\n  ".join(pragmas) + "\n")
    (args.out / "stream_rom.cpp").write_text(
        f"// Généré par tools/gen_stream_rom.py depuis {src} : ne pas éditer.\n"
        '#include "stream_rom.hpp"\n\nnamespace stream_rom {\n\n' + "\n".join(defs)
        + "\n}  // namespace stream_rom\n")
    total = sum(p["stages"][k]["weight_bytes"] for k in onchip)
    print(f"ROM : étages {onchip}, {total / 2**20:.2f} Mo → {args.out}")


if __name__ == "__main__":
    main()
