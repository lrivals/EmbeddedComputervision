"""Per-operation traceability: cfg operation → Python, golden C++, HLS, `LayerDesc` (T28.3).

    python tools/traceability.py            # writes docs/traceability.md
    python tools/traceability.py --check    # CI: fails when a symbol is missing or the doc is stale

Each cfg operation of the networks (conv, its output stage, maxpool, route, upsample, the
detection heads) is mapped to the function that implements it at every stage, with file and
line. The table below is the only hand-written part: the lines are found by searching the
definitions, so `--check` fails when a symbol disappears (renamed, deleted) and when the
generated document no longer matches the sources (a symbol moved: run the tool again).
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "docs" / "traceability.md"

PY_INT = "python/yolo/quant/int_layers.py"
PY_MODEL = "python/yolo/quant/int_model.py"
GOLDEN_CONV = "cpp/golden/include/golden/conv.hpp"
GOLDEN_ENGINE = "cpp/golden/src/engine.cpp"
GOLDEN_HWPP = "cpp/golden/include/golden/hw_postproc.hpp"
HLS_PE = "hls/kernels/conv_pe.cpp"
HLS_OUT = "hls/kernels/output_stage.hpp"
HLS_POST = "hls/kernels/postproc.cpp"
DESC = "hls/kernels/layer_desc.hpp"

# (cfg operation, what it computes, Python, golden, HLS, LayerDesc fields).
# A symbol is "path::name"; name is a function, "Class.method" (Python), "Class::method" (C++)
# or a struct field (LayerDesc).
TABLE = [
    ("[convolutional] (MACs)", "int32 accumulator Σ w·x + b",
     [f"{PY_INT}::conv_acc"], [f"{GOLDEN_CONV}::conv_layer"],
     [f"{HLS_PE}::compute", f"{HLS_PE}::conv_tile", f"{HLS_PE}::load_input",
      f"{HLS_PE}::load_weights"],
     ["k", "pad", "cin", "cout", "h", "w", "w_off", "b_off", "tr", "tc", "fold", "wbits"]),
    ("[convolutional] requantization", "(acc·M0 + 2ⁿ⁻¹) >> n",
     [f"{PY_INT}::requantize", f"{PY_INT}::rshift_round"], [f"{GOLDEN_CONV}::requantize"],
     [f"{HLS_OUT}::store_tile"], ["shift", "m0_off"]),
    ("activation=leaky", "y > 0 ? y : (13y + 64) >> 7",
     [f"{PY_INT}::leaky_int"], [f"{GOLDEN_CONV}::leaky_int"], [f"{HLS_OUT}::store_tile"],
     ["leaky"]),
    ("activation=linear (heads)", "saturation only",
     [f"{PY_INT}::conv_out"], [f"{GOLDEN_CONV}::conv_layer"], [f"{HLS_OUT}::store_tile"],
     ["leaky"]),
    ("activation=silu / swish (T26.2)", "requant to the table scale, 256-entry int8 table",
     [f"{PY_INT}::silu_int", "python/yolo/quant/lut.py::silu_lut",
      "python/yolo/layers/activations.py::silu_forward"],
     [f"{GOLDEN_CONV}::silu_lookup"], [f"{HLS_OUT}::store_tile"], ["leaky", "m0_off"]),
    ("saturation", "clip to [−qmax, qmax]",
     [f"{PY_INT}::clip_q"], [f"{GOLDEN_CONV}::clip_q"], [f"{HLS_OUT}::store_tile"], ["qmax"]),
    ("ABFT checksum (T25.2)", "Σ_f acc_f = conv(x, Σ_f w_f) + Σ_f b_f (mod 2³²)",
     ["python/yolo/quant/abft.py::check", "python/yolo/quant/abft.py::AbftMonitor"],
     [f"{GOLDEN_CONV}::conv_layer"],
     [f"{HLS_PE}::load_cs", f"{HLS_PE}::compute", f"{HLS_OUT}::AbftTile", f"{HLS_PE}::conv_layer"],
     ["m0_off"]),
    ("[maxpool] size=2 stride=2 / 1", "fused into the conv output stage",
     [f"{PY_INT}::maxpool_int", "python/yolo/layers/pool.py::maxpool_forward"],
     [f"{GOLDEN_CONV}::conv_layer"], [f"{HLS_OUT}::store_tile"], ["pool_k", "pool_s"]),
    ("[maxpool] size=5 stride=1 (SPP, T26.0)", "centred window, edge replication",
     ["python/yolo/layers/pool.py::pool_pads"], [f"{GOLDEN_CONV}::conv_layer"],
     [f"{HLS_OUT}::store_tile"], ["pool_k", "pool_s", "pool_lo"]),
    ("[maxpool] pre-pool map", "map before pooling written for a route",
     [f"{PY_MODEL}::IntNetwork.forward"], [f"{GOLDEN_ENGINE}::Engine::place_views"],
     [f"{HLS_OUT}::store_tile"], ["prepool_off"]),
    ("[route] (concat, groups)", "views over the sources, no copy",
     [f"{PY_INT}::route_int", "python/yolo/layers/route.py::route_forward"],
     [f"{GOLDEN_ENGINE}::concat", f"{GOLDEN_ENGINE}::Engine::place_views"],
     [f"{HLS_PE}::load_input"],
     ["nseg", "seg0_off", "seg0_c", "seg0_h", "seg0_w", "seg1_off", "seg1_c", "seg1_h",
      "seg1_w"]),
    ("[upsample] stride=2", "address ÷ 2^up when reading",
     [f"{PY_INT}::upsample_int", "python/yolo/layers/upsample.py::upsample_forward"],
     [f"{GOLDEN_ENGINE}::Engine::place_views"], [f"{HLS_PE}::load_input"],
     ["seg0_up", "seg1_up"]),
    ("[region] / [yolo] decode", "tables σ, e^t, softmax; Q4 corners, Q16 scores",
     ["python/yolo/infer/hw_postproc.py::decode_head", "python/yolo/quant/lut.py::sigmoid_lut"],
     [f"{GOLDEN_HWPP}::decode_cell", f"{GOLDEN_HWPP}::cell_scores"], [f"{HLS_POST}::yolo_post"],
     []),
    ("NMS", "unsorted, 256 slots",
     ["python/yolo/infer/hw_postproc.py::nms_nosort"], [f"{GOLDEN_HWPP}::Selector::push"],
     [f"{HLS_POST}::yolo_post"], []),
]


def _def_patterns(path, name):
    """Regexes of a definition of `name` in `path` (outer class first for methods)."""
    if path.endswith(".py"):
        parts = name.split(".")
        return [rf"^\s*(def|class)\s+{re.escape(p)}\b" for p in parts]
    if path.endswith(DESC.rsplit("/", 1)[1]) and "::" not in name:
        # LayerDesc field, or helper method
        return [rf"^\s*int32_t\b[^;(]*\b{re.escape(name)}\b|^\s*int\s+{re.escape(name)}\s*\("]
    parts = name.split("::")
    outer = [rf"^\s*(struct|class)\s+{re.escape(p)}\b" for p in parts[:-1]]
    last = re.escape(parts[-1])
    qual = re.escape("::".join(parts))
    # free function, method defined out of class (Class::name), or inline method (name(...) {)
    fn = rf"^[^;/]*\b({qual}|{last})\s*\([^;]*$|^\s*(struct|class)\s+{last}\b"
    return (outer if not re.search(qual, (ROOT / path).read_text()) else []) + [fn]


def locate(sym):
    """'path::name' → (path, line), or (path, None) when the definition is not found."""
    path, _, name = sym.partition("::")
    f = ROOT / path
    if not f.exists():
        return path, None
    lines = f.read_text().splitlines()
    start = 0
    for pat in _def_patterns(path, name):
        rx = re.compile(pat)
        for k in range(start, len(lines)):
            if rx.search(lines[k]):
                start = k
                break
        else:
            return path, None
    return path, start + 1


def _cell(syms):
    out = []
    for s in syms:
        path, line = locate(s)
        name = s.partition("::")[2]
        out.append(f"`{name}` ([{Path(path).name}:{line}](../{path}#L{line}))" if line
                   else f"`{name}` (MISSING in {path})")
    return "<br>".join(out) or "—"


def render():
    missing = []
    rows = []
    for op, what, py, golden, hls, desc in TABLE:
        for s in py + golden + hls + [f"{DESC}::{x}" for x in desc]:
            if locate(s)[1] is None:
                missing.append(s)
        fields = ", ".join(f"`{x}`" for x in desc) or "—"
        rows.append(f"| {op} | {what} | {_cell(py)} | {_cell(golden)} | {_cell(hls)} | {fields} |")
    text = "\n".join([
        "# Traceability: cfg operation → implementation at every stage (T28.3)",
        "",
        "Generated by `python tools/traceability.py`; checked in `make ci` "
        "(`--check`: a symbol missing or moved fails). Python is the specification "
        "(`python/yolo/quant/int_layers.py`), the golden C++ its bit-exact mirror, the HLS "
        "kernel reuses the golden scalar operators (`store_tile`). Equality of the three: "
        "`results/equivalence.md`.",
        "",
        "| cfg operation | Computes | Python (integer) | Golden C++ | HLS | `LayerDesc` fields |",
        "|---|---|---|---|---|---|",
        *rows, ""])
    return text, missing


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    text, missing = render()
    for s in missing:
        print(f"missing symbol: {s}")
    if args.check:
        stale = not DOC.exists() or DOC.read_text() != text
        if stale:
            print(f"{DOC.relative_to(ROOT)} is stale: python tools/traceability.py")
        sys.exit(1 if missing or stale else 0)
    DOC.write_text(text)
    print(f"{DOC.relative_to(ROOT)}: {len(TABLE)} operations, {len(missing)} missing symbols")
    sys.exit(1 if missing else 0)


if __name__ == "__main__":
    main()
