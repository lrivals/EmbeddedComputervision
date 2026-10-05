"""Compare sw/driver/regmap.hpp à l'en-tête `xyolo_conv_hw.h` généré par Vitis HLS (T7.2).

Le driver écrit les registres s_axilite aux offsets de regmap.hpp ; s'ils diffèrent de ceux
de l'IP exportée (`make hls-export`), le noyau lirait des paramètres faux. Sans en-tête
généré (Vitis absent), le contrôle est sauté (code 0). Code 1 au moindre écart.
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGMAP = ROOT / "sw" / "driver" / "regmap.hpp"

# Nom dans regmap.hpp → suffixe de la macro Vitis XYOLO_CONV_CONTROL_ADDR_<…>.
FIELDS = {
    "CTRL": "AP_CTRL",
    "GIE": "GIE",
    "IER": "IER",
    "ISR": "ISR",
    "ACT_IN": "ACT_IN_DATA",
    "ACT_OUT": "ACT_OUT_DATA",
    "WTS": "WTS_DATA",
    "PRM": "PRM_DATA",
    "D": "D_DATA",
    "DESCS": "DESCS_DATA",
    "N_CALLS": "N_CALLS_DATA",
}
D_BITS = 30 * 32


def regmap_offsets(path=REGMAP):
    text = path.read_text()
    found = dict(re.findall(r"constexpr uint32_t (\w+) = (0x[0-9a-fA-F]+);", text))
    return {k: int(found[k], 16) for k in FIELDS}


def generated_header(board):
    pats = [f"hls/proj_{board}_export/**/xyolo_conv_hw.h", "build/hls/**/xyolo_conv_hw.h"]
    for pat in pats:
        hits = sorted(ROOT.glob(pat))
        if hits:
            return hits[0]
    return None


def header_values(path):
    text = path.read_text()
    vals = dict(re.findall(r"#define\s+XYOLO_CONV_CONTROL_(\w+)\s+(0x[0-9a-fA-F]+|\d+)", text))
    return {k: int(v, 0) for k, v in vals.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--board", default="kv260")
    ap.add_argument("--header", type=Path, help="xyolo_conv_hw.h (défaut : recherche)")
    args = ap.parse_args()

    hdr = args.header or generated_header(args.board)
    if hdr is None or not hdr.exists():
        print("xyolo_conv_hw.h absent (make hls-export, Vitis requis) : contrôle sauté")
        return 0
    ours, gen = regmap_offsets(), header_values(hdr)
    bad = 0
    for name, macro in FIELDS.items():
        key = f"ADDR_{macro}"
        if key not in gen:
            print(f"{name} : {key} absent de {hdr}")
            bad += 1
        elif gen[key] != ours[name]:
            print(f"{name} : regmap.hpp 0x{ours[name]:02x}, Vitis 0x{gen[key]:02x}")
            bad += 1
    bits = gen.get("BITS_D_DATA")
    if bits is not None and bits != D_BITS:
        print(f"d : {bits} bits au lieu de {D_BITS} (LayerDesc agrégé, 30 × int32)")
        bad += 1
    print(f"{hdr} : {'OK' if bad == 0 else f'{bad} écart(s)'}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
