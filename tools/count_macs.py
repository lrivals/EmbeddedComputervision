"""Paramètres et MACs des réseaux Tiny-YOLO, au format des tableaux du §3 (T0.5).

    python tools/count_macs.py --net all
    python tools/count_macs.py --manifest model/example_manifest.json
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from yolo.models.specs import NETWORKS, infer_shapes, layer_cost  # noqa: E402


def net_from_manifest(path):
    """Reconstruit une description `specs` à partir d'un manifest (§10.5, docs/conventions.md)."""
    m = json.loads(Path(path).read_text())
    layers = []
    for entry in m["layers"]:
        layer = {k: v for k, v in entry.items() if k in ("type", "k", "s", "cout", "act", "from")}
        if entry["type"] == "conv":
            layer["bn"] = entry["act"] != "linear"
        layers.append(layer)
    return {"name": m["network"], "input": m["input"]["shape"][1:], "layers": layers}


def describe(layer):
    t = layer["type"]
    if t == "conv":
        return "conv linéaire" if layer["act"] == "linear" else "conv"
    if t == "route":
        return "route " + ", ".join(str(j) for j in layer["from"])
    if t == "upsample":
        return f"upsample ×{layer['s']}"
    return t


def table(net):
    """Lignes (id, couche, k/s, cin→cout, sortie, paramètres, MACs) et totaux."""
    rows = []
    for i, (layer, (ins, outs)) in enumerate(zip(net["layers"], infer_shapes(net))):
        params, macs = layer_cost(layer, ins, outs)
        ks = f"{layer['k']}×{layer['k']}/{layer['s']}" if "k" in layer else ""
        chans = f"{ins[0]}→{outs[0]}" if layer["type"] == "conv" else ""
        out = f"{outs[1]}×{outs[2]}×{outs[0]}"
        rows.append((i, describe(layer), ks, chans, out, params, macs))
    return rows


def print_table(net):
    rows = table(net)
    print(f"## {net['name']}\n")
    print("| # | Couche | k/s | C_in→C_out | Sortie | Paramètres | MACs (M) |")
    print("|---|---|---|---|---|---|---|")
    for i, name, ks, chans, out, params, macs in rows:
        p = f"{params:,}".replace(",", " ") if params else ""
        mm = f"{macs / 1e6:,.1f}".replace(",", " ").replace(".", ",") if macs else ""
        print(f"| {i} | {name} | {ks} | {chans} | {out} | {p} | {mm} |")
    total_p = sum(r[5] for r in rows)
    total_m = sum(r[6] for r in rows)
    print(f"| | **Total** | | | | **{total_p / 1e6:.2f} M** | **{total_m / 1e9:.2f} G** |\n"
          .replace(".", ","))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--net", choices=[*NETWORKS, "all"], default="all")
    src.add_argument("--manifest", type=Path)
    args = ap.parse_args()
    if args.manifest:
        nets = [net_from_manifest(args.manifest)]
    elif args.net == "all":
        nets = list(NETWORKS.values())
    else:
        nets = [NETWORKS[args.net]]
    for net in nets:
        print_table(net)


if __name__ == "__main__":
    main()
