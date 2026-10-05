#!/usr/bin/env bash
# Paquet firmware pour `xmutil loadapp yolo` (T7.1) : yolo.bit.bin, yolo.dtbo, shell.json.
#   hw/boards/kv260/firmware.sh            (make fpga-firmware, après make vivado-build)
# Sortie : build/vivado/kv260/firmware/yolo/ → à copier dans /lib/firmware/xilinx/yolo/ sur
# la carte. Outils : bootgen (Vivado/Vitis), dtc ≥ 1.4.7 (-@ pour les overlays).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
root="$(cd "$here/../../.." && pwd)"
out="$root/build/vivado/kv260"
fw="$out/firmware/yolo"

[ -f "$out/yolo.bit" ] || { echo "bitstream absent : make vivado-build" >&2; exit 1; }
mkdir -p "$fw"
printf 'all:\n{\n  %s\n}\n' "$out/yolo.bit" > "$out/yolo.bif"
bootgen -arch zynqmp -image "$out/yolo.bif" -process_bitstream bin -w on
mv -f "$out/yolo.bit.bin" "$fw/yolo.bit.bin"
dtc -@ -O dtb -o "$fw/yolo.dtbo" "$here/pl.dtsi"
cp "$here/shell.json" "$fw/shell.json"
echo "firmware : $fw (copier dans /lib/firmware/xilinx/yolo/)"
