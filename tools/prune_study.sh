#!/usr/bin/env bash
# Étude d'élagage de Tiny-YOLOv2 (T10.11), palier M de M12 : pour chaque taux, élagage,
# mAP entière sans affinage, affinage flottant court, calibration, mAP entière (sous-ensemble
# de VOC2007 test), export et cycles du noyau actuel. Reprise : chaque étape faite est sautée.
#
#   bash tools/prune_study.sh                 # taux 30 50 70, 300 itérations, 500 images
#   RATES="50" ITERS=600 SUBSET=0 bash tools/prune_study.sh
#   RATES="50:10,12,13" bash tools/prune_study.sh   # taux 50 % sur L10, L12, L13 → r50-t10
set -euo pipefail
cd "$(dirname "$0")/.."
RATES=${RATES:-"30 50 70"}
ITERS=${ITERS:-300}
SUBSET=${SUBSET:-500}
JOBS=${JOBS:-8}
OUT=build/m10/prune

eval_int() {  # cfg weights calib out.json
  [ -f "$4" ] || python tools/eval_quant.py --net "$1" --weights "$2" --calib "$3" \
    --variants int --resize stretch --subset "$SUBSET" --jobs "$JOBS" --blas-threads 1 \
    --out "$4" | grep mAP
}

for spec in $RATES; do
  r=${spec%%:*}
  layers=""
  d=$OUT/r$r
  if [ "$spec" != "$r" ]; then
    layers=${spec#*:}
    d=$OUT/r$r-t${layers%%,*}
  fi
  [ -f $d/pruned.weights ] || python tools/prune.py --rate 0.$r --layers "$layers" --out $d
  cfg=$d/pruned.cfg
  # Sans affinage : calibration et mAP entière des poids élagués.
  [ -f $d/calib_notune.json ] || python tools/calibrate.py --net $cfg --weights $d/pruned.weights \
    --out $d/calib_notune.json --markdown $d/calib_notune.md > $d/calib_notune.log
  eval_int $cfg $d/pruned.weights $d/calib_notune.json $d/map_notune_s$SUBSET.json
  # Affinage flottant (BN active), lr 5e-4 après 50 itérations de montée.
  [ -f $d/train/final.weights ] || python tools/train.py --net $cfg --init $d/pruned.weights \
    --iters $ITERS --batch 8 --lr 5e-4 --burn-in 50 --save-every 100 --out $d/train \
    > $d/train.log 2>&1
  w=$d/train/final.weights
  [ -f $d/calib.json ] || python tools/calibrate.py --net $cfg --weights $w \
    --out $d/calib.json --markdown $d/calib.md > $d/calib.log
  eval_int $cfg $w $d/calib.json $d/map_s$SUBSET.json
  [ -f $d/model/manifest.json ] || python tools/export_model.py --net $cfg --weights $w \
    --calib $d/calib.json --out $d/model > $d/export.log
  python tools/perf_model.py --manifest $d/model/manifest.json > $d/perf.json
  echo "$(basename $d) : $(python -c "import json;p=json.load(open('$d/perf.json'))[0];print(f\"{p['ms']:.1f} ms, {p['fps']:.1f} img/s\")")"
done
