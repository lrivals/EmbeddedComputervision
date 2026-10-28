#!/usr/bin/env bash
# T39.4: label-smoothing fine-tune of tiny-yolov4-coco against a fine-tune without it (same
# recipe: COCO calib2017, 200 it, batch 8, lr 1e-4), then labels, heads and the calibration
# table on PREWID test and COCO val2017. Run from the repo root (≈ 5 h on the PC CPU).
set -euo pipefail
NET=tiny-yolov4-coco
L=build/m39/ls
LAB=build/m39/labels
SETS="prewid:test coco:val2017"
for eps in 0 0.1; do
  d=$L/eps$eps
  tag=$NET-ls$eps
  [ -f $d/final.weights ] || python tools/train.py --net $NET --init weights/yolov4-tiny.weights \
    --dataset coco --splits calib2017 --batch 8 --iters 200 --lr 1e-4 --burn-in 20 \
    --label-smoothing $eps --workers 4 --no-figures --out $d
  [ -f $d/calib.json ] || python tools/calibrate.py --net $NET --weights $d/final.weights \
    --dataset coco --out $d/calib.json --markdown $d/calib.md
  for s in $SETS; do
    out=$LAB/${tag}_${s/:/-}.npz
    [ -f $out ] || python tools/failure_labels.py label --net $NET --weights $d/final.weights \
      --calib $d/calib.json --set $s --conf 0.0953 --jobs 12 --out $out
    python tools/failure_prediction.py fit --labels $out
  done
done
for s in $SETS; do
  n=${s/:/-}
  python tools/failure_prediction.py calib --labels $LAB/$NET-ls0_$n.npz \
    --smoothed $LAB/$NET-ls0.1_$n.npz
  python tools/failure_prediction.py calib --labels $LAB/${NET}_$n.npz \
    --smoothed $LAB/$NET-ls0.1_$n.npz
done
