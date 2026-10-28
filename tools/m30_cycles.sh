#!/usr/bin/env bash
# T30.1: post-processing cycles of the C-sim (`yolo_bench --hw-post`, sim backend) over
# ≥ 10,000 real images: tiny-yolov2-voc on VOC2007 test (4,952), COCO val2017 (5,000) and
# PREWID test (592). The accelerator cycles do not depend on the image (T30.0); this run
# measures the data-dependent part, at the default threshold of yolo_bench (0.005, the mAP
# one: the heaviest realistic case). Resumes like bench_sim.sh. Run from the repo root.
#   tools/m30_cycles.sh [JOBS]        (≈ 1-1.5 h on 20 processes)
# Outputs: build/m30/cycles/<set>/times_*.csv (post_cycles, nms_cycles, candidates,
# survivors per image), then the summary of tools/m30_cycles.py.
set -euo pipefail
jobs=${1:-20}
NET=tiny-yolov2-voc
OUT=build/m30/cycles
[ -x build/sw-fast/yolo_bench ] || { echo "build/sw-fast/yolo_bench missing (make bench-sim)"; exit 1; }
declare -A DIRS=([voc]=build/m8/$NET [coco]=$OUT/inputs/coco [prewid]=$OUT/inputs/prewid)
for set in voc coco prewid; do
  d=${DIRS[$set]}
  [ -f "$d/ids.txt" ] || python tools/make_inputs.py --net $NET --dataset $set --out "$d"
  echo "== $set: $(wc -w < "$d/ids.txt") images"
  OUT=$OUT/$set BENCH=build/sw-fast/yolo_bench EXTRA=--hw-post \
    bash tools/bench_sim.sh "$d" model/$NET "$jobs"
done
python tools/m30_cycles.py $OUT/voc $OUT/coco $OUT/prewid --net $NET --json $OUT/summary.json
