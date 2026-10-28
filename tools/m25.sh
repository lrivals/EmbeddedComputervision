#!/usr/bin/env bash
# Dependability profiles (M25, docs/tasks/M25-dependability.md).
#
#   tools/m25.sh <profile>
#
# Profiles (tier R < 5 min, M < 1 h, N several hours):
#   smoke                 T25.1  reduced campaign: 100 images, 1 seed, few points (R)
#   plan                  T25.1  cost of each grid on SUBSET images (R)
#   ber | layer | bit     T25.1  full grids on SUBSET images, 3 seeds (M to N, see `plan`)
#   report                T25.1  tables and report.json of every grid run so far
#   abft                  T25.2  C++ ABFT tests, golden-check, checksum widths (R)
#   ood                   T25.5  OOD monitor: features, fit, evaluation (M)
#   person                T25.6  layer and bit grids with the person criterion (N, ≈ 4 h)
#   monitor               T25.9  faults in the failure head of T39.1, PREWID and COCO (R)
#
# Outputs in build/m25/<profile>/ (log.txt, JSONL of the runs; monitor: build/m25/monitor/
# *.json); nothing is written in results/. Variables: JOBS (default 16), SUBSET (default 500), SEEDS (default 0,1,2),
# NET (default tiny-yolov2-voc).
set -euo pipefail
cd "$(dirname "$0")/.."

NET=${NET:-tiny-yolov2-voc}
JOBS=${JOBS:-16}
SUBSET=${SUBSET:-500}
SEEDS=${SEEDS:-0,1,2}
M25=build/m25

# Log: command and duration of each step, in $LOG.
step() {
  echo "\$ $*" | tee -a "$LOG"
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  echo "  ($((SECONDS - t0)) s)" | tee -a "$LOG"
}

setup() {
  mkdir -p "$M25/$1"
  LOG=$M25/$1/log.txt
  echo "=== $1 — $(date -Is)" >> "$LOG"
}

campaign() {  # grid [options]
  local g=$1
  shift
  step python tools/fault_campaign.py run --net "$NET" --grid "$g" --subset "$SUBSET" \
    --jobs "$JOBS" --seeds "$SEEDS" --out "$M25/faults/$g.jsonl" "$@"
}

profile=${1:-}
shift || true
case "$profile" in
smoke)
  setup smoke
  step python tools/fault_campaign.py run --net "$NET" --grid ber --subset 100 --jobs "$JOBS" \
    --seeds 0 --bers 1e-7,1e-5 --out "$M25/smoke/ber.jsonl"
  step python tools/fault_campaign.py run --net "$NET" --grid bit --subset 100 --jobs "$JOBS" \
    --seeds 0 --out "$M25/smoke/bit.jsonl"
  step python tools/fault_campaign.py run --net "$NET" --grid layer --subset 100 \
    --jobs "$JOBS" --seeds 0 --sites act,acc --layer-bers 1e-5 --out "$M25/smoke/layer.jsonl"
  step python tools/fault_campaign.py report "$M25"/smoke/*.jsonl
  ;;
plan)
  setup faults
  step python tools/fault_campaign.py plan --net "$NET" --subset "$SUBSET" --jobs "$JOBS" \
    --seeds "$SEEDS"
  ;;
ber | layer | bit)
  setup faults
  campaign "$profile"
  ;;
report)
  setup faults
  step python tools/fault_campaign.py report "$M25"/faults/*.jsonl
  ;;
abft)
  setup abft
  step make test-cpp
  step make golden-check
  step python -m pytest python/tests/test_abft.py python/tests/test_faults.py -q
  step python tools/fault_campaign.py widths model/tiny-yolov2-voc model/tiny-yolov3-coco \
    model/tiny-yolov4-coco
  ;;
person)
  setup person
  P="--criterion person --jobs $JOBS --subset $SUBSET --net $NET"
  step python tools/fault_campaign.py run --grid layer $P --seeds 0,1 \
    --out "$M25/person/layer.jsonl"
  step python tools/fault_campaign.py run --grid bit $P --seeds "$SEEDS" \
    --out "$M25/person/bit.jsonl"
  step python tools/fault_campaign.py report "$M25"/person/*.jsonl
  ;;
monitor)
  setup monitor
  step python tools/fault_campaign.py monitor build/m39/labels/tiny-yolov4-coco_prewid-test.npz \
    build/m39/labels/tiny-yolov4-coco_prewid-lab.npz \
    build/m39/labels/tiny-yolov4-coco_coco-val2017.npz
  ;;
ood)
  setup ood
  step python tools/ood_monitor.py extract --jobs "$JOBS"
  step python tools/ood_monitor.py fit
  step python tools/ood_monitor.py eval
  ;;
*)
  sed -n '2,18p' "$0"
  exit 1
  ;;
esac
