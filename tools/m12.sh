#!/usr/bin/env bash
# Profils de test sur PC avant la carte (M12, docs/tasks/M12-profils-pc.md).
#
#   tools/m12.sh <profil> [essai]
#
# Profils (palier R < 5 min, M < 1 h, N plusieurs heures) :
#   fumee | bitexact | avant-carte        T12.1   non-régression (R)
#   map-c                                 T12.2-c fidélité du sous-ensemble (M, puis N pour 0)
#   map-a | map-b | map-d | map-e         T12.2   prétraitement, seuils, calibration, tête (M)
#   fq                                    T12.3   complément fake-quant (M)
#   ptq                                   T12.4   PTQ basse précision, 500 images (M)
#   ptq-full <variante>…                  T12.4   variantes retenues sur 4 952 images (N)
#   qat <essai> | admm <essai>            T12.5 / T12.6 : entraînement, export, mAP 500 (M / N)
#   hwpp                                  T12.7   post-traitement matériel (R puis M)
#   perf                                  T12.8   performance estimée (R)
#   sim                                   T12.9   chaîne carte simulée, 100 puis 500 images (M)
#   repetition                            T12.10  répétition du protocole carte (M)
#
# Sorties dans build/m12/<profil>/ (journal log.txt, JSON des mAP, CSV) ; rien n'est écrit
# dans results/. Variables : JOBS (évaluations, défaut 16), DEVICE (cpu | gpu, entraînements,
# défaut cpu), SUBSET (défaut 500), NET (défaut tiny-yolov2-voc).
set -euo pipefail
cd "$(dirname "$0")/.."

NET=${NET:-tiny-yolov2-voc}
JOBS=${JOBS:-16}
SUBSET=${SUBSET:-500}
DEVICE=${DEVICE:-cpu}
M12=build/m12
INIT=weights/yolov2-tiny-voc.weights
W4A4_STEPS=build/m9/models/$NET-w4a4-ptq/steps.json
POW2_PTQ=build/m9/models/$NET-pow2-ptq
M8=build/m8/$NET

# Journal : commande et durée de chaque étape, dans $LOG.
step() {
  echo "\$ $*" | tee -a "$LOG"
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  echo "  ($((SECONDS - t0)) s)" | tee -a "$LOG"
}

setup() {
  mkdir -p "$M12/$1"
  LOG=$M12/$1/log.txt
  echo "=== $1 — $(date -Is)" >> "$LOG"
}

# Évaluation entière (référence : stretch, conf 0,005, NMS 0,45) ; options en plus à la suite.
evalq() {
  local out=$1
  shift
  step python tools/eval_quant.py --net "$NET" --resize stretch --jobs "$JOBS" \
    --blas-threads 1 --out "$out" "$@"
}

report() {
  python tools/m12_report.py map "$@" | tee -a "$LOG"
}

profile=${1:-}
shift || true
case "$profile" in

# ------------------------------------------------------------------ T12.1 non-régression
fumee)
  setup regress
  step make test-py
  step make test-cpp
  ;;
bitexact)
  setup regress
  step make golden-check
  step make csim-gcc
  step make sw-sim
  ;;
avant-carte)
  setup regress
  "$0" bitexact
  step make test-slow
  step make perf-model
  step make check-regmap
  ;;

# ------------------------------------------------------------- T12.2 mAP flottante et entière
map-c)  # fidélité du sous-ensemble, à faire en premier
  setup map
  for n in 100 500 1000 0; do
    evalq "$M12/map/subset_int_$n.json" --variants int --subset "$n"
    evalq "$M12/map/subset_pow2_$n.json" --variants int --subset "$n" --model-dir "$POW2_PTQ"
  done
  report "$M12"/map/subset_*.json
  ;;
map-a)  # prétraitement
  setup map
  for r in letterbox stretch; do
    step python tools/eval_quant.py --net "$NET" --variants float,int --resize "$r" \
      --subset "$SUBSET" --jobs "$JOBS" --blas-threads 1 --out "$M12/map/pre_$r.json"
  done
  for i in pil darknet; do
    step python tools/eval_voc.py --net "$NET" --resize stretch --interp "$i" --subset "$SUBSET"
  done
  report "$M12"/map/pre_*.json
  ;;
map-b)  # seuils
  setup map
  for c in 0.005 0.01 0.25; do
    for u in 0.40 0.45 0.50; do
      evalq "$M12/map/thr_c${c}_i${u}.json" --variants int --subset "$SUBSET" --conf "$c" --iou "$u"
    done
  done
  report "$M12"/map/thr_*.json
  ;;
map-d)  # calibration, une variable à la fois autour de mse / 500 images / graine 0
  setup map
  calib() {  # tag choice images seed
    local d=$M12/calib/$1
    step python tools/calibrate.py --net "$NET" --choice "$2" --images "$3" --seed "$4" \
      --out "$d/calib.json" --markdown "$d/calibration.md"
    evalq "$M12/map/calib_$1.json" --variants int --subset "$SUBSET" --calib "$d/calib.json"
  }
  for c in mse p99 p99.9 p99.99 max; do calib "choice_$c" "$c" 500 0; done
  for n in 100 2000; do calib "images_$n" mse "$n" 0; done
  for s in 1 2; do calib "seed_$s" mse 500 "$s"; done
  report "$M12"/map/calib_*.json
  ;;
map-e)  # échelle des têtes (défaut 1/8)
  setup map
  for h in 0.0625 0.125 0.25; do
    d=$M12/calib/head_$h
    step python tools/calibrate.py --net "$NET" --head-scale "$h" --out "$d/calib.json" \
      --markdown "$d/calibration.md"
    evalq "$M12/map/head_$h.json" --variants int --subset "$SUBSET" --calib "$d/calib.json"
  done
  report "$M12"/map/head_*.json
  ;;

# ------------------------------------------------------------------- T12.3 sensibilité
fq)
  setup sens
  evalq "$M12/sens/fq.json" --variants float,fq:all,fq:each --subset "$SUBSET"
  report "$M12/sens/fq.json"
  ;;

# ------------------------------------------------------------------- T12.4 PTQ basse précision
ptq)
  setup ptq
  echo '{"0": "int8"}' > "$M12/ptq/plan_t12_3.json"  # T10.10 : L00 perd > 1 point en mixed6
  q() {  # variante, options de quant_lowbit.py
    local v=$1
    shift
    step python tools/quant_lowbit.py --net "$NET" "$@" --out "$M12/ptq/$v"
    evalq "$M12/ptq/map_$v.json" --variants int --subset "$SUBSET" --model-dir "$M12/ptq/$v"
  }
  for s in w8a8 w8a4 w4a8 w4a4; do q "$s" --scheme "$s"; done
  q uniform6 --weights pow2 --weights-plan build/m9/plan_uniform6.json
  q mixed6 --weights pow2
  q mixed6-sens --weights pow2 --weights-plan "$M12/ptq/plan_t12_3.json"
  q w4a4-int8ends --scheme w4a4 --int8-layers 0,14
  report "$M12"/ptq/map_*.json
  ;;
ptq-full)  # les deux meilleures variantes de ptq sur les 4 952 images (palier N)
  setup ptq
  for v in "$@"; do
    evalq "$M12/ptq/map_${v}_full.json" --variants int --model-dir "$M12/ptq/$v"
  done
  report "$M12"/ptq/map_*_full.json
  ;;

# ---------------------------------------------------------------------------- T12.5 QAT
qat)  # essai : ref | lr3e-4 | lr1e-3 | it300 | it2000 | b16 | nosteps
  essai=${1:?essai QAT}
  lr=1e-4 iters=600 batch=8 steps=(--qat-steps "$W4A4_STEPS")
  case "$essai" in
    ref) ;;
    lr3e-4) lr=3e-4 ;;
    lr1e-3) lr=1e-3 ;;
    it300) iters=300 ;;
    it2000) iters=2000 ;;
    b16) batch=16 ;;
    nosteps) steps=() ;;
    *) echo "essai QAT inconnu : $essai" >&2; exit 2 ;;
  esac
  d=$M12/qat/$essai
  setup "qat/$essai"
  step python tools/train.py --net "$NET" --init "$INIT" --qat w4a4 "${steps[@]}" \
    --lr "$lr" --burn-in 50 --batch "$batch" --iters "$iters" --save-every 100 --workers 2 \
    --device "$DEVICE" --out "$d"
  step python tools/quant_lowbit.py --net "$NET" --scheme w4a4 \
    --checkpoint "$d/checkpoint.npz" --out "$d/model"
  evalq "$d/map.json" --variants int --subset "$SUBSET" --model-dir "$d/model"
  python tools/m12_report.py loss "$d/loss.csv" | tee -a "$LOG"
  report "$d/map.json"
  ;;

# --------------------------------------------------------------------------- T12.6 ADMM
admm)  # essai : ref | A | B | C | D | E (table de T12.6)
  essai=${1:?essai ADMM}
  case "$essai" in   #   ρ₀    growth every iters
    ref) set -- 1e-3 1.3 50 600 ;;
    A) set -- 1e-2 1.3 50 600 ;;
    B) set -- 5e-2 1.3 50 600 ;;
    C) set -- 5e-2 1.3 25 300 ;;
    D) set -- 1e-1 1.5 50 300 ;;
    E) set -- 5e-2 1.3 100 600 ;;
    *) echo "essai ADMM inconnu : $essai" >&2; exit 2 ;;
  esac
  d=$M12/admm/$essai
  setup "admm/$essai"
  echo '{}' > "$d/plan_mixed6.json"
  step python tools/train.py --net "$NET" --init "$INIT" --admm "$d/plan_mixed6.json" \
    --admm-rho "$1" --admm-growth "$2" --admm-every "$3" --lr 1e-4 --burn-in 50 --batch 8 \
    --iters "$4" --save-every 100 --workers 2 --device "$DEVICE" --out "$d"
  step python tools/quant_lowbit.py --net "$NET" --weights pow2 \
    --checkpoint "$d/checkpoint.npz" --out "$d/model"
  evalq "$d/map.json" --variants int --subset "$SUBSET" --model-dir "$d/model"
  python tools/m12_report.py admm "$d/admm.csv" | tee -a "$LOG"
  report "$d/map.json"
  ;;

# ------------------------------------------------------------- T12.7 post-traitement matériel
hwpp)
  setup hwpp
  for c in 0.005 0.25; do
    for cap in 64 128 256 1024; do
      evalq "$M12/hwpp/cap${cap}_c$c.json" --variants int,int-hwpp --subset "$SUBSET" \
        --hw-cap "$cap" --conf "$c"
    done
  done
  report "$M12"/hwpp/cap*.json
  # Cycles du pire cas (845 candidates) et robustesse, noyau compilé à chaque capacité.
  for cap in 64 128 256; do
    b=$M12/hwpp/build_cap$cap
    step cmake -S hls -B "$b" -DCMAKE_CXX_FLAGS="-DHWPP_CAP=$cap"
    step cmake --build "$b" -j --target tb_post
    step "$b/tb_post" --net "$NET" --random "${RANDOM_HEADS:-800}"
  done
  ;;

# ---------------------------------------------------------------- T12.8 performance estimée
perf)
  setup perf
  step make perf-model
  step python tools/roofline.py --out "$M12/perf"
  for wa in "8 8" "4 4" "4 8" "8 4"; do
    set -- $wa
    step python tools/stream_model.py --net "$NET" --board kv260 --wbits "$1" --abits "$2" \
      --json "$M12/perf/stream_${NET}_w$1a$2.json"
  done
  for n in tiny-yolov3-voc tiny-yolov3-coco; do
    if [ -f "model/$n/manifest.json" ]; then
      step python tools/stream_model.py --net "$n" --board kv260 \
        --json "$M12/perf/stream_${n}_w8a8.json"
    else
      echo "$n : model/$n absent (make export), sauté" | tee -a "$LOG"
    fi
  done
  ;;

# --------------------------------------------------------------- T12.9 chaîne carte simulée
sim)
  setup sim
  [ -f "$M8/inputs.bin" ] || { echo "$M8/inputs.bin absent : make m8-inputs" >&2; exit 1; }
  [ -f "$M8/int.jsonl" ] || { echo "$M8/int.jsonl absent : make m8-int" >&2; exit 1; }
  step make sw-sim
  CXXFLAGS="-DACC_NO_APINT -march=native" step cmake -S sw -B build/sw-fast -DSW_BACKEND=sim
  step cmake --build build/sw-fast -j --target yolo_bench
  # Paliers d'images (ap_int), puis ACC_NO_APINT sur les mêmes 500.
  for n in 100 500; do
    COUNT=$n OUT=$M12/sim/apint_$n step tools/bench_sim.sh "$M8" "model/$NET" "$JOBS"
    step python tools/m12_report.py dets "$M8/int.jsonl" "$M12"/sim/apint_$n/dets_*.jsonl
  done
  COUNT=500 OUT=$M12/sim/fast_500 BENCH=build/sw-fast/yolo_bench \
    step tools/bench_sim.sh "$M8" "model/$NET" "$JOBS"
  step python tools/m12_report.py dets "$M8/int.jsonl" "$M12"/sim/fast_500/dets_*.jsonl
  # Post-traitement matériel : yolo_bench --hw-post contre eval_quant int-hwpp.
  evalq "$M12/sim/hwpp_500.json" --variants int-hwpp --subset 500 \
    --save-dets "$M12/sim/hwpp_500.jsonl"
  COUNT=500 OUT=$M12/sim/hwpost_500 BENCH=build/sw-fast/yolo_bench EXTRA=--hw-post \
    step tools/bench_sim.sh "$M8" "model/$NET" "$JOBS"
  step python tools/m12_report.py dets "$M12/sim/hwpp_500.jsonl" "$M12"/sim/hwpost_500/dets_*.jsonl
  # Modèles basse précision exportés par T12.4 / T12.5 / T12.6 (MODELS="dir1 dir2").
  for m in ${MODELS:-$POW2_PTQ}; do
    t=$(basename "$m")
    evalq "$M12/sim/${t}_100.json" --variants int --subset 100 --model-dir "$m" \
      --save-dets "$M12/sim/${t}_100.jsonl"
    COUNT=100 OUT=$M12/sim/${t}_100 BENCH=build/sw-fast/yolo_bench \
      step tools/bench_sim.sh "$M8" "$m" "$JOBS"
    step python tools/m12_report.py dets "$M12/sim/${t}_100.jsonl" "$M12/sim/${t}_100"/dets_*.jsonl
  done
  ;;

# ---------------------------------------------------- T12.10 répétition du protocole carte
repetition)
  setup repetition
  d=$M12/repetition
  n=${COUNT:-100}
  mkdir -p "$d/board"
  head -n "$n" "$M8/ids.txt" | sed "s|^|data/VOCdevkit/VOC2007/JPEGImages/|; s|$|.jpg|" \
    > "$d/images.txt"
  step make sw-sim
  # JPEG décodé par stb sur « l'ARM » contre entrées int8 préparées en PIL.
  step build/sw/yolo_bench --model "model/$NET" --images "$d/images.txt" --count "$n" \
    --warmup 5 --times "$d/times_jpeg.csv" --dets "$d/board/dets_jpeg.jsonl"
  step build/sw/yolo_bench --model "model/$NET" --inputs "$M8/inputs.bin" --ids "$M8/ids.txt" \
    --count "$n" --warmup 5 --times "$d/times_inputs.csv" --dets "$d/board/dets_inputs.jsonl"
  python tools/m12_report.py dets "$M8/int.jsonl" "$d/board/dets_inputs.jsonl" | tee -a "$LOG"
  python tools/m12_report.py dets "$M8/int.jsonl" "$d/board/dets_jpeg.jsonl" | tee -a "$LOG" || true
  # Les outils de rapport lisent les sorties sans modification (chemins simulés).
  step python tools/bench_report.py --net "$NET" --times "$d/times_*.csv" \
    --csv "$d/benchmarks.csv" --md "$d/mesures.md"
  ;;

*)
  sed -n '2,20p' "$0"
  exit 2
  ;;
esac
