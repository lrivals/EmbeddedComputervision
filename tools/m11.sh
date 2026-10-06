#!/usr/bin/env bash
# Jeux de données au-delà de VOC (M11, docs/tasks/M11-jeux-de-donnees.md).
#
#   tools/m11.sh <profil>
#
# Profils (données : tools/get_datasets.sh ; ordre recommandé de M11) :
#   voc-regress                    T11.0  mAP VOC2007 test inchangée (56,30 / 55,66)
#   coco-float | coco-int          T11.1  tiny-yolov3-coco sur val2017, flottant puis entier
#   coco-calib                     T11.1  calibration VOC contre 500 images de train2017
#   coco-fpga                      T11.1  stade FPGA en C-sim, égalité image par image
#   hors-domaine                   T11.2  poids VOC et COCO sans affinage (ExDark, KITTI, VisDrone)
#   exdark                         T11.3  calibration ExDark, histogrammes L00-L04
#   crowdhuman                     T11.6  NMS sans tri et capacité de sélection, scènes denses
#   kitti-prep | visdrone-prep | flir-prep    ancres k-means et cfg d'affinage
#   kitti-train | visdrone-train | flir-train   affinage (long : plusieurs heures à jours)
#   visdrone-size                  T11.5  entrée 416 / 608 / 832 des poids affinés
#
# Sorties dans build/m11/<profil>/ (journal log.txt, JSON des mAP) ; rien n'est écrit dans
# results/ sauf par calibrate.py et kmeans_anchors.py (rapports de calibration et
# d'ancres). Variables : JOBS (défaut 16), SUBSET (défaut 0 = tout), DEVICE (cpu | gpu),
# ITERS (affinages, défaut 4000), BATCH (défaut 16).
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-16}
SUBSET=${SUBSET:-0}
DEVICE=${DEVICE:-cpu}
ITERS=${ITERS:-4000}
BATCH=${BATCH:-16}
M11=build/m11
V2=tiny-yolov2-voc
V3=tiny-yolov3-coco

step() {
  echo "\$ $*" | tee -a "$LOG"
  local t0=$SECONDS
  "$@" 2>&1 | tee -a "$LOG"
  echo "  ($((SECONDS - t0)) s)" | tee -a "$LOG"
}

setup() {
  mkdir -p "$M11/$1"
  LOG=$M11/$1/log.txt
  echo "=== $1 — $(date -Is)" >> "$LOG"
}

need() {  # jeux requis : vérifiés seulement, jamais téléchargés ici
  local ds
  for ds in "$@"; do
    tools/get_datasets.sh check | grep -q "^$ds : prêt" || {
      echo "$ds absent : tools/get_datasets.sh $ds" >&2
      exit 1
    }
  done
}

# Évaluation entière (référence : stretch, conf 0,005, NMS 0,45) ; options en plus à la suite.
evalq() {  # net sortie options…
  local net=$1 out=$2
  shift 2
  step python tools/eval_quant.py --net "$net" --resize stretch --jobs "$JOBS" \
    --blas-threads 1 --subset "$SUBSET" --out "$out" "$@"
}

report() {
  python tools/m12_report.py map "$@" | tee -a "$LOG"
}

# Ancres à 6 (Tiny-YOLOv3) de results/anchors_<jeu>.md, au format de make_cfg.py.
anchors6() {
  python - "$1" <<'EOF'
import re, sys
for line in open(sys.argv[1]):
    if line.startswith("| 6 |"):
        print(re.findall(r"`([^`]*)`", line)[0])
        break
EOF
}

prep() {  # jeu : ancres sur le split d'entraînement, cfg Tiny-YOLOv3 à N classes
  local ds=$1
  setup "$ds"
  need "$ds"
  step python tools/kmeans_anchors.py --dataset "$ds"
  step python tools/make_cfg.py --base tiny-yolov3-voc --dataset "$ds" \
    --anchors "$(anchors6 "results/anchors_$ds.md")" --out "$M11/cfg/tiny-yolov3-$ds.cfg"
}

train() {  # jeu : affinage depuis les poids COCO (têtes réinitialisées), puis mAP flottante
  local ds=$1 d=$M11/$1/train cfg=$M11/cfg/tiny-yolov3-$1.cfg
  setup "$ds"
  need "$ds"
  [[ -f $cfg ]] || { echo "$cfg absent : tools/m11.sh $ds-prep" >&2; exit 1; }
  local resume=()
  [[ -f $d/checkpoint.npz ]] && resume=(--resume)
  step python tools/train.py --net "$cfg" --dataset "$ds" --init coco "${resume[@]}" \
    --iters "$ITERS" --batch "$BATCH" --lr 1e-3 --burn-in 500 --multiscale --workers 4 \
    --device "$DEVICE" --out "$d"
  step python tools/eval_voc.py --net "$cfg" --weights "$d/final.weights" --dataset "$ds" \
    --resize stretch --subset "$SUBSET"
}

profile=${1:-}
shift || true
case "$profile" in

# ------------------------------------------------------------- T11.0 non-régression VOC
voc-regress)
  setup voc-regress
  step python tools/eval_voc.py --net $V2 --weights weights/yolov2-tiny-voc.weights \
    --resize stretch --dataset voc --out $M11/voc-regress/dets
  evalq $V2 $M11/voc-regress/float_int.json --variants float,int --dataset voc
  report $M11/voc-regress/float_int.json
  ;;
# ------------------------------------------------------------------------ T11.1 COCO
coco-float)  # AP50 attendu ≈ 33 (Darknet, yolov3-tiny 416), en letterbox
  setup coco
  need coco
  for r in letterbox stretch; do
    step python tools/eval_voc.py --net $V3 --weights weights/yolov3-tiny.weights \
      --dataset coco --resize "$r" --subset "$SUBSET" --markdown $M11/coco/float.md
  done
  ;;
coco-int)  # calibration actuelle (500 images VOC2007 trainval)
  setup coco
  need coco
  evalq $V3 $M11/coco/float_int_calibvoc.json --variants float,int --dataset coco
  report $M11/coco/float_int_calibvoc.json
  ;;
coco-calib)  # calibration VOC contre 500 images COCO train2017
  setup coco
  need coco
  [[ -f data/coco/annotations/instances_calib2017.json ]] || \
    step python tools/coco_subset.py --images 500
  step python tools/calibrate.py --net $V3 --dataset coco
  evalq $V3 $M11/coco/int_calibcoco.json --variants int --dataset coco \
    --calib build/quant/$V3/calib-coco.json
  report $M11/coco/float_int_calibvoc.json $M11/coco/int_calibcoco.json
  ;;
coco-fpga)  # modèle exporté model/tiny-yolov3-coco (calibration VOC), C-sim derrière le driver
  setup coco
  need coco
  d=$M11/coco/$V3
  step python tools/make_inputs.py --net $V3 --dataset coco --subset "$SUBSET" --out "$d"
  step python tools/eval_quant.py --net $V3 --dataset coco --variants float,int \
    --resize stretch --jobs "$JOBS" --blas-threads 1 --subset "$SUBSET" \
    --save-dets "$d/int.jsonl" --out "$d/eval_float_int.json"
  CXXFLAGS="-DACC_NO_APINT -march=native" step cmake -S sw -B build/sw-fast -DSW_BACKEND=sim
  step cmake --build build/sw-fast -j --target yolo_bench
  BENCH=build/sw-fast/yolo_bench step tools/bench_sim.sh "$d" model/$V3 "$JOBS"
  step python tools/map_stades.py --net $V3 --dataset coco --dir "$d" \
    --out $M11/coco/map_stades.md
  ;;
# ------------------------------------------------------- T11.2 hors domaine, sans affinage
hors-domaine)
  setup hors-domaine
  for ds in voc exdark kitti visdrone; do
    [[ $ds == voc ]] || need "$ds"
    for net in $V2 $V3; do
      evalq $net "$M11/hors-domaine/${ds}_$net.json" --variants float,int --dataset "$ds"
    done
  done
  report $M11/hors-domaine/*.json
  ;;
# ---------------------------------------------------------------------- T11.3 ExDark
exdark)
  setup exdark
  need exdark
  for net in $V2 $V3; do
    step python tools/calibrate.py --net $net --dataset exdark
    evalq $net "$M11/exdark/int_calibexdark_$net.json" --variants int --dataset exdark \
      --calib "build/quant/$net/calib-exdark.json"
    step python tools/act_hist.py --net $net --datasets voc,exdark \
      --out "$M11/exdark/act_calibvoc_$net"
    step python tools/act_hist.py --net $net --datasets voc,exdark \
      --calib "build/quant/$net/calib-exdark.json" --out "$M11/exdark/act_calibexdark_$net"
  done
  report $M11/hors-domaine/exdark_*.json $M11/exdark/int_calibexdark_*.json
  ;;
# ------------------------------------------------------------------ T11.6 CrowdHuman
crowdhuman)  # NMS sans tri face à la NMS triée, débordements de la sélection
  setup crowdhuman
  need crowdhuman
  for net in $V2 $V3; do
    for cap in 256 1024; do
      evalq $net "$M11/crowdhuman/hwpp_cap${cap}_$net.json" --variants int,int-hwpp \
        --dataset crowdhuman --hw-cap "$cap"
    done
  done
  report $M11/crowdhuman/*.json
  ;;
# ------------------------------------------------- T11.4, T11.5, T11.7 préparation, affinage
kitti-prep | visdrone-prep | flir-prep)
  prep "${profile%-prep}"
  ;;
kitti-train | visdrone-train | flir-train)
  train "${profile%-train}"
  ;;
visdrone-size)  # T11.5 : entrée plus grande, mêmes poids
  setup visdrone
  need visdrone
  for s in 416 608 832; do
    step python tools/eval_voc.py --net $M11/cfg/tiny-yolov3-visdrone.cfg \
      --weights $M11/visdrone/train/final.weights --dataset visdrone --resize stretch \
      --size "$s" --subset "$SUBSET" --markdown $M11/visdrone/size.md
  done
  ;;
*)
  sed -n '2,20p' "$0"
  exit 1
  ;;
esac
