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
#   crowdhuman-cycles              T11.6  cycles de yolo_post par image en C-sim (N images)
#   <jeu>-prep (kitti, visdrone, flir, exdark, crowdhuman)    ancres k-means et cfg d'affinage
#   <jeu>-train (mêmes jeux)       affinage (long : plusieurs heures à jours)
#   visdrone-size                  T11.5  entrée 416 / 608 / 832 des poids affinés
#
# Sorties dans build/m11/<profil>/ (journal log.txt, JSON des mAP) ; rien n'est écrit dans
# results/ sauf par calibrate.py et kmeans_anchors.py (rapports de calibration et
# d'ancres). Variables : JOBS (défaut 16), SUBSET (défaut 0 = tout), DEVICE (cpu | gpu),
# ITERS (affinages, défaut 4000), BATCH (défaut 16), NET (réseau des profils -prep, -train
# et visdrone-size : tiny-yolov3, défaut, ou tiny-yolov2), N (images de crowdhuman-cycles,
# défaut 200), SIZE (entrée LxH non carrée des profils -prep et -train, ex. 640x192 pour
# KITTI, T11.4 ; défaut : 416 × 416), CH (canaux d'entrée, 1 pour FLIR, T11.7 ; défaut 3).
# SIZE et CH s'ajoutent au nom de la cfg et du dossier d'affinage.
set -euo pipefail
cd "$(dirname "$0")/.."

JOBS=${JOBS:-16}
SUBSET=${SUBSET:-0}
DEVICE=${DEVICE:-cpu}
ITERS=${ITERS:-4000}
BATCH=${BATCH:-16}
NET=${NET:-tiny-yolov3}
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

# Ancres à k de results/anchors_<jeu>.md (5 : Tiny-YOLOv2, 6 : Tiny-YOLOv3), au format de
# make_cfg.py.
anchors_k() {  # k fichier
  python - "$1" "$2" <<'EOF'
import re, sys
for line in open(sys.argv[2]):
    if line.startswith(f"| {sys.argv[1]} |"):
        print(re.findall(r"`([^`]*)`", line)[0])
        break
EOF
}

case "$NET" in
  tiny-yolov3) K=6; INIT=(--init coco) ;;
  # Pas de poids COCO Tiny-YOLOv2 dans weights/ : départ des poids VOC, hors tête.
  tiny-yolov2) K=5; INIT=(--init weights/yolov2-tiny-voc.weights --init-net $V2) ;;
  *) echo "NET inconnu : $NET (tiny-yolov3 | tiny-yolov2)" >&2; exit 1 ;;
esac

# Variante d'entrée : suffixe des cfg et dossiers (-640x192, -c1), options de make_cfg.
VAR=${SIZE:+-$SIZE}${CH:+-c$CH}
CFG_OPTS=(${SIZE:+--size "$SIZE"} ${CH:+--channels "$CH"})

# Dossier d'affinage : build/m11/<jeu>/train (Tiny-YOLOv3), train-tiny-yolov2 sinon ; suffixe
# de la variante d'entrée.
train_dir() {
  local d=$M11/$1/train
  [[ $NET == tiny-yolov3 ]] || d=$d-$NET
  echo "$d$VAR"
}

prep() {  # jeu : ancres sur le split d'entraînement (à l'entrée SIZE), cfg $NET à N classes
  local ds=$1 anchors=results/anchors_$1${SIZE:+_$SIZE}.md
  setup "$ds"
  need "$ds"
  step python tools/kmeans_anchors.py --dataset "$ds" --size "${SIZE:-416}" --out "$anchors"
  step python tools/make_cfg.py --base "$NET-voc" --dataset "$ds" "${CFG_OPTS[@]}" \
    --anchors "$(anchors_k $K "$anchors")" --out "$M11/cfg/$NET-$ds$VAR.cfg"
}

train() {  # jeu : affinage (couches de forme différente, les têtes, réinitialisées), mAP
  # Entrée à un canal : L00 copiée des poids RGB par somme sur les canaux (copy_matching).
  local ds=$1 d cfg=$M11/cfg/$NET-$1$VAR.cfg
  d=$(train_dir "$ds")
  setup "$ds"
  need "$ds"
  [[ -f $cfg ]] || {
    echo "$cfg absent : NET=$NET SIZE=${SIZE:-} CH=${CH:-} tools/m11.sh $ds-prep" >&2
    exit 1
  }
  local resume=()
  [[ -f $d/checkpoint.npz ]] && resume=(--resume)
  step python tools/train.py --net "$cfg" --dataset "$ds" "${INIT[@]}" "${resume[@]}" \
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
crowdhuman-cycles)  # T11.6 : cycles de yolo_post par image (C-sim), N images (défaut 200)
  setup crowdhuman
  need crowdhuman
  CXXFLAGS="-DACC_NO_APINT -march=native" step cmake -S sw -B build/sw-fast -DSW_BACKEND=sim
  step cmake --build build/sw-fast -j --target yolo_bench
  for net in $V2 $V3; do
    d=$M11/crowdhuman/fpga_$net
    step python tools/make_inputs.py --net $net --dataset crowdhuman --subset "${N:-200}" \
      --out "$d"
    # times_c*.csv : colonnes overflow, post_cycles, nms_cycles, candidates, survivors.
    BENCH=build/sw-fast/yolo_bench EXTRA=--hw-post step tools/bench_sim.sh "$d" \
      model/$net "$JOBS"
  done
  ;;
# ------------------------------------------------- T11.4, T11.5, T11.7 préparation, affinage
kitti-prep | visdrone-prep | flir-prep | exdark-prep | crowdhuman-prep)
  prep "${profile%-prep}"
  ;;
kitti-train | visdrone-train | flir-train | exdark-train | crowdhuman-train)
  train "${profile%-train}"
  ;;
visdrone-size)  # T11.5 : entrée plus grande, mêmes poids ; AP par taille (métrique COCO)
  setup visdrone
  need visdrone
  for s in 416 608 832; do
    step python tools/eval_voc.py --net "$M11/cfg/$NET-visdrone.cfg" \
      --weights "$(train_dir visdrone)/final.weights" --dataset visdrone --resize stretch \
      --size "$s" --metric coco --subset "$SUBSET" --markdown "$M11/visdrone/size-$NET.md"
  done
  ;;
*)
  sed -n '2,26p' "$0"
  exit 1
  ;;
esac
