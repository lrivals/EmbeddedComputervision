#!/usr/bin/env bash
# Jeux de données de M11 dans data/ (docs/tasks/M11-jeux-de-donnees.md).
#
#   tools/get_datasets.sh <jeu>…     coco | kitti | visdrone | crowdhuman | exdark | flir
#   tools/get_datasets.sh check      état de chaque jeu (rien n'est téléchargé)
#
# Téléchargés ici (idempotent, comme get_voc.sh) : COCO val2017 + annotations 2017 (≈ 1,0 +
# 0,25 Go ; les 500 images de calibration de train2017 : tools/coco_subset.py) et KITTI 2D
# object (images ≈ 12 Go + labels). VisDrone, CrowdHuman, ExDark et FLIR demandent une
# inscription ou passent par Google Drive : le script donne la source et l'arborescence
# attendue, puis vérifie qu'elle est en place. Licences et tailles à vérifier au
# téléchargement (tableau de synthèse de M11).
set -euo pipefail

DATA_DIR="$(cd "$(dirname "$0")/.." && pwd)/data"

download() {  # url fichier
  [[ -f "$2" ]] && return 0
  echo "téléchargement $1"
  curl -fL --retry 3 --connect-timeout 20 -C - -o "$2.part" "$1"
  mv "$2.part" "$2"
}

fetch() {  # dossier url témoin : télécharge et décompresse si le témoin manque
  local dir=$1 url=$2 marker=$3 name
  name=$(basename "$url")
  mkdir -p "$DATA_DIR/$dir"
  cd "$DATA_DIR/$dir"
  if [[ -e "$marker" ]]; then
    echo "$dir/$name : déjà extrait"
  else
    download "$url" "$name"
    echo "extraction $name"
    unzip -q -o "$name"
  fi
  cd - > /dev/null
}

# Témoin de chaque jeu (relatif à data/), source et arborescence attendue.
declare -A MARKER=(
  [coco]=coco/annotations/instances_val2017.json
  [kitti]=kitti/training/image_2/000000.png
  [visdrone]=visdrone/VisDrone2019-DET-val/annotations
  [crowdhuman]=crowdhuman/annotation_val.odgt
  [exdark]=exdark/imageclasslist.txt
  [flir]=flir/images_thermal_val/coco.json
)
declare -A HOWTO=(
  [visdrone]="https://github.com/VisDrone/VisDrone-Dataset (Task 1, DET) : VisDrone2019-DET-train.zip et -val.zip
    data/visdrone/VisDrone2019-DET-{train,val}/{images,annotations}/"
  [crowdhuman]="https://www.crowdhuman.org/download.html : CrowdHuman_train0{1,2,3}.zip, CrowdHuman_val.zip, annotation_{train,val}.odgt
    data/crowdhuman/annotation_{train,val}.odgt, data/crowdhuman/Images/<ID>.jpg"
  [exdark]="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset : ExDark.zip, ExDark_Annno.zip, imageclasslist.txt
    data/exdark/ExDark/<Classe>/, data/exdark/ExDark_Annno/<Classe>/<image>.txt, data/exdark/imageclasslist.txt"
  [flir]="https://www.flir.com/oem/adas/adas-dataset-form/ (inscription) : FLIR ADAS v2
    data/flir/images_thermal_{train,val}/coco.json et data/"
)

check() {
  local ds=$1
  if [[ -e "$DATA_DIR/${MARKER[$ds]}" ]]; then
    echo "$ds : prêt (${MARKER[$ds]})"
  else
    echo "$ds : absent (${MARKER[$ds]})"
    return 1
  fi
}

[[ $# -gt 0 ]] || { sed -n '2,12p' "$0"; exit 1; }
status=0
for ds in "$@"; do
  case "$ds" in
  check)
    for d in coco kitti visdrone crowdhuman exdark flir; do check "$d" || true; done
    [[ -e "$DATA_DIR/coco/annotations/instances_calib2017.json" ]] \
      && echo "coco calib2017 : prêt" || echo "coco calib2017 : absent (tools/coco_subset.py)"
    ;;
  coco)
    fetch coco http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
      annotations/instances_val2017.json
    fetch coco http://images.cocodataset.org/zips/val2017.zip val2017
    echo "calibration : python tools/coco_subset.py --images 500"
    ;;
  kitti)
    base=https://s3.eu-central-1.amazonaws.com/avg-kitti
    fetch kitti $base/data_object_label_2.zip training/label_2
    fetch kitti $base/data_object_image_2.zip training/image_2
    ;;
  visdrone|crowdhuman|exdark|flir)
    if ! check "$ds"; then
      echo "  téléchargement manuel : ${HOWTO[$ds]}"
      status=1
    fi
    ;;
  *)
    echo "jeu inconnu : $ds" >&2
    exit 1
    ;;
  esac
done
exit $status
