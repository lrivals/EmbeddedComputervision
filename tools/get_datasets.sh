#!/usr/bin/env bash
# Jeux de données de M11 dans data/ (docs/tasks/M11-jeux-de-donnees.md).
#
#   tools/get_datasets.sh <jeu>…     coco | kitti | visdrone | crowdhuman | exdark | flir
#   tools/get_datasets.sh check      état de chaque jeu (rien n'est téléchargé)
#   tools/get_datasets.sh kaggle     relie les versions Kaggle (CrowdHuman, VisDrone, ExDark)
#   tools/get_datasets.sh kaggle-download <jeu>…   crowdhuman | visdrone | exdark, par l'API
#                                    Kaggle (ExDark : images seules), puis kaggle
#   tools/get_datasets.sh ready <jeu>…              code de retour 0 si tous sont prêts
#   tools/get_datasets.sh pack <jeu>…               data/<racine> → data_archives/<jeu>.tar
#   tools/get_datasets.sh unpack <dossier> <jeu>…   <dossier>/<jeu>.tar → data/ (Colab : Drive)
#   tools/get_datasets.sh push <jeu>…               pack, puis rclone vers $DRIVE_REMOTE
#   tools/get_datasets.sh pull <jeu>…               si absent : rclone depuis $DRIVE_REMOTE, unpack
#                                    (docs/tasks/donnees-drive.md)
#
# Téléchargés ici (idempotent, comme get_voc.sh) : COCO val2017 + annotations 2017 (≈ 1,0 +
# 0,25 Go ; les 500 images de calibration de train2017 : tools/coco_subset.py) et KITTI 2D
# object (images ≈ 12 Go + labels). VisDrone, CrowdHuman, ExDark et FLIR demandent une
# inscription ou passent par Google Drive : le script donne la source et l'arborescence
# attendue, puis vérifie qu'elle est en place. Licences et tailles à vérifier au
# téléchargement (tableau de synthèse de M11).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-$ROOT/data}"           # surchargés par les tests
ARCHIVE_DIR="${ARCHIVE_DIR:-$ROOT/data_archives}"
DRIVE_REMOTE="${DRIVE_REMOTE:-gdrive:EmbeddedCV/data}"  # remote rclone des archives

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
  [voc]=VOCdevkit/VOC2007/ImageSets/Main/test.txt
  [coco]=coco/annotations/instances_val2017.json
  [kitti]=kitti/training/image_2/000000.png
  [visdrone]=visdrone/VisDrone2019-DET-val/annotations
  [crowdhuman]=crowdhuman/annotation_val.odgt
  [exdark]=exdark/imageclasslist.txt
  [flir]=flir/images_thermal_val/coco.json
)
declare -A HOWTO=(
  [visdrone]="https://github.com/VisDrone/VisDrone-Dataset (Task 1, DET) : VisDrone2019-DET-train.zip et -val.zip
    (ou kaggle.com/datasets/kushagrapandya/visdrone-dataset dans data/VisDrone Dataset/, puis : $0 kaggle)
    data/visdrone/VisDrone2019-DET-{train,val}/{images,annotations}/"
  [crowdhuman]="https://www.crowdhuman.org/download.html : CrowdHuman_train0{1,2,3}.zip, CrowdHuman_val.zip, annotation_{train,val}.odgt
    (ou kaggle.com/datasets/leducnhuan/crowdhuman dans data/CrowdHuman/, puis : $0 kaggle)
    data/crowdhuman/annotation_{train,val}.odgt, data/crowdhuman/Images/<ID>.jpg"
  [exdark]="https://github.com/cs-chan/Exclusively-Dark-Image-Dataset : ExDark.zip, ExDark_Annno.zip, imageclasslist.txt
    (kaggle.com/datasets/washingtongold/exdark-dataset dans data/ExDark Dataset/ n'a que les images :
    $0 kaggle les relie, ExDark_Annno/ et imageclasslist.txt viennent toujours de GitHub, Groundtruth/)
    data/exdark/ExDark/<Classe>/, data/exdark/ExDark_Annno/<Classe>/<image>.txt, data/exdark/imageclasslist.txt"
  [flir]="https://www.flir.com/oem/adas/adas-dataset-form/ (inscription) : FLIR ADAS v2
    data/flir/images_thermal_{train,val}/coco.json et data/"
)

# Dossier de chaque jeu dans data/ (Dataset.root de python/yolo/data/datasets.py) : contenu
# de son archive.
declare -A DIR=([voc]=VOCdevkit)

# Versions Kaggle (kaggle-download) : jeu → identifiant et dossier de data/ attendu par kaggle().
declare -A KAGGLE_ID=(
  [crowdhuman]=leducnhuan/crowdhuman
  [visdrone]=kushagrapandya/visdrone-dataset
  [exdark]=washingtongold/exdark-dataset
)
declare -A KAGGLE_DIR=([crowdhuman]=CrowdHuman [visdrone]="VisDrone Dataset" [exdark]="ExDark Dataset")

known() {
  [[ -n "${MARKER[$1]:-}" ]] || { echo "jeu inconnu : $1" >&2; exit 1; }
}

# Archive d'un jeu prêt : un tar non compressé (images déjà en JPEG ou PNG), liens suivis
# (-h) pour que l'arborescence attendue tienne sans les dossiers Kaggle d'origine ; les
# archives téléchargées (data/kitti/*.zip…) restent dehors.
pack() {
  local ds=$1 dir=${DIR[$1]:-$1}
  check "$ds" > /dev/null || { echo "$ds : absent, rien à archiver" >&2; return 1; }
  mkdir -p "$ARCHIVE_DIR"
  echo "archivage data/$dir → $ARCHIVE_DIR/$ds.tar"
  tar -chf "$ARCHIVE_DIR/$ds.tar.part" -C "$DATA_DIR" --exclude='*.zip' --exclude='*.part' \
    --exclude='*.tar' --exclude=__MACOSX "$dir"
  mv "$ARCHIVE_DIR/$ds.tar.part" "$ARCHIVE_DIR/$ds.tar"
}

# Extraction depuis un dossier d'archives (Drive monté sur Colab) : une lecture séquentielle
# du tar, puis les images sont lues sur le disque local, jamais à travers le montage.
unpack() {
  local src=$1 ds=$2
  if check "$ds" > /dev/null; then
    echo "$ds : déjà prêt"
    return 0
  fi
  [[ -f "$src/$ds.tar" ]] || { echo "$ds : pas d'archive $src/$ds.tar" >&2; return 1; }
  mkdir -p "$DATA_DIR"
  echo "extraction $src/$ds.tar"
  tar -xf "$src/$ds.tar" -C "$DATA_DIR"
  check "$ds"
}

need_rclone() {
  command -v rclone > /dev/null || {
    echo "rclone absent : sudo apt install rclone, puis rclone config create gdrive drive scope=drive" >&2
    return 1
  }
  rclone listremotes | grep -qx "${DRIVE_REMOTE%%:*}:" || {
    echo "remote ${DRIVE_REMOTE%%:*} absent : rclone config create ${DRIVE_REMOTE%%:*} drive scope=drive" >&2
    return 1
  }
}

# Export d'un jeu prêt vers Google Drive (PC) : l'archive reste dans data_archives/.
push() {
  local ds=$1
  need_rclone || return 1
  pack "$ds"
  echo "envoi $ARCHIVE_DIR/$ds.tar → $DRIVE_REMOTE"
  rclone copy "$ARCHIVE_DIR/$ds.tar" "$DRIVE_REMOTE" --progress
}

# Jeu absent en local : son archive est rapatriée de Google Drive, puis extraite.
pull() {
  local ds=$1
  if check "$ds" > /dev/null; then
    echo "$ds : déjà prêt"
    return 0
  fi
  need_rclone || return 1
  mkdir -p "$ARCHIVE_DIR"
  echo "rapatriement $DRIVE_REMOTE/$ds.tar → $ARCHIVE_DIR"
  rclone copy "$DRIVE_REMOTE/$ds.tar" "$ARCHIVE_DIR" --progress || return 1
  unpack "$ARCHIVE_DIR" "$ds"
}

kaggle_download() {
  local ds=$1
  [[ -n "${KAGGLE_ID[$ds]:-}" ]] || { echo "$ds : pas de version Kaggle" >&2; return 1; }
  check "$ds" > /dev/null && { echo "$ds : déjà prêt"; return 0; }
  command -v kaggle > /dev/null \
    || { echo "CLI kaggle absente : pip install kaggle (jeton : ~/.kaggle/kaggle.json)" >&2; return 1; }
  kaggle datasets download "${KAGGLE_ID[$ds]}" -p "$DATA_DIR/${KAGGLE_DIR[$ds]}" --unzip
}

link() {  # cible lien : lien relatif, seulement si la cible existe et que le lien manque
  local target=$1 name=$2
  [[ -e "$DATA_DIR/$name" || ! -e "$(dirname "$DATA_DIR/$name")/$target" ]] && return 0
  ln -s "$target" "$DATA_DIR/$name"
  echo "lien $name -> $target"
}

# Versions Kaggle : dossiers d'autre nom, souvent à double niveau ; on les relie aux
# arborescences attendues sans rien déplacer.
kaggle() {
  link CrowdHuman/CrowdHuman crowdhuman
  link CrowdHuman crowdhuman  # archive à un seul niveau
  if [[ -d "$DATA_DIR/VisDrone Dataset" ]]; then
    mkdir -p "$DATA_DIR/visdrone"
    for s in train val test-dev; do
      link "../VisDrone Dataset/VisDrone2019-DET-$s/VisDrone2019-DET-$s" \
        "visdrone/VisDrone2019-DET-$s"
      link "../VisDrone Dataset/VisDrone2019-DET-$s" "visdrone/VisDrone2019-DET-$s"
    done
  fi
  if [[ -d "$DATA_DIR/ExDark Dataset" ]]; then
    mkdir -p "$DATA_DIR/exdark"
    link "../ExDark Dataset" exdark/ExDark
    # Annotations GitHub décompressées à côté des images (zip à double niveau).
    link "../ExDark Dataset/ExDark_Annno/ExDark_Annno" exdark/ExDark_Annno
    link "../ExDark Dataset/ExDark_Annno" exdark/ExDark_Annno
    link "../ExDark Dataset/imageclasslist.txt" exdark/imageclasslist.txt
  fi
}

check() {
  local ds=$1
  if [[ -e "$DATA_DIR/${MARKER[$ds]}" ]]; then
    echo "$ds : prêt (${MARKER[$ds]})"
  else
    echo "$ds : absent (${MARKER[$ds]})"
    return 1
  fi
}

[[ $# -gt 0 ]] || { sed -n '2,21p' "$0"; exit 1; }
case "$1" in
ready)
  shift
  for ds in "$@"; do known "$ds"; check "$ds" || exit 1; done
  exit 0
  ;;
pack)
  shift
  for ds in "$@"; do known "$ds"; pack "$ds"; done
  exit 0
  ;;
push)
  shift
  for ds in "$@"; do known "$ds"; push "$ds"; done
  exit 0
  ;;
pull)
  shift
  status=0
  for ds in "$@"; do known "$ds"; pull "$ds" || status=1; done
  exit $status
  ;;
unpack)
  [[ $# -ge 3 ]] || { echo "usage : $0 unpack <dossier> <jeu>…" >&2; exit 1; }
  src=$2; shift 2
  status=0
  for ds in "$@"; do known "$ds"; unpack "$src" "$ds" || status=1; done
  exit $status
  ;;
kaggle-download)
  shift
  for ds in "$@"; do known "$ds"; kaggle_download "$ds"; done
  kaggle
  status=0
  for ds in "$@"; do check "$ds" || { echo "  ${HOWTO[$ds]}"; status=1; }; done
  exit $status
  ;;
esac
status=0
for ds in "$@"; do
  case "$ds" in
  kaggle)
    kaggle
    ;&
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
