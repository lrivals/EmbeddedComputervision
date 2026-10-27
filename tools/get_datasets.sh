#!/usr/bin/env bash
# Jeux de données de M11 dans data/ (docs/tasks/M11-jeux-de-donnees.md).
#
#   tools/get_datasets.sh <jeu>…     coco | kitti | visdrone | crowdhuman | exdark | flir
#                                    | auair | dronevehicle | hituav | uavdt (M18) | asl (M21)
#                                    | trashnet | pets (M32: download + tools/prep_datasets.py)
#                                    | prewid (M38: hand-copied folder + tools/prep_datasets.py)
#                                    | mot17 (M27, T27.4: MOT17 train videos, 5.5 GB)
#                                    | safe_unsafe | soda (M41, M42: hand-copied, checked here)
#                                    | construction_ppe (M43: download + tools/prep_datasets.py)
#   tools/get_datasets.sh check      état de chaque jeu (rien n'est téléchargé)
#   tools/get_datasets.sh kaggle     relie les versions Kaggle (CrowdHuman, VisDrone, ExDark, FLIR)
#                                    et les dossiers bruts de M18, puis lance leurs prétraitements
#                                    (tools/prep_datasets.py : DroneVehicle, UAVDT, ASL)
#   tools/get_datasets.sh kaggle-download <jeu>…   crowdhuman | visdrone | exdark | flir | asl, par l'API
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
# téléchargement (tableau de synthèse de M11). Jeux drone et thermiques de M18
# (docs/tasks/M18-jeux-drone.md) : dossiers bruts déposés dans data/, reliés et prétraités
# par `kaggle`. ASL Alphabet (M21, docs/tasks/M21-asl-alphabet.md): hand pseudo-boxes made
# by `kaggle` (tools/prep_datasets.py asl, needs `pip install -e "python[asl]"`).
# TrashNet and Oxford-IIIT Pet (M32, docs/tasks/M32-trashnet-pets.md): downloaded here, then
# boxed by tools/prep_datasets.py (TrashNet: M22 labeller, `pip install -e
# "python[autolabel]"`, ≈ 1 h on CPU; Pet: trimaps, seconds).
# PREWID (M38, docs/tasks/M38-prewid.md): downloaded by hand from Mendeley Data (the folder
# keeps the truncated archive name), then split by tools/prep_datasets.py (seconds).
# Industrial sets of M41-M43: Safe and Unsafe Behaviours (Mendeley, CC BY 4.0) and SODA
# (authors' OneDrive, no licence stated) are copied by hand, their prep comes with T41.1 and
# T42.1; Construction-PPE (Ultralytics, AGPL-3.0: never in the repository, private Drive
# archive) is downloaded here and remapped by tools/prep_datasets.py (seconds).
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
PREWID_RAW="PREWID PRE-industry Worker Image Dataset for Indus/PREWID"  # name of the archive
SAFE_UNSAFE_RAW="Video Dataset for Safe and Unsafe Behaviours"  # folder of the Mendeley zip
SODA_RAW="SODA"
CPPE_RAW="construction-ppe"  # folder of the Ultralytics zip (tools/prep_datasets.py CPPE_RAW)
CPPE_URL=https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
declare -A MARKER=(
  [voc]=VOCdevkit/VOC2007/ImageSets/Main/test.txt
  [coco]=coco/annotations/instances_val2017.json
  [kitti]=kitti/training/image_2/000000.png
  [visdrone]=visdrone/VisDrone2019-DET-val/annotations
  [crowdhuman]=crowdhuman/annotation_val.odgt
  [exdark]=exdark/imageclasslist.txt
  [flir]=flir/images_thermal_val/coco.json
  [auair]=auair/annotations.json
  [dronevehicle]=dronevehicle/test/labels
  [hituav]=hituav/labels/test
  [uavdt]=uavdt/annotations_test.json
  [asl]=asl/annotations_val.json
  [trashnet]=trashnet/annotations_val.json
  [pets]=pets/annotations_val.json
  [prewid]=prewid/annotations_val.json
  [mot17]=mot17/train/MOT17-02-FRCNN/gt/gt.txt
  [safe_unsafe]=safe_unsafe/annotations_val.json
  [soda]=soda/annotations_val.json
  [construction_ppe]=construction_ppe/annotations_val.json
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
    (ou kaggle.com/datasets/samdazel/teledyne-flir-adas-thermal-dataset-v2 dans data/FLIR Dataset/, puis : $0 kaggle)
    data/flir/images_thermal_{train,val}/coco.json et data/"
  [auair]="https://bozcani.github.io/auairdataset : auair2019annotations.zip et auair2019data.zip
    dans data/AU-AIR dataset/AU-AIR dataset/{annotations.json,images/}, puis : $0 kaggle
    data/auair/annotations.json, data/auair/images/"
  [dronevehicle]="https://github.com/VisDrone/DroneVehicle (version YOLO-OBB de Roboflow, RGB seul)
    dans data/DroneVehicle Dataset/DroneVehiclesDatasetYOLO/{train,val,test}/{images,labels},
    puis : $0 kaggle (recadrage et boîtes englobantes : tools/prep_datasets.py dronevehicle)
    data/dronevehicle/{train,val,test}/{images,labels}/"
  [hituav]="https://github.com/suojiashun/HIT-UAV-Infrared-Thermal-Dataset (version YOLO)
    dans data/HIT-UAV: A High-altitude Infrared Thermal Dataset/hit-uav/, puis : $0 kaggle
    data/hituav/{images,labels}/{train,val,test}/"
  [uavdt]="https://sites.google.com/view/grli-uavdt (export Supervisely de DatasetNinja)
    dans data/UAVDT Dataset/{train,test}/{img,ann}, puis : $0 kaggle
    (séquences M seules : tools/prep_datasets.py uavdt)
    data/uavdt/annotations_{train,test}.json, data/uavdt/images/{train,test}/"
  [asl]="https://www.kaggle.com/datasets/grassknoted/asl-alphabet ($0 kaggle-download asl)
    in data/ASL Alphabet/asl_alphabet_{train,test}/, then: $0 kaggle
    (hand pseudo-boxes: pip install -e \"python[asl]\"; tools/prep_datasets.py asl)
    data/asl/annotations_{train,val,test}.json, data/asl/images/{train,val,test}/"
  [trashnet]="https://github.com/garythung/trashnet (data/dataset-resized.zip): $0 trashnet
    in data/TrashNet/dataset-resized/<class>/, boxes: pip install -e \"python[autolabel]\";
    tools/prep_datasets.py trashnet
    data/trashnet/annotations_{train,val,test}.json, data/trashnet/images/{train,val,test}/"
  [pets]="https://www.robots.ox.ac.uk/~vgg/data/pets/ (images.tar.gz, annotations.tar.gz): $0 pets
    in data/Oxford-IIIT Pet/{images,annotations}/, boxes: tools/prep_datasets.py pets
    data/pets/annotations_{train,val,test}.json, data/pets/images/{train,val,test}/"
  [prewid]="https://data.mendeley.com/datasets/xgt8kyxyrz/1 (Download All, doi:10.17632/xgt8kyxyrz.1,
    CC BY 4.0: cite Silvestre-Blanes, Albero-Albero, Zafrilla-Muñoz, Perez Llorens (2026), PREWID,
    Mendeley Data V1), unzipped as data/$PREWID_RAW/{01_Images,02_GroundTruth}/, then: $0 prewid
    data/prewid/annotations_{train,val,test,lab}.json, data/prewid/images/{train,val,test,lab}/"
  [safe_unsafe]="https://data.mendeley.com/datasets/xjmtb22pff/1 (Download All, doi:10.17632/xjmtb22pff.1,
    CC BY 4.0: cite Önal, Dandıl (2024), Video dataset for the detection of safe and unsafe
    behaviours in workplaces, Data in Brief 56, 110756, doi:10.1016/j.dib.2024.110756;
    also Hugging Face Voxel51/Safe_and_Unsafe_Behaviours), unzipped as data/$SAFE_UNSAFE_RAW/,
    then: $0 safe_unsafe; frames and boxes: T41.1-T41.3 (docs/tasks/M41-safe-unsafe-behaviours.md)
    data/safe_unsafe/annotations_{train,val,test}.json"
  [soda]="https://linjiarui.net/en/portfolio/2022-02-22-SODA-site-object-detection-dataset-for-deep-learning-in-construction
    (HKUST OneDrive link, Baidu copy; arXiv:2202.09554, VOC xml). No licence stated: ask the
    authors before any copy (T42.0, docs/tasks/M42-soda.md); unzipped as data/$SODA_RAW/,
    then: $0 soda; boxes and split: T42.1
    data/soda/annotations_{train,val,test}.json"
  [construction_ppe]="https://docs.ultralytics.com/datasets/detect/construction-ppe/ (AGPL-3.0): $0 construction_ppe
    downloads $CPPE_URL into data/, then tools/prep_datasets.py construction_ppe
    data/construction_ppe/annotations{,_ppe}_{train,val,test}.json, data/construction_ppe/images/{train,val,test}/"
)

# Dossier de chaque jeu dans data/ (Dataset.root de python/yolo/data/datasets.py) : contenu
# de son archive.
declare -A DIR=([voc]=VOCdevkit)

# Versions Kaggle (kaggle-download) : jeu → identifiant et dossier de data/ attendu par kaggle().
declare -A KAGGLE_ID=(
  [crowdhuman]=leducnhuan/crowdhuman
  [visdrone]=kushagrapandya/visdrone-dataset
  [exdark]=washingtongold/exdark-dataset
  [flir]=samdazel/teledyne-flir-adas-thermal-dataset-v2
  [asl]=grassknoted/asl-alphabet
)
declare -A KAGGLE_DIR=([crowdhuman]=CrowdHuman [visdrone]="VisDrone Dataset" [exdark]="ExDark Dataset"
  [flir]="FLIR Dataset" [asl]="ASL Alphabet")

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
    || { echo "CLI kaggle absente : pip install kaggle (jeton : KAGGLE_API_TOKEN ou ~/.kaggle/access_token)" >&2; return 1; }
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
  # FLIR : profondeur du miroir Kaggle incertaine (FLIR_ADAS_v2/…) ; data/flir pointe sur le
  # dossier qui contient images_thermal_val/.
  local d
  if [[ -d "$DATA_DIR/FLIR Dataset" ]]; then
    d=$(find "$DATA_DIR/FLIR Dataset" -maxdepth 3 -type d -name images_thermal_val -print -quit)
    if [[ -n $d ]]; then link "$(dirname "${d#"$DATA_DIR"/}")" flir; fi
  fi
  m18
  m21
}

# Jeux de M18 : dossiers bruts (noms d'origine) reliés ou prétraités vers data/<jeu>.
m18() {
  link "AU-AIR dataset/AU-AIR dataset" auair
  link "HIT-UAV: A High-altitude Infrared Thermal Dataset/hit-uav" hituav
  local prep=()
  [[ -d "$DATA_DIR/DroneVehicle Dataset" && ! -e "$DATA_DIR/${MARKER[dronevehicle]}" ]] \
    && prep+=(dronevehicle)
  [[ -d "$DATA_DIR/UAVDT Dataset" && ! -e "$DATA_DIR/${MARKER[uavdt]}" ]] && prep+=(uavdt)
  if [[ ${#prep[@]} -gt 0 ]]; then
    python "$ROOT/tools/prep_datasets.py" "${prep[@]}"
  fi
  return 0
}

# ASL Alphabet (M21): hand pseudo-boxes, once (the marker is written last).
m21() {
  if [[ -d "$DATA_DIR/ASL Alphabet" && ! -e "$DATA_DIR/${MARKER[asl]}" ]]; then
    python "$ROOT/tools/prep_datasets.py" asl
  fi
  return 0
}

# TrashNet and Oxford-IIIT Pet (M32): download, extract, then the prep (marker written last).
trashnet() {
  mkdir -p "$DATA_DIR/TrashNet"
  if [[ ! -d "$DATA_DIR/TrashNet/dataset-resized" ]]; then
    download https://github.com/garythung/trashnet/raw/master/data/dataset-resized.zip \
      "$DATA_DIR/TrashNet/dataset-resized.zip"
    unzip -q -o "$DATA_DIR/TrashNet/dataset-resized.zip" -d "$DATA_DIR/TrashNet"
  fi
  [[ -e "$DATA_DIR/${MARKER[trashnet]}" ]] || python "$ROOT/tools/prep_datasets.py" trashnet
}

pets() {
  local d="$DATA_DIR/Oxford-IIIT Pet" f
  mkdir -p "$d"
  for f in images annotations; do
    if [[ ! -d "$d/$f" ]]; then
      download "https://www.robots.ox.ac.uk/~vgg/data/pets/data/$f.tar.gz" "$d/$f.tar.gz"
      tar -xzf "$d/$f.tar.gz" -C "$d"
    fi
  done
  [[ -e "$DATA_DIR/${MARKER[pets]}" ]] || python "$ROOT/tools/prep_datasets.py" pets
}

# MOT17 (M27, T27.4): ground-level pedestrian videos, read by `load_mot17` from
# data/mot17/train/<SEQ>-FRCNN/. The archive holds a top folder MOT17/: moved up one level.
mot17() {
  fetch mot17 https://motchallenge.net/data/MOT17.zip train/MOT17-02-FRCNN/gt/gt.txt
  if [[ -d "$DATA_DIR/mot17/MOT17/train" && ! -e "$DATA_DIR/${MARKER[mot17]}" ]]; then
    mv "$DATA_DIR/mot17/MOT17/"* "$DATA_DIR/mot17/" && rmdir "$DATA_DIR/mot17/MOT17"
  fi
}

# PREWID (M38): no automatic download; frame counts of the hand-copied folder, then the prep.
prewid() {
  local d="$DATA_DIR/$PREWID_RAW" cam
  [[ -d "$d/01_Images" ]] || { echo "prewid : $d absent"; return 1; }
  for cam in CAM0 CAM1 CAM2 CAM3 CAMLAB01; do
    echo "prewid $cam : $(find "$d/01_Images/$cam" -name '*.jpg' | wc -l) frames"
  done
  [[ -e "$DATA_DIR/${MARKER[prewid]}" ]] || python "$ROOT/tools/prep_datasets.py" prewid
}

# Safe and Unsafe Behaviours (M41) and SODA (M42): no automatic download; file counts of the
# hand-copied folder (to check the Summary of T41.0 / T42.0). Their prep comes with T41.1 /
# T42.1.
safe_unsafe() {
  local d="$DATA_DIR/$SAFE_UNSAFE_RAW"
  [[ -d "$d" ]] || { echo "safe_unsafe : $d absent"; return 1; }
  echo "safe_unsafe : $(find "$d" -iname '*.mp4' -o -iname '*.avi' | wc -l) clips"
  find "$d" -mindepth 1 -maxdepth 2 -type d | sed "s|$d/|  |"
}

soda() {
  local d="$DATA_DIR/$SODA_RAW"
  [[ -d "$d" ]] || { echo "soda : $d absent"; return 1; }
  echo "soda : $(find "$d" -iname '*.jpg' | wc -l) images, $(find "$d" -iname '*.xml' | wc -l) xml"
}

# Construction-PPE (M43): download, extract, then the remap (marker written last).
construction_ppe() {
  if [[ ! -d "$DATA_DIR/$CPPE_RAW/images" ]]; then
    download "$CPPE_URL" "$DATA_DIR/construction-ppe.zip"
    unzip -q -o "$DATA_DIR/construction-ppe.zip" -d "$DATA_DIR"
  fi
  [[ -e "$DATA_DIR/${MARKER[construction_ppe]}" ]] \
    || python "$ROOT/tools/prep_datasets.py" construction_ppe
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

[[ $# -gt 0 ]] || { sed -n '2,41p' "$0"; exit 1; }
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
    for d in coco kitti visdrone crowdhuman exdark flir auair dronevehicle hituav uavdt asl \
      trashnet pets prewid mot17 safe_unsafe soda construction_ppe; do
      check "$d" || true
    done
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
  trashnet|pets|mot17|construction_ppe)
    "$ds"
    check "$ds" || status=1
    ;;
  prewid|safe_unsafe|soda)
    "$ds" || echo "  téléchargement manuel : ${HOWTO[$ds]}"
    check "$ds" || status=1
    ;;
  visdrone|crowdhuman|exdark|flir|auair|dronevehicle|hituav|uavdt|asl)
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
