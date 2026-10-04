#!/usr/bin/env bash
# Télécharge PASCAL VOC2007 (trainval + test) et VOC2012 (trainval) dans data/ (T0.6, §8.3).
# Idempotent : une archive déjà téléchargée ou déjà extraite n'est pas reprise.
set -euo pipefail

DATA_DIR="${1:-$(dirname "$0")/../data}"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

MIRRORS=(
  "http://host.robots.ox.ac.uk/pascal/VOC"
  "https://pjreddie.com/media/files"
)

# archive  chemin sur le miroir officiel  répertoire témoin après extraction
ARCHIVES=(
  "VOCtrainval_06-Nov-2007.tar voc2007/VOCtrainval_06-Nov-2007.tar VOCdevkit/VOC2007/ImageSets/Main/trainval.txt"
  "VOCtest_06-Nov-2007.tar voc2007/VOCtest_06-Nov-2007.tar VOCdevkit/VOC2007/ImageSets/Main/test.txt"
  "VOCtrainval_11-May-2012.tar voc2012/VOCtrainval_11-May-2012.tar VOCdevkit/VOC2012/ImageSets/Main/trainval.txt"
)

download() {
  local name="$1" official="$2"
  for mirror in "${MIRRORS[@]}"; do
    local url
    if [[ "$mirror" == *robots.ox.ac.uk* ]]; then url="$mirror/$official"; else url="$mirror/$name"; fi
    echo "téléchargement $url"
    if curl -fL --retry 3 --connect-timeout 20 -C - -o "$name.part" "$url"; then
      mv "$name.part" "$name"
      return 0
    fi
  done
  echo "échec du téléchargement de $name" >&2
  return 1
}

for entry in "${ARCHIVES[@]}"; do
  read -r name official marker <<<"$entry"
  if [[ -f "$marker" ]]; then
    echo "$name : déjà extrait"
    continue
  fi
  [[ -f "$name" ]] || download "$name" "$official"
  echo "extraction $name"
  tar -xf "$name"
done

echo "VOC prêt dans $(pwd)/VOCdevkit"
