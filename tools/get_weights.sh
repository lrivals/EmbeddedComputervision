#!/usr/bin/env bash
# Télécharge les poids Darknet pré-entraînés (hors base, T1.9) dans weights/.
#   yolov3-tiny.weights      : Tiny-YOLOv3 COCO   (python/yolo/models/cfg/yolov3-tiny.cfg)
#   yolov2-tiny-voc.weights  : Tiny-YOLOv2 VOC    (python/yolo/models/cfg/yolov2-tiny-voc.cfg)
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/weights"
mkdir -p "$DEST"
for f in yolov3-tiny.weights yolov2-tiny-voc.weights; do
    if [[ -s "$DEST/$f" ]]; then
        echo "$f : déjà présent"
    else
        curl -fL --retry 3 -o "$DEST/$f.part" "https://pjreddie.com/media/files/$f"
        mv "$DEST/$f.part" "$DEST/$f"
    fi
done
ls -l "$DEST"/*.weights
