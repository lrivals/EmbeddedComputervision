#!/usr/bin/env bash
# Télécharge les poids Darknet pré-entraînés (hors base, T1.9) dans weights/.
#   yolov3-tiny.weights      : Tiny-YOLOv3 COCO   (python/yolo/models/cfg/yolov3-tiny.cfg)
#   yolov2-tiny-voc.weights  : Tiny-YOLOv2 VOC    (python/yolo/models/cfg/yolov2-tiny-voc.cfg)
#   yolov4-tiny.weights      : YOLOv4-tiny COCO   (python/yolo/models/cfg/yolov4-tiny.cfg, T24.0,
#                              release AlexeyAB/darknet)
set -euo pipefail
DEST="$(cd "$(dirname "$0")/.." && pwd)/weights"
mkdir -p "$DEST"
declare -A URL=(
  [yolov3-tiny.weights]=https://pjreddie.com/media/files/yolov3-tiny.weights
  [yolov2-tiny-voc.weights]=https://pjreddie.com/media/files/yolov2-tiny-voc.weights
  [yolov4-tiny.weights]=https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v4_pre/yolov4-tiny.weights
)
for f in yolov3-tiny.weights yolov2-tiny-voc.weights yolov4-tiny.weights; do
    if [[ -s "$DEST/$f" ]]; then
        echo "$f : déjà présent"
    else
        curl -fL --retry 3 -o "$DEST/$f.part" "${URL[$f]}"
        mv "$DEST/$f.part" "$DEST/$f"
    fi
done
ls -l "$DEST"/*.weights
