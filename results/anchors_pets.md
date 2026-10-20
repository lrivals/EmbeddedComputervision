# Ancres par k-means sur pets (T11)

`python tools/kmeans_anchors.py --dataset pets --seed 0` — 3302 boîtes non-difficult de pets train, en pixels de l'image letterbox 416×416 ; distance 1 − IoU (§5.2), initialisation k-means++.

| k | k-means (w,h px à 416×416) | IoU moyenne | Darknet | ancres Darknet | IoU moyenne |
|---|---|---|---|---|---|
| 5 | `151,206  227,263  328,233  268,374  376,312` | 0.8138 | yolov2-tiny-voc.cfg | `35,38  109,141  212,364  301,164  532,337` | 0.6623 |
| 6 | `151,194  214,279  304,213  267,373  357,278  383,356` | 0.8235 | yolov3-tiny.cfg | `10,14  23,27  37,58  81,82  135,169  344,319` | 0.6832 |
