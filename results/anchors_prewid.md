# Ancres par k-means sur prewid (T11)

`python tools/kmeans_anchors.py --dataset prewid --seed 0` — 1581 boîtes non-difficult de prewid train, en pixels de l'image letterbox 416×416 ; distance 1 − IoU (§5.2), initialisation k-means++.

| k | k-means (w,h px à 416×416) | IoU moyenne | Darknet | ancres Darknet | IoU moyenne |
|---|---|---|---|---|---|
| 5 | `16,41  24,65  35,100  82,89  59,145` | 0.7557 | yolov2-tiny-voc.cfg | `35,38  109,141  212,364  301,164  532,337` | 0.4196 |
| 6 | `15,37  20,56  29,79  40,110  82,88  63,149` | 0.7783 | yolov3-tiny.cfg | `10,14  23,27  37,58  81,82  135,169  344,319` | 0.5304 |
