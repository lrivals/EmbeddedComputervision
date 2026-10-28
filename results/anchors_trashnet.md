# Ancres par k-means sur trashnet (T11)

`python tools/kmeans_anchors.py --dataset trashnet --seed 0` — 1765 boîtes non-difficult de trashnet train, en pixels de l'image letterbox 416×416 ; distance 1 − IoU (§5.2), initialisation k-means++.

| k | k-means (w,h px à 416×416) | IoU moyenne | Darknet | ancres Darknet | IoU moyenne |
|---|---|---|---|---|---|
| 5 | `112,106  185,208  314,155  274,259  387,293` | 0.7874 | yolov2-tiny-voc.cfg | `35,38  109,141  212,364  301,164  532,337` | 0.6512 |
| 6 | `104,101  179,207  282,134  259,264  348,211  386,302` | 0.8050 | yolov3-tiny.cfg | `10,14  23,27  37,58  81,82  135,169  344,319` | 0.6707 |
