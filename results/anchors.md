# Ancres par k-means sur VOC (T2.2)

`python tools/kmeans_anchors.py --seed 0` — 40058 boîtes non-difficult de VOC2007 + VOC2012 trainval, en pixels de l'image letterbox 416 ; distance 1 − IoU (§5.2), initialisation k-means++.

| k | k-means (w,h px à 416) | IoU moyenne | Darknet | ancres Darknet | IoU moyenne |
|---|---|---|---|---|---|
| 5 | `38,49  90,112  141,225  264,135  312,278` | 0.6113 | yolov2-tiny-voc.cfg | `35,38  109,141  212,364  301,164  532,337` | 0.5591 |
| 6 | `35,47  84,95  109,192  252,129  193,263  341,269` | 0.6307 | yolov3-tiny.cfg | `10,14  23,27  37,58  81,82  135,169  344,319` | 0.6029 |
