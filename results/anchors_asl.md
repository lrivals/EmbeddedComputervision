# Ancres par k-means sur asl (T11)

`python tools/kmeans_anchors.py --dataset asl --seed 0` — 67434 boîtes non-difficult de asl train, en pixels de l'image letterbox 416×416 ; distance 1 − IoU (§5.2), initialisation k-means++.

| k | k-means (w,h px à 416×416) | IoU moyenne | Darknet | ancres Darknet | IoU moyenne |
|---|---|---|---|---|---|
| 5 | `151,144  105,218  233,201  159,321  297,329` | 0.7512 | yolov2-tiny-voc.cfg | `35,38  109,141  212,364  301,164  532,337` | 0.6373 |
| 6 | `99,163  191,152  127,245  246,224  171,347  310,339` | 0.7677 | yolov3-tiny.cfg | `10,14  23,27  37,58  81,82  135,169  344,319` | 0.6152 |
