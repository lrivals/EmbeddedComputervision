# mAP aux trois stades (T8.2)

tiny-yolov2-voc, VOC2007 test (4952 images), redimensionnement direct 416×416 (`stretch`, Pillow), seuil 0,005, NMS 0,45, AP 11 points. Généré par `python tools/map_stades.py` (`make map-stades`).

Les stades entier et FPGA reçoivent **les mêmes octets** : entrées int8 de `tools/make_inputs.py` (`preprocess` + `quantize_input`), lues par `sw/app/yolo_bench --inputs` ; le décodage JPEG de la carte (stb) n'intervient pas. Égalité exigée : boîtes, scores et classes identiques, image par image (flottants comparés exactement). Stade C-sim : `make bench-sim` (noyau compilé sans `ap_int`, `-DACC_NO_APINT`, même arithmétique ; les 1 832 premières images ont aussi été calculées avec `ap_int`, mêmes détections).

| Stade | mAP | Égalité avec l'entier |
|---|---|---|
| Flottant (BN fusionnée, NumPy float32) | **56.30** | — |
| Entier Python (`IntNetwork`, bit-exact) | **55.66** | référence |
| FPGA, C-sim (driver ARM, backend sim, PC) | **55.66** | 4952 / 4952 images identiques |
| FPGA, KV260 (backend uio) | — | à mesurer |

## AP par classe

| Classe | Flottant | Entier Python | FPGA, C-sim |
|---|---|---|---|
| aeroplane | 61.4 | 60.1 | 60.1 |
| bicycle | 71.6 | 73.1 | 73.1 |
| bird | 48.5 | 47.3 | 47.3 |
| boat | 41.5 | 41.8 | 41.8 |
| bottle | 21.6 | 21.3 | 21.3 |
| bus | 67.8 | 67.5 | 67.5 |
| car | 67.0 | 66.9 | 66.9 |
| cat | 70.4 | 68.6 | 68.6 |
| chair | 35.3 | 34.8 | 34.8 |
| cow | 54.6 | 52.9 | 52.9 |
| diningtable | 57.5 | 55.8 | 55.8 |
| dog | 62.3 | 62.2 | 62.2 |
| horse | 70.9 | 71.3 | 71.3 |
| motorbike | 70.5 | 69.1 | 69.1 |
| person | 60.6 | 58.9 | 58.9 |
| pottedplant | 27.5 | 26.6 | 26.6 |
| sheep | 56.5 | 55.0 | 55.0 |
| sofa | 52.1 | 51.5 | 51.5 |
| train | 69.7 | 70.1 | 70.1 |
| tvmonitor | 58.6 | 58.4 | 58.4 |

## Reproduire

```
make m8-inputs          # build/m8/<net>/inputs.bin (2,6 Go) + ids.txt
make m8-int             # eval_quant float,int --resize stretch --save-dets
make bench-sim          # stade FPGA en C-sim, lots parallèles
# carte : yolo_bench --inputs ... --dets build/m8/<net>/board/dets_0.jsonl (results/protocole.md)
make map-stades
```
