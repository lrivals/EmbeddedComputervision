# mAP flottante de référence (T3.4)

Tiny-YOLOv2 VOC, poids Darknet `yolov2-tiny-voc.weights` (pjreddie.com), modèle NumPy
float32 de `python/yolo/`, VOC2007 test (4 952 images), entrée 416×416, seuil de confiance
0,005, NMS par classe 0,45, AP 11 points (VOC2007), objets `difficult` ignorés.

```
python tools/eval_voc.py --net tiny-yolov2-voc --weights weights/yolov2-tiny-voc.weights \
    --resize stretch            # référence
python tools/eval_voc.py ... --resize letterbox [--interp darknet]
```

## Résultat

**Référence pour M4 (entier) et M8 (matériel) : mAP = 56,30** (redimensionnement direct,
interpolation Pillow). Valeur publiée par Darknet pour Tiny-YOLOv2 VOC : **57,1**.
Écart : **0,8 point**.

| Prétraitement | Interpolation | mAP |
|---|---|---|
| redimensionnement direct (`stretch`) | Pillow bilinéaire | **56,30** |
| redimensionnement direct (`stretch`) | Darknet (`resize_image`) | 56,32 |
| letterbox | Pillow bilinéaire | 54,18 |
| letterbox | Darknet (`resize_image`) | 53,98 |

## Explication de l'écart

- **Prétraitement (≈ 2,1 points)** : ces poids ont été entraînés et évalués par le Darknet
  de l'époque YOLOv2, qui redimensionne l'image au carré sans conserver le rapport
  d'aspect. Avec la letterbox (celle de notre pipeline d'entraînement), les objets sont
  plus petits et déformés autrement qu'à l'entraînement : −2,1 points. La référence utilise
  donc `stretch`, qui est aussi le plus simple à reproduire en matériel.
- **Interpolation (≈ 0)** : reproduire `resize_image` de Darknet (bilinéaire à coins
  alignés, sans anti-crénelage ; testé contre un port ligne à ligne) au lieu du bilinéaire
  de Pillow ne change la mAP que de ±0,2 point. Ce n'est pas la cause de l'écart.
- **Reste (0,8 point, sous la tolérance de 1 point)** : causes plausibles, non mesurées ici —
  décodage JPEG (stb_image dans Darknet, libjpeg dans Pillow), détails de la boucle
  d'évaluation de Darknet (`validate_detector`, version du code ayant produit le 57,1) et
  script d'évaluation utilisé pour le chiffre publié. Notre évaluation est identique au
  devkit officiel à 1e-12 près (`python/tests/test_metrics.py`).

## AP par classe

| Classe | stretch | stretch (Darknet) | letterbox | letterbox (Darknet) |
|---|---|---|---|---|
| aeroplane | 61,4 | 61,9 | 57,9 | 58,0 |
| bicycle | 71,6 | 71,8 | 67,8 | 67,4 |
| bird | 48,5 | 48,4 | 47,2 | 46,1 |
| boat | 41,5 | 42,1 | 36,5 | 34,8 |
| bottle | 21,6 | 21,7 | 21,2 | 19,9 |
| bus | 67,8 | 67,9 | 67,7 | 66,9 |
| car | 67,0 | 67,2 | 63,0 | 62,9 |
| cat | 70,4 | 70,7 | 66,4 | 66,8 |
| chair | 35,3 | 35,0 | 31,3 | 31,8 |
| cow | 54,6 | 53,8 | 53,5 | 53,8 |
| diningtable | 57,5 | 58,2 | 55,4 | 57,0 |
| dog | 62,3 | 62,4 | 62,0 | 62,4 |
| horse | 70,9 | 71,0 | 70,1 | 70,0 |
| motorbike | 70,5 | 68,8 | 69,8 | 68,8 |
| person | 60,6 | 59,9 | 58,3 | 58,1 |
| pottedplant | 27,5 | 27,8 | 26,8 | 27,1 |
| sheep | 56,5 | 55,6 | 50,7 | 52,2 |
| sofa | 52,1 | 52,0 | 51,8 | 50,7 |
| train | 69,7 | 70,7 | 68,1 | 67,7 |
| tvmonitor | 58,6 | 59,2 | 58,1 | 57,2 |
| **mAP** | **56,30** | **56,32** | **54,18** | **53,98** |

Temps : ≈ 170 ms/image (passe avant NumPy float32, 22 cœurs), ≈ 15 min par passe.
