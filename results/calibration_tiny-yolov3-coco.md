# Calibration INT8 — tiny-yolov3-coco (T4.2)

### tiny-yolov3-coco — 500 images VOC2007 trainval, choix « mse »

Erreur quadratique moyenne FP32 / INT8 de la sortie de chaque convolution, pour chaque seuil d'écrêtage c (percentile de |x|, s = c/127). **Gras** : seuil retenu.

| Couche | act | MSE p90 | MSE p95 | MSE p99 | MSE p99.9 | MSE p99.99 | MSE max | c retenu | s_y final | écrêté |
|---|---|---|---|---|---|---|---|---|---|---|
| L00 | leaky | 2.55e+00 | 2.09e+00 | 8.93e-01 | 5.19e-02 | **1.37e-02** | 1.57e-02 | p99.99 = 36.7 | 0.2893 | 0.010 % |
| L02 | leaky | 3.37e-01 | 2.35e-01 | 7.49e-02 | 1.26e-02 | **2.61e-03** | 8.04e-03 | p99.99 = 15 | 0.1183 | 0.010 % |
| L04 | leaky | 4.10e-01 | 2.08e-01 | 6.48e-02 | 8.07e-03 | **2.38e-03** | 8.64e-03 | p99.99 = 13.3 | 0.105 | 0.010 % |
| L06 | leaky | 3.10e-01 | 1.70e-01 | 4.20e-02 | 4.96e-03 | **1.00e-03** | 4.86e-03 | p99.99 = 11.3 | 0.08879 | 0.010 % |
| L08 | leaky | 2.03e-01 | 1.26e-01 | 3.58e-02 | 5.97e-03 | **1.72e-03** | 7.13e-03 | p99.99 = 9.22 | 0.07257 | 0.010 % |
| L10 | leaky | 4.88e-02 | 3.73e-02 | 1.27e-02 | 2.34e-03 | **5.34e-04** | 2.72e-03 | p99.99 = 5.74 | 0.0452 | 0.010 % |
| L12 | leaky | 9.44e-01 | 5.51e-01 | 1.44e-01 | 2.28e-02 | **4.71e-03** | 1.78e-02 | p99.99 = 20.5 | 0.1612 | 0.010 % |
| L13 | leaky | 3.43e-02 | 1.80e-02 | 5.00e-03 | 9.47e-04 | **2.13e-04** | 6.23e-04 | p99.99 = 4.29 | 0.03377 | 0.010 % |
| L14 | leaky | 8.72e-02 | 4.36e-02 | 9.88e-03 | 1.32e-03 | **3.47e-04** | 1.02e-03 | p99.99 = 6.37 | 0.05018 | 0.010 % |
| L15 | linear | 6.29e-01 | 3.34e-01 | 6.82e-02 | 5.35e-03 | 2.69e-03 | 4.60e-03 | fixe (tête) | 0.125 | 0.452 % |
| L18 | leaky | 7.91e-02 | 4.05e-02 | 1.07e-02 | 2.30e-03 | **6.54e-04** | 1.61e-03 | p99.99 = 6.62 | 0.07257 | 0.002 % |
| L21 | leaky | 9.79e-02 | 4.63e-02 | 9.19e-03 | 1.14e-03 | **3.31e-04** | 9.41e-04 | p99.99 = 6.5 | 0.05119 | 0.010 % |
| L22 | linear | 5.29e-01 | 2.93e-01 | 6.68e-02 | 4.58e-03 | 2.48e-03 | 4.77e-03 | fixe (tête) | 0.125 | 0.346 % |

Échelle commune imposée aux sources des routes : {L08, L18} (la plus grande des deux).

Têtes linéaires : échelle fixe 0.125 (voir le docstring de `yolo.quant.calibrate`) ; la colonne « écrêté » donne la part des sorties hors de ±127·s.
