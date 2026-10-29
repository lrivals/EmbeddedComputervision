# Calibration INT8 — tiny-yolov3-coco (T4.2)

### tiny-yolov3-coco — 500 images coco calib2017, choix « mse »

Erreur quadratique moyenne FP32 / INT8 de la sortie de chaque convolution, pour chaque seuil d'écrêtage c (percentile de |x|, s = c/127). **Gras** : seuil retenu.

| Couche | act | MSE p90 | MSE p95 | MSE p99 | MSE p99.9 | MSE p99.99 | MSE max | c retenu | s_y final | écrêté |
|---|---|---|---|---|---|---|---|---|---|---|
| L00 | leaky | 2.61e+00 | 2.10e+00 | 8.93e-01 | 6.11e-02 | **1.61e-02** | 1.61e-02 | p99.99 = 37.8 | 0.2976 | 0.010 % |
| L02 | leaky | 3.71e-01 | 2.54e-01 | 7.61e-02 | 1.18e-02 | **2.54e-03** | 9.94e-03 | p99.99 = 14.9 | 0.1173 | 0.010 % |
| L04 | leaky | 4.25e-01 | 2.25e-01 | 7.06e-02 | 8.73e-03 | **2.25e-03** | 5.48e-03 | p99.99 = 13.8 | 0.109 | 0.010 % |
| L06 | leaky | 3.20e-01 | 1.76e-01 | 4.48e-02 | 5.69e-03 | **1.11e-03** | 4.17e-03 | p99.99 = 11.7 | 0.0918 | 0.010 % |
| L08 | leaky | 2.07e-01 | 1.31e-01 | 3.72e-02 | 5.91e-03 | **1.66e-03** | 4.28e-03 | p99.99 = 9.29 | 0.07314 | 0.010 % |
| L10 | leaky | 5.25e-02 | 4.02e-02 | 1.38e-02 | 2.52e-03 | **5.66e-04** | 1.99e-03 | p99.99 = 5.89 | 0.04637 | 0.010 % |
| L12 | leaky | 1.05e+00 | 6.24e-01 | 1.68e-01 | 2.93e-02 | **7.48e-03** | 2.55e-02 | p99.99 = 21.7 | 0.1708 | 0.010 % |
| L13 | leaky | 3.98e-02 | 2.15e-02 | 6.22e-03 | 1.17e-03 | **2.79e-04** | 1.06e-03 | p99.99 = 4.57 | 0.03596 | 0.010 % |
| L14 | leaky | 1.08e-01 | 5.69e-02 | 1.50e-02 | 2.55e-03 | **6.94e-04** | 2.14e-03 | p99.99 = 7.43 | 0.0585 | 0.010 % |
| L15 | linear | 7.82e-01 | 4.12e-01 | 8.38e-02 | 1.06e-02 | 4.28e-03 | 9.26e-03 | fixe (tête) | 0.125 | 0.755 % |
| L18 | leaky | 9.37e-02 | 4.99e-02 | 1.46e-02 | 3.24e-03 | **9.01e-04** | 1.95e-03 | p99.99 = 7.43 | 0.07314 | 0.003 % |
| L21 | leaky | 1.16e-01 | 5.71e-02 | 1.26e-02 | 1.85e-03 | **5.28e-04** | 3.31e-03 | p99.99 = 7.4 | 0.05826 | 0.010 % |
| L22 | linear | 6.59e-01 | 3.46e-01 | 6.68e-02 | 7.12e-03 | 3.26e-03 | 6.82e-03 | fixe (tête) | 0.125 | 0.584 % |

Échelle commune imposée aux sources des routes : {L08, L18} (la plus grande des deux).

Têtes linéaires : échelle fixe 0.125 (voir le docstring de `yolo.quant.calibrate`) ; la colonne « écrêté » donne la part des sorties hors de ±127·s.
