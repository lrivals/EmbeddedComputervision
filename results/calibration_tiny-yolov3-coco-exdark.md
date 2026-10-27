# Calibration INT8 — tiny-yolov3-coco (T4.2)

### tiny-yolov3-coco — 500 images exdark train, choix « mse »

Erreur quadratique moyenne FP32 / INT8 de la sortie de chaque convolution, pour chaque seuil d'écrêtage c (percentile de |x|, s = c/127). **Gras** : seuil retenu.

| Couche | act | MSE p90 | MSE p95 | MSE p99 | MSE p99.9 | MSE p99.99 | MSE max | c retenu | s_y final | écrêté |
|---|---|---|---|---|---|---|---|---|---|---|
| L00 | leaky | 1.85e+00 | 1.51e+00 | 9.02e-01 | 2.57e-02 | **1.03e-02** | 3.07e-02 | p99.99 = 31.2 | 0.2456 | 0.010 % |
| L02 | leaky | 3.20e-01 | 2.26e-01 | 8.19e-02 | 1.23e-02 | **2.24e-03** | 7.74e-03 | p99.99 = 14.7 | 0.1159 | 0.010 % |
| L04 | leaky | 3.42e-01 | 1.99e-01 | 6.29e-02 | 7.56e-03 | **2.48e-03** | 8.78e-03 | p99.99 = 12.5 | 0.09862 | 0.010 % |
| L06 | leaky | 2.65e-01 | 1.54e-01 | 4.10e-02 | 4.60e-03 | **1.10e-03** | 4.18e-03 | p99.99 = 10.7 | 0.08394 | 0.010 % |
| L08 | leaky | 1.76e-01 | 1.11e-01 | 3.26e-02 | 6.05e-03 | **1.56e-03** | 3.89e-03 | p99.99 = 8.85 | 0.06966 | 0.010 % |
| L10 | leaky | 4.41e-02 | 3.32e-02 | 1.20e-02 | 2.81e-03 | **9.09e-04** | 1.92e-03 | p99.99 = 5.42 | 0.04267 | 0.010 % |
| L12 | leaky | 7.84e-01 | 4.44e-01 | 1.09e-01 | 1.63e-02 | **3.81e-03** | 1.66e-02 | p99.99 = 17.7 | 0.1398 | 0.010 % |
| L13 | leaky | 3.05e-02 | 1.61e-02 | 4.28e-03 | 7.83e-04 | **2.24e-04** | 5.91e-04 | p99.99 = 3.9 | 0.03071 | 0.010 % |
| L14 | leaky | 7.91e-02 | 3.90e-02 | 8.76e-03 | 1.19e-03 | **3.25e-04** | 9.14e-04 | p99.99 = 6.09 | 0.04798 | 0.010 % |
| L15 | linear | 5.83e-01 | 3.31e-01 | 8.45e-02 | 5.56e-03 | 2.67e-03 | 4.63e-03 | fixe (tête) | 0.125 | 0.363 % |
| L18 | leaky | 7.13e-02 | 3.59e-02 | 8.65e-03 | 1.65e-03 | **4.56e-04** | 1.25e-03 | p99.99 = 6.15 | 0.06966 | 0.001 % |
| L21 | leaky | 9.27e-02 | 4.39e-02 | 8.58e-03 | 9.53e-04 | **2.89e-04** | 7.69e-04 | p99.99 = 6.31 | 0.04966 | 0.010 % |
| L22 | linear | 5.13e-01 | 2.74e-01 | 6.20e-02 | 4.27e-03 | 2.38e-03 | 4.49e-03 | fixe (tête) | 0.125 | 0.340 % |

Échelle commune imposée aux sources des routes : {L08, L18} (la plus grande des deux).

Têtes linéaires : échelle fixe 0.125 (voir le docstring de `yolo.quant.calibrate`) ; la colonne « écrêté » donne la part des sorties hors de ±127·s.
