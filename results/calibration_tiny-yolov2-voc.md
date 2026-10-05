# Calibration INT8 — tiny-yolov2-voc (T4.2)

### tiny-yolov2-voc — 500 images VOC2007 trainval, choix « mse »

Erreur quadratique moyenne FP32 / INT8 de la sortie de chaque convolution, pour chaque seuil d'écrêtage c (percentile de |x|, s = c/127). **Gras** : seuil retenu.

| Couche | act | MSE p90 | MSE p95 | MSE p99 | MSE p99.9 | MSE p99.99 | MSE max | c retenu | s_y final | écrêté |
|---|---|---|---|---|---|---|---|---|---|---|
| L00 | leaky | 3.39e+00 | 2.82e+00 | 1.27e+00 | 6.25e-02 | **1.94e-02** | 2.33e-02 | p99.99 = 42.9 | 0.338 | 0.010 % |
| L02 | leaky | 2.68e-01 | 1.88e-01 | 6.72e-02 | 1.12e-02 | **2.01e-03** | 7.37e-03 | p99.99 = 13.8 | 0.1088 | 0.010 % |
| L04 | leaky | 3.26e-01 | 2.12e-01 | 7.13e-02 | 7.60e-03 | **1.81e-03** | 7.48e-03 | p99.99 = 13.3 | 0.105 | 0.010 % |
| L06 | leaky | 2.54e-01 | 1.46e-01 | 3.58e-02 | 3.47e-03 | **8.92e-04** | 5.09e-03 | p99.99 = 9.69 | 0.07627 | 0.010 % |
| L08 | leaky | 1.29e-01 | 8.38e-02 | 2.04e-02 | 2.97e-03 | **7.75e-04** | 4.21e-03 | p99.99 = 7.19 | 0.05661 | 0.010 % |
| L10 | leaky | 4.23e-02 | 3.36e-02 | 1.16e-02 | 1.68e-03 | **3.98e-04** | 1.66e-03 | p99.99 = 5.17 | 0.04069 | 0.010 % |
| L12 | leaky | 1.15e+00 | 6.13e-01 | 1.50e-01 | 2.25e-02 | **5.28e-03** | 2.90e-02 | p99.99 = 21.3 | 0.1674 | 0.010 % |
| L13 | leaky | 7.63e-02 | 3.75e-02 | 7.72e-03 | 1.02e-03 | **2.77e-04** | 7.45e-04 | p99.99 = 5.66 | 0.04455 | 0.010 % |
| L14 | linear | 1.29e+00 | 5.11e-01 | 6.23e-02 | 1.27e-02 | 3.02e-03 | 6.00e-03 | fixe (tête) | 0.125 | 0.034 % |

Têtes linéaires : échelle fixe 0.125 (voir le docstring de `yolo.quant.calibrate`) ; la colonne « écrêté » donne la part des sorties hors de ±127·s.
