# Calibration INT8 — tiny-yolov2-voc (T4.2)

### tiny-yolov2-voc — 500 images exdark train, choix « mse »

Erreur quadratique moyenne FP32 / INT8 de la sortie de chaque convolution, pour chaque seuil d'écrêtage c (percentile de |x|, s = c/127). **Gras** : seuil retenu.

| Couche | act | MSE p90 | MSE p95 | MSE p99 | MSE p99.9 | MSE p99.99 | MSE max | c retenu | s_y final | écrêté |
|---|---|---|---|---|---|---|---|---|---|---|
| L00 | leaky | 2.46e+00 | 1.99e+00 | 1.33e+00 | 3.83e-02 | **1.52e-02** | 4.62e-02 | p99.99 = 36.5 | 0.2874 | 0.010 % |
| L02 | leaky | 2.30e-01 | 1.82e-01 | 6.88e-02 | 1.23e-02 | **2.00e-03** | 6.90e-03 | p99.99 = 14 | 0.1102 | 0.010 % |
| L04 | leaky | 2.76e-01 | 1.90e-01 | 7.18e-02 | 8.40e-03 | **1.72e-03** | 8.00e-03 | p99.99 = 13.2 | 0.1038 | 0.010 % |
| L06 | leaky | 2.31e-01 | 1.44e-01 | 3.97e-02 | 4.13e-03 | **1.23e-03** | 4.65e-03 | p99.99 = 9.68 | 0.07621 | 0.010 % |
| L08 | leaky | 1.18e-01 | 7.72e-02 | 1.95e-02 | 2.69e-03 | **6.47e-04** | 2.15e-03 | p99.99 = 6.99 | 0.05502 | 0.010 % |
| L10 | leaky | 3.73e-02 | 2.92e-02 | 9.28e-03 | 1.37e-03 | **4.05e-04** | 1.52e-03 | p99.99 = 4.52 | 0.03562 | 0.010 % |
| L12 | leaky | 9.83e-01 | 4.95e-01 | 1.09e-01 | 1.49e-02 | **4.07e-03** | 1.69e-02 | p99.99 = 18.4 | 0.1452 | 0.010 % |
| L13 | leaky | 6.62e-02 | 3.14e-02 | 6.06e-03 | 6.48e-04 | **1.84e-04** | 1.22e-03 | p99.99 = 5.14 | 0.04047 | 0.010 % |
| L14 | linear | 1.02e+00 | 3.86e-01 | 3.66e-02 | 6.27e-03 | 2.07e-03 | 3.44e-03 | fixe (tête) | 0.125 | 0.009 % |

Têtes linéaires : échelle fixe 0.125 (voir le docstring de `yolo.quant.calibrate`) ; la colonne « écrêté » donne la part des sorties hors de ±127·s.
