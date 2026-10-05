# Précision mixte par couche (T10.10)

Tiny-YOLOv2 VOC. Poids par couche en INT8, `uniform6` (entiers ±31), `mixed6` (puissances de
2 de REQ-YOLO, T9.2) ou `uniform4` (±7, paquetés deux par octet). Activations INT8
calibrées (`build/quant/tiny-yolov2-voc/calib.json`). Le facteur d'échelle est choisi par
canal, sur la grille de `pow2.project`. Prétraitement `stretch`.

Production : `tools/mixed_precision.py sens | front | full`, avec les résultats dans
`build/m10/mixed/`.

## Sensibilité (500 images de VOC2007 test)

Une seule couche en bas bit, les autres en INT8 projeté. Référence : 57.78 sur ces 500
images. Le tableau donne la perte de mAP en points ; une valeur négative est un gain, dans le
bruit du sous-ensemble.

| conv | L00 | L02 | L04 | L06 | L08 | L10 | L12 | L13 | L14 |
|---|---|---|---|---|---|---|---|---|---|
| uniform4 | +4.23 | +2.88 | +2.56 | +0.14 | +0.06 | -0.38 | +0.03 | +0.29 | +0.43 |
| uniform6 | +1.27 | -0.18 | +0.67 | +0.17 | -0.24 | -0.05 | +0.26 | +0.02 | +0.39 |
| mixed6 | +3.35 | +0.49 | -0.20 | -0.71 | -0.26 | +0.15 | -0.23 | +0.28 | +0.20 |

Lecture :

- Les trois premières convs (L00, L02, L04) sont les plus sensibles, de 2,6 à 4,2 points en
  4 bits. L00 perd aussi 1,3 point en 6 bits.
- La queue, L06 à L13, perd moins de 0,3 point en 4 bits. Elle porte pourtant 98 % des
  poids.
- `mixed6` ne se distingue pas de `uniform6`, sauf sur L00 (−3,4 points).

## Front glouton (500 images)

Les passages (couche, niveau uniforme) sont rangés par perte isolée / bits gagnés, puis
appliqués un à un ; chaque préfixe est évalué.

- *Octets moteur* : stockage de weights.bin pour le moteur unique (6 bits sur un octet).
- *Octets au plus juste* : stockage du streaming (6 bits = 0,75 octet).
- DSP et img/s : plan `stream_model` KV260 à poids par couche, deux MAC par DSP en 4 bits.

| étape | changement | mAP | perte | octets moteur | octets au plus juste | DSP streaming | img/s streaming |
|---|---|---|---|---|---|---|---|
| 0 | INT8 (référence) | 57.78 | +0.00 | 15.86 Mo | 15.86 Mo | 968 | 32.1 |
| 1 | L02 → uniform6 | 57.96 | -0.18 | 15.86 Mo | 15.86 Mo | 968 | 32.1 |
| 2 | L08 → uniform6 | 57.95 | -0.17 | 15.86 Mo | 15.78 Mo | 968 | 32.1 |
| 3 | L10 → uniform4 | 57.99 | -0.21 | 15.27 Mo | 15.19 Mo | 992 | 32.1 |
| 4 | L13 → uniform6 | 58.04 | -0.26 | 15.27 Mo | 12.83 Mo | 992 | 32.1 |
| 5 | L12 → uniform4 | 57.91 | -0.13 | 12.91 Mo | 10.47 Mo | 992 | 32.1 |
| 6 | L13 → uniform4 | 56.81 | +0.97 | 8.19 Mo | 8.11 Mo | 992 | 64.2 |
| 7 | L08 → uniform4 | 55.46 | +2.32 | 8.04 Mo | 8.04 Mo | 992 | 64.2 |
| 8 | L06 → uniform4 | 55.35 | +2.43 | 8.00 Mo | 8.00 Mo | 996 | 64.2 |
| 9 | L14 → uniform4 | 53.48 | +4.29 | 7.94 Mo | 7.94 Mo | 996 | 64.2 |
| 10 | L04 → uniform6 | 53.49 | +4.29 | 7.94 Mo | 7.94 Mo | 996 | 64.2 |
| 11 | L04 → uniform4 | 50.51 | +7.27 | 7.93 Mo | 7.93 Mo | 988 | 64.2 |
| 12 | L02 → uniform4 | 48.15 | +9.62 | 7.93 Mo | 7.93 Mo | 996 | 64.2 |
| - | tout uniform4 | 43.19 | +14.59 | 7.93 Mo | 7.93 Mo | 996 | 64.2 |

## Configuration retenue

Critère : la plus petite empreinte du moteur unique avec une perte d'au plus 1 point sur le
sous-ensemble. C'est l'étape 6 :

- L10, L12 et L13 en 4 bits paquetés ;
- L02 et L08 en 6 bits ;
- le reste en INT8.

| | mAP VOC2007 test (4 952 images) | poids | DSP streaming | img/s streaming |
|---|---|---|---|---|
| INT8 projeté (même grille d'échelles) | 55,69 | 15,86 Mo | 968 | 32,1 |
| **mixte retenue** | **54,68** (−1,01) | **8,19 Mo** (−48 %) | 992 | **64,2** |
| tout uniform4, PTQ (sous-ensemble) | 43,19 (−14,6 sur 500 images) | 7,93 Mo | 996 | 64,2 |
| w4a4 QAT 600 itérations (`quant_4bits.md`) | 36,38 | 7,9 Mo | 996 | 64,2 |

Lecture :

- **Streaming.** L13 fixe l'II du plan 8 bits. En 4 bits, elle fait deux MAC par DSP et
  l'II tombe à celui des autres étages. La configuration mixte obtient ainsi le débit du
  tout 4 bits (64,2 img/s) pour 1 point de mAP au lieu de 14,6, sans réentraînement.
- **Moteur unique.** Les poids en DDR sont divisés par deux, mais les cycles ne bougent
  pas (7 443 739, `perf_model --manifest`) : le noyau est limité par le calcul et les
  chargements de poids sont recouverts. Le gain porte sur la mémoire et la bande passante,
  pas sur la latence.
- **Contrat entier.** L'export paqueté (`wbits` = 4 dans le manifest) donne 0 écart, sur 3
  images, entre golden C++, C-sim (`tb_net_mixed`) et dumps Python.
- **Limite.** La perte atteint 1,01 point sur le test complet, juste au-dessus de la
  tolérance fixée sur le sous-ensemble (0,97). Un affinage QAT par couche la réduirait,
  mais le QAT actuel n'a qu'un schéma global (`lowbit.qat_from_steps`).
