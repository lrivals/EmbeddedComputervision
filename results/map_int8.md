# mAP du modèle entier — Tiny-YOLOv2 VOC (T4.5)

Poids Darknet `yolov2-tiny-voc.weights`, VOC2007 test (4 952 images), 416×416 (PIL),
seuil 0,005, NMS 0,45, AP 11 points (`tools/eval_voc.py`, T3.3). Modèle entier : BN fusionnée
(T4.1), poids int8 par canal, activations int8 calibrées sur 500 images VOC2007 trainval
(p99,99, critère MSE, `results/calibration_tiny-yolov2-voc.md`), têtes au pas 1/8,
requantification M0/décalage, leaky 13/128 arrondie, décodage par LUT (T4.3, T4.4),
moteur `IntNetwork` bit-exact (`tools/eval_quant.py`, `make eval-int`).

## Résultat

| Prétraitement | Flottant (BN fusionnée) | **Entier INT8** | Écart |
|---|---|---|---|
| stretch (référence T3.4, `make eval-float`) | 56,30 | **55,66** | **−0,64 point** |
| letterbox (celui de l'entraînement) | 54,18 | **53,68** | **−0,50 point** |

Les mAP flottantes sont exactement celles de `results/map_float.md` (la fusion BN ne change
rien). La calibration est faite en letterbox ; elle tient aussi en stretch. L'écart est sous
la cible de 1 point (conventions.md) dans les deux cas : **le QAT (T4.6) n'est pas
nécessaire**. `make eval-int` mesure le cas stretch.

## Par classe (letterbox)

| Classe | Flottant | Entier | Écart |
|---|---|---|---|
| aeroplane | 57.9 | 57.9 | -0.0 |
| bicycle | 67.8 | 66.7 | -1.1 |
| bird | 47.2 | 46.3 | -0.9 |
| boat | 36.5 | 37.7 | +1.2 |
| bottle | 21.2 | 21.7 | +0.5 |
| bus | 67.7 | 66.6 | -1.1 |
| car | 63.0 | 62.6 | -0.4 |
| cat | 66.4 | 65.9 | -0.5 |
| chair | 31.3 | 31.5 | +0.2 |
| cow | 53.5 | 51.3 | -2.3 |
| diningtable | 55.4 | 54.5 | -0.9 |
| dog | 62.0 | 61.0 | -1.0 |
| horse | 70.1 | 69.7 | -0.4 |
| motorbike | 69.8 | 68.8 | -1.0 |
| person | 58.3 | 57.7 | -0.6 |
| pottedplant | 26.8 | 28.1 | +1.4 |
| sheep | 50.7 | 50.1 | -0.6 |
| sofa | 51.8 | 51.6 | -0.1 |
| train | 68.1 | 67.5 | -0.6 |
| tvmonitor | 58.1 | 56.3 | -1.8 |

## Sensibilité par couche

Une seule convolution quantifiée à la fois (poids fake-quantifiés par canal, leaky 13/128,
sortie fake-quantifiée à s_y), les autres en flottant ; 1 000 premières images de VOC2007
test (simulation flottante `fake_quant_forward`, plus rapide que le modèle entier).

| Couche quantifiée | mAP | Écart au flottant |
|---|---|---|
| aucune (flottant) | 56.19 | — |
| L00 | 56.12 | -0.07 |
| L02 | 56.21 | +0.02 |
| L04 | 55.94 | -0.25 |
| L06 | 56.21 | +0.02 |
| L08 | 55.84 | -0.34 |
| **L10** | 55.58 | -0.61 |
| L12 | 56.13 | -0.06 |
| L13 | 56.06 | -0.13 |
| L14 | 56.38 | +0.19 |
| toutes (simulation) | 55.64 | -0.55 |

Couche la plus sensible : **L10** (−0,61 point seule), puis L08 et L04. Les écarts par
couche (≤ 0,6 point) sont du même ordre que le bruit d'un sous-ensemble de 1 000 images
(L14 seule *gagne* 0,19) : aucune couche ne domine la perte, qui se répartit sur le réseau.

## Échelle des têtes

500 premières images de VOC2007 test, modèle entier (`IntNetwork`) ; seule l'échelle de la
tête (L14) change.

| Échelle de tête | Plage de t | Sorties écrêtées | mAP entière |
|---|---|---|---|
| 1/16 (pas de la LUT du §9.4) | ±7,94 | 2,5 % | 49.66 |
| **1/8 (retenue)** | ±15,9 | 0,03 % | **57.26** |
| calibrée p99,99 (0,149) | ±18,9 | 0,01 % | 56.59 |
| flottant | — | — | 57.24 |

Au pas 1/16, les logits de classe de Tiny-YOLOv2 (|t| jusqu'à ~19) saturent ; plusieurs
classes atteignent le même maximum et le softmax s'aplatit : les scores plafonnent vers
0,45 et la mAP perd ~7,5 points. Au pas 1/8 la table de 256 entrées couvre ±15,9 ; l'erreur
de σ due au pas d'entrée passe de 1/128 à 1/64, sans effet mesurable sur la mAP. La table
exponentielle est alors en Q12 (`exp_frac`) pour tenir sur 32 bits.

## Choix d'arithmétique qui comptent

- **Leaky arrondie.** `(13·y + 64) >> 7` plutôt que `(13·y) >> 7` (§9.3) : le plancher
  biaise chaque valeur négative de −½ pas et le biais se cumule de couche en couche
  (écart moyen aux têtes ×1,7 sur une image test).
- **Têtes au pas 1/8** (ci-dessus).
- Activations à p99,99 de |x| : p90-p99 écrêtent trop (MSE ×10 à ×1000), le max gaspille la
  plage (MSE ×2 à ×5).
