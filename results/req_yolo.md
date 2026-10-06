# REQ-YOLO : poids sur des puissances de 2 — Tiny-YOLOv2 VOC (M9.2)

Reproduction de la quantification de 2019-ding [2019-ding#008.1, #009.1, #009.4]. Les
activations restent en **INT8 calibré**, comme le modèle de référence : seul l'effet des
poids est mesuré.

**Niveaux** (`python/yolo/quant/pow2.py`) :

- `mixed6` : poids de 6 bits, |q| ∈ {0, 2^a, 2^a + 2^{a−k}}, avec a ≤ 6 (3 bits primaires)
  et k ∈ {1, 2, 3} (2 bits secondaires). Cela fait 45 niveaux signés, |q| ≤ 96. Le produit
  q·x vaut ±((x ≪ a) + (x ≪ (a − k))) : **deux décalages et une addition**.
- `uniform6` : niveaux équidistants ±31.

Dans les deux cas, α est choisi par canal de sortie : erreur quadratique minimale sur
32 échelles.

## Résultats (VOC2007 test, 4 952 images, stretch)

| Poids | Méthode | mAP | Écart à INT8 |
|---|---|---|---|
| INT8 par canal (référence) | PTQ | 55,66 | — |
| uniform6 (±31) | PTQ | 54,35 | −1,31 |
| **mixed6 (puissances de 2)** | PTQ, projection directe | 52,46 | −3,20 |
| mixed6 | ADMM, 600 itérations, puis projection | 43,07 | −12,59 |

**ADMM** (`python/yolo/train/admm.py`, `tools/train.py --admm`) :

- poids fusionnés, mixed6 sur toutes les convs ;
- ρ = 1e-3 au départ, × 1,3 à chaque pas Z/U (toutes les 50 itérations) ;
- lr 1e-4, lots de 8, 600 itérations.

**L'ADMM n'a pas convergé.** Le résidu ‖W − Z‖/‖W‖ par couche (`build/train/admm-mixed6/admm.csv`)
devrait tendre vers 0. Il passe au contraire de 0,05-0,06 (itération 50) à 0,07-0,09
(itération 600). ρ final : 0,023, trop faible face au gradient de la perte de détection, et
les poids ne sont pas attirés vers les niveaux.

**Le résultat est pire que la projection directe** : 43,07 contre 52,46. La perte de
détection est elle-même montée pendant l'affinage, de 7,3 à 8,1 en moyenne sur 50
itérations : les 600 itérations ont dégradé le réseau au lieu de le rapprocher des niveaux.

Causes probables, **non vérifiées** (la mAP flottante du checkpoint n'a pas été mesurée) :

- affinage du réseau **à BN fusionnée**, sans BN, avec les augmentations du loader, depuis
  des poids Darknet déjà convergés ;
- lots de 8, trop petits pour ce réseau sans BN ;
- ρ trop faible pour compenser.

L'algorithme lui-même est validé sur un problème quadratique (`test_admm.py` : résidu
< 1e-3). En l'état, la meilleure option puissances de 2 reste la PTQ à 52,46.

Corrections prévues :

- mesurer d'abord la mAP flottante d'un affinage sans ADMM, pour isoler l'effet de
  l'affinage ;

- ρ de départ 10 à 100 fois plus grand (0,05 → 1) ;
- davantage d'itérations ;
- choix du schéma par couche d'après la sensibilité de chaque couche (étude préparée,
  annulée faute de temps).

## Matériel (T9.2.4)

**Bit-exactitude.** Les poids mixed6 tiennent dans l'int8 du moteur. Le golden et le noyau
existants restent exacts sans modification : `tb_net`, 0 écart sur 3 images.

**Variante `ACC_WMODE_POW2`** (`hls/kernels/accel.hpp`, `mul_w`) : la PE remplace chaque
multiplication par deux décalages et une addition.

- C-sim (`tb_net_pow2`, ctest) : **0 écart** avec le golden sur les 3 images du modèle
  mixed6.
- Économie de DSP **estimée** sur le moteur unique KV260 (Tm·Tn = 768) :

| | INT8 | mixed6 (décalages) |
|---|---|---|
| DSP des MAC | 768 | **0** |
| DSP de requantification | 128 (4·Tm) | 128 |
| LUT par MAC (ordre de grandeur, hors base) | ≈ 0 (DSP) | ≈ 40 (2 décaleurs 7:1 sur 8 bits + additionneur 15 bits) |
| LUT des MAC | — | ≈ 31 000, soit 26 % des 117 120 LUT du XCK26 |

Les 768 DSP libérés peuvent servir à doubler Tm·Tn, ou être laissés libres sur une plus
petite puce. 2019-ding (Virtex-7, FFT + puissances de 2) annonce 314 images/s
[2019-ding, Tables 1-2], mais sur une autre architecture.

## Reste

- ADMM convergé (ρ plus grand, plus d'itérations, plan par couche), sans dégrader le
  réseau flottant.
- Synthèse des deux variantes de PE : DSP, LUT et timing mesurés.
- Codes 6 bits stockés en mémoire au lieu de l'int8 (gain de 25 % sur les poids).

Figures : `results/figures/resultats/map_formats.png` (T13.14) et `entrainement_admm-mixed6.png` (T13.26, résidus ADMM), `python -m tools.figures map_formats entrainement`.
