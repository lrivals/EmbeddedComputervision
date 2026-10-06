# Quantification 4 bits — Tiny-YOLOv2 VOC (M9.3)

Schéma « w4a4 » (`python/yolo/quant/lowbit.py`) :

- **Poids** : toutes les convs sur 4 bits, symétriques par canal, entiers dans [−7, 7].
- **Activations** : sorties des convs sur 4 bits (`qmax` = 7 dans le manifest), avec un
  **pas puissance de 2** [2024-yan#006.0].
- **Exceptions** :
  - la tête reste sur 8 bits au pas 1/8 des LUT (§9.4) ;
  - l'entrée reste sur 8 bits (pixels).
- **Première couche** : elle est calculable par deux convolutions à entrées 4 bits,
  x = 16·(x ≫ 4) + (x & 15) [2024-yan#007.3]. L'égalité est vérifiée exhaustivement
  (`golden::mul_split4`, `int_layers.conv_acc_split4`).

VOC2007 test, 4 952 images, prétraitement stretch, seuil 0,005, mAP VOC07 sur 11 points.

## Résultats

| Modèle | mAP | Écart à INT8 |
|---|---|---|
| INT8 PTQ (référence, `results/map_int8.md`) | 55,66 | — |
| w4a4 PTQ (pas 2^j calibrés, minimum d'erreur quadratique, 500 images trainval) | 17,06 | −38,6 |
| **w4a4 QAT, 600 itérations** | **36,38** | −19,3 |

**Conditions du QAT** (`tools/train.py --qat w4a4`, `python/yolo/quant/fake_quant.py`) :

- réseau à BN fusionnée, initialisé depuis les poids Darknet, avec les pas de la PTQ ;
- SGD avec lr 1e-4 [2026-fata#015.3], montée en 50 itérations, lots de 8 à 416 ;
- VOC07+12 trainval, **600 itérations seulement** (4 800 images, ≈ 0,3 époque, ≈ 1 h sur
  CPU) : le budget de temps a été réduit volontairement.
- La perte moyenne passe de 16,0 (50 premières itérations) à 10,1 (50 dernières). Elle
  décroît encore : un QAT plus long devrait améliorer la mAP.

Le QAT récupère 19 points sur la PTQ (17,1 → 36,4), mais reste loin de l'INT8 : avec
0,3 époque, ce n'est qu'un début d'affinage. 2024-yan atteint le 4 bits sur YOLOv5s avec un
entraînement complet [2024-yan#006.0].

**Fidélité au matériel.** Le fake-quant suit l'ordre du modèle entier : arrondi au pas,
puis leaky entière (13r + 64) ≫ 7, puis écrêtage. Les gradients passent par l'estimateur
straight-through, et par LSQ pour le pas.

- Gradcheck de la fonction de substitution : `test_fake_quant.py`.
- Sur un réseau aléatoire, plus de 95 % des sorties sont égales à l'entier du modèle exporté.
- L'exposant du pas est appris sans le facteur 1/√(n·qmax) de LSQ : avec lui, il ne bougeait
  pas (|δk| ~ 1e-8 par itération).

**Chaîne entière vérifiée.** Le modèle 4 bits exporté
(`tools/quant_lowbit.py --scheme w4a4 [--checkpoint …]`) a un `qmax` par couche dans le
manifest. Le golden C++ et la C-sim HLS (`tb_net`) le reproduisent **à l'octet**
(0 écart sur 3 images).

## Gains matériels estimés (hors base, synthèse à faire)

| | INT8 | 4 bits |
|---|---|---|
| mémoire des poids | 15,9 Mo | 7,9 Mo (paquetage de 2 poids par octet, pas encore fait dans le moteur) |
| MAC par DSP | 1 | 2 (deux produits 4 bits par DSP48, comme le packing de 2024-zhang) |
| moteur unique Tm·Tn = 768 | 768 DSP | 384 DSP, ou Tm·Tn doublé à DSP constant |
| streaming KV260 (`results/streaming.md`) | 32,1 img/s | 64,2 img/s |

## Reste

- Un QAT plus long : plusieurs époques, ou une montée progressive 8 → 6 → 4 bits.
- Le paquetage 4 bits dans le moteur : conteneur int8 aujourd'hui, donc pas de gain de
  bande passante.
- Le packing de 2 MAC par DSP dans le HLS, puis la synthèse.

Figures : `results/figures/resultats/map_formats.png` (T13.14) et `entrainement_qat-w4a4.png` (T13.26), `python -m tools.figures map_formats entrainement`.
