# Élagage de canaux (T10.11)

Tiny-YOLOv2 VOC, poids Darknet. Élagage structuré par norme L1 du filtre après fusion de la
BN (`python/yolo/prune.py`, `tools/prune.py`). Les filtres gardés sont arrondis au multiple
de Tm = 32 le plus proche ; L00, L02 et la tête ne sont pas élaguées. Chaque variante suit
la même chaîne (`tools/prune_study.sh`, palier M de M12) :

1. affinage flottant de 300 itérations, lots de 8, lr 5·10⁻⁴ ;
2. calibration INT8 sur 500 images ;
3. mAP entière sur 500 images de VOC2007 test (`--resize stretch`) ;
4. export, puis cycles du noyau actuel (`perf_model --manifest`, 200 MHz).

Référence : INT8 non élagué, **57,78** sur les mêmes 500 images ; 7 443 739 cycles,
37,2 ms, 26,9 img/s.

| variante | filtres gardés (L04, L06, L08, L10, L12, L13) | poids | mAP sans affinage | mAP après affinage | cycles | ms | img/s | GOPS effectifs | efficacité MAC |
|---|---|---|---|---|---|---|---|---|---|
| r30 (toutes) | 32, 96, 192, 352, 704, 704 | 7,59 Mo | 2,48 | 11,89 | 4 483 801 | 22,4 | 44,6 | 161 | 52,5 % |
| r50 (toutes) | 32, 64, 128, 256, 512, 512 | 4,00 Mo | 1,04 | 8,02 | 3 308 491 | 16,5 | 60,5 | 137 | 44,6 % |
| r70 (toutes) | 32, 32, 64, 160, 320, 320 | 1,56 Mo | 0,56 | 1,61 | 2 423 017 | 12,1 | 82,5 | 110 | 35,9 % |

Variante « queue » (50 % sur L10, L12 et L13 seulement, 4,59 M paramètres) : élaguée
(`RATES="50:10,12,13" bash tools/prune_study.sh`), puis arrêtée au début de l'affinage ;
elle n'est pas mesurée.

Lecture :

- **Cycles.** Ils baissent moins vite que les MACs : à 70 %, il reste 2,4 M cycles pour
  0,67 GMAC par image (3,49 non élagué), soit 36 % d'efficacité contre 61 % non élagué. Avec moins de canaux,
  les tuiles Tm = 32 et Tn = 24 sont moins remplies : L04 garde 32 filtres et nourrit L06
  avec 32 canaux, soit 2 tuiles Tn dont la seconde n'est remplie qu'au tiers. Les chargements et
  les couches à 13 × 13 pèsent davantage.
- **mAP.** Un élagage uniforme détruit le réseau (moins de 3 points sans affinage, même à
  30 %), et 300 itérations (2 400 images, soit 0,15 époque de VOC 07 + 12) n'en récupèrent
  qu'une petite partie. Au palier M, aucune variante n'est utilisable : il faut un
  affinage de plusieurs époques (palier N), ou un élagage progressif.
- **Comparaison.** 2026-fata (élagué à 70 %, QAT, DPU de la KV260) annonce 24,3 img/s. Ici,
  le noyau fait 82,5 img/s à 70 % sur la même carte (projection C-sim, sans le
  post-traitement ni les transferts), mais avec une mAP inutilisable. La comparaison ne
  vaudra qu'à mAP comparable, après un affinage long. 2023-zhai (INT16, véhicules) reste
  hors de portée d'une comparaison directe.
- **Contrat entier.** Sur l'export r30, golden C++, C-sim (`tb_net --model
  build/m10/prune/r30 --net model`) et dumps Python donnent 0 écart sur 3 images. Les
  cycles C-sim sont égaux au modèle (4 483 801).

## Reste

- Variante « queue » à mesurer, puis affinage long (palier N, plusieurs époques) et mAP
  sur les 4 952 images.
- Élagage guidé par la sensibilité par couche, ou progressif (petits pas et affinage entre
  deux).
- Poids propres affinés sur VOC (T10.12, renvoi vers T2.9).
