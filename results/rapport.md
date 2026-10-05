# Rapport comparatif (T8.3)

Ce rapport situe l'accélérateur de ce dépôt (moteur unique couche par couche, Vitis HLS,
Kria KV260) par rapport aux travaux de la base (§10.4), et chiffre les optimisations qui
restent à faire.

**État des mesures (2026-10-05).** La KV260 et Vivado/Vitis ne sont pas encore disponibles.
Les chiffres de « ce travail » sont donc une **projection** : celle du modèle de cycles
`tools/perf_model.py`, égal cycle pour cycle aux compteurs C-sim du noyau. Cette projection
est une **borne basse du temps accélérateur** : elle suppose une boucle interne en II = 1 et
ignore les profondeurs de pipeline, la latence DDR et le pilotage ARM. Les mesures de la
carte remplaceront ces lignes en suivant [protocole.md](protocole.md) ; les sections qui en
dépendent sont marquées *à mesurer*.

## 1. Ce travail

| | Tiny-YOLOv2 VOC | Tiny-YOLOv3 COCO |
|---|---|---|
| MACs (convolutions) | 3,486 G | 2,782 G |
| Format | INT8 (poids par canal, activations par couche), accumulateur 32 bits | idem |
| Tuiles, MAC/cycle | Tm = 32, Tn = 24, Tr = Tc = 13 → 768 | idem |
| Horloge | 200 MHz (timing à confirmer, T7.1) | idem |
| Accélérateur, projection | **206,5 ms**, 4,8 img/s, **33,8 GOPS**, efficacité 11,0 % | 203,2 ms, 4,9 img/s, 27,4 GOPS, 8,9 % |
| Post-traitement ARM | < 0,02 ms sur x86 (T7.3) ; A53 *à mesurer* | idem |
| Puissance | *à mesurer* (puce : `power.rpt` ; SOM : INA260) | — |
| LUT / DSP / BRAM | *après synthèse* ; modèle roofline : 896 DSP, 112 BRAM18 | — |
| mAP VOC2007 test | flottant 56,30, entier 55,66, **FPGA (C-sim) 55,66** ([map_stades.md](map_stades.md)) | (COCO, non évalué) |

Le bit-exact est vérifié de bout en bout. Sur les **4 952 images** de VOC2007 test, le
driver ARM pilotant le noyau C-sim produit des détections **identiques** à celles du modèle
entier Python, et donc la même mAP (55,66). Le noyau était compilé en entiers natifs
(`-DACC_NO_APINT`, même arithmétique) ; les 1 832 premières images ont aussi été calculées
avec `ap_int`, avec le même résultat. La perte de précision vient donc tout entière de la
quantification (§9, T4.5). La mAP de la carte sera celle de l'entier si `run_compare` y
confirme 0 écart.

## 2. Comparaison avec la base

Source : [benchmarks.csv](benchmarks.csv), §10.4. GOPS = 2 × MACs / temps, sauf indication
contraire de l'article.

| Travail | Modèle | FPGA | Format | img/s | ms | GOPS | W (périmètre) | Ce qui empêche la comparaison directe |
|---|---|---|---|---|---|---|---|---|
| **ce travail** (projection) | Tiny-YOLOv2 VOC 416 | KV260 (XCK26), 200 MHz | INT8 | 4,8 | 206,5 (accél.) | 33,8 | *à mesurer* | projection C-sim, accélérateur seul, pas de mesure |
| **ce travail** (projection) | Tiny-YOLOv3 COCO 416 | KV260 | INT8 | 4,9 | 203,2 (accél.) | 27,4 | *à mesurer* | idem ; poids COCO (80 classes) |
| 2024-zhang | YOLOv2-Tiny 416, lot 1 | Kintex-7 325T | INT8 | n/r | 14,49 | 406 | 12,3 (carte) ; puce 7,8 (Vivado) | FPGA sans ARM, 687 DSP avec deux MAC par DSP (*packing*), post-traitement matériel compris ; jeu de données à vérifier ; fréquence non relevée |
| 2024-zhang | YOLOv3-Tiny 416, lot 1 | Kintex-7 325T | INT8 | n/r | 15,2 | 401 | 12,3 (carte) | idem |
| 2023-zhai | YOLOv3-tiny **élagué**, 1 module | Zynq XC7Z035 | INT16 | 91,65 | — | 67,91 | 12,51 (non précisé) | réseau élagué (moins de MACs : GOPS et img/s ne se convertissent pas), INT16, jeu de véhicules (détection et suivi), périmètre de puissance inconnu |
| 2023-zhai | YOLOv3-tiny élagué, 2 modules | Zynq XC7Z035 | INT16 | 168,72 | — | 124,01 | 15,18 (non précisé) | idem |
| 2026-fata | YOLOv3-tiny **élagué à 70 %** | KV260 (DPU) | INT8 QAT | 24,3 | — | 60,18 | 2,13 (non précisé) | même carte, mais DPU Vitis AI préconfiguré (pas conçu à la main), élagage à 70 %, QAT |
| 2025-kim | YOLOv2 **complet** | Zybo Z7-20 | INT16 | ≈ 0,08 | ≈ 12 000 | — | — | prototype fonctionnel lent, réseau complet (pas Tiny) : pas une référence de performance |

**Lecture.**

- **2024-zhang** est la seule référence au même modèle (Tiny-YOLOv2 416, INT8, lot 1). Elle
  est **14× plus rapide** que notre projection (14,49 contre 206,5 ms). Elle n'a pourtant
  que 1,8× plus de multiplieurs (687 DSP en *packing*, soit environ 1 374 MAC/cycle, contre
  768). L'essentiel de l'écart ne tient donc pas au calcul : notre noyau passe 80 % de son
  temps hors de la boucle MAC (41,3 Mcycles, dont 8,2 de calcul), à attendre ses ports
  mémoire et son étage de sortie (section 3).
- **2026-fata** tourne sur la **même carte** : 60 GOPS et 24 img/s avec un DPU sur un
  YOLOv3-tiny élagué à 70 %. Nos 27,4 GOPS projetés sur Tiny-YOLOv3 non élagué sont 2,2×
  en dessous. Les img/s ne se comparent pas : l'élagage divise les MACs par environ 3.
  Notre point roofline sur cette carte (31,9 ms pour Tiny-YOLOv2, environ 218 GOPS) est
  au-dessus du DPU : l'écart est à rattraper du côté de l'implémentation, pas du choix de
  la carte.
- **2023-zhai** : INT16 et réseau élagué. Seul l'ordre de grandeur des GOPS se compare
  (68 à 124 GOPS sur un Zynq-7035 contre nos 34 projetés).
- **Puissance** : seule 2024-zhang donne deux périmètres (puce estimée et carte mesurée). Nos
  deux périmètres (puce par `report_power`, SOM par l'INA260) se comparent à ces deux
  chiffres respectivement. Aucun ne se compare aux valeurs *unspec.* de 2023-zhai et
  2026-fata.
- **mAP** : la base ne relève pas de mAP VOC2007 pour ces lignes (colonne vide dans
  `benchmarks.csv`), et les jeux diffèrent (véhicules chez 2023-zhai). Notre chaîne mesure
  la perte de quantification seule : −0,64 point de l'entier par rapport au flottant, puis
  0 entre l'entier et le FPGA en C-sim (4 952 images identiques).

## 3. Où part le temps, et pistes chiffrées

Modèle : `python tools/perf_model.py` (`make perf-model`). Scénarios appliqués au noyau
actuel, sans changer les tuiles ; cycles à 200 MHz, accélérateur seul.

### Tiny-YOLOv2 VOC

| Scénario | Mcycles | ms | img/s | GOPS | efficacité MAC |
|---|---|---|---|---|---|
| actuel (ports 8 bits) | 41,30 | 206,5 | 4,8 | 33,8 | 11,0 % |
| canaux valides seulement (trim) | 32,67 | 163,3 | 6,1 | 42,7 | 13,9 % |
| ports 64 bits | 11,89 | 59,4 | 16,8 | 117,3 | 38,2 % |
| ports 128 bits + trim | 11,67 | 58,3 | 17,1 | 119,5 | 38,9 % |
| ports 128 bits + trim + requantification ×8 (+28 DSP) | 8,62 | 43,1 | 23,2 | 161,7 | 52,6 % |
| borne calcul (chargements gratuits) | 8,23 | 41,1 | 24,3 | 169,5 | 55,2 % |
| point roofline KV260 ([roofline.md](roofline.md)) | — | 31,9 | 31,3 | 218,5 | — |
| idéal (MACs / 768, efficacité 100 %) | 4,54 | 22,7 | 44,1 | 307,2 | 100 % |

Tiny-YOLOv3 suit la même échelle : 203,2 → 56,9 ms (ports 64 bits) → 39,2 ms (+ requantification
×8) ; borne calcul 36,5 ms.

**Pistes, par gain décroissant :**

1. **Ports m_axi larges (×3,5).** Les quatre ports lisent et écrivent un octet par cycle,
   soit 0,2 Go/s par bundle, alors que le roofline suppose 13,4 Go/s de DDR. Les charger
   en 64 bits (`in_buf` par lignes, `w_buf` dans l'ordre des tuiles, réordonné une fois par
   le driver) ramène Tiny-YOLOv2 de 206,5 à 59,4 ms. Passer à 128 bits n'apporte presque
   rien de plus (58,3 ms) : le goulot passe alors à l'étage de sortie.
2. **Requantification parallèle (−26 %, après la piste 1).** `store_tile` requantifie un
   canal à la fois avec un seul multiplieur 32 × 31 bits (4 DSP), soit Tm × 169 cycles par
   tuile. Avec 8 multiplieurs (+28 DSP, ressource abondante : 896 utilisés sur 1 248), on
   descend à 43,1 ms, à 5 % de la borne calcul.
3. **Canaux valides seulement (−21 % sur le noyau actuel, ≈ 0 après la piste 1).** L00 a
   cin = 3, mais le noyau charge Tn = 24 canaux : 87 % du chargement de L00 est inutile.
   La piste ne compte qu'aussi longtemps que les ports restent étroits.
4. **Borne calcul (41,1 ms) contre point roofline (31,9 ms).** Restent deux pertes de
   tuilage que le roofline ignore. L00 (cin = 3, cout = 16) n'occupe que 6 % des 768
   multiplieurs : 1,86 Mcycles, soit 23 % de la borne calcul, pour 2 % des MACs. Avec un
   pooling 2×2/2 et Tr = Tc = 13, chaque tuile ne produit que 12 × 12 sorties utiles sur
   13 × 13. Les remèdes sont un Tn réduit ou une PE dédiée pour L00 (comme la PE 3×3 de
   2025-kim) et Tr = Tc = 12 ou 14 pour les couches poolées.
5. **Côté ARM (sur la carte).** Recouvrir le prétraitement de l'image i + 1 avec
   l'accélérateur de l'image i (deux threads, deux arènes) ferait du débit le maximum des
   étages plutôt que leur somme. Le gain sera chiffré avec les temps `pre` de la carte. Si
   la copie et le post-traitement en mémoire non cachée (`O_SYNC`) pèsent, il faudra passer
   à un tampon caché avec `sync_for_cpu/device` (u-dma-buf).

Après les pistes 1 et 2, Tiny-YOLOv2 projeté atteint **≈ 23 img/s et ≈ 160 GOPS**, au niveau
de 2026-fata (DPU, même carte, réseau élagué) en débit de calcul, et sans élagage.

Note : `results/hls_report.md` donne 210,5 ms au lieu de 206,5 ms. Son `tb_conv` écrit la
carte avant pooling de toutes les convs poolées pour tester ce chemin. Le programme du
driver ne l'écrit que pour L08 de Tiny-YOLOv3 : c'est ce cas que suit la projection.

## 4. Ce qui reste à mesurer

| Chiffre | Commande | Dépend de |
|---|---|---|
| Timing tenu à 200 MHz, LUT/DSP/BRAM, puissance puce | `make vivado-build` | Vitis HLS + Vivado |
| Cycles réels par couche (II, pipeline) | `make hls-cosim`, puis `run_compare --csv` sur carte | Vitis, KV260 |
| Latence par étage (moyenne, p99), FPS, puissance SOM | `yolo_bench --images` (1 000 images), [protocole.md](protocole.md) | KV260 |
| mAP sur FPGA | `yolo_bench --inputs` puis `make map-stades` | KV260 |

Une fois ces fichiers rapatriés, `tools/bench_report.py` régénère `benchmarks.csv` et
`mesures.md`. Il restera à remplacer la ligne « projection » des tableaux ci-dessus par la
mesure et à confronter l'efficacité par couche au modèle de cycles.
