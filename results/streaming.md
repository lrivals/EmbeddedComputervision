# Architecture streaming face au moteur unique — Tiny-YOLOv2 VOC, KV260 (M9.4)

Seconde famille d'architecture du §10.1 :

- un étage matériel par conv, en pipeline [2018-venieris#004.0] ;
- fenêtre glissante par line buffer [2023-montgomerie-corcoran#011.0] ;
- maxpool fusionné, y compris le 2×2/1 de L11 (réplication du bord, §10.3).

Fichiers :

- modèle de ressources : `tools/stream_model.py` ;
- noyau : `hls/stream/` (`conv_stage.hpp`, `yolo_stream.cpp`) ;
- testbench : `hls/tb/tb_stream.cpp`.

**Toutes les ressources ci-dessous sont des estimations** : Vitis est absent, donc pas de
synthèse.

## Vérification (C-sim g++)

- La sortie de chaque étage et la tête sont **égales à l'octet aux dumps** (== golden ==
  modèle entier) sur les 3 images (`tb_stream`, ctest).
- Les cycles C-sim par étage (itérations PE × SIMD, II = 1) sont **égaux** à ceux du modèle
  (`python/tests/test_stream_model.py`). Les repliements codés dans `yolo_stream.hpp` sont
  ceux du plan.

## Plan pour la KV260, poids et activations 8 bits

Budgets : 80 % des 1 248 DSP, 80 % de BRAM + URAM (2,3 Mo). Une MAC int8 par DSP, plus
4 DSP par PE pour la requantification. Horloge 200 MHz.

| conv | K | C_in → C_out | H×W | PE × SIMD | DSP | cycles | poids | tampon (Ko) |
|---|---|---|---|---|---|---|---|---|
| L00 | 3 | 3 → 16 | 416×416 | 8 × 3 | 56 | 3 115 008 | puce (0 Ko) | 8.9 |
| L02 | 3 | 16 → 32 | 208×208 | 4 × 16 | 80 | 3 115 008 | puce (4 Ko) | 13.0 |
| L04 | 3 | 32 → 64 | 104×104 | 2 × 32 | 72 | 3 115 008 | puce (18 Ko) | 13.0 |
| L06 | 3 | 64 → 128 | 52×52 | 1 × 64 | 68 | 3 115 008 | puce (72 Ko) | 13.0 |
| L08 | 3 | 128 → 256 | 26×26 | 1 × 64 | 68 | 3 115 008 | puce (288 Ko) | 13.0 |
| L10 | 3 | 256 → 512 | 13×13 | 1 × 64 | 68 | 3 115 008 | puce (1 152 Ko) | 13.0 |
| L12 | 3 | 512 → 1024 | 13×13 | 1 × 256 | 260 | 3 115 008 | ddr (4 608 Ko) | 84.5 |
| L13 | 3 | 1024 → 1024 | 13×13 | 1 × 256 | 260 | 6 230 016 | ddr (9 216 Ko) | 169.0 |
| L14 | 1 | 1024 → 125 | 13×13 | 1 × 32 | 36 | 676 000 | puce (125 Ko) | 0.0 |

- II = 6 230 016 cycles → **32.1 img/s** à 200 MHz ; latence ≈ 85.2 ms
- DSP 968 / 998 (80 %) ; mémoire sur puce 1.94 / 2.31 Mo ; DDR 13.50 Mo de poids par image
- 223.8 GOPS, efficacité MAC 63.0 %
- moteur unique (perf model, ports 8 bits) : 41 299 241 cycles → 4.8 img/s ; streaming × 6.6 en débit


**Les poids ne tiennent pas sur la puce.** Tiny-YOLOv2 a 15,9 Mo de poids int8 pour
2,3 Mo de mémoire utilisable. L12 (4,5 Mo) et L13 (9 Mo) en contiennent 86 %. SATAY garde
tous ses paramètres sur la puce [2023-montgomerie-corcoran#006.0], mais sur un VCU110, bien
plus grand. Sur la KV260, il faut donc un **hybride** :

- les 7 étages à poids sur la puce fonctionnent en flux pur ;
- L12 et L13 gardent leur carte d'entrée (13×13×C, 85 et 169 Ko) et lisent leurs poids une
  fois par image en DDR, soit 13,5 Mo par image ;
- à 32 img/s, cela fait 0,43 Go/s, très loin des 13,4 Go/s utiles.

Le prix est la latence : L12 et L13 attendent chacune une carte entière.

## Avec les poids et activations sur 4 bits (M9.3)

Hypothèse : 2 MAC 4 bits par DSP.

| conv | K | C_in → C_out | H×W | PE × SIMD | DSP | cycles | poids | tampon (Ko) |
|---|---|---|---|---|---|---|---|---|
| L00 | 3 | 3 → 16 | 416×416 | 16 × 3 | 88 | 1 557 504 | puce (0 Ko) | 4.5 |
| L02 | 3 | 16 → 32 | 208×208 | 16 × 16 | 192 | 778 752 | puce (2 Ko) | 6.5 |
| L04 | 3 | 32 → 64 | 104×104 | 4 × 32 | 80 | 1 557 504 | puce (9 Ko) | 6.5 |
| L06 | 3 | 64 → 128 | 52×52 | 2 × 64 | 72 | 1 557 504 | puce (36 Ko) | 6.5 |
| L08 | 3 | 128 → 256 | 26×26 | 1 × 128 | 68 | 1 557 504 | puce (144 Ko) | 6.5 |
| L10 | 3 | 256 → 512 | 13×13 | 1 × 128 | 68 | 1 557 504 | puce (576 Ko) | 6.5 |
| L12 | 3 | 512 → 1024 | 13×13 | 1 × 256 | 132 | 3 115 008 | ddr (2 304 Ko) | 42.2 |
| L13 | 3 | 1024 → 1024 | 13×13 | 1 × 512 | 260 | 3 115 008 | ddr (4 608 Ko) | 84.5 |
| L14 | 1 | 1024 → 125 | 13×13 | 1 × 64 | 36 | 338 000 | puce (62 Ko) | 0.0 |

- II = 3 115 008 cycles → **64.2 img/s** à 200 MHz ; latence ≈ 50.3 ms
- DSP 996 / 998 (80 %) ; mémoire sur puce 0.97 / 2.31 Mo ; DDR 6.75 Mo de poids par image
- 447.6 GOPS, efficacité MAC 67.9 %
- moteur unique (perf model, ports 8 bits) : 41 299 241 cycles → 4.8 img/s ; streaming × 13.3 en débit


## Comparaison

| | Moteur unique (M6-M8) | Streaming 8 bits | Streaming 4 bits |
|---|---|---|---|
| cycles par image | 41,3 M (noyau M6, ports 8 bits) ; 7,44 M (noyau M10) | II = 6,23 M | II = 3,12 M |
| débit | 4,8 img/s ; 26,9 img/s | **32,1 img/s** | **64,2 img/s** |
| latence | 206 ms ; 43 ms | ≈ 85 ms | ≈ 50 ms |
| DSP | 896 (Tm·Tn = 768 + 4·Tm requantification) | 968 | 996 |
| mémoire sur puce | tampons ping-pong (roofline : 112 BRAM18 ≈ 0,25 Mo) | 1,9 Mo (L12-L13 : cartes entières) | 1,0 Mo |
| poids lus en DDR par image | 15,9 Mo, plus les activations à chaque couche | 13,5 Mo (L12, L13) | 6,75 Mo |
| efficacité MAC | 11 % ; 53 % | 63 % | 68 % |
| flexibilité | un bitstream pour v2 et v3 | un bitstream par réseau (dimensions en paramètres de gabarit) | idem |

Lecture :

- **Débit.** À DSP comparables, le streaming fait **6,6 fois** mieux que le moteur actuel et
  **1,4 fois** mieux que sa meilleure piste chiffrée. Toutes les couches calculent en même
  temps, et le trafic DDR des activations disparaît.
- **Ce qui limite le streaming.** Le repliement se fait par diviseurs de C_in et C_out.
  L13 (1024 → 1024) passe de 256 à 512 MAC par cycle d'un seul coup, et ce saut ne tient pas
  dans le budget. L13 fixe donc l'II à 2 fois celui des autres étages : l'efficacité est de
  63 %, pas de 100 %.
- **Latence.** Elle reste inférieure au moteur actuel, mais le moteur optimisé (43 ms) fait
  mieux que le streaming 8 bits (85 ms), à cause des deux étages à carte entière.
- **4 bits.** Ils divisent la mémoire des poids par 2 et doublent les MAC par DSP, ce qui
  double le débit, sous réserve de la mAP du QAT 4 bits (`results/quant_4bits.md`).

## Reste

- Synthèse de `yolo_stream` : DSP, LUT, BRAM/URAM réels et fréquence atteinte (les
  boucles PE × SIMD déroulées de 256 sont un risque de timing).
- Poids sur la puce initialisés comme ROM (aujourd'hui des pointeurs m_axi en C-sim) ;
  ordre « PE extérieur » des étages L12-L13 en matériel.
- Co-simulation RTL du DATAFLOW (profondeurs des FIFO).
- Intégration (AXI-Stream ↔ DMA) et mesure sur carte.
