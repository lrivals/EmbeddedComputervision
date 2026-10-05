# Accélérateur HLS — rapport (T6.5)

Généré par `python tools/hls_report.py --board kv260` (`make hls-report`). Carte `kv260` : part `xck26-sfvc784-2LV-c`, horloge 200 MHz, tuiles Tm = 32, Tn = 24, Tr = 13, Tc = 13 ([ADR 0003](../docs/adr/0003-choix-carte.md)).

## Synthèse

À produire : `make hls-synth` (Vitis HLS non installé sur la machine de développement).

## Co-simulation RTL

À produire : `make hls-cosim` (une image complète de Tiny-YOLOv2, chaque couche comparée au golden).

## Cycles par couche (estimation C-sim)

Compteurs du noyau en C-sim (`accel::sim_cycles`, II = 1, profondeurs de pipeline ignorées). *Calcul* = K²·Tr·Tc par (tuile, ti) : la borne du toit de calcul ; *ping-pong* = chargement de ti + 1 recouvert par le calcul de ti, stockage de la tuile k − 1 recouvert par la tuile k ; *séquentiel* = sans recouvrement. Les chargements lisent un octet par cycle (ports m_axi de 8 bits).

### tiny-yolov2-voc

| Couche | K | Cin → Cout | Entrée | MMAC | calcul | ping-pong | séquentiel | ms (ping-pong) | efficacité MAC |
|---|---|---|---|---|---|---|---|---|---|
| L00 | 3 | 3 → 16 | 416×416 | 74.8 | 1,863,225 | 10,334,409 | 23,718,945 | 51.67 | 0.9% |
| L02 | 3 | 16 → 32 | 208×208 | 199.4 | 492,804 | 3,528,755 | 7,964,644 | 17.64 | 7.4% |
| L04 | 3 | 32 → 64 | 104×104 | 199.4 | 492,804 | 2,493,858 | 6,223,268 | 12.47 | 10.4% |
| L06 | 3 | 64 → 128 | 52×52 | 199.4 | 456,300 | 2,231,748 | 5,123,340 | 11.16 | 11.6% |
| L08 | 3 | 128 → 256 | 26×26 | 199.4 | 657,072 | 3,101,064 | 6,581,552 | 15.51 | 8.4% |
| L10 | 3 | 256 → 512 | 13×13 | 199.4 | 1,070,784 | 4,968,864 | 10,257,600 | 24.84 | 5.2% |
| L12 | 3 | 512 → 1024 | 13×13 | 797.4 | 1,070,784 | 4,925,536 | 10,084,544 | 24.63 | 21.1% |
| L13 | 3 | 1024 → 1024 | 13×13 | 1594.9 | 2,092,896 | 9,570,400 | 19,380,320 | 47.85 | 21.7% |
| L14 | 1 | 1024 → 125 | 13×13 | 21.6 | 29,068 | 939,278 | 1,132,214 | 4.70 | 3.0% |
| **total** | | | | 3486 | 8,225,737 | 42,093,912 | 90,466,427 | **210.5** | 10.8% |

Borne calcul seule : 41.1 ms ; le reste est le temps de chargement non recouvert.

### tiny-yolov3-coco

| Couche | K | Cin → Cout | Entrée | MMAC | calcul | ping-pong | séquentiel | ms (ping-pong) | efficacité MAC |
|---|---|---|---|---|---|---|---|---|---|
| L00 | 3 | 3 → 16 | 416×416 | 74.8 | 1,863,225 | 10,334,409 | 23,718,945 | 51.67 | 0.9% |
| L02 | 3 | 16 → 32 | 208×208 | 199.4 | 492,804 | 3,528,755 | 7,964,644 | 17.64 | 7.4% |
| L04 | 3 | 32 → 64 | 104×104 | 199.4 | 492,804 | 2,493,858 | 6,223,268 | 12.47 | 10.4% |
| L06 | 3 | 64 → 128 | 52×52 | 199.4 | 456,300 | 2,231,748 | 5,123,340 | 11.16 | 11.6% |
| L08 | 3 | 128 → 256 | 26×26 | 199.4 | 657,072 | 3,101,064 | 6,581,552 | 15.51 | 8.4% |
| L10 | 3 | 256 → 512 | 13×13 | 199.4 | 1,070,784 | 4,968,864 | 10,257,600 | 24.84 | 5.2% |
| L12 | 3 | 512 → 1024 | 13×13 | 797.4 | 1,070,784 | 4,925,536 | 10,084,544 | 24.63 | 21.1% |
| L13 | 1 | 1024 → 256 | 13×13 | 44.3 | 58,136 | 1,869,768 | 2,266,456 | 9.35 | 3.1% |
| L14 | 3 | 256 → 512 | 13×13 | 199.4 | 267,696 | 1,251,664 | 2,607,664 | 6.26 | 20.7% |
| L15 | 1 | 512 → 255 | 13×13 | 22.1 | 29,744 | 962,230 | 1,201,502 | 4.81 | 3.0% |
| L18 | 1 | 256 → 128 | 13×13 | 5.5 | 7,436 | 249,092 | 322,092 | 1.25 | 2.9% |
| L21 | 3 | 384 → 256 | 26×26 | 598.1 | 778,752 | 3,598,432 | 7,428,608 | 17.99 | 21.6% |
| L22 | 1 | 256 → 255 | 26×26 | 44.1 | 59,488 | 1,916,686 | 2,575,384 | 9.58 | 3.0% |
| **total** | | | | 2782 | 7,305,025 | 41,432,106 | 86,355,599 | **207.2** | 8.7% |

Borne calcul seule : 36.5 ms ; le reste est le temps de chargement non recouvert.

## Lecture

- Le noyau est **limité par les chargements**, pas par le calcul : avec des ports de 8 bits, charger `in_buf` (Tn·IR·IC octets) et `w_buf` (Tm·Tn·K² octets) prend plus de cycles que les K²·Tr·Tc cycles de calcul d'une ti, même en ping-pong.
- Pistes (M8/M9) : ports m_axi larges (64-128 bits) avec poids réordonnés par le driver dans l'ordre des tuiles ; réutilisation de `in_buf` entre tuiles `to` (boucle to la plus interne, entrée inchangée) ; Tn adapté à la couche 0 (cin = 3).
