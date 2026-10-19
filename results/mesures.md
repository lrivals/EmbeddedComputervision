# Mesures de ce travail (T8.1)

Généré par `python tools/bench_report.py` (`make bench-report`) ; protocole : [protocole.md](protocole.md). Kria KV260, tuiles Tm = 32, Tn = 24, Tr = Tc = 13 (14 pour les convs poolées en stride 2), 200 MHz.

## Projection (modèle de cycles, accélérateur seul)

`tools/perf_model.py`, noyau actuel (ports 64 bits, trim, requant ×8, pliage L00, tuiles 14 poolées), II = 1, profondeurs de pipeline et pilotage ARM ignorés : borne basse du temps accélérateur mesurable.

| Réseau | GMAC | Mcycles | ms | img/s | GOPS | efficacité |
|---|---|---|---|---|---|---|
| tiny-yolov2-voc | 3.486 | 7.44 | 37.2 | 26.87 | 187.3 | 61.0 % |
| tiny-yolov3-coco | 2.782 | 7.21 | 36.1 | 27.72 | 154.3 | 50.2 % |

## Projection, deux MAC par DSP (T23.3)

PE paquetée (`ACC_PACK=2`, results/dsp_packing.md), mêmes hypothèses ; DSP estimés (PE + requantification), à confirmer en synthèse.

| Réseau | Option | Tm, Tn | DSP (modèle) | ms | img/s | GOPS |
|---|---|---|---|---|---|---|
| tiny-yolov2-voc | packed, same tiles (half the DSPs) | 32, 24 | 416 | 37.2 | 26.87 | 187.3 |
| tiny-yolov2-voc | packed, Tm 64 (same DSPs) | 64, 24 | 800 | 24.7 | 40.53 | 282.5 |
| tiny-yolov2-voc | packed, Tn 32 (roofline point) | 32, 32 | 544 | 30.0 | 33.31 | 232.2 |
| tiny-yolov3-coco | packed, same tiles (half the DSPs) | 32, 24 | 416 | 36.1 | 27.72 | 154.3 |
| tiny-yolov3-coco | packed, Tm 64 (same DSPs) | 64, 24 | 800 | 23.8 | 41.95 | 233.4 |
| tiny-yolov3-coco | packed, Tn 32 (roofline point) | 32, 32 | 544 | 30.2 | 33.06 | 184.0 |

## Mesures sur la carte

À mesurer : aucun fichier `build/m8/tiny-yolov2-voc/board/times_*.csv` (KV260 non disponible). Procédure : [protocole.md](protocole.md).
