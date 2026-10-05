# Mesures de ce travail (T8.1)

Généré par `python tools/bench_report.py` (`make bench-report`) ; protocole : [protocole.md](protocole.md). Kria KV260, tuiles Tm = 32, Tn = 24, Tr = Tc = 13 (14 pour les convs poolées en stride 2), 200 MHz.

## Projection (modèle de cycles, accélérateur seul)

`tools/perf_model.py`, noyau actuel (ports 64 bits, trim, requant ×8, pliage L00, tuiles 14 poolées), II = 1, profondeurs de pipeline et pilotage ARM ignorés : borne basse du temps accélérateur mesurable.

| Réseau | GMAC | Mcycles | ms | img/s | GOPS | efficacité |
|---|---|---|---|---|---|---|
| tiny-yolov2-voc | 3.486 | 7.44 | 37.2 | 26.87 | 187.3 | 61.0 % |
| tiny-yolov3-coco | 2.782 | 7.21 | 36.1 | 27.72 | 154.3 | 50.2 % |

## Mesures sur la carte

À mesurer : aucun fichier `build/m8/tiny-yolov2-voc/board/times_*.csv` (KV260 non disponible). Procédure : [protocole.md](protocole.md).
