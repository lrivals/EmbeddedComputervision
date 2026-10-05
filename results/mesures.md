# Mesures de ce travail (T8.1)

Généré par `python tools/bench_report.py` (`make bench-report`) ; protocole : [protocole.md](protocole.md). Kria KV260, tuiles Tm = 32, Tn = 24, Tr = Tc = 13, 200 MHz.

## Projection (modèle de cycles, accélérateur seul)

`tools/perf_model.py`, ports m_axi de 8 bits, II = 1, profondeurs de pipeline et pilotage ARM ignorés : borne basse du temps accélérateur mesurable.

| Réseau | GMAC | Mcycles | ms | img/s | GOPS | efficacité |
|---|---|---|---|---|---|---|
| tiny-yolov2-voc | 3.486 | 41.30 | 206.5 | 4.84 | 33.8 | 11.0 % |
| tiny-yolov3-coco | 2.782 | 40.64 | 203.2 | 4.92 | 27.4 | 8.9 % |

## Mesures sur la carte

À mesurer : aucun fichier `build/m8/tiny-yolov2-voc/board/times_*.csv` (KV260 non disponible). Procédure : [protocole.md](protocole.md).
