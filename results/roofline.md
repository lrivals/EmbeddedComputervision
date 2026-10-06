# Roofline des cartes candidates (T5.6)

Généré par `python tools/roofline.py` (`make roofline`) ; modèle et hypothèses dans l'en-tête du script. Ressources : `hw/boards/*.yaml`. Points retenus : ≤ 80 % des DSP et des BRAM18 ; meilleur point = temps minimal pour Tiny-YOLOv2.

Ordre de grandeur du §10.2 : Tm = Tn = 16, 200 MHz, efficacité 100 % → 3.49 GMAC / 256 = 68.1 ms (Tiny-YOLOv2).

| Carte | DSP | BRAM18 | BW eff. (Go/s) | f (MHz) | Tm, Tn, Tr, Tc | DSP util. | BRAM18 util. | CTC (op/o) | toit calcul (GOPS) | atteint (GOPS) | tiny-yolov2-voc (ms) | tiny-yolov3-voc (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Kria KV260 | 1248 | 288 | 13.44 | 200 | 32, 24, 13, 13 | 896 | 112 | 114.6 | 218.5 | 218.5 | 31.9 | 27.0 |
| PYNQ-Z2 | 220 | 280 | 2.94 | 150 | 8, 16, 13, 13 | 160 | 48 | 68.9 | 35.1 | 35.1 | 198.4 | 159.4 |
| Ultra96-V2 | 360 | 432 | 5.95 | 200 | 8, 32, 13, 13 | 288 | 80 | 60.2 | 81.0 | 80.9 | 86.2 | 71.7 |
| Zybo Z7-20 | 220 | 280 | 2.98 | 150 | 8, 16, 13, 13 | 160 | 48 | 68.9 | 35.1 | 35.1 | 198.4 | 159.4 |

Graphiques : `roofline_kv260.png`, `roofline_pynq-z2.png`, `roofline_ultra96-v2.png`, `roofline_zybo-z7-20.png`

Roofline par couche, avant et après M10 : `results/figures/resultats/roofline_couches_kv260.png` (`python -m tools.figures roofline_couches`, T13.22).
