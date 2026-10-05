# M8 — Mesures et rapport

Objectif : des chiffres comparables à ceux de la base (§10.4), avec un périmètre de mesure
explicite.

### [ ] T8.1 — Performance, puissance, ressources
- **Spec** : §10.4, §10.5 étape 6 · **Dépend de** : T7.4 · **Taille** : M
- **Livrables** : ligne(s) dans `results/benchmarks.csv` ; protocole dans `results/protocole.md`
- **Acceptation** : FPS et latence (moyenne et p99 sur ≥ 1 000 images, en séparant
  prétraitement, accélérateur et post-traitement) ; GOPS = 2 × MACs / temps ; puissance avec
  **périmètre précisé** (puce estimée par Vivado *et* carte mesurée si possible) ;
  LUT/DSP/BRAM du rapport d'implémentation
- **Notes** : efficacité = cycles théoriques ($\text{MACs}/(T_mT_n)$) / cycles mesurés
- **Fait** : `sw/app/yolo_bench` (temps par image et par étage : prétraitement, copie DDR,
  accélérateur, post-traitement ; moyenne et p99 ; `--images` JPEG ou `--inputs` int8 ;
  `--power` : INA260 du SOM via hwmon, au repos puis en charge ; détections JSONL ; lots
  `--start/--count`). `tools/perf_model.py` : modèle de cycles **égal cycle pour cycle** aux
  compteurs C-sim (`make perf-model`), scénarios d'optimisation. `tools/bench_report.py` :
  `results/benchmarks.csv` (8 lignes publiées du §10.4 + ce travail), `results/mesures.md`
  (étages, efficacité par couche), lecture de `utilization.rpt` / `power.rpt`.
  `results/protocole.md` : périmètres (étages, puissance puce estimée / SOM mesuré),
  conditions, commandes carte. Sans carte : ligne « projection C-sim » (Tiny-YOLOv2 :
  206,5 ms accélérateur, 4,8 img/s, 33,8 GOPS, efficacité 11,0 %).
- **Reste** : mesures sur la KV260 (≥ 1 000 images, puissance SOM) ; LUT/DSP/BRAM et
  puissance puce de `make vivado-build`.

### [ ] T8.2 — mAP aux trois stades
- **Spec** : §11 · **Dépend de** : T8.1, T4.5 · **Taille** : S
- **Livrables** : `results/map_stades.md`
- **Acceptation** : mAP VOC2007 test en flottant, en entier Python et sur FPGA ; les deux
  dernières doivent être **identiques** (bit-exact)
- **Fait** : `tools/make_inputs.py` (entrées int8 de VOC2007 test, `stretch` PIL, lues par la
  carte au lieu des JPEG : stb ≠ PIL), `tools/eval_quant.py --save-dets` (détections
  entières par image), `tools/map_stades.py` → `results/map_stades.md` (égalité image par
  image + mAP). Flottant 56,30, entier 55,66 (= T3.4, T4.5). Stade FPGA en C-sim (`make bench-sim`, driver
  ARM + noyau C-sim, 22 processus, ≈ 40 min) : **mAP 55,66, 4 952 / 4 952 images identiques**
  à l'entier.
- **Reste** : le même `yolo_bench --inputs` sur la KV260 (`board/dets_*.jsonl`).
- **Profils PC** : chaîne simulée par paliers [T12.9](M12-profils-pc.md#-t129--chaîne-carte-simulée), répétition du protocole [T12.10](M12-profils-pc.md#-t1210--répétition-générale-du-protocole-carte)

### [ ] T8.3 — Rapport comparatif
- **Spec** : §10.4 · **Dépend de** : T8.2 · **Taille** : M
- **Livrables** : `results/rapport.md`
- **Acceptation** : comparaison avec 2023-zhai, 2024-zhang, 2025-kim et 2026-fata en
  signalant ce qui n'est pas comparable (jeu de données, élagage, fréquence, périmètre de
  puissance) ; pistes d'optimisation chiffrées par le roofline
- **Fait** : `results/rapport.md` : comparaison avec la projection (en attendant la carte),
  colonne « non comparable », pistes chiffrées par `tools/perf_model.py` (ports 64/128 bits,
  canaux valides, requantification parallèle) et rapprochées du roofline.
- **Reste** : remplacer la projection par les mesures de la carte.
