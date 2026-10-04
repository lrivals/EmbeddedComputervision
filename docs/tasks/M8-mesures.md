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

### [ ] T8.2 — mAP aux trois stades
- **Spec** : §11 · **Dépend de** : T8.1, T4.5 · **Taille** : S
- **Livrables** : `results/map_stades.md`
- **Acceptation** : mAP VOC2007 test en flottant, en entier Python et sur FPGA ; les deux
  dernières doivent être **identiques** (bit-exact)

### [ ] T8.3 — Rapport comparatif
- **Spec** : §10.4 · **Dépend de** : T8.2 · **Taille** : M
- **Livrables** : `results/rapport.md`
- **Acceptation** : comparaison avec 2023-zhai, 2024-zhang, 2025-kim et 2026-fata en
  signalant ce qui n'est pas comparable (jeu de données, élagage, fréquence, périmètre de
  puissance) ; pistes d'optimisation chiffrées par le roofline
