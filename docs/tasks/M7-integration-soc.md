# M7 — Intégration SoC

Objectif : l'accélérateur tourne sur la carte, piloté par l'ARM, et produit en DDR exactement
les mêmes entiers que le golden model.

### [ ] T7.1 — Block design Vivado
- **Spec** : §10.1 · **Dépend de** : T6.5 · **Taille** : M
- **Livrables** : `hw/boards/<carte>/build.tcl` (PS, IP HLS, interconnexions AXI,
  horloges, reset) ; bitstream + `.xsa`
- **Acceptation** : bitstream généré depuis le tcl seul (reproductible) ; timing tenu

### [ ] T7.2 — Driver ARM
- **Spec** : §10.1, §10.3 · **Dépend de** : T7.1, T5.1 · **Taille** : L
- **Livrables** : `sw/driver/` : chargement du manifest et des blobs en mémoire contiguë
  (CMA / XRT / PYNQ selon la carte), programmation des registres AXI-Lite, ordonnancement
  des couches, attente de fin
- **Acceptation** : une couche isolée exécutée sur carte == dump golden
- **Notes** : l'allocation des tampons doit respecter le placement contigu de la route

### [ ] T7.3 — Post-traitement sur l'ARM
- **Spec** : §10.3 (décodage + NMS sur ARM, comme 2025-kim et 2024-yan) · **Dépend de** :
  T5.5, T7.2 · **Taille** : S
- **Livrables** : `sw/postproc/` (réutilise `postproc.hpp`), `sw/app/` (image ou caméra →
  boîtes)
- **Acceptation** : boîtes == `dumps/heads.json` ; temps du post-traitement mesuré

### [ ] T7.4 — Bout-en-bout sur carte
- **Spec** : §10.5 étape 4, §11 · **Dépend de** : T7.3 · **Taille** : M
- **Livrables** : `sw/app/run_compare` (exécute et compare chaque couche au golden)
- **Acceptation** : **0 écart** sur toutes les couches pour les 3 images de référence ;
  démo de détection sur une image réelle
