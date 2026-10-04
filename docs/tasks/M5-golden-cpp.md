# M5 — Golden model C++

Objectif : un moteur C++ qui exécute le réseau entier **comme le matériel** (tuilage,
tampons, maxpool fusionné, route par adressage) et reproduit les dumps Python à l'entier
près. Il sert de testbench à l'étage HLS.

### [ ] T5.1 — Lecture du manifest et des blobs
- **Spec** : — · **Dépend de** : T4.7, T0.3 · **Taille** : S
- **Livrables** : `cpp/golden/include/golden/model.hpp`, `src/model.cpp` (lecteur JSON
  minimal ou nlohmann/json en en-tête), lecteur `.npy` pour les dumps
- **Acceptation** : toutes les couches, les échelles et les décalages relus == valeurs
  Python

### [ ] T5.2 — Convolution tuilée
- **Spec** : §10.2, §9.3 · **Dépend de** : T5.1 · **Taille** : L
- **Livrables** : `cpp/golden/include/golden/conv.hpp` : nid de boucles (Tm, Tn, Tr, Tc)
  avec tampons `in_buf`, `w_buf`, `out_buf` de taille fixe ; étage de sortie biais +
  requantification + leaky + maxpool fusionné
- **Acceptation** : sortie == dump Python pour chaque couche conv, **pour plusieurs jeux de
  tuiles** (dont des tuiles qui ne divisent pas les dimensions) ; tailles de tampons ==
  formules $B_{in}$, $B_w$, $B_{out}$ du §10.2
- **Notes** : pas d'allocation dynamique dans le noyau
  ([ADR 0002](../adr/0002-hls.md)) ; relire `2015-zhang#014.0`

### [ ] T5.3 — Couches spécifiques à YOLO
- **Spec** : §10.3 · **Dépend de** : T5.2 · **Taille** : M
- **Livrables** : maxpool 2×2/1 (réplication du bord), upsample par division d'adresse,
  route par placement contigu en mémoire
- **Acceptation** : dumps des couches 11, 19 et 20 identiques ; aucune copie mémoire pour
  l'upsample ni la route (vérifié par compteur d'accès)

### [ ] T5.4 — Réseau complet et comparaison
- **Spec** : §10.5 étape 3, §11 · **Dépend de** : T5.3 · **Taille** : M
- **Livrables** : `cpp/golden/src/engine.cpp`, `tools/compare_dumps.py` (rapport par
  couche : nombre de valeurs différentes, écart max)
- **Acceptation** : **0 valeur différente** sur toutes les couches pour les 3 images de
  T4.7, pour Tiny-YOLOv2 et Tiny-YOLOv3

### [ ] T5.5 — Post-traitement C++
- **Spec** : §8.1, §8.2, §9.4 · **Dépend de** : T5.4 · **Taille** : M
- **Livrables** : `cpp/golden/include/golden/postproc.hpp` (seuil sur logit, LUT, décodage,
  NMS)
- **Acceptation** : boîtes == `dumps/heads.json` ; code réutilisable tel quel dans
  `sw/postproc` (sans dépendance au reste du golden)

### [ ] T5.6 — Explorateur roofline
- **Spec** : §10.2 · **Dépend de** : T0.5 · **Taille** : M
- **Livrables** : `tools/roofline.py` ; fichiers de ressources de cartes candidates
  `hw/boards/<carte>.yaml` (DSP, BRAM, bande passante DDR, fréquence)
- **Acceptation** : pour chaque carte, énumère $(T_m, T_n, T_r, T_c)$, calcule BRAM, DSP,
  CTC et performance atteignable ; retrouve l'ordre de grandeur du §10.2 (Tm = Tn = 16 →
  ≈ 68 ms à 200 MHz pour Tiny-YOLOv2) ; graphique roofline
- **Notes** : peut démarrer dès M0 ; c'est l'entrée de la décision T6.0
