# M5 — Golden model C++

Objectif : un moteur C++ qui exécute le réseau entier **comme le matériel** (tuilage,
tampons, maxpool fusionné, route par adressage) et reproduit les dumps Python à l'entier
près. Il sert de testbench à l'étage HLS.

### [x] T5.1 — Lecture du manifest et des blobs
- **Spec** : — · **Dépend de** : T4.7, T0.3 · **Taille** : S
- **Livrables** : `cpp/golden/include/golden/model.hpp`, `src/model.cpp` (lecteur JSON
  minimal ou nlohmann/json en en-tête), lecteur `.npy` pour les dumps
- **Acceptation** : toutes les couches, les échelles et les décalages relus == valeurs
  Python
- **Fait** : `json.hpp` (lecteur minimal, nombres par `strtod` : aller-retour exact des
  `repr` Python ; pas de nlohmann, la lib reste sans dépendance), `npy.hpp` (int8, lecture et
  écriture), `model.hpp` (`Model::load` : couches, échelles, blobs little-endian ; contrôle
  de version, alignement 64, tailles des blobs, tenseurs dans leurs tampons). CLI
  `golden_run inspect` ; `python/tests/test_golden_cpp.py` le compare à `load_model` (champs,
  échelles à l'égalité, sommes/premier/dernier des poids, biais, M0, tables).

### [x] T5.2 — Convolution tuilée
- **Spec** : §10.2, §9.3 · **Dépend de** : T5.1 · **Taille** : L
- **Livrables** : `cpp/golden/include/golden/conv.hpp` : nid de boucles (Tm, Tn, Tr, Tc)
  avec tampons `in_buf`, `w_buf`, `out_buf` de taille fixe ; étage de sortie biais +
  requantification + leaky + maxpool fusionné
- **Acceptation** : sortie == dump Python pour chaque couche conv, **pour plusieurs jeux de
  tuiles** (dont des tuiles qui ne divisent pas les dimensions) ; tailles de tampons ==
  formules $B_{in}$, $B_w$, $B_{out}$ du §10.2
- **Notes** : pas d'allocation dynamique dans le noyau
  ([ADR 0002](../adr/0002-hls.md)) ; relire `2015-zhang#014.0`
- **Fait** : `conv.hpp` (en-tête seul), `conv_layer<Tiles<Tm, Tn, Tr, Tc>>`, tampons
  `Tiles::Buffers` dont la taille est vérifiée par `static_assert` contre B_in, B_w, B_out ;
  aucune allocation dans le noyau. Tests : chaque conv des deux réseaux, seule, sur l'entrée
  dumpée == dump (carte poolée et carte avant pooling) pour `16,16,13,13`, `7,5,6,9` et
  `32,8,26,4` ; cas synthétiques (1×1, pad, pool s2 et s1 au bord) contre une référence
  naïve pour quatre jeux de tuiles dont `1,1,2,2`.
  **Écarts au pseudo-code du §10.2** : (1) ordre des boucles de tuiles row → col → to → ti
  (2015-zhang) : l'ordre (to, ti, row, col) du §10.2 avec stockage à la dernière ti exigerait
  un `out_buf` de la carte entière ; (2) Tr, Tc = tuile de sortie de la conv **avant**
  pooling (formules littérales) : une conv poolée (k_p, s_p) produit
  P = ⌊(Tr − k_p)/s_p⌋ + 1 lignes poolées par tuile et avance de P·s_p lignes ; en stride 1
  la dernière ligne de la tuile est recalculée par la suivante ; Tr, Tc ≥ 2 ; (3) dans une
  tuile, MAC dans l'ordre (too, tii, i, j, trr, tcc), simple commutation de sommes entières
  exactes. `2015-zhang#014.0` non relu (`bin/pdb` absent du dépôt).

### [x] T5.3 — Couches spécifiques à YOLO
- **Spec** : §10.3 · **Dépend de** : T5.2 · **Taille** : M
- **Livrables** : maxpool 2×2/1 (réplication du bord), upsample par division d'adresse,
  route par placement contigu en mémoire
- **Acceptation** : dumps des couches 11, 19 et 20 identiques ; aucune copie mémoire pour
  l'upsample ni la route (vérifié par compteur d'accès)
- **Fait** : `engine.hpp` : la sortie de chaque couche est une vue (`InView`, ≤ 2 segments
  `{base, C, H, W, up}`) ; maxpool 2×2/1 dans l'étage de sortie de la conv (fenêtre bornée
  = réplication) ; upsample = vue de la source avec `up + 1` ; route = concaténation des
  segments, fusionnés quand ils sont bout à bout (route 17 = L13 en place). Couches 11 (v2,
  v3), 19 et 20 == dumps ; compteurs `AccessStats` : 0 octet lu ou écrit par les couches 17,
  19 et 20, octets écrits par le réseau = Σ sorties des convs + prépool de L08, et la zone
  `R:[0, 86528)` que le manifest réservait à la copie ×2 reste intacte.
  **Écart au manifest** : l'`out` de l'upsample n'est pas écrit ; la route 20 est lue en deux
  segments (L18 dans B vue ×2, prépool de L08 à `R+86528`) — conventions.md mis à jour.

### [x] T5.4 — Réseau complet et comparaison
- **Spec** : §10.5 étape 3, §11 · **Dépend de** : T5.3 · **Taille** : M
- **Livrables** : `cpp/golden/src/engine.cpp`, `tools/compare_dumps.py` (rapport par
  couche : nombre de valeurs différentes, écart max)
- **Acceptation** : **0 valeur différente** sur toutes les couches pour les 3 images de
  T4.7, pour Tiny-YOLOv2 et Tiny-YOLOv3
- **Fait** : `Engine::run<Tiles>` (tuiles par défaut 16,16,13,13), CLI
  `golden_run run <model> <input.npy> <out>` (`Lxx.npy` + `detections.json`),
  `tools/compare_dumps.py`, `make golden-check` : **0 valeur différente** sur toutes les
  couches, 3 images × (Tiny-YOLOv2 VOC, Tiny-YOLOv3 COCO — l'export de T4.7), détections
  identiques. Aussi vérifié dans `make test-cpp` (`test_engine.cpp`). ≈ 1,5 s par image.

### [x] T5.5 — Post-traitement C++
- **Spec** : §8.1, §8.2, §9.4 · **Dépend de** : T5.4 · **Taille** : M
- **Livrables** : `cpp/golden/include/golden/postproc.hpp` (seuil sur logit, LUT, décodage,
  NMS)
- **Acceptation** : boîtes == `dumps/heads.json` ; code réutilisable tel quel dans
  `sw/postproc` (sans dépendance au reste du golden)
- **Fait** : `postproc.hpp` (en-tête seul, STL uniquement, espace `postproc`) : seuil sur
  l'entier t_o, tables σ / e^t / e^{d·s} (softmax v2 en trois étages), décodage, NMS par
  classe à tri stable ; copie opération par opération de `decode_head_int` et
  `filter_and_nms`. Boîtes, scores et classes **égaux bit à bit** à `detections.json` (le
  fichier réel ; `heads.json` n'existe pas) pour 3 images × 2 réseaux, têtes lues dans les
  dumps : le test n'utilise pas le moteur.

### [x] T5.6 — Explorateur roofline
- **Spec** : §10.2 · **Dépend de** : T0.5 · **Taille** : M
- **Livrables** : `tools/roofline.py` ; fichiers de ressources de cartes candidates
  `hw/boards/<carte>.yaml` (DSP, BRAM, bande passante DDR, fréquence)
- **Acceptation** : pour chaque carte, énumère $(T_m, T_n, T_r, T_c)$, calcule BRAM, DSP,
  CTC et performance atteignable ; retrouve l'ordre de grandeur du §10.2 (Tm = Tn = 16 →
  ≈ 68 ms à 200 MHz pour Tiny-YOLOv2) ; graphique roofline
- **Notes** : peut démarrer dès M0 ; c'est l'entrée de la décision T6.0
- **Fait** : `tools/roofline.py` (`make roofline`) → `results/roofline.md` et
  `results/roofline_<carte>.png` ; cartes `hw/boards/{pynq-z2,zybo-z7-20,ultra96-v2,kv260}.yaml`
  (fiches techniques publiques, efficacité DDR 0,7 et horloge supposées). Modèle : cycles
  ⌈M/Tm⌉⌈N/Tn⌉⌈R/Tr⌉⌈C/Tc⌉K²TrTc, trafic DDR B_in, B_w, B_out par tuile, temps
  max(calcul, transfert) par couche, DSP = TmTn + 4Tm (requantification), BRAM18 des bancs
  `in_buf`/`out_buf` en ping-pong, ≤ 80 % des ressources. Ordre de grandeur retrouvé :
  3,49 GMAC / 256 = 68,1 ms à 200 MHz. Meilleurs points (Tiny-YOLOv2) : KV260 32,24,13,13 →
  ≈ 32 ms ; Ultra96-V2 8,32,13,13 → ≈ 86 ms ; Zynq-7020 8,16,13,13 → ≈ 198 ms (bornés par
  les DSP, pas par la DDR).
