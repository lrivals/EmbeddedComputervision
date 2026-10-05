# M6 — Accélérateur Vitis HLS

Objectif : un noyau HLS « moteur unique couche par couche » (§10.1), validé en C-sim puis en
co-simulation RTL contre le golden model, avec des rapports de synthèse exploitables.

### [ ] T6.0 — Choix de la carte
- **Spec** : §10.1, §10.4 · **Dépend de** : T5.6 · **Taille** : S
- **Livrables** : [ADR 0003](../adr/0003-choix-carte.md) complété ;
  `hls/configs/<carte>.tcl` (part, horloge, tuiles retenues)
- **Acceptation** : décision argumentée selon les critères de l'ADR ; installation des
  outils Vivado/Vitis correspondants vérifiée
- **Fait** : ADR 0003 accepté : **KV260**, tuiles 32, 24, 13, 13 à 200 MHz (point
  roofline de T5.6, 6× plus rapide que les Zynq-7020 au même prix) ;
  `hls/configs/kv260.tcl`, plus `zybo-z7-20.tcl` (8, 16, 13, 13) pour tester des tuiles
  plus petites.
- **Reste** : Vitis/Vivado absent de la machine de développement — installation à faire et
  à vérifier (`which vitis_hls`).

### [ ] T6.1 — PE de convolution
- **Spec** : §10.2 · **Dépend de** : T6.0, T5.2 · **Taille** : L
- **Livrables** : `hls/kernels/conv_pe.cpp`, `hls/tb/tb_conv.cpp`, `hls/scripts/csim.tcl`
- **Acceptation** : C-sim == golden pour toutes les couches conv ; boucle interne
  `PIPELINE II=1`, `UNROLL` de Tm × Tn, II atteint dans le rapport de synthèse
- **Notes** : `ap_int<8>` / `ap_int<32>` ; arbre d'additions sur Tn ; partition des
  tampons alignée sur le déroulage
- **Fait** : `hls/kernels/conv_pe.cpp` (+ `accel.hpp`, `layer_desc.hpp`) : même parcours
  de tuiles que `golden::conv_layer` ; boucle (i, j, trr, tcc) aplatie en `PIPELINE II=1`,
  too et tii `UNROLL`, somme sur Tn en `acc_t` ; `in_buf` partitionné dim 1, `w_buf` dims 1-2,
  `out_buf` dim 1, `DEPENDENCE inter false`. Tailles des tampons vérifiées contre B_in, B_w,
  B_out. `hls/tb/tb_conv.cpp` : chaque conv des deux réseaux, 3 images, sortie == golden ==
  dump (carte poolée et carte avant pooling), tuiles KV260 et Zybo. C-sim avec g++ et les
  en-têtes `ap_int` open source (`hls/CMakeLists.txt`, `make csim-gcc`) ;
  `hls/scripts/csim.tcl` (`make csim`) prêt pour Vitis.
- **Reste** : II = 1 atteint dans le rapport de synthèse (`make hls-synth`, Vitis requis).

### [ ] T6.2 — Interfaces mémoire et contrôle
- **Spec** : §10.2 (double tampon) · **Dépend de** : T6.1 · **Taille** : L
- **Livrables** : `m_axi` pour poids, activations et requant ; `s_axilite` pour les
  paramètres de couche (forme, offsets, échelle, options) ; ping-pong
  chargement/calcul/stockage
- **Acceptation** : co-sim == golden ; recouvrement transferts/calcul visible dans les
  traces ; débit mesuré en cycles par couche
- **Fait** : top `yolo_conv` : `m_axi` `gmem_in` / `gmem_out` (même arène d'activations,
  deux bundles pour recouvrir chargement et stockage), `gmem_w` (poids), `gmem_p` (biais puis
  M0) ; `s_axilite` : `LayerDesc` à plat (forme, 2 segments d'entrée, offsets, décalage,
  options). Ping-pong `in_buf`/`w_buf` (chargement de ti + 1 pendant le calcul de ti) et
  `out_buf` (stockage de la tuile k − 1 pendant la tuile k). C-sim toujours == golden.
  Estimation de cycles par couche en C-sim (`accel::sim_cycles`, `make hls-cycles`) :
  ≈ 210 ms/image Tiny-YOLOv2 à 200 MHz contre 41 ms de calcul pur — le noyau est limité par
  les chargements octet par octet (`results/hls_report.md`).
- **Reste** : co-sim, traces et cycles réels (`make hls-cosim`, Vitis requis).

### [ ] T6.3 — Étage de sortie fusionné
- **Spec** : §9.3, §10.3 · **Dépend de** : T6.1 · **Taille** : M
- **Livrables** : biais + requantification M0/décalage + leaky 13/128 + maxpool 2×2 en
  stride 2 **et 1**
- **Acceptation** : couches 0 à 11 == golden en C-sim ; coût DSP de la requantification
  rapporté
- **Fait** : `hls/kernels/output_stage.hpp` : `golden::requantize` / `leaky_int` / `clip8`
  réutilisés tels quels, maxpool 2×2 s2 et s1 (réplication du bord), carte avant pooling en
  DDR (L08 → R). Couches 0 à 11 (et toutes les autres) == golden en C-sim, v2 et v3. Un seul
  multiplieur acc·M0 (64 bits) partagé par les Tm canaux : 4 DSP48 attendus.
- **Reste** : coût DSP mesuré dans le rapport de synthèse (`store_tile`, `make hls-report`).

### [x] T6.4 — Upsample et route par adressage
- **Spec** : §10.3 · **Dépend de** : T6.2 · **Taille** : M
- **Livrables** : modes d'adressage dans le chargeur d'entrée (lecture à l'adresse / 2) ;
  allocation contiguë par le driver
- **Acceptation** : réseau Tiny-YOLOv3 complet == golden en C-sim, sans noyau dédié à
  l'upsample ni à la route
- **Fait** : chargeur à 2 segments `{offset, c, h, w, up}`, lecture à
  `(c·h + (r >> up))·w + (col >> up)` ; `sw/driver/program.{hpp,cpp}` : arène DDR contiguë
  (tampons du manifest bout à bout, alignés à 64), vues par couche (même logique que
  `Engine::place_views`), un `LayerDesc` par conv. `hls/tb/tb_net.cpp` : Tiny-YOLOv2 (9
  appels) et Tiny-YOLOv3 (13 appels) complets, chaque couche vérifiée juste après son appel
  == golden == dump, 3 images, route 20 en deux segments (L18 vue ×2, prépool de L08).

### [ ] T6.5 — Co-simulation et synthèse
- **Spec** : §10.4 · **Dépend de** : T6.3, T6.4 · **Taille** : M
- **Livrables** : `hls/scripts/{synth,cosim,export}.tcl` ; rapport `results/hls_report.md`
  (LUT, FF, DSP, BRAM, fréquence, latence en cycles par couche)
- **Acceptation** : co-sim RTL == golden sur une image complète ; timing tenu à la
  fréquence cible ; IP exportée pour Vivado
- **Fait** : `hls/scripts/{common,csim,synth,cosim,export}.tcl` (`make csim | hls-synth |
  hls-cosim | hls-export BOARD=kv260`) ; `tools/hls_report.py` (`make hls-report`) →
  `results/hls_report.md` : section cycles par couche (estimation C-sim) remplie ; sections
  synthèse (LUT, FF, DSP, BRAM, période, II des boucles, DSP de `store_tile`) et co-sim
  lues dans les rapports Vitis dès qu'ils existent.
- **Reste** : tout ce qui demande Vitis : synthèse, co-sim d'une image (longue : ~4·10⁷
  cycles estimés), timing à 200 MHz, export de l'IP.
