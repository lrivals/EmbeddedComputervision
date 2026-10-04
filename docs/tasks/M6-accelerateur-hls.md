# M6 — Accélérateur Vitis HLS

Objectif : un noyau HLS « moteur unique couche par couche » (§10.1), validé en C-sim puis en
co-simulation RTL contre le golden model, avec des rapports de synthèse exploitables.

### [ ] T6.0 — Choix de la carte
- **Spec** : §10.1, §10.4 · **Dépend de** : T5.6 · **Taille** : S
- **Livrables** : [ADR 0003](../adr/0003-choix-carte.md) complété ;
  `hls/configs/<carte>.tcl` (part, horloge, tuiles retenues)
- **Acceptation** : décision argumentée selon les critères de l'ADR ; installation des
  outils Vivado/Vitis correspondants vérifiée

### [ ] T6.1 — PE de convolution
- **Spec** : §10.2 · **Dépend de** : T6.0, T5.2 · **Taille** : L
- **Livrables** : `hls/kernels/conv_pe.cpp`, `hls/tb/tb_conv.cpp`, `hls/scripts/csim.tcl`
- **Acceptation** : C-sim == golden pour toutes les couches conv ; boucle interne
  `PIPELINE II=1`, `UNROLL` de Tm × Tn, II atteint dans le rapport de synthèse
- **Notes** : `ap_int<8>` / `ap_int<32>` ; arbre d'additions sur Tn ; partition des
  tampons alignée sur le déroulage

### [ ] T6.2 — Interfaces mémoire et contrôle
- **Spec** : §10.2 (double tampon) · **Dépend de** : T6.1 · **Taille** : L
- **Livrables** : `m_axi` pour poids, activations et requant ; `s_axilite` pour les
  paramètres de couche (forme, offsets, échelle, options) ; ping-pong
  chargement/calcul/stockage
- **Acceptation** : co-sim == golden ; recouvrement transferts/calcul visible dans les
  traces ; débit mesuré en cycles par couche

### [ ] T6.3 — Étage de sortie fusionné
- **Spec** : §9.3, §10.3 · **Dépend de** : T6.1 · **Taille** : M
- **Livrables** : biais + requantification M0/décalage + leaky 13/128 + maxpool 2×2 en
  stride 2 **et 1**
- **Acceptation** : couches 0 à 11 == golden en C-sim ; coût DSP de la requantification
  rapporté

### [ ] T6.4 — Upsample et route par adressage
- **Spec** : §10.3 · **Dépend de** : T6.2 · **Taille** : M
- **Livrables** : modes d'adressage dans le chargeur d'entrée (lecture à l'adresse / 2) ;
  allocation contiguë par le driver
- **Acceptation** : réseau Tiny-YOLOv3 complet == golden en C-sim, sans noyau dédié à
  l'upsample ni à la route

### [ ] T6.5 — Co-simulation et synthèse
- **Spec** : §10.4 · **Dépend de** : T6.3, T6.4 · **Taille** : M
- **Livrables** : `hls/scripts/{synth,cosim,export}.tcl` ; rapport `results/hls_report.md`
  (LUT, FF, DSP, BRAM, fréquence, latence en cycles par couche)
- **Acceptation** : co-sim RTL == golden sur une image complète ; timing tenu à la
  fréquence cible ; IP exportée pour Vivado
