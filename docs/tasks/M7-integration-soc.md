# M7 — Intégration SoC

Objectif : l'accélérateur tourne sur la carte, piloté par l'ARM, et produit en DDR exactement
les mêmes entiers que le golden model.

### [ ] T7.1 — Block design Vivado
- **Spec** : §10.1 · **Dépend de** : T6.5 · **Taille** : M
- **Livrables** : `hw/boards/<carte>/build.tcl` (PS, IP HLS, interconnexions AXI,
  horloges, reset) ; bitstream + `.xsa`
- **Acceptation** : bitstream généré depuis le tcl seul (reproductible) ; timing tenu
- **Fait** : `hw/boards/kv260/build.tcl` (Vivado batch) : PS avec preset de carte, IP
  `yolo_conv` de `make hls-export`, `s_axi_control` sur HPM0_FPD à 0xA000_0000,
  `gmem_in`/`gmem_out` → HP0, `gmem_w`/`gmem_p` → HP1 (smartconnects), `pl_clk0` 200 MHz,
  `proc_sys_reset`, interruption → `pl_ps_irq0` ; bitstream, `.xsa`, rapports timing /
  utilisation / puissance ; échec si WNS, TNS ou WHS < 0. `pl.dtsi` (overlay : `generic-uio`
  + IRQ, u-dma-buf 32 Mo, horloge), `shell.json`, `firmware.sh` (`make fpga-firmware`),
  procédure carte dans `hw/boards/kv260/README.md`. `make vivado-build`.
- **Reste** : tout ce qui demande Vivado : génération du bitstream, timing tenu à 200 MHz.
  Le script n'a jamais été exécuté (Vivado absent) ; `pl.dtsi` non compilé (dtc absent).

### [ ] T7.2 — Driver ARM
- **Spec** : §10.1, §10.3 · **Dépend de** : T7.1, T5.1 · **Taille** : L
- **Livrables** : `sw/driver/` : chargement du manifest et des blobs en mémoire contiguë
  (CMA / XRT / PYNQ selon la carte), programmation des registres AXI-Lite, ordonnancement
  des couches, attente de fin
- **Acceptation** : une couche isolée exécutée sur carte == dump golden
- **Notes** : l'allocation des tampons doit respecter le placement contigu de la route
- **Fait** : pile retenue : Ubuntu Kria + UIO + u-dma-buf. `sw/driver/` : `regmap.hpp`
  (offsets s_axilite attendus de Vitis HLS, `LayerDesc` agrégé en 26 mots ; contrôle contre
  l'en-tête généré : `make check-regmap`, `tools/check_regmap.py`), `device.hpp` (HAL :
  tampon contigu, registres, attente), `uio_device.cpp` (carte : `/dev/uioN` + IRQ,
  `/dev/udmabufN` en O_SYNC), `sim_device.cpp` (PC : registres émulés, traduction des
  adresses physiques fictives > 4 Go, appel du noyau C-sim à ap_start),
  `accel_driver.{hpp,cpp}` (`Accelerator` : un tampon [arène de `driver::build` | poids |
  paramètres], contrôle des bornes de chaque `LayerDesc`, `run_layer` chronométré).
  Sur PC : `run_compare --layer N` (arène pré-remplie avec les dumps des couches < N) ==
  dump pour L00 (v2, v3), L13 (v2) et L21 (v3, route en 2 segments dont un upsample).
  `make sw-sim`, `make sw-board` (build natif, backend uio).
- **Reste** : exécution sur la KV260.

### [ ] T7.3 — Post-traitement sur l'ARM
- **Spec** : §10.3 (décodage + NMS sur ARM, comme 2025-kim et 2024-yan) · **Dépend de** :
  T5.5, T7.2 · **Taille** : S
- **Livrables** : `sw/postproc/` (réutilise `postproc.hpp`), `sw/app/` (image ou caméra →
  boîtes)
- **Acceptation** : boîtes == `dumps/heads.json` ; temps du post-traitement mesuré
- **Fait** : `sw/postproc/heads.{hpp,cpp}` : têtes lues en place dans l'arène (vues du
  driver), puis `postproc::postprocess` (`postproc.hpp` tel quel) ; `golden/detections_io.hpp`
  (écriture partagée avec `golden_run`, lecture et égalité exacte). `sw/app/yolo_app` :
  `--input x.npy` ou `--image` (stb_image, `stretch` Darknet comme `resize_darknet`,
  quantification `quantize_input`), `--draw` (boîtes sur l'image), temps prétraitement /
  accélérateur / post-traitement. Sur PC : détections == `detections.json` pour les 3 images
  des deux réseaux ; post-traitement < 0,02 ms sur x86.
- **Reste** : temps du post-traitement mesuré sur l'A53.

### [ ] T7.4 — Bout-en-bout sur carte
- **Spec** : §10.5 étape 4, §11 · **Dépend de** : T7.3 · **Taille** : M
- **Livrables** : `sw/app/run_compare` (exécute et compare chaque couche au golden)
- **Acceptation** : **0 écart** sur toutes les couches pour les 3 images de référence ;
  démo de détection sur une image réelle
- **Fait** : `sw/app/run_compare` : passe complète, chaque couche en DDR comparée au dump
  juste après sa conv (même parcours que `hls/tb/tb_net.cpp`), puis détections ; `--csv`
  des temps par couche (M8). Backend sim : **87 sorties de couche, 0 écart**, détections
  identiques, Tiny-YOLOv2 VOC et Tiny-YOLOv3 COCO, 3 images. Démo `yolo_app --image
  000004.jpg` : 7 voitures détectées.
- **Reste** : 0 écart sur la KV260 (`hw/boards/kv260/README.md`, étape 3) ; démo sur carte.
