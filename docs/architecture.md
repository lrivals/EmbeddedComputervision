# Architecture

## 1. Quatre étages, un seul contrat

Le projet suit la feuille de route du §10.5 de la spec. Chaque étage est la référence du
suivant :

| Étage | Dossier | Rôle | Référence de comparaison |
|---|---|---|---|
| 1. Flottant | `python/yolo/` | réseau, perte, entraînement, mAP de référence | gradcheck (différences finies) |
| 2. Entier | `python/yolo/quant/` | fusion BN, int8, M0/décalage, LUT | étage 1 (écart de mAP) |
| 3. Golden | `cpp/golden/` | moteur tuilé en C++, comme le matériel | étage 2 (**0 écart**) |
| 4. Matériel | `hls/`, `sw/`, `hw/` | accélérateur + driver ARM | étage 3 (**0 écart**) |

Les étages 2, 3 et 4 doivent produire **les mêmes entiers**, couche par couche. Toute la
vérification repose sur ce principe.

## 2. Format d'échange (`model/<nom>/`)

Produit par l'export Python (T4.7), lu par le golden (T5.1), le testbench HLS et le driver ARM.

```
model/tiny-yolov3-coco/        # make export : tiny-yolov2-voc, tiny-yolov3-coco
├── manifest.json              # description du réseau (ci-dessous)
├── weights.bin                # poids int8, concaténés, alignés sur 64 octets
├── bias.bin                   # biais int32
├── requant.bin                # M0 int32 par canal de sortie
├── luts.bin                   # tables σ, e^t, e^{d·s} des têtes (uint32)
└── dumps/000001/ … 000003/    # 3 images de VOC2007 test
    ├── input.npy              # image prétraitée quantifiée (int8, NCHW)
    ├── L00.npy … L23.npy      # sortie entière de chaque couche (conv fusionnée : avant pool)
    └── detections.json        # boîtes après décodage + NMS (référence de post-traitement)
```

Extrait de `manifest.json` (schéma : [manifest.schema.json](manifest.schema.json), exemple
complet : `model/example_manifest.json`, sémantique : [conventions.md](conventions.md)) :

```json
{
  "format_version": 1,
  "network": "tiny-yolov3-voc",
  "classes": 20,
  "input": {"shape": [1, 3, 416, 416], "scale": 0.0078125, "buf": {"buf": "A", "offset": 0}},
  "anchors": [[10,14],[23,27],[37,58],[81,82],[135,169],[344,319]],
  "blobs": {"weights.bin": 8707264, "bias.bin": 13376, "requant.bin": 13376},
  "buffers": {"A": 519168, "B": 692224, "H13": 12675, "H26": 50700, "R": 259584, "S": 43264},
  "layers": [
    {"id": 0, "type": "conv", "k": 3, "s": 1, "pad": 1, "cin": 3, "cout": 16, "act": "leaky", "fused_pool": {"layer": 1, "k": 2, "s": 2}, "w_offset": 0, "b_offset": 0, "m0_offset": 0, "shift": 31, "in_scale": 0.0078125, "out_scale": 0.0625, "in": {"buf": "A", "offset": 0}, "out": {"buf": "B", "offset": 0}, "out_shape": [16, 208, 208]},
    {"id": 1, "type": "maxpool", "k": 2, "s": 2, "fused_into": 0, "out_shape": [16, 208, 208]},
    …
    {"id": 8, "type": "conv", "k": 3, "s": 1, "pad": 1, "cin": 128, "cout": 256, "act": "leaky", "fused_pool": {"layer": 9, "k": 2, "s": 2}, "prepool_out": {"buf": "R", "offset": 86528}, "w_offset": 97216, "b_offset": 960, "m0_offset": 960, "shift": 31, "in_scale": 0.0625, "out_scale": 0.0625, "in": {"buf": "A", "offset": 0}, "out": {"buf": "B", "offset": 0}, "out_shape": [256, 13, 13]},
    …
    {"id": 20, "type": "route", "from": [19, 8], "layout": "contiguous", "scale": 0.0625, "out": {"buf": "R", "offset": 0}, "out_shape": [384, 26, 26]}
  ]
}
```

Les `*_offset` pointent dans les blobs binaires. `in`/`out`/`prepool_out` placent chaque
tenseur dans un tampon DDR (ping-pong entre couches, placement contigu de 19 puis 8 pour la
route 20 du §10.3).

## 3. Moteur matériel : couche par couche

Choix du §10.1 : **un moteur unique** plutôt qu'un pipeline dataflow. Il comprend :

- une PE conv paramétrée (Tm × Tn MAC par cycle, nid de boucles du §10.2) ;
- un étage de sortie fusionné : biais, requantification M0/décalage, leaky 13/128, maxpool
  2×2 en stride 2 ou 1 ;
- upsample et route **par adressage**, sans copie (§10.3) ;
- un contrôleur AXI-Lite qui reçoit les paramètres de couche du driver ARM.

Le décodage et la NMS tournent d'abord sur l'ARM (§10.3), avec le code de
`cpp/golden` réutilisé dans `sw/postproc`.

Carte : Kria KV260 ([ADR 0003](adr/0003-choix-carte.md)). Les tailles de tuiles restent
des paramètres (`hls/configs/<carte>.tcl` → `-DACC_TM=…`), fixés par l'outil roofline (T5.6).

### Noyau `yolo_conv` (M6)

Un appel = une conv (+ maxpool fusionné), même parcours de tuiles que
`golden::conv_layer` (row → col → to → ti), donc mêmes entiers.

| Fichier | Rôle |
|---|---|
| `hls/kernels/layer_desc.hpp` | registres s_axilite (`LayerDesc`) : forme, 2 segments d'entrée `{offset, c, h, w, up}`, offsets de sortie et de paramètres — C++ simple, partagé avec le driver |
| `hls/kernels/conv_pe.cpp` | top, chargeurs, PE Tm × Tn (`PIPELINE II=1`), ping-pong `in_buf`/`w_buf` sur ti et `out_buf` sur les tuiles |
| `hls/kernels/output_stage.hpp` | biais + requantification + leaky + saturation, maxpool 2×2 s2/s1, carte avant pooling |
| `sw/driver/program.hpp` | arène DDR contiguë (tampons du manifest bout à bout), vues par couche, un `LayerDesc` par conv |

Interfaces : `m_axi` `gmem_in` / `gmem_out` (même arène d'activations, deux bundles pour
recouvrir chargement et stockage), `gmem_w` (weights.bin), `gmem_p` (bias.bin puis
requant.bin) ; adresses = indices dans ces tableaux. Upsample et route ne sont que des
segments d'entrée de la conv suivante : aucun appel du noyau.

Vérification : `hls/tb/tb_conv.cpp` (chaque conv seule) et `hls/tb/tb_net.cpp` (réseau via
le driver, chaque couche contrôlée juste après son appel, avant que les tampons ping-pong
A/B soient réécrits). C-sim avec g++ (`make csim-gcc`, en-têtes `ap_int` open source) ou
Vitis (`make csim`) ; synthèse, co-sim, export : `make hls-synth | hls-cosim | hls-export`,
rapport `make hls-report` → `results/hls_report.md`.

### Intégration SoC (M7)

Pile sur la KV260 : Ubuntu Kria, overlay chargé par `xmutil loadapp yolo`, registres du
noyau par `/dev/uioN` (`generic-uio`), mémoire contiguë par `/dev/udmabuf0` (u-dma-buf).

```
PS (A53) ── HPM0_FPD ─► s_axi_control @0xA000_0000 ─┐
          ◄─ pl_ps_irq0 ─ interrupt ─────────────────┤ yolo_conv (200 MHz, pl_clk0)
DDR ◄─ HP0_FPD ◄─ gmem_in, gmem_out ─────────────────┤
    ◄─ HP1_FPD ◄─ gmem_w, gmem_p ────────────────────┘
```

| Fichier | Rôle |
|---|---|
| `hw/boards/kv260/build.tcl` | block design, bitstream, `.xsa`, rapports ; échoue si le timing n'est pas tenu (`make vivado-build`) |
| `hw/boards/kv260/pl.dtsi`, `firmware.sh` | overlay (UIO + IRQ, u-dma-buf 32 Mo, horloge) et paquet `xmutil` (`make fpga-firmware`) |
| `sw/driver/regmap.hpp` | offsets s_axilite ; `LayerDesc` agrégé = 26 mots ; contrôlés contre l'en-tête Vitis (`make check-regmap`) |
| `sw/driver/device.hpp` | accès matériel : `uio` (carte) ou `sim` (PC : registres émulés → noyau C-sim) |
| `sw/driver/accel_driver.hpp` | `Accelerator` : un tampon contigu [arène \| poids \| paramètres], bornes de chaque `LayerDesc` vérifiées, une conv par `run_layer` |
| `sw/postproc/heads.hpp` | têtes lues en place dans l'arène → `postproc.hpp` (décodage + NMS sur l'ARM) |
| `sw/app/run_compare.cpp` | chaque couche en DDR == dump, puis détections == `detections.json` ; `--layer N` : une conv isolée |
| `sw/app/yolo_app.cpp` | démo image → boîtes, temps prétraitement / accélérateur / post-traitement |

Le backend `sim` passe par le même chemin que la carte (encodage des registres, adresses
physiques, ap_start / ap_done) : `make sw-sim` valide tout sauf le matériel lui-même.
Procédure carte : [hw/boards/kv260/README.md](../hw/boards/kv260/README.md).

### Mesures (M8)

| Fichier | Rôle |
|---|---|
| `sw/app/yolo_bench.cpp` | suite d'images (`--images` JPEG ou `--inputs` int8) : temps par étage et par image (pre, load, acc, post), moyenne et p99, détections JSONL au seuil de la mAP, puissance INA260 du SOM (`--power`) |
| `tools/perf_model.py` | modèle de cycles du noyau, égal aux compteurs C-sim (`make perf-model`) ; scénarios d'optimisation chiffrés |
| `tools/make_inputs.py` | entrées int8 de VOC2007 test (prétraitement du modèle entier), lues par la carte à la place des JPEG |
| `tools/bench_sim.sh` | stade FPGA en C-sim sur PC : paquets distribués aux cœurs, reprise (`make bench-sim`) |
| `tools/map_stades.py` | mAP flottant / entier / FPGA, égalité image par image → `results/map_stades.md` |
| `tools/bench_report.py` | `results/benchmarks.csv` (base + ce travail), `results/mesures.md` ; lit `times*.csv`, `utilization.rpt`, `power.rpt` |

Périmètres de mesure : [results/protocole.md](../results/protocole.md) ; comparaison et
pistes : [results/rapport.md](../results/rapport.md).

## 4. Golden model C++ (`cpp/golden/`, M5)

Le golden exécute le réseau comme le moteur matériel et sert de testbench au HLS :

| En-tête | Rôle |
|---|---|
| `model.hpp`, `json.hpp`, `npy.hpp` | lecture du manifest, des blobs et des dumps (T5.1) |
| `conv.hpp` | PE conv tuilée `conv_layer<Tiles<Tm, Tn, Tr, Tc>>` : tampons de taille fixe, étage de sortie biais + requantification + leaky + maxpool fusionné ; en-tête seul, sans allocation, partagé avec `hls/` (T5.2) |
| `engine.hpp` | réseau complet sur une arène « DDR » par tampon du manifest ; chaque couche a une vue de sortie (`InView`, ≤ 2 segments) : upsample = division d'adresse, route = segments bout à bout, aucune copie ; compteurs d'accès DDR (T5.3, T5.4) |
| `postproc.hpp` | seuil sur le logit, LUT, décodage, NMS ; STL seule, réutilisable dans `sw/postproc` (T5.5) |

`golden_run run <model> <input.npy> <out>` écrit `Lxx.npy` et `detections.json` ;
`make golden-check` les compare aux dumps Python avec `tools/compare_dumps.py`. Les écarts
assumés au pseudo-code du §10.2 (ordre des boucles de tuiles, tuiles avant pooling) sont
décrits dans [T5.2](tasks/M5-golden-cpp.md).

## 5. Flux de vérification

```
pytest (gradcheck)          ─► étage 1 correct
mAP flottant vs entier      ─► perte de quantification mesurée (§9)
tools/compare_dumps.py      ─► golden == dumps Python, couche par couche (make golden-check)
C-sim / co-sim HLS          ─► noyau == golden == dumps (make csim-gcc, make hls-cosim)
run_compare (sim puis uio)  ─► sortie DDR == golden == dumps (make sw-sim, puis sur carte)
yolo_bench + map_stades     ─► mAP aux trois stades (§11), FPGA == entier image par image (M8)
bench_report                ─► benchmarks.csv : mesures carte, ou projection perf_model (M8)
```
