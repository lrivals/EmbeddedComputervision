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
`golden::conv_layer` (row → col → to → ti), donc mêmes entiers. La tuile (Tr = Tc = 13,
ou 14 pour les convs suivies d'un maxpool de stride 2) et le pliage de L00 (voies = canal ×
ligne du noyau) sont choisis par couche par le driver (M10, T10.4). Avec `n_calls > 0`, un
seul appel enchaîne toutes les convs à partir d'une table de descripteurs en DDR
(séquenceur, T10.7).

| Fichier | Rôle |
|---|---|
| `hls/kernels/accel_config.hpp` | configuration de compilation : tuiles, octets par mot des ports, trim, requant ×RQ, pliage, tuile des convs poolées — lue aussi par le driver et `tools/perf_model.py` |
| `hls/kernels/layer_desc.hpp` | registres s_axilite (`LayerDesc`, 31 mots) : forme, 2 segments d'entrée `{offset, c, h, w, up}`, offsets de sortie et de paramètres, tuile et pliage de la couche — C++ simple, partagé avec le driver |
| `hls/kernels/weight_layout.hpp` | poids dans l'ordre des tuiles : un bloc contigu par (to, ti), réordonné une fois par le driver (T10.1) |
| `hls/kernels/conv_pe.cpp` | top, séquenceur, chargeurs en mots de 64 bits ligne par ligne, PE Tm × Tn (`PIPELINE II=1`), ping-pong `in_buf`/`w_buf` sur ti et `out_buf` sur les tuiles |
| `hls/kernels/output_stage.hpp` | biais + requantification (8 canaux par cycle) + leaky + saturation, maxpool 2×2 s2/s1, carte avant pooling, écritures en mots |
| `sw/driver/program.hpp` | arène DDR contiguë (tampons du manifest bout à bout, plus une marge de lecture), vues par couche, un `LayerDesc` par conv, poids réordonnés, table du séquenceur |

Interfaces : `m_axi` `gmem_in` / `gmem_out` (même arène d'activations, deux bundles pour
recouvrir chargement et stockage), `gmem_w` (poids dans l'ordre des tuiles), en mots de
64 bits ; `gmem_p` (bias.bin puis requant.bin, et la table des descripteurs) ; adresses =
indices d'octet dans ces tableaux. Upsample et route ne sont que des
segments d'entrée de la conv suivante : aucun appel du noyau.

Vérification : `hls/tb/tb_conv.cpp` (chaque conv seule) et `hls/tb/tb_net.cpp` (réseau via
le driver, chaque couche contrôlée juste après son appel, avant que les tampons ping-pong
A/B soient réécrits). C-sim avec g++ (`make csim-gcc`, en-têtes `ap_int` open source) ou
Vitis (`make csim`) ; synthèse, co-sim, export : `make hls-synth | hls-cosim | hls-export`,
rapport `make hls-report` → `results/hls_report.md`.

### Architecture streaming (M9.4, T10.8, T10.9)

Seconde famille du §10.1, à côté du moteur unique : `yolo_stream` (hls/stream/) enchaîne un
étage matériel par conv de Tiny-YOLOv2 sous DATAFLOW, repliements PE × SIMD du plan de
`tools/stream_model.py` (KV260). Les poids des étages sur la puce sont une ROM générée
depuis l'export (`tools/gen_stream_rom.py`, `-DSTREAM_ROM`) ; L12 et L13 gardent leur carte
d'entrée et lisent leurs poids en DDR une fois par image (ordre PE extérieur, L12 → L13 en
CHW). Entrée et tête en AXI-Stream d'un octet (TLAST), apportées par un AXI DMA en mode
direct.

| Fichier | Rôle |
|---|---|
| `hls/stream/conv_stage.hpp` | étage conv : line buffer ou carte entière (FRAME, PE extérieur), poids en ROM `[og][ig][i][j][pe·SIMD + s]` ou par pointeur, maxpool fusionné en flux |
| `hls/stream/yolo_stream.hpp` | tables des étages (PE, SIMD, FRAME, CHW, profondeurs des FIFO == `stream_model.fifo_depths`) |
| `hls/stream/stream_desc.hpp` | registres `StreamDesc` (45 mots), partagés avec le driver |
| `sw/driver/stream_driver.hpp` | `StreamAccelerator` : entrée HWC, ap_start, S2MM puis MM2S, attente de la fin du S2MM, tête HWC → CHW |
| `sw/driver/stream_regmap.hpp` | offsets de `yolo_stream` (à vérifier contre l'en-tête Vitis) et de l'AXI DMA (PG021) |

Vérification : `tb_stream` et `tb_stream_rom` (chaque étage == dumps, cycles == modèle, un
seul TLAST), `yolo_bench --engine stream` en backend sim (DMA émulé, détections == dumps).
Synthèse : `make hls-synth-stream | hls-cosim-stream | hls-export-stream`, block design
`make vivado-build ENGINE=stream` (overlay `pl_stream.dtsi`), non exécutés (Vitis absent).

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
| `sw/driver/regmap.hpp` | offsets s_axilite ; `LayerDesc` agrégé = 31 mots (`qmax` en T9.3, `tr`, `tc`, `fold` en T10.4, `wbits` en T10.10), puis `DESCS` et `N_CALLS` (séquenceur, T10.7) ; contrôlés contre l'en-tête Vitis (`make check-regmap`) |
| `sw/driver/device.hpp` | accès matériel : `uio` (carte ; `--cached` : u-dma-buf caché et synchronisations par plage, T10.6) ou `sim` (PC : registres émulés → noyau C-sim) |
| `sw/driver/accel_driver.hpp` | `Accelerator` : un tampon contigu [arène(s) \| poids \| paramètres \| descripteurs], bornes de chaque `LayerDesc` vérifiées ; une conv par `run_layer`, ou toute la passe par `run_all` (une IRQ) ; deux arènes pour le pipeline inter-images (T10.5) |
| `sw/postproc/heads.hpp` | têtes lues en place dans l'arène → `postproc.hpp` (décodage + NMS sur l'ARM) |
| `sw/app/run_compare.cpp` | chaque couche en DDR == dump, puis détections == `detections.json` ; `--layer N` : une conv isolée |
| `sw/app/yolo_app.cpp` | démo image → boîtes, temps prétraitement / accélérateur / post-traitement ; `--pipeline` |
| `sw/app/yolo_bench.cpp` | mesures par étage (pre, load, acc, post, temps ARM) sur une suite d'images ; `--pipeline` : deux threads, deux arènes ; `--engine stream` : architecture streaming (T10.9) |

Le backend `sim` passe par le même chemin que la carte (encodage des registres, adresses
physiques, ap_start / ap_done) : `make sw-sim` valide tout sauf le matériel lui-même.
Procédure carte : [hw/boards/kv260/README.md](../hw/boards/kv260/README.md).

### Mesures (M8)

| Fichier | Rôle |
|---|---|
| `sw/app/yolo_bench.cpp` | suite d'images (`--images` JPEG ou `--inputs` int8) : temps par étage et par image (pre, load, acc, post), moyenne et p99, détections JSONL au seuil de la mAP, puissance INA260 du SOM (`--power`) |
| `tools/perf_model.py` | modèle de cycles du noyau, égal aux compteurs C-sim (`make perf-model`) ; découpage en tuiles (`tile_grid`) ; scénarios d'optimisation chiffrés ; `--manifest` : cycles d'un export quelconque (élagué, précision mixte) |
| `tools/mixed_precision.py` | précision mixte par couche (T10.10) : sensibilité, front glouton, mAP complète de la configuration retenue → `results/precision_mixte.md` |
| `tools/prune.py`, `tools/prune_study.sh` | élagage structuré par norme de filtre (T10.11), puis affinage, calibration, mAP et cycles par taux → `results/elagage.md` |
| `tools/make_ci_model.py` | export synthétique de `model/` pour la CI (T10.13), sans VOC ni poids Darknet |
| `tools/make_inputs.py` | entrées int8 de VOC2007 test (prétraitement du modèle entier), lues par la carte à la place des JPEG |
| `tools/bench_sim.sh` | stade FPGA en C-sim sur PC : paquets distribués aux cœurs, reprise (`make bench-sim`) |
| `tools/map_stades.py` | mAP flottant / entier / FPGA, égalité image par image → `results/map_stades.md` |
| `tools/bench_report.py` | `results/benchmarks.csv` (base + ce travail), `results/mesures.md` ; lit `times*.csv`, `utilization.rpt`, `power.rpt` |
| `tools/figures/` | figures M13 (`make figures`) : un module par famille (modèles, matériel, résultats, projet, maths, réseaux), lues dans les sources ci-dessus → `results/figures/`, galerie `results/figures.md` |

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
