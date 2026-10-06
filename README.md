# YOLO embarqué de zéro

[![ci](https://github.com/lrivals/EmbeddedComputervision/actions/workflows/ci.yml/badge.svg)](https://github.com/lrivals/EmbeddedComputervision/actions/workflows/ci.yml)

Réécriture à la main de **Tiny-YOLOv2** et **Tiny-YOLOv3** sans bibliothèque d'apprentissage
(NumPy), puis quantification entière bit-exacte et accélérateur **FPGA en Vitis HLS**.

La spécification technique complète est [yolo-embarque-de-zero.md](yolo-embarque-de-zero.md) ;
la documentation du projet y renvoie par numéro de section (§4.1, §9.3…).

## Chaîne de développement

```
python/  modèle flottant ──► modèle entier bit-exact ──export──► model/ (manifest + blobs + dumps)
                                                                     │
cpp/golden/  moteur C++ tuilé, bit-exact ◄───────────────────────────┤ comparaison couche par couche
hls/         accélérateur Vitis HLS (testbench = golden) ◄───────────┤ C-sim / co-sim
sw/          driver ARM + post-traitement ◄──────────────────────────┘ test sur carte
```

Détails : [docs/architecture.md](docs/architecture.md) · conventions :
[docs/conventions.md](docs/conventions.md) · décisions : [docs/adr/](docs/adr/).

## Plan de travail

Le travail est découpé en jalons M0 à M13 dans [docs/tasks/](docs/tasks/README.md). Chaque
tâche indique la section de la spec, ses dépendances, ses livrables et un critère
d'acceptation mesurable.

- **M0 à M8** : la chaîne de base, du NumPy aux mesures sur carte.
- **M9.1 à M9.4** : extensions de recherche (post-traitement matériel, REQ-YOLO, 4 bits,
  streaming).
- **[M10](docs/tasks/M10-ameliorations.md)** : améliorations d'ingénierie chiffrées par
  `tools/perf_model.py` (ports m_axi larges, requantification parallèle, pipeline ARM),
  streaming synthétisable, précision mixte, élagage, CI (`.github/workflows/ci.yml`).
- **[M11](docs/tasks/M11-jeux-de-donnees.md)** : jeux de données au-delà de VOC (COCO,
  ExDark, KITTI, VisDrone, CrowdHuman, FLIR), chacun choisi pour éprouver une partie de la
  chaîne.
- **[M12](docs/tasks/M12-profils-pc.md)** : profils de test sur PC (R/M/N), avec un critère
  et une décision par évaluation, avant de passer sur la carte.
- **[M13](docs/tasks/M13-figures.md)** : figures régénérables par `make figures`
  (architecture des modèles et de l'accélérateur, résultats, suivi du projet,
  réimplémentation de zéro expliquée par ses formules, réseaux couche par couche) ;
  galerie dans [results/figures.md](results/figures.md).

## Arborescence

| Dossier | Contenu |
|---|---|
| `python/yolo/` | couches, modèles, données, entraînement, inférence, quantification, export |
| `python/tests/` | tests pytest, dont les vérifications de gradients |
| `cpp/golden/` | golden model C++17 bit-exact (bibliothèque + tests Catch2) |
| `hls/` | noyaux Vitis HLS, testbenchs, scripts tcl, configurations par carte |
| `sw/` | code ARM : driver de l'accélérateur, post-traitement, application |
| `hw/boards/` | ressources des cartes candidates (yaml, roofline) — carte retenue : KV260 ([ADR 0003](docs/adr/0003-choix-carte.md)) ; `kv260/` : block design Vivado, overlay, [procédure carte](hw/boards/kv260/README.md) |
| `tools/` | scripts : comptage des MACs, roofline, comparaison des dumps |
| `model/` | modèles exportés (non versionnés) |
| `results/` | `benchmarks.csv` des mesures (format du §10.4), protocole, rapports mAP, roofline, rapport HLS, rapport comparatif (`rapport.md`) |
| `data/`, `weights/` | jeux de données et poids (non versionnés) |

## Démarrage rapide

```bash
pip install -e "python[dev,data]"   # numpy ; dev : pytest, ruff, jsonschema ; data : pillow
make test-py                        # tests NumPy
make test-cpp                       # build + tests du golden model (télécharge Catch2)
make golden-check                   # golden C++ == dumps Python de model/ (après make export)
make roofline                       # tuiles et ms/image par carte → results/roofline.md
make csim-gcc                       # C-sim du noyau HLS avec g++ : noyau == golden == dumps
make hls-cycles hls-report          # cycles par couche (C-sim) → results/hls_report.md
make csim hls-synth hls-cosim       # même chose dans Vitis HLS (BOARD=kv260 par défaut)
make sw-sim                         # driver ARM sur PC (registres émulés) : DDR == dumps, détections
make vivado-build fpga-firmware     # bitstream + overlay KV260 (Vivado), puis make sw-board sur la carte
make perf-model                     # modèle de cycles == C-sim, pistes d'optimisation (M8)
make m8-inputs m8-int bench-sim map-stades   # mAP flottant / entier / FPGA (C-sim) → results/map_stades.md
make bench-report                   # results/benchmarks.csv, results/mesures.md (protocole : results/protocole.md)
make ci-model ci                    # CI (T10.13) : export synthétique, sans VOC ni poids Darknet
make count-macs                     # tableaux du §3 (paramètres, MACs)
make figures                        # figures M13 → results/figures/, galerie results/figures.md
make test-durations                 # suite complète (slow compris) : durées des tests pour la figure T13.30
tools/get_voc.sh                    # PASCAL VOC 2007 + 2012 dans data/ (~3,6 Go)
python tools/voc_stats.py --show 10 # comptes par split + images annotées dans build/
tools/get_datasets.sh check         # jeux de M11 (COCO, KITTI… ; tools/get_datasets.sh coco)
tools/m11.sh coco-float             # profils de M11 (liste : tools/m11.sh)
```
