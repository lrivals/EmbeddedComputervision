# YOLO embarqué de zéro

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

Le travail est découpé en jalons M0 à M9 dans [docs/tasks/](docs/tasks/README.md). Chaque
tâche indique la section de la spec, ses dépendances, ses livrables et un critère
d'acceptation mesurable.

## Arborescence

| Dossier | Contenu |
|---|---|
| `python/yolo/` | couches, modèles, données, entraînement, inférence, quantification, export |
| `python/tests/` | tests pytest, dont les vérifications de gradients |
| `cpp/golden/` | golden model C++17 bit-exact (bibliothèque + tests Catch2) |
| `hls/` | noyaux Vitis HLS, testbenchs, scripts tcl, configurations par carte |
| `sw/` | code ARM : driver de l'accélérateur, post-traitement, application |
| `hw/boards/` | block designs Vivado (tcl), un par carte |
| `tools/` | scripts : comptage des MACs, roofline, comparaison des dumps |
| `model/` | modèles exportés (non versionnés) |
| `results/` | `benchmarks.csv` des mesures (format du §10.4) |
| `data/`, `weights/` | jeux de données et poids (non versionnés) |

## Démarrage rapide

```bash
pip install -e "python[dev]"   # numpy, pytest, ruff
make test-py                   # tests NumPy
make test-cpp                  # build + tests du golden model (télécharge Catch2)
```
