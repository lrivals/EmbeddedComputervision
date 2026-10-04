# M0 — Infrastructure

Objectif : un dépôt outillé où `make test` passe, et des chiffres de référence (paramètres,
MACs) qui confirment la lecture des architectures du §3.

### [x] T0.1 — Dépôt, arborescence, README
- **Spec** : — · **Dépend de** : — · **Taille** : S
- **Livrables** : `git init`, arborescence de [architecture.md](../architecture.md),
  `README.md`, `.gitignore`, `Makefile`
- **Acceptation** : `git status` propre après le premier commit ; `data/`, `weights/`,
  `model/` et `build/` ignorés

### [x] T0.2 — Environnement Python
- **Spec** : — · **Dépend de** : T0.1 · **Taille** : S
- **Livrables** : `python/pyproject.toml` (numpy ; dev : pytest, ruff), installation
  éditable
- **Acceptation** : `pip install -e "python[dev]"` puis `make test-py` et `make lint`
  passent
- **Notes** : aucune bibliothèque d'apprentissage ([ADR 0001](../adr/0001-numpy-pur.md))

### [x] T0.3 — Toolchain C++
- **Spec** : — · **Dépend de** : T0.1 · **Taille** : S
- **Livrables** : `cpp/golden/CMakeLists.txt`, Catch2 via FetchContent, test de fumée
- **Acceptation** : `make test-cpp` compile et passe

### [x] T0.4 — Conventions et format d'échange
- **Spec** : §0 (notations), §10.5 · **Dépend de** : T0.1 · **Taille** : M
- **Livrables** : [conventions.md](../conventions.md) finalisé ;
  `model/example_manifest.json` (Tiny-YOLOv3 complet, valeurs fictives) ; schéma JSON
  `docs/manifest.schema.json`
- **Acceptation** : le manifest d'exemple décrit les 24 couches du §3.2 et passe la
  validation du schéma
- **Notes** : prévoir dès maintenant le placement contigu de la route 20 (couches 19 et 8)
  et le maxpool fusionné, pour ne pas changer le format plus tard

### [x] T0.5 — Comptage des paramètres et des MACs
- **Spec** : §3.1, §3.2 · **Dépend de** : T0.2 · **Taille** : S
- **Livrables** : `tools/count_macs.py` (lit une description de couches, imprime le tableau
  du §3)
- **Acceptation** : Tiny-YOLOv2 VOC = **15,86 M** paramètres / **3,49 G** MACs ;
  Tiny-YOLOv3 VOC = **8,71 M** / **2,74 G** ; chaque ligne égale à celle des tableaux
- **Notes** : BN = 2 paramètres par canal, pas de biais de conv avant BN (§4.2). La ligne
  « 0–11 » du §3.2 affichait 1 071,8 M (somme de valeurs arrondies) ; corrigée en 1 071,6 M
  (1 071 562 752 MACs), les totaux sont inchangés.

### [x] T0.6 — Jeu de données PASCAL VOC
- **Spec** : §1, §8.3 · **Dépend de** : T0.2 · **Taille** : M
- **Livrables** : `python/yolo/data/voc.py` (parsing XML → boîtes normalisées),
  `tools/get_voc.sh` (VOC2007 + 2012 trainval, VOC2007 test)
- **Acceptation** : nombre d'images et d'objets par split conforme aux chiffres officiels ;
  10 images tirées au hasard affichées avec leurs boîtes correctes
- **Notes** : garder le drapeau `difficult`, ignoré par l'évaluation VOC
