# M13 — Figures (optionnel)

Objectif : expliquer le projet en images. Les chiffres existent déjà dans `results/*.md`,
[benchmarks.csv](../../results/benchmarks.csv), `build/` et `tools/perf_model.py`, mais une
seule famille de figures est produite aujourd'hui : les roofline de `tools/roofline.py`
(`results/roofline_*.png`). M13 ajoute des figures sur cinq axes :

- l'**architecture** des modèles et de l'accélérateur (sections B, C) ;
- les **résultats** des tests et des mesures (D) ;
- le **développement** du projet (E) ;
- la **réimplémentation de zéro** : chaque brique essentielle reprogrammée en NumPy pur,
  sans PyTorch, OpenCV, scikit-learn ni devkit, expliquée par ses fonctions mathématiques (F) ;
- l'**architecture des réseaux en détail**, couche par couche et tenseur par tenseur (G).

Chaque figure est **régénérable par une commande** et lit ses données dans un fichier
versionné ou produit par un outil existant. Aucun chiffre n'est recopié à la main.

## Conventions communes

### Outillage

- **Paquet** : `tools/figures/`. Il contient un module par famille (`modeles.py`,
  `materiel.py`, `resultats.py`, `projet.py`, `maths.py`, `reseaux.py`) et un `style.py`
  commun.
- **Commandes** : `python -m tools.figures <nom|famille|all>`, et `make figures` qui
  régénère tout ce dont les données sont présentes. Une figure dont la source manque est
  sautée avec un message ; ce n'est pas une erreur.
- **Import de matplotlib** : comme dans `tools/roofline.py`, `matplotlib.use("Agg")` et
  import paresseux dans la fonction de tracé. Le paquet `yolo` reste en NumPy pur
  ([ADR 0001](../adr/0001-numpy-pur.md)) : matplotlib n'est jamais importé depuis
  `python/yolo/`.
- **Dépendances** : extra `plots = ["matplotlib>=3.7"]` dans `python/pyproject.toml`. Les
  schémas se font en Graphviz (`dot`), ou en mermaid quand ils restent dans un `.md`.
- **Style** (`style.py`) :
  - une palette catégorielle sûre pour les daltoniens ;
  - une couleur fixe par stade : flottant, entier, C-sim, carte ;
  - une couleur fixe par réseau : v2, v3 ;
  - largeur 1 colonne (6,4 in) ou pleine page (10 in) ;
  - enregistrement en PNG 150 dpi et en SVG.

### Sorties

- **Fichiers** : `results/figures/<famille>/<nom>.png` et `.svg`.
- **Index** : `results/figures.md`, une galerie avec pour chaque figure l'image, une phrase
  d'explication, la commande et la source des données.
- **Renvois** : chaque `.md` de résultat concerné renvoie aussi à sa figure (par exemple
  `map_stades.md` → T13.13).

### Règles

- **Projection ou mesure** : un chiffre qui n'a pas été mesuré sur la carte est marqué sur
  la figure (« projection C-sim », « estimation », « à mesurer »), avec un marqueur ou un
  trait distinct. Une figure ne mélange jamais projection et mesure sans légende.
- **Sous-ensembles** : les règles de [M12](M12-profils-pc.md#règles) s'appliquent. Une figure
  publiée dans `results/` montre des mAP sur les 4 952 images. Une figure sur `--subset`
  reste dans `build/figures/`.
- **Coût** : les paliers R/M/N de M12 s'appliquent. `make figures` ne lance que des
  générateurs R. Ceux qui demandent une évaluation (M ou N) lisent les sorties déjà
  présentes dans `build/`.
- **Acceptation commune** (toutes les tâches) :
  - la figure est produite par `make figures` ;
  - elle est référencée dans `results/figures.md` ;
  - ses valeurs sont égales à celles de la source (un test relit la source) ;
  - elle reste lisible en niveaux de gris.
- **Fiche mathématique** (sections F et G) : chaque planche porte, en plus de la figure :
  - les **équations** exactes, en mathtext matplotlib, recopiées de la docstring du module
    (une seule source : la docstring) ;
  - le **chemin** `module:fonction` du code tracé ;
  - la **bibliothèque évitée** (par exemple `torch.nn.Conv2d`, `cv2.resize`,
    `torchvision.ops.nms`, `sklearn.cluster.KMeans`) ;
  - le **test** qui vérifie l'implémentation (`python/tests/test_*.py`).

  Les courbes sont calculées **en appelant le code du dépôt** (jamais une réécriture dans
  le générateur), sur des entrées jouets à graine fixe : la figure prouve ce que fait le code.

---

## A. Infrastructure

### [ ] T13.0 — Paquet `tools/figures/` et galerie
- **Spec** : — · **Dépend de** : — · **Taille** : S
- **Livrables** :
  - `tools/figures/__init__.py` et `__main__.py`, qui tiennent le registre nom → générateur ;
  - `style.py` ;
  - la cible `make figures` ;
  - `results/figures.md` ;
  - `python/tests/test_figures.py`.
- **Acceptation** :
  - `make figures` tourne sans données (tout est sauté proprement) ;
  - le test de fumée appelle chaque générateur sur des données jouets, dans un répertoire
    temporaire ;
  - le test est marqué `skip` si matplotlib est absent.
- **Notes** : un générateur a la forme `fn(out_dir) -> list[Path]`. La galerie
  `results/figures.md` est réécrite à partir du registre, ce qui évite les liens morts.

## B. Architecture des modèles

### [ ] T13.1 — Schéma couche par couche de Tiny-YOLOv2 et Tiny-YOLOv3
- **Spec** : §3 · **Dépend de** : T1.7 · **Taille** : M
- **Livrables** : `figures/modeles/graphe_<net>.{svg,png}`, généré en Graphviz depuis
  `python/yolo/models/cfg.py` et `graph.py`
- **Acceptation** : un nœud par couche du cfg. Chaque nœud porte :
  - le type de couche (conv, maxpool, route, upsample, yolo) ;
  - K et le stride ;
  - la forme de sortie C×H×W.

  En v3, la route et l'upsample relient les deux têtes (13×13 et 26×26).
- **Notes** :
  - couleur par type de couche ;
  - épaisseur des nœuds proportionnelle aux MACs (T13.2) ;
  - variante « compacte » : un bloc par étage de résolution.

### [ ] T13.2 — Profil par couche : MACs, paramètres, activations
- **Spec** : §3 · **Dépend de** : T0.5 · **Taille** : S
- **Livrables** : `figures/modeles/profil_couches.png`, en trois panneaux à axe x commun
  (indice de couche) :
  - MACs ;
  - nombre de paramètres ;
  - taille de la sortie en octets int8.
- **Acceptation** :
  - les données viennent de `tools/count_macs.py` (`net_from_manifest`) sur
    `model/tiny-yolov2-voc` et `model/tiny-yolov3-coco` ;
  - le total des MACs est égal au §3 (3,486 G et 2,782 G) ;
  - v2 et v3 sont superposés.
- **Notes** : ajouter le cumul de MACs en courbe secondaire. Il montre que les couches
  13×13 portent l'essentiel des poids, et les premières couches l'essentiel des
  activations.

### [ ] T13.3 — Empreinte mémoire face au budget de la carte
- **Spec** : §10.2 · **Dépend de** : T13.2, T5.6 · **Taille** : S
- **Livrables** : `figures/modeles/memoire.png`, en barres empilées par couche (poids int8,
  biais int32, activations entrée et sortie) avec deux lignes de référence :
  - la BRAM du noyau (112 BRAM18 selon le modèle roofline) ;
  - la taille de l'arène DDR du driver.
- **Acceptation** : la taille de l'arène est égale à celle calculée par `sw/driver/program.cpp`
  (lue dans le journal de `yolo_bench` ou recalculée en Python).
- **Notes** : la figure justifie le moteur couche par couche, puisque rien ne tient en BRAM
  d'un seul coup. Elle sert aussi de point de départ à T13.11 (streaming).

### [ ] T13.4 — Ancres : boîtes VOC et k-means
- **Spec** : §5.2 · **Dépend de** : T2.2 · **Taille** : S
- **Livrables** :
  - `figures/modeles/ancres.png` : nuage (w, h) des 40 058 boîtes VOC07+12 (hexbin), avec
    les ancres k-means (k = 5, 6) et les ancres Darknet en surimpression ;
  - `ancres_k.png` : IoU moyenne en fonction de k, de 1 à 12.
- **Acceptation** : les ancres et les IoU affichées sont égales à
  [anchors.md](../../results/anchors.md) (0,6113 et 0,6307).
- **Notes** :
  - réutiliser les fonctions de `tools/kmeans_anchors.py` (ne pas dupliquer le k-means) ;
  - la courbe en k est un palier M, car elle relance un k-means par valeur ;
  - nécessite `data/VOCdevkit`.

### [ ] T13.5 — Sortie YOLO expliquée sur une image
- **Spec** : §5.1, §8.1 · **Dépend de** : T3.1 · **Taille** : M
- **Livrables** : `figures/modeles/sortie_yolo.png`, une planche en quatre vues :
  - l'image et la grille 13×13 ;
  - les cellules responsables et leurs 5 ancres ;
  - toutes les boîtes décodées au-dessus de `--conf`, avant NMS ;
  - les boîtes après NMS.
- **Acceptation** : les boîtes finales sont égales à celles de `tools/detect.py` sur la même
  image.
- **Notes** : passer par `python/yolo/infer/decode.py` et `nms.py`. En v3, une seconde
  planche montre les deux échelles.

### [ ] T13.6 — Planche d'augmentations
- **Spec** : §7.1 · **Dépend de** : T2.6 · **Taille** : S
- **Livrables** : `figures/modeles/augmentations.png`, une grille image d'origine →
  augmentations avec les boîtes cibles
- **Acceptation** : graine fixe, et mêmes tirages que `tools/show_augment.py`
- **Notes** : reprendre `tools/show_augment.py` (sorties dans `build/augment_samples`) au
  lieu de le réécrire.

### [ ] T13.7 — Statistiques du jeu VOC
- **Spec** : §8.3 · **Dépend de** : T2.6 · **Taille** : S
- **Livrables** : `figures/modeles/voc_stats.png`, en trois panneaux :
  - nombre d'objets par classe, en trainval et en test ;
  - histogramme des aires de boîtes (petits, moyens, grands) ;
  - nombre d'objets par image.
- **Acceptation** : les totaux sont égaux à la sortie de `tools/voc_stats.py`
- **Notes** : utile pour lire T13.13. Les classes rares ou à petits objets (bottle,
  pottedplant) ont les AP les plus faibles.

## C. Architecture matérielle

### [ ] T13.8 — Schéma du SoC
- **Spec** : §10.1 · **Dépend de** : T7.1 · **Taille** : S
- **Livrables** : `figures/materiel/soc.svg`, un schéma blocs avec :
  - le PS : ARM A53, driver `sw/driver/`, application `yolo_app` ;
  - l'interface AXI-Lite (registres de `sw/driver/regmap.hpp`) ;
  - le PL (noyau `accel`) ;
  - les ports m_axi vers la DDR ;
  - le découpage de l'arène (entrée, activations, poids, sortie).
- **Acceptation** : les noms des registres et des ports sont ceux de `regmap.hpp` et de
  `hls/kernels/accel.hpp`
- **Notes** : variantes `sim` et `uio` de `device.hpp`, sous forme de deux encadrés sur le
  même schéma.

### [ ] T13.9 — Schéma du moteur unique
- **Spec** : §10.2, §10.3 · **Dépend de** : T6.2, T10.1, T10.2 · **Taille** : M
- **Livrables** : `figures/materiel/moteur.svg`, qui montre :
  - les chargeurs `in_buf` et `w_buf` en ping-pong (mots de 64 bits) ;
  - le réseau Tm × Tn de MACs (32 × 24) ;
  - l'accumulateur 32 bits ;
  - l'étage de sortie `store_tile` (requantification ×8, LeakyReLU, maxpool) ;
  - l'écriture en mots.
- **Acceptation** : chaque bloc porte le nom de la fonction ou du tampon dans
  `hls/kernels/conv_pe.cpp` et `output_stage.hpp`. Les constantes affichées sont lues dans
  `accel_config.hpp`.
- **Notes** : ajouter un chronogramme du recouvrement chargement / calcul / stockage sur
  3 tuiles. C'est la vue « ping-pong » de [hls_report.md](../../results/hls_report.md).

### [ ] T13.10 — Tuilage illustré
- **Spec** : §10.2 · **Dépend de** : T6.2, T10.3 · **Taille** : S
- **Livrables** : `figures/materiel/tuilage.png`, un cube C×H×W d'une couche découpé en
  tuiles Tm, Tn, Tr, Tc, avec deux cas particuliers :
  - **L00** : cin = 3 ; pliage et trim ;
  - **L13** : 13×13, où une seule tuile spatiale couvre la carte.
- **Acceptation** : le nombre de tuiles est égal à `layer_tiling` de `tools/perf_model.py`
- **Notes** : montrer les voies inutiles (87 % de L00 sans trim) en hachuré.

### [ ] T13.11 — Streaming face au moteur unique
- **Spec** : §10.1 · **Dépend de** : T9.4.x · **Taille** : M
- **Livrables** :
  - `figures/materiel/streaming.svg` : un étage par conv, line buffers, maxpool fusionné ;
  - un chronogramme comparé des deux architectures sur une image.
- **Acceptation** : les profondeurs de line buffer et les ressources par étage viennent de
  `tools/stream_model.py`
- **Notes** : à mettre en regard de [streaming.md](../../results/streaming.md). Les
  chiffres sont des estimations, à marquer comme tels.

### [ ] T13.12 — Chaîne de vérification bit-exact
- **Spec** : §10.5 · **Dépend de** : M5, M6, M8 · **Taille** : S
- **Livrables** : `figures/materiel/chaine_verif.svg`. Les stades sont :
  - NumPy flottant ;
  - entier Python (`IntNetwork`) ;
  - golden C++ ;
  - C-sim du noyau ;
  - driver en backend sim ;
  - carte.

  Chaque flèche porte son critère et son état (vert = vérifié, gris = à faire). Exemples de
  critères : « 0 écart à l'octet, 3 dumps » ; « 4 952 / 4 952 images identiques ».
- **Acceptation** : les états sont lus dans le tableau de suivi du
  [README](README.md#suivi), pas codés en dur
- **Notes** : c'est la figure qui résume le mieux le projet ; la placer en tête de
  `results/figures.md`.

## D. Résultats

### [ ] T13.13 — mAP aux trois stades et AP par classe
- **Spec** : §8.3 · **Dépend de** : T8.2 · **Taille** : S
- **Livrables** : `figures/resultats/map_stades.png`, des barres groupées par classe
  (flottant, entier, C-sim, puis carte quand elle sera mesurée) avec la mAP globale en
  encart (56,30 / 55,66 / 55,66)
- **Acceptation** : les valeurs sont égales à [map_stades.md](../../results/map_stades.md)
- **Notes** :
  - ajouter `--json` à `tools/map_stades.py`, pour ne pas reparser le Markdown ;
  - variante en écart (entier − flottant) par classe, triée.

### [ ] T13.14 — Précision selon le format numérique
- **Spec** : §9 · **Dépend de** : T4.5, M9.2, M9.3 · **Taille** : S
- **Livrables** : `figures/resultats/map_formats.png`, une mAP par variante :
  - flottant ;
  - INT8 ;
  - puissances de 2 (PTQ, ADMM) ;
  - 4 bits (PTQ, QAT).

  Le coût matériel estimé (DSP ou bits de poids) est porté en second axe.
- **Acceptation** : les valeurs sont égales à `map_float.md`, `map_int8.md`, `req_yolo.md`
  et `quant_4bits.md`, ou aux JSON de `build/m9/map_*.json`
- **Notes** :
  - marquer « non convergé » sur l'ADMM tant que T9.2.3 est ouverte ;
  - une version en nuage (DSP en x, mAP en y) donne la frontière de Pareto.

### [ ] T13.15 — Courbes précision-rappel
- **Spec** : §8.3 · **Dépend de** : T3.3 · **Taille** : S
- **Livrables** : `figures/resultats/pr_<variante>.png`, une petite courbe par classe
  (grille 4 × 5) avec les 11 points VOC07
- **Acceptation** : l'aire sous chaque courbe redonne l'AP de la classe, calculée par
  `python/yolo/infer/metrics.py`
- **Notes** :
  - source : `build/eval/<variante>/comp4_det_test_*.txt` ;
  - comparer stretch et letterbox, et notre prétraitement face à celui de Darknet (les
    quatre variantes de `build/eval/`).

### [ ] T13.16 — Sensibilité par couche à la quantification
- **Spec** : §9.2 · **Dépend de** : T4.5, T12.3 · **Taille** : S
- **Livrables** : `figures/resultats/sensibilite.png`, une barre par couche. La barre donne
  la perte de mAP quand cette couche seule est quantifiée, ou quand elle seule reste en
  flottant ; il y a une variante par format (INT8, pow2, 4 bits).
- **Acceptation** : les valeurs viennent de `build/quant/eval-sens.log` ou des JSON de
  T12.3
- **Notes** : à rapprocher de T13.2 (couches lourdes et couches sensibles).

### [ ] T13.17 — Distributions et échelles de calibration
- **Spec** : §9.2 · **Dépend de** : T4.2 · **Taille** : S
- **Livrables** :
  - `figures/resultats/calibration.png` : par couche, l'histogramme (log) des activations
    avec le seuil de saturation retenu ;
  - échelles des poids par canal, en boîtes à moustaches par couche.
- **Acceptation** : les seuils sont égaux à `build/quant/<net>/calib.json` et à
  `results/calibration_*.md`
- **Notes** : montrer la part des valeurs saturées par couche.

### [ ] T13.18 — Erreur flottant → entier par couche
- **Spec** : §9 · **Dépend de** : T4.7, T5.3 · **Taille** : S
- **Livrables** : `figures/resultats/erreur_couches.png`, qui trace par couche :
  - le SNR (dB) de la sortie entière déquantifiée face au flottant ;
  - l'écart max.

  Un second panneau donne l'écart entier Python ↔ golden C++, qui doit être de 0 partout.
- **Acceptation** : 3 images de dumps (`model/<net>/dumps/`) ; les écarts sont égaux à ceux
  de `tools/compare_dumps.py`
- **Notes** : le panneau « 0 partout » est la preuve visuelle du bit-exact.

### [ ] T13.19 — Détections côte à côte
- **Spec** : §8 · **Dépend de** : T3.4, T8.2 · **Taille** : S
- **Livrables** : `figures/resultats/detections.png`, une grille de 4 images ×
  (vérité terrain, flottant, entier, C-sim)
- **Acceptation** : les boîtes viennent de `tools/detect.py` et des sorties de `yolo_bench`.
  Les colonnes entier et C-sim sont identiques.
- **Notes** : choisir une image où flottant et entier diffèrent, pour montrer l'effet de la
  quantification.

### [ ] T13.20 — Cycles par couche
- **Spec** : §10.2 · **Dépend de** : T6.3, T8.1 · **Taille** : S
- **Livrables** : `figures/resultats/cycles_couches.png`, des barres empilées par couche
  (load_in, load_w, compute, store), le temps recouvert étant marqué d'un point. Deux
  panneaux : v2 et v3.
- **Acceptation** : les valeurs sont égales à `build/hls/cycles_conv.csv` (`make
  hls-cycles`) et à `tools/perf_model.py --check`
- **Notes** : on y voit quel goulot domine chaque couche (mémoire au début, calcul à 13×13).

### [ ] T13.21 — Cascade des optimisations
- **Spec** : §10.3 · **Dépend de** : M10 · **Taille** : S
- **Livrables** : `figures/resultats/cascade_m10.png`, un waterfall en ms sur Tiny-YOLOv2.
  - Il part de 206,5 ms (noyau M6), puis passe par les ports 64 bits, le trim, la
    requantification ×8, le pliage de L00, etc.
  - Il porte les bornes horizontales : calcul (41,1 ms) et roofline KV260 (31,9 ms).
  - img/s en second axe.
- **Acceptation** : une marche par entrée de `SCENARIOS` de `tools/perf_model.py`, avec des
  valeurs égales à `scenario_table`
- **Notes** :
  - deux teintes : piste faite (code dans `hls/kernels/`) et piste projetée ;
  - mettre à jour à chaque tâche M10 fermée.

### [ ] T13.22 — Roofline par couche
- **Spec** : §10.2 · **Dépend de** : T5.6, T13.20 · **Taille** : S
- **Livrables** : `figures/resultats/roofline_couches_<carte>.png`. C'est le roofline de
  `tools/roofline.py`, avec :
  - un point par couche (intensité opérationnelle, GOPS projetés) ;
  - une flèche « avant / après M10 » ;
  - un point « mesuré » quand la carte sera disponible.
- **Acceptation** : le toit et la pente sont égaux à `results/roofline_<carte>.png`, et les
  points à T13.20
- **Notes** : étendre `tools/roofline.py` plutôt que le dupliquer.

### [ ] T13.23 — Comparaison à l'état de l'art
- **Spec** : §10.4 · **Dépend de** : T8.3 · **Taille** : S
- **Livrables** : `figures/resultats/etat_art.png`, en trois panneaux :
  - GOPS face à la puissance, avec des iso-GOPS/W ;
  - img/s face à la mAP ;
  - GOPS par DSP.
- **Acceptation** :
  - une marque par ligne de [benchmarks.csv](../../results/benchmarks.csv) ;
  - marqueur creux pour « projection » ;
  - lignes sans valeur omises du panneau, mais citées en légende.
- **Notes** : l'annotation donne l'auteur et l'année (colonne `travail`). Les colonnes vides
  (puissance de ce travail) restent « à mesurer ».

### [ ] T13.24 — Ressources face au budget KV260
- **Spec** : §10.1 · **Dépend de** : M9.2, M9.3, M9.4 · **Taille** : S
- **Livrables** : `figures/resultats/ressources.png`, des barres DSP / BRAM / LUT
  rapportées au budget XCK26 (1 248 DSP, etc.) pour quatre variantes :
  - moteur unique INT8 ;
  - pow2 ;
  - 4 bits ;
  - streaming.
- **Acceptation** :
  - les valeurs viennent de `tools/stream_model.py`, de `req_yolo.md` et de
    `quant_4bits.md` ;
  - les barres sont hachurées tant que ce sont des estimations ;
  - elles deviennent pleines avec `hls_report.md` après synthèse.
- **Notes** : à recouper avec T13.14 pour l'arbitrage précision / surface.

### [ ] T13.25 — Post-traitement matériel
- **Spec** : §10.3 · **Dépend de** : M9.1 · **Taille** : S
- **Livrables** : `figures/resultats/hw_postproc.png`, en deux panneaux :
  - la mAP avec et sans plafond de boîtes ;
  - le nombre de boîtes par image face au plafond.
- **Acceptation** : les valeurs sont égales à `build/m9/hwpp_map.json` et
  `hwpp_map_nocap.json`
- **Notes** : justifie la taille du plafond retenue dans [postproc_hw.md](../../results/postproc_hw.md).

### [ ] T13.26 — Courbes d'entraînement
- **Spec** : §7 · **Dépend de** : T2.7, M9.2, M9.3 · **Taille** : S
- **Livrables** : `figures/resultats/entrainement_<run>.png`, avec :
  - la perte totale et ses composantes (coordonnées, objectness, classes), lissées ;
  - le taux d'apprentissage en second axe ;
  - pour l'ADMM, les résidus primal et dual.
- **Acceptation** : les données sont lues dans `loss.csv` et `admm.csv` du répertoire
  `--out` de `tools/train.py` (`build/train/…`)
- **Notes** :
  - superposer plusieurs runs (QAT 300 face à 600 itérations, CPU face à GPU de T12.11) ;
  - utile au diagnostic de T12.6 (ADMM non convergé).

## E. Développement du projet

### [ ] T13.27 — Chronologie des jalons
- **Spec** : — · **Dépend de** : — · **Taille** : S
- **Livrables** : `figures/projet/chronologie.png`, un Gantt des jalons M0 → M9, reconstruit
  à partir des commits « Mx : » de `git log --date=iso`, avec les jalons de documentation en
  pointillé
- **Acceptation** : une barre par jalon. Ses dates sont égales à celles des commits.
- **Notes** :
  - granularité à l'heure : beaucoup de jalons tombent le même jour ;
  - les jalons non commencés (M10 à M13) sont en gris.

### [ ] T13.28 — Avancement par jalon
- **Spec** : — · **Dépend de** : — · **Taille** : S
- **Livrables** : `figures/projet/avancement.png`, des barres horizontales par jalon (tâches
  faites, vérifiées sur PC, en attente de la carte, ouvertes)
- **Acceptation** :
  - les comptes sont parsés depuis les titres `### [x]` et `### [ ]` des `docs/tasks/M*.md` ;
  - un test vérifie qu'ils sont égaux au tableau de suivi du [README](README.md#suivi), et
    signale les écarts.
- **Notes** : la catégorie « vérifiée sur PC, carte en attente » est lue dans les mots-clés
  des notes du suivi (C-sim, sim). Si la règle est trop fragile, ajouter une colonne au
  tableau de suivi.

### [ ] T13.29 — Volume de code au fil des commits
- **Spec** : — · **Dépend de** : — · **Taille** : S
- **Livrables** : `figures/projet/code.png`, des aires empilées (lignes par dossier :
  `python/`, `tools/`, `hls/`, `sw/`, `docs/`, `results/`) commit par commit
- **Acceptation** : les données viennent de `git log --numstat`. Le dernier point est égal
  au décompte actuel.
- **Notes** : variante par langage (Python, C++, CMake, Markdown).

### [ ] T13.30 — Tests et dépendances
- **Spec** : — · **Dépend de** : T0.3 · **Taille** : S
- **Livrables** :
  - `figures/projet/tests.png` : nombre de tests par module Python
    (`pytest --collect-only -q`, slow inclus) et par exécutable C++ (`ctest -N`), avec la
    durée relevée par `--durations` ;
  - `figures/projet/dependances.svg` : le graphe mermaid des jalons du README, rendu en
    image et coloré selon T13.28.
- **Acceptation** : le total des tests est égal à la collecte pytest + ctest
- **Notes** : le graphe reste défini une seule fois, dans le README ; la figure le parse.

## F. Réimplémentation de zéro : fonctions mathématiques

Toute la chaîne flottante et entière est écrite à la main en NumPy
([ADR 0001](../adr/0001-numpy-pur.md)), sans autodiff ni bibliothèque de vision. Chaque
planche de cette section explique une brique : **la formule**, **le code qui la calcule**,
et **la preuve** qu'il la calcule juste. Les planches suivent la règle « fiche
mathématique » (voir [Règles](#règles)). Module : `tools/figures/maths.py`, sorties dans
`results/figures/maths/`.

### [ ] T13.31 — Carte de la réimplémentation
- **Spec** : §4 à §9, §11 · **Dépend de** : M1 à M4 · **Taille** : S
- **Livrables** : `figures/maths/carte.svg`, un schéma en colonnes :
  - brique (convolution, BN, activations, pool, route/upsample, graphe et rétropropagation,
    perte, SGD, k-means, NMS, mAP, prétraitement, quantification, entier, LUT, formats) ;
  - module et fonction (`python/yolo/…`) ;
  - bibliothèque évitée (`torch.nn`, `torch.autograd`, `torch.optim`, `cv2`, `torchvision.ops`,
    `sklearn`, devkit VOC MATLAB, `onnx`, Vitis AI) ;
  - test qui la vérifie, et planche de cette section qui l'explique.
- **Acceptation** :
  - une ligne par module de `python/yolo/` (aucun oublié : le générateur parcourt le
    paquet) ;
  - chaque test cité existe ;
  - chaque ligne renvoie à une tâche T13.32 à T13.47.
- **Notes** : c'est la porte d'entrée de la section F ; la placer en tête de la galerie
  « maths ».

### [ ] T13.32 — Convolution : boucles et im2col
- **Spec** : §4.1 · **Dépend de** : T1.2 · **Taille** : M
- **Livrables** : `figures/maths/convolution.png`, en trois panneaux :
  - la formule y[f,i,j] = b_f + Σ_c Σ_u Σ_v W[f,c,u,v] · x[c, s·i+u−p, s·j+v−p], avec un
    patch 3×3 sur une carte 6×6 et le pas s ;
  - le dépliage im2col : chaque patch devient une colonne (`sliding_window_view`), les
    poids une matrice (F, C·k²), et la convolution un produit de matrices ; formes
    annotées pour L00 de Tiny-YOLOv2 ;
  - le temps de `conv_forward_naive` face à `conv_forward` selon la taille de la carte
    (échelle log).
- **Acceptation** :
  - les deux implémentations donnent la même sortie, écart max affiché (≤ 1e-12 en
    float64) ;
  - le temps est mesuré sur la machine, palier R (cartes ≤ 52×52 pour la version naïve).
- **Notes** :
  - code : `python/yolo/layers/conv.py` ; test : `test_conv.py` ;
  - ajouter la passe arrière : dW = dY · colᵀ, dX = col2im(Wᵀ · dY), avec le repli des
    colonnes (accumulation des recouvrements) dessiné.

### [ ] T13.33 — Rétropropagation écrite à la main
- **Spec** : §4, §11 · **Dépend de** : T1.1, T1.7 · **Taille** : M
- **Livrables** :
  - `figures/maths/retropropagation.svg` : le graphe d'une tranche de réseau (conv → BN →
    leaky → maxpool, plus une route et un upsample de v3) avec, sur chaque arête, la passe
    avant et la formule du gradient renvoyé. Les gradients d'une carte lue deux fois
    (route) s'additionnent, comme dans `Network.backward` ;
  - `figures/maths/gradcheck.png` : par couche, l'erreur relative
    ‖g_analytique − g_numérique‖ / max(‖g_a‖, ‖g_n‖) en log, avec la différence centrée
    (f(x+h) − f(x−h)) / 2h, h = 1e-6, et le seuil 1e-7 en trait.
- **Acceptation** :
  - les erreurs viennent de `python/yolo/testing/gradcheck.py` (`check_layer`,
    `check_grad`) en float64 ;
  - toutes sont sous 1e-7, sauf la ligne témoin « gradient faux » de
    `test_gradcheck.py::test_detects_wrong_gradient`, tracée en rouge au-dessus de 1e-5.
- **Notes** : code : `python/yolo/models/graph.py` et `python/yolo/layers/*.py` ; tests :
  `test_gradcheck.py`, `test_graph.py`.

### [ ] T13.34 — Activations, sigmoïde stable, BCE et softmax
- **Spec** : §4.3, §4.5, §6.2 · **Dépend de** : T1.4, T1.6, T2.4 · **Taille** : S
- **Livrables** : `figures/maths/activations.png`, en quatre panneaux :
  - leaky ReLU f(x) = max(x, 0,1·x) et sa dérivée (1 ou 0,1) ;
  - sigmoïde σ(t) = 1/(1 + e^{−t}) calculée en deux branches (e^{−t} pour t ≥ 0,
    e^{t}/(1 + e^{t}) sinon), face à la forme naïve qui déborde pour t ≪ 0 en float32 ;
    σ'(t) = σ(t)(1 − σ(t)) superposée ;
  - softplus et BCE sur logits : BCE(t, y) = softplus(t) − y·t, et son gradient σ(t) − y ;
  - softmax stable (soustraction du max) pour v2, sur un vecteur de 20 logits.
- **Acceptation** : courbes calculées par `layers/activations.py` (`leaky_forward`,
  `sigmoid`) et `train/loss.py` (`softplus`, `bce_logits`) ; aucun `inf` ni `nan` sur
  t ∈ [−100, 100] pour la version stable.
- **Notes** : tests : `test_activations.py`, `test_loss.py`.

### [ ] T13.35 — Batch normalization et sa fusion
- **Spec** : §4.2, §9.1 · **Dépend de** : T1.3, T4.1 · **Taille** : S
- **Livrables** : `figures/maths/batchnorm.png`, en trois panneaux :
  - normalisation par canal sur (N, H, W) : histogrammes avant et après
    x̂ = (x − μ_B)/√(σ²_B + ε), puis y = γ x̂ + β ;
  - moyennes glissantes μ ← 0,9 μ + 0,1 μ_B sur 100 itérations, face à la vraie moyenne ;
  - fusion dans la conv : s_f = γ_f / √(σ²_f + ε), W'_f = s_f W_f,
    b'_f = β_f + s_f (b_f − μ_f) ; nuage sortie (conv + BN) face à sortie (conv fusionnée).
- **Acceptation** : code `layers/batchnorm.py` (`bn_forward`) et `quant/fuse_bn.py`
  (`fuse_bn`) ; nuage sur la diagonale, écart max affiché.
- **Notes** :
  - tests : `test_batchnorm.py`, `test_fuse_bn.py` ;
  - la passe arrière de la BN (trois termes du gradient) est l'une des formules les plus
    longues du dépôt ; l'écrire en entier dans la planche.

### [ ] T13.36 — Maxpool, upsample et route
- **Spec** : §4.4, §4.5 · **Dépend de** : T1.5, T1.6 · **Taille** : S
- **Livrables** : `figures/maths/pool_upsample_route.png`, des grilles annotées de valeurs :
  - maxpool 2×2/2 sur une carte 4×4 : max de chaque fenêtre ; le gradient remonte
    **uniquement** vers l'argmax ;
  - maxpool 2×2/1 de la couche 11 : complétion d'une ligne et d'une colonne par réplication
    du bord, taille conservée (13×13 → 13×13) ;
  - upsample ×2 au plus proche voisin ; en arrière, somme des 4 gradients de chaque bloc ;
  - route : concaténation des canaux, et découpage du gradient.
- **Acceptation** : valeurs affichées calculées par `layers/pool.py`, `upsample.py`,
  `route.py`
- **Notes** : tests : `test_pool.py`, `test_upsample_route.py`.

### [ ] T13.37 — Boîtes, IoU et décodage
- **Spec** : §5.2, §8.1 · **Dépend de** : T2.1, T3.1 · **Taille** : S
- **Livrables** : `figures/maths/boites_decodage.png`, en trois panneaux :
  - IoU = aire(A ∩ B) / aire(A ∪ B), dessinée sur deux boîtes, et `iou_wh` (boîtes centrées,
    utilisée par les ancres) ;
  - décodage : b_x = (σ(t_x) + j)/S, b_y = (σ(t_y) + i)/S ; le centre reste dans sa cellule ;
  - b_w = p_w e^{t_w}, b_h = p_h e^{t_h} : taille de la boîte selon t_w pour les 5 ancres.
- **Acceptation** : code `infer/boxes.py` (`iou`, `iou_wh`, `cxcywh_to_xyxy`) et
  `infer/decode.py` (`decode_head`)
- **Notes** : tests : `test_boxes.py`, `test_decode.py`. Lien avec T13.5 (même décodage sur
  une vraie image).

### [ ] T13.38 — Cibles et perte YOLO
- **Spec** : §5.1, §6.2 · **Dépend de** : T2.3, T2.4, T2.5 · **Taille** : M
- **Livrables** : `figures/maths/perte.png`, en quatre panneaux :
  - encodage d'une vérité : cellule (i, j), ancre responsable (meilleure IoU de forme),
    cibles x* = g_x·S − j, t_w* = ln(g_w / p_w) (`build_targets`) ;
  - la perte complète, écrite terme par terme :
    L = λ_coord Σ_obj ω [(σ(t_x) − x*)² + (σ(t_y) − y*)² + (t_w − t_w*)² + (t_h − t_h*)²]
    + Σ_obj BCE(σ(t_o), 1) + Σ_noobj BCE(σ(t_o), 0) + Σ_obj L_cls ;
  - la surface du poids ω = 2 − g_w g_h (les petites boîtes comptent double) ;
  - le masque *ignore* : ancres non responsables dont la boîte prédite dépasse
    `ignore_thresh` d'IoU avec une vérité, sur une image.
- **Acceptation** :
  - code `data/targets.py` et `train/loss.py` (`yolo_loss`, `ignore_mask`) ;
  - un cinquième panneau donne la part de chaque terme (coord, obj, noobj, cls) sur un lot,
    et la somme est égale à `LossResult.total`.
- **Notes** : tests : `test_targets.py`, `test_loss.py`. Variante v2 : L_cls = −ln
  softmax(t)_{c*}.

### [ ] T13.39 — Optimiseur et taux d'apprentissage
- **Spec** : §7.1 · **Dépend de** : T2.7 · **Taille** : S
- **Livrables** : `figures/maths/optimisation.png`, en trois panneaux :
  - SGD avec momentum et weight decay, v ← μ v − lr (g + wd·p), p ← p + v (μ = 0,9,
    wd = 5e-4) : trajectoires sur une quadratique 2D mal conditionnée, avec et sans
    momentum ;
  - `lr_at` : montée lr·(it / burn_in)⁴ puis paliers, sur un entraînement complet ;
  - tailles multi-échelles (320 à 608, pas de 32, tirées toutes les 10 itérations) par
    `multiscale_size`.
- **Acceptation** : code `train/optim.py` (`SGD`), `train/schedule.py` (`lr_at`),
  `train/trainer.py` (`multiscale_size`)
- **Notes** : test : `test_train.py`. Relier aux courbes réelles de T13.26.

### [ ] T13.40 — k-means des ancres avec la distance 1 − IoU
- **Spec** : §5.2 · **Dépend de** : T2.2 · **Taille** : S
- **Livrables** : `figures/maths/kmeans.png`, en deux panneaux :
  - les mêmes boîtes regroupées avec la distance d = 1 − IoU(boîte, centre) et avec la
    distance euclidienne en (w, h) : l'euclidienne attire les centres vers les grandes
    boîtes ;
  - l'initialisation k-means++ (tirage ∝ d²) puis les itérations de mise à jour des
    centres, sur 300 itérations au plus.
- **Acceptation** : code `data/anchors.py` (`kmeans_anchors`, `_init_plusplus`,
  `mean_best_iou`) ; l'IoU moyenne affichée est égale à celle de `anchors.md` sur les
  données VOC, ou à celle du test sur les données jouets.
- **Notes** : test : `test_anchors.py`. Complète T13.4, qui montre le résultat sur VOC.

### [ ] T13.41 — NMS et mAP VOC
- **Spec** : §8.2, §8.3 · **Dépend de** : T3.2, T3.3 · **Taille** : M
- **Livrables** : `figures/maths/nms_map.png`, en trois panneaux :
  - NMS pas à pas sur une image : tri par score, garde de la meilleure, suppression des
    boîtes d'IoU > 0,45, par classe ;
  - appariement détections / vérités (IoU ≥ 0,5, une vérité ne compte qu'une fois,
    `difficult` ignorés), et la courbe précision-rappel qui en sort ;
  - AP VOC07 sur 11 points, AP = (1/11) Σ_{r ∈ {0, 0,1, …, 1}} max_{r̃ ≥ r} p(r̃), face à
    l'aire sous l'enveloppe (VOC2010+).
- **Acceptation** : code `infer/nms.py` (`nms`, `filter_and_nms`) et `infer/metrics.py`
  (`voc_ap`, `eval_class`) ; l'AP affichée est égale à celle de la réécriture du devkit
  `python/tests/voc_eval_ref.py`.
- **Notes** : tests : `test_nms.py`, `test_metrics.py`. Le devkit officiel est en MATLAB :
  l'égalité avec la référence est le point clé de la planche.

### [ ] T13.42 — Prétraitement et augmentations
- **Spec** : §1, §7.1 · **Dépend de** : T2.6, T3.4 · **Taille** : S
- **Livrables** : `figures/maths/pretraitement.png`, en quatre panneaux :
  - letterbox (facteur min(416/w, 416/h), bandes grises, boîtes recalées) face à stretch ;
  - redimensionnement bilinéaire de `resize_darknet`, avec les poids d'interpolation d'un
    pixel ;
  - aller-retour RGB → HSV → RGB, gains de saturation et d'exposition ;
  - matrice affine (échelle et translation jusqu'à 20 %, retournement) appliquée à l'image
    et aux boîtes, avec l'élimination des boîtes trop petites.
- **Acceptation** :
  - code `data/letterbox.py`, `infer/pipeline.py` (`resize_darknet`, `preprocess`),
    `data/augment.py` (`affine`, `transform_boxes`, `rgb_to_hsv`, `hsv_to_rgb`) ;
  - l'erreur de l'aller-retour HSV est affichée.
- **Notes** : test : `test_data_pipeline.py`. Pillow ne sert qu'à décoder le JPEG.

### [ ] T13.43 — Quantification symétrique et calibration
- **Spec** : §9.2 · **Dépend de** : T4.2 · **Taille** : S
- **Livrables** : `figures/maths/quantification.png`, en trois panneaux :
  - l'escalier q = clip(round_half_up(x / s), −127, 127), avec round_half_up(v) = ⌊v + ½⌋
    et l'erreur x − s·q en dents de scie ;
  - les échelles par canal s_w,f = max|W_f| / 127 d'une couche, face à une échelle unique
    par couche (erreur de chaque option) ;
  - le choix du seuil de clip d'une activation : MSE de quantification selon le seuil
    candidat, minimum marqué, taux de saturation en second axe.
- **Acceptation** : code `quant/quantize.py` (`round_half_up`, `quantize`,
  `weight_scales`) et `quant/calibrate.py` (`clip_candidates`, `quant_mse`, `clip_rate`,
  `choose_scales`)
- **Notes** : test : `test_quantize.py`. Complète T13.17 (échelles réelles du réseau).

### [ ] T13.44 — Arithmétique entière du matériel
- **Spec** : §9.3 · **Dépend de** : T4.3 · **Taille** : M
- **Livrables** : `figures/maths/entier.png`, en quatre panneaux :
  - la chaîne d'une sortie : acc = Σ q_x q_w + q_b (int32), y = (acc · M0 + 2ⁿ⁻¹) ≫ n,
    leaky, clip à [−127, 127] ;
  - le multiplicateur fixe : M0 = round(s_x s_w / s_y · 2ⁿ) < 2³¹, n ≤ 31 le plus grand
    possible ; erreur relative de M0 / 2ⁿ face à l'échelle flottante, par canal ;
  - la leaky entière : y si y > 0, sinon (13 y + 64) ≫ 7, soit une pente 13/128 ≈ 0,1016,
    face à 0,1 ;
  - l'histogramme des accumulateurs d'une couche face aux bornes int32 (marge en bits).
- **Acceptation** :
  - code `quant/int_layers.py` (`conv_acc`, `requantize`, `rshift_round`, `leaky_int`,
    `clip_q`) et `quant/quantize.py` (`requant_params`) ;
  - mêmes valeurs que `golden::requantize` et `golden::leaky_int`
    (`cpp/golden/include/golden/conv.hpp`).
- **Notes** : tests : `test_int_layers.py`, `test_golden_cpp.py`. C'est ce contrat que le
  golden C++ et le noyau HLS recopient à l'identique (T13.52).

### [ ] T13.45 — Sigmoïde et exponentielle par tables
- **Spec** : §9.4 · **Dépend de** : T4.4 · **Taille** : S
- **Livrables** : `figures/maths/lut.png`, en trois panneaux :
  - la table σ à 256 entrées (indice q + 128, valeurs Q16), face à σ(q·s) flottante ;
  - la table exp pour b_w = p_w e^{t_w} et la table de softmax (indice d + 255) ;
  - l'erreur : ≤ 2⁻¹⁷ aux points de la grille, et s/8 de bout en bout à cause du pas s
    (1/64 pour s = 1/8).
- **Acceptation** : code `quant/lut.py` (`sigmoid_lut`, `exp_lut`, `softmax_exp_lut`,
  `logit_threshold_q`) ; bornes d'erreur égales à celles de la docstring
- **Notes** :
  - test : `test_lut.py` ;
  - ajouter le seuil sans sigmoïde : σ(q s) > θ ⟺ q > logit(θ)/s, une seule comparaison
    entière.

### [ ] T13.46 — QAT, puissances de 2 et 4 bits
- **Spec** : §9.2 · **Dépend de** : M9.2, M9.3 · **Taille** : M
- **Livrables** : `figures/maths/basse_precision.png`, en quatre panneaux :
  - fake-quant : passe avant en escalier, passe arrière par estimateur straight-through
    (gradient 1 dans la plage, 0 hors de la plage) ;
  - niveaux équidistants face aux niveaux « mixed powers-of-two », et la multiplication
    par décalages et additions (`shift_add_mul`) ;
  - ADMM : W, Z = Π_S(W + U), U ← U + W − Z ; trajectoire d'un poids et résidus primal et
    dual ;
  - découpage 4 bits de l'entrée : x = 16·(x ≫ 4) + (x & 15), deux convolutions 4 bits.
- **Acceptation** : code `quant/fake_quant.py` (`fq_weight`, `fq_act_forward`,
  `fq_act_backward`), `quant/pow2.py` (`levels`, `project`, `shift_add_mul`),
  `train/admm.py` (`ADMM`), `quant/int_layers.py` (`split4`, `conv_acc_split4`)
- **Notes** : tests : `test_fake_quant.py`, `test_pow2.py`, `test_admm.py`. Les mAP qui
  en résultent sont en T13.14.

### [ ] T13.47 — Formats de fichiers lus et écrits à la main
- **Spec** : §7.1, §10.2 · **Dépend de** : T1.7, T1.9, T4.7 · **Taille** : S
- **Livrables** : `figures/maths/formats.svg`, en trois bandes :
  - un `.weights` Darknet octet par octet : en-tête (major, minor, revision, seen), puis
    par conv [β, γ, μ, σ²] ou [biais], puis les poids (F, C, k, k), en float32 ;
  - le parser `.cfg` : sections `[convolutional]`, `[maxpool]`, `[route]`, `[upsample]`,
    `[yolo]`/`[region]` → dicts de `models/specs.py` ;
  - l'arène exportée : décalage et taille de chaque tampon d'activation, poids et LUT,
    alignés.
- **Acceptation** : code `io/darknet_weights.py` (`read_header`, `load_darknet_weights`),
  `models/cfg.py` (`parse_cfg`), `io/export.py` (`layout`, `build_manifest`) ; décalages
  égaux au `manifest.json` de `model/tiny-yolov2-voc`
- **Notes** : tests : `test_darknet_weights.py`, `test_cfg.py`, `test_export.py`.

## G. Architecture des réseaux en détail

Complète T13.1 à T13.3 : ici, chaque figure décrit Tiny-YOLOv2 VOC et Tiny-YOLOv3 COCO
au niveau de la couche et du tenseur. Les données viennent de `python/yolo/models/specs.py`
(`infer_shapes`, `layer_cost`) appliqué aux `.cfg` de `python/yolo/models/cfg/`. Module :
`tools/figures/reseaux.py`, sorties dans `results/figures/reseaux/`.

### [ ] T13.48 — Fiche couche par couche
- **Spec** : §3.1, §3.2 · **Dépend de** : T1.8 · **Taille** : S
- **Livrables** : `figures/reseaux/fiche_<net>.png`, une figure-tableau, une ligne par
  couche :
  - indice et type ; k, stride, pad ; cin → cout ; H×W en entrée et en sortie ;
  - paramètres, MACs (et part du total en barre dans la cellule) ;
  - champ réceptif et pas cumulé (stride effectif) ;
  - activation et présence de BN.
- **Acceptation** : totaux égaux au §3 et à `tools/count_macs.py` ; formes égales à
  `infer_shapes`
- **Notes** : test : `test_tiny_yolo.py`. Même contenu exporté en CSV à côté de l'image.

### [ ] T13.49 — Champ réceptif et résolution
- **Spec** : §3 · **Dépend de** : T13.48 · **Taille** : S
- **Livrables** : `figures/reseaux/champ_receptif.png`, en deux panneaux :
  - croissance du champ réceptif par couche (r_l = r_{l−1} + (k_l − 1)·j_{l−1}, j_l =
    j_{l−1}·s_l), v2 et v3 ;
  - sur une image 416×416 : une cellule 13×13 (pas de 32 px), son champ réceptif, et une
    cellule 26×26 de v3 (pas de 16 px), avec les ancres de chaque tête.
- **Acceptation** : r et j calculés depuis les specs ; pas finaux = 32 (v2, v3 tête 1) et
  16 (v3 tête 2)
- **Notes** : explique pourquoi v3 détecte mieux les petits objets (lien avec T13.7).

### [ ] T13.50 — Flux des tenseurs en 3D
- **Spec** : §3.1, §3.2 · **Dépend de** : T13.48 · **Taille** : M
- **Livrables** : `figures/reseaux/flux_<net>.png`, un bloc par tenseur dont la largeur
  suit C et la hauteur suit H = W (échelle log) :
  - v2 : 3×416×416 → … → 1024×13×13 → 125×13×13 ;
  - v3 : branche 13×13, route vers la couche 8, upsample 13 → 26, concaténation, seconde
    tête 255×26×26.
- **Acceptation** : chaque bloc porte sa forme, égale à `infer_shapes`
- **Notes** : variante en ligne v2 et v3 côte à côte pour la comparaison.

### [ ] T13.51 — La tête de sortie
- **Spec** : §2.2, §5.1, §8.1 · **Dépend de** : T3.1 · **Taille** : S
- **Livrables** : `figures/reseaux/tete.png`, en deux panneaux :
  - la sortie (A·(5 + C), S, S) vue en (A, 5 + C, S, S) : canaux t_x, t_y, t_w, t_h, t_o
    puis les classes ; 5 × (5 + 20) = 125 pour v2 VOC, 3 × (5 + 80) = 255 par tête pour v3
    COCO ;
  - le vecteur d'une cellule et d'une ancre, décodé : sigmoïdes sur t_x, t_y, t_o, exp sur
    t_w, t_h, puis softmax des classes (v2) ou sigmoïdes indépendantes (v3).
- **Acceptation** : ordre des canaux égal à `docs/conventions.md` et à
  `infer/decode.py` (`decode_head`, modes `v2` et `v3`)
- **Notes** : ajouter la version entière à côté (même vecteur en int8, décodé par les LUT
  de T13.45).

### [ ] T13.52 — Une couche à travers toutes les représentations
- **Spec** : §9, §10.2 · **Dépend de** : T4.3, T5.3, T6.2 · **Taille** : M
- **Livrables** : `figures/reseaux/une_couche.svg`. Une même couche (conv 3×3 + BN + leaky
  + maxpool, par exemple L02 de v2) suivie à travers les stades :
  - flottant : conv, BN, leaky, maxpool (float32) ;
  - BN fusionnée : W', b' ;
  - entier Python : `conv_acc` (int8 × int8 → int32) → `requantize` → `leaky_int` →
    `clip_q` → `maxpool_int` ;
  - golden C++ : `golden::conv_layer`, `golden::requantize`, `golden::leaky_int` ;
  - HLS : `load_input`, `load_weights`, `compute`, puis l'étage de sortie qui appelle
    `golden::requantize` (`hls/kernels/conv_pe.cpp`, `output_stage.hpp`).

  Chaque flèche porte le type et la largeur en bits du tenseur, et l'écart mesuré entre
  stades (flottant ↔ entier : SNR en dB ; entier ↔ golden ↔ HLS : 0).
- **Acceptation** : noms des fonctions vérifiés dans le code par le générateur (grep) ;
  écarts lus sur les dumps de `model/<net>/dumps/` (T13.18)
- **Notes** : c'est le pendant détaillé de T13.12 : une seule couche, mais toutes les
  formules et tous les types.
