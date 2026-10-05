# M13 — Figures (optionnel)

Objectif : expliquer le projet en images. Les chiffres existent déjà dans `results/*.md`,
[benchmarks.csv](../../results/benchmarks.csv), `build/` et `tools/perf_model.py`, mais une
seule famille de figures est produite aujourd'hui : les roofline de `tools/roofline.py`
(`results/roofline_*.png`). M13 ajoute des figures sur trois axes :

- l'**architecture** des modèles et de l'accélérateur ;
- les **résultats** des tests et des mesures ;
- le **développement** du projet.

Chaque figure est **régénérable par une commande** et lit ses données dans un fichier
versionné ou produit par un outil existant. Aucun chiffre n'est recopié à la main.

## Conventions communes

### Outillage

- **Paquet** : `tools/figures/`. Il contient un module par famille (`modeles.py`,
  `materiel.py`, `resultats.py`, `projet.py`) et un `style.py` commun.
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
- **Spec** : §6 · **Dépend de** : T2.5 · **Taille** : S
- **Livrables** : `figures/modeles/augmentations.png`, une grille image d'origine →
  augmentations avec les boîtes cibles
- **Acceptation** : graine fixe, et mêmes tirages que `tools/show_augment.py`
- **Notes** : reprendre `tools/show_augment.py` (sorties dans `build/augment_samples`) au
  lieu de le réécrire.

### [ ] T13.7 — Statistiques du jeu VOC
- **Spec** : §8.3 · **Dépend de** : T2.1 · **Taille** : S
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
- **Spec** : §9.1 · **Dépend de** : T4.3 · **Taille** : S
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
