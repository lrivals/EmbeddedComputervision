# Figures (M13)

Galerie générée par `python -m tools.figures all` (`make figures`) à partir du registre de `tools/figures/` ; ne pas éditer à la main. Chaque figure se régénère par sa commande et lit ses données dans la source indiquée. Les figures absentes attendent leur donnée (voir la commande de la tâche).

## Résultats (D)

### T13.13 — `map_stades`

AP par classe aux stades flottant, entier et C-sim (la carte quand elle sera mesurée), et écart entier − flottant trié.

- Commande : `python -m tools.figures map_stades`
- Source : `build/m8/<net>/map_stades.json (tools/map_stades.py --json)`

![map_stades](figures/resultats/map_stades.png)

### T13.14 — `map_formats`

mAP de Tiny-YOLOv2 selon le format des poids (flottant, INT8, puissances de 2, 4 bits), et frontière précision / bits.

- Commande : `python -m tools.figures map_formats`
- Source : `build/m8/<net>/eval_float_int.json, build/m9/map_*.json`

![map_formats](figures/resultats/map_formats.png)

### T13.15 — `pr`

Courbes précision-rappel par classe (11 points VOC07) pour chaque prétraitement évalué, puis les quatre superposées.

- Commande : `python -m tools.figures pr`
- Source : `build/eval/<net>-<variante>/comp4_det_test_*.txt + data/VOCdevkit`

![pr_letterbox-darknet](figures/resultats/pr_letterbox-darknet.png)
![pr_letterbox](figures/resultats/pr_letterbox.png)
![pr_stretch-darknet](figures/resultats/pr_stretch-darknet.png)
![pr_stretch](figures/resultats/pr_stretch.png)
![pr_variantes](figures/resultats/pr_variantes.png)

### T13.16 — `sensibilite`

Perte de mAP quand une seule couche est quantifiée (INT8 en fake-quant face au flottant ; uniform6, uniform4, mixed6 face à l'INT8). Sous-ensembles de VOC2007 test.

- Commande : `python -m tools.figures sensibilite`
- Source : `build/quant/<net>/eval_test_*_fq*.json, build/m10/mixed/sensitivity.csv`
- Sous-ensemble d'images : sortie dans `build/figures/resultats/`, non publiée (règle M12).

### T13.17 — `calibration`

Distribution des activations par couche et seuil de saturation retenu par la calibration ; échelles des poids par canal.

- Commande : `python -m tools.figures calibration`
- Source : `build/quant/<net>/calib.json, build/m9/stats_<net>.npz, weights/`

![calibration_tiny-yolov2-voc](figures/resultats/calibration_tiny-yolov2-voc.png)
![calibration_tiny-yolov3-coco](figures/resultats/calibration_tiny-yolov3-coco.png)

### T13.18 — `erreur_couches`

SNR par couche entre le flottant et l'entier déquantifié, et écart entier Python ↔ golden C++ (0 partout : bit-exact).

- Commande : `python -m tools.figures erreur_couches`
- Source : `model/<net>/dumps/, build/golden/out/<net>/, weights/`

![erreur_couches_tiny-yolov2-voc](figures/resultats/erreur_couches_tiny-yolov2-voc.png)
![erreur_couches_tiny-yolov3-coco](figures/resultats/erreur_couches_tiny-yolov3-coco.png)

### T13.20 — `cycles_couches`

Cycles par couche du noyau actuel, décomposés en chargements, calcul et stockage ; le losange donne le temps réel avec recouvrement.

- Commande : `python -m tools.figures cycles_couches`
- Source : `build/hls/cycles_conv.csv (make hls-cycles)`

![cycles_couches](figures/resultats/cycles_couches.png)

### T13.21 — `cascade_m10`

Temps par image de Tiny-YOLOv2 à chaque piste d'optimisation M10, face à la borne de calcul et au roofline KV260.

- Commande : `python -m tools.figures cascade_m10`
- Source : `tools/perf_model.py (SCENARIOS, scenario_table), tools/roofline.py`

![cascade_m10](figures/resultats/cascade_m10.png)

### T13.22 — `roofline_couches`

Roofline KV260 avec un point par couche de Tiny-YOLOv2, avant et après les optimisations M10.

- Commande : `python -m tools.figures roofline_couches`
- Source : `hw/boards/kv260.yaml, model/<net>/manifest.json, tools/perf_model.py`

![roofline_couches_kv260](figures/resultats/roofline_couches_kv260.png)

### T13.23 — `etat_art`

Comparaison aux accélérateurs publiés : GOPS et puissance, débit et mAP, GOPS par DSP (marque creuse : projection de ce travail).

- Commande : `python -m tools.figures etat_art`
- Source : `results/benchmarks.csv`

![etat_art](figures/resultats/etat_art.png)

### T13.24 — `ressources`

DSP et mémoire sur puce estimés des architectures (moteur unique INT8 et 4 bits, streaming) face au budget de la KV260.

- Commande : `python -m tools.figures ressources`
- Source : `hw/boards/kv260.yaml, tools/roofline.py, build/m9/stream_plan_w*.json`

![ressources](figures/resultats/ressources.png)

### T13.25 — `hw_postproc`

mAP avec le post-traitement matériel, avec et sans plafond de boîtes, et nombre de boîtes par image face au plafond.

- Commande : `python -m tools.figures hw_postproc`
- Source : `build/m9/hwpp_map.json, hwpp_map_nocap.json, build/m8/<net>/int.jsonl`

![hw_postproc](figures/resultats/hw_postproc.png)

### T13.26 — `entrainement`

Courbes d'entraînement : perte et composantes lissées, taux d'apprentissage, résidus ADMM ; puis tous les runs superposés.

- Commande : `python -m tools.figures entrainement`
- Source : `build/train/<run>/loss.csv, admm.csv (tools/train.py --out)`

![entrainement_admm-mixed6](figures/resultats/entrainement_admm-mixed6.png)
![entrainement_qat-w4a4](figures/resultats/entrainement_qat-w4a4.png)
![entrainement_runs](figures/resultats/entrainement_runs.png)

## Modèles (B)

### T13.2 — `profil_couches`

MACs, paramètres et taille des sorties int8 par couche, Tiny-YOLOv2 et Tiny-YOLOv3 superposés, avec les cumuls.

- Commande : `python -m tools.figures profil_couches`
- Source : `model/<net>/manifest.json via tools/count_macs.py`

![profil_couches](figures/modeles/profil_couches.png)

### T13.1 — `graphe`

Graphe couche par couche (type, noyau et stride, forme de sortie C×H×W ; bord épais = beaucoup de MACs).

- Commande : `python -m tools.figures graphe`
- Source : `model/<net>/manifest.json, yolo.models.specs (Graphviz dot)`

![graphe_tiny-yolov2-voc](figures/modeles/graphe_tiny-yolov2-voc.png)
![graphe_tiny-yolov3-coco](figures/modeles/graphe_tiny-yolov3-coco.png)

### T13.3 — `memoire`

Empreinte mémoire par conv (poids, biais, activations) face aux tampons BRAM du noyau et à l'arène DDR du driver.

- Commande : `python -m tools.figures memoire`
- Source : `model/<net>/manifest.json, sw/driver/program.hpp, tools/roofline.py`

![memoire_tiny-yolov2-voc](figures/modeles/memoire_tiny-yolov2-voc.png)
![memoire_tiny-yolov3-coco](figures/modeles/memoire_tiny-yolov3-coco.png)

### T13.4 — `ancres`

Nuage (w, h) des boîtes VOC avec les ancres k-means (distance 1 − IoU) et les ancres Darknet.

- Commande : `python -m tools.figures ancres`
- Source : `data/VOCdevkit, results/anchors.md, tools/kmeans_anchors.py:dataset_wh`

![ancres](figures/modeles/ancres.png)

### T13.7 — `voc_stats`

Statistiques de VOC : objets par classe en trainval et en test, aires des boîtes, objets par image.

- Commande : `python -m tools.figures voc_stats`
- Source : `data/VOCdevkit (comptes égaux à tools/voc_stats.py)`

![voc_stats](figures/modeles/voc_stats.png)

## Développement du projet (E)

### T13.27 — `chronologie`

Gantt des jalons reconstruit depuis les commits « Mx : », avec les commits de documentation en pointillé et les jalons non commencés en gris.

- Commande : `python -m tools.figures chronologie`
- Source : `git log --date=iso`

![chronologie](figures/projet/chronologie.png)

### T13.28 — `avancement`

Tâches faites, partielles, vérifiées sur PC (carte en attente) et ouvertes, par jalon.

- Commande : `python -m tools.figures avancement`
- Source : `docs/tasks/M*.md, tableau de suivi de docs/tasks/README.md`

![avancement](figures/projet/avancement.png)

### T13.29 — `code`

Lignes par dossier (python, tools, cpp, hls, sw, docs, results) commit par commit.

- Commande : `python -m tools.figures code`
- Source : `git log --numstat`

![code](figures/projet/code.png)

### T13.30 — `tests`

Nombre de tests par module Python (pytest, slow compris) et par build C++ (ctest).

- Commande : `python -m tools.figures tests`
- Source : `pytest --collect-only -q, ctest -N`

![tests](figures/projet/tests.png)
