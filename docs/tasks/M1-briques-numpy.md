# M1 — Briques NumPy (passes avant et arrière)

Objectif : toutes les couches du §4, vérifiées par différences finies, puis assemblées en
Tiny-YOLOv2 et Tiny-YOLOv3. Interface commune : `forward(x) -> (y, cache)`,
`backward(dy, cache) -> (dx, grads)`.

### [x] T1.1 — Utilitaire de vérification des gradients
- **Spec** : §11 · **Dépend de** : T0.2 · **Taille** : S
- **Livrables** : `python/yolo/testing/gradcheck.py`, `python/tests/test_gradcheck.py`
- **Acceptation** : différences centrées h = 1e-6 en float64 ; erreur relative
  $|a-n|/\max(|a|,|n|,\epsilon)$ ; validé sur $f(x)=\sum x^3$
- **Notes** : vérifier par échantillonnage d'indices pour les gros tenseurs
- **Fait** : $|\cdot|$ = norme max sur le tenseur (un gradient accidentellement petit rendrait
  l'erreur élément par élément dominée par le bruit d'arrondi de h = 1e-6) ; la perte peut
  être rendue terme à terme pour que les termes non affectés s'annulent exactement.
  `check_layer` fait le gradcheck d'une couche avec $L=\sum y\cdot R$.

### [x] T1.2 — Convolution 2D (naïf + im2col)
- **Spec** : §4.1 · **Dépend de** : T1.1 · **Taille** : M
- **Livrables** : `python/yolo/layers/conv.py`, `python/tests/test_conv.py`
- **Acceptation** : gradcheck ≤ 1e-7 sur $W$, $b$, $X$ (k = 1 et 3, s = 1 et 2) ;
  im2col == naïf à 1e-12 ; temps d'une conv 13×13×1024→1024 mesuré
- **Notes** : la version naïve reste comme référence de test ; im2col est celle utilisée en
  entraînement
- **Mesure** (`python tools/bench_conv.py`, Intel Core Ultra 7 155H, OpenBLAS) : lot 1,
  1,59 GMAC — float32 : avant 13 ms (≈ 120 GMAC/s), arrière 37 ms ; float64 : avant 30 ms,
  arrière 58 ms. Lot 8 en float32 : avant 107 ms, arrière 226 ms.

### [x] T1.3 — Batch normalization
- **Spec** : §4.2 · **Dépend de** : T1.1 · **Taille** : M
- **Livrables** : `python/yolo/layers/batchnorm.py` + tests
- **Acceptation** : gradcheck ≤ 1e-7 sur $\gamma$, $\beta$, $x$ ; mode inférence avec les
  moyennes glissantes ; test que le gradient d'un biais conv avant BN est ≈ 1e-15
- **Notes** : momentum des moyennes glissantes et $\varepsilon$ en paramètres

### [x] T1.4 — Leaky ReLU
- **Spec** : §4.3 · **Dépend de** : T1.1 · **Taille** : S
- **Livrables** : `python/yolo/layers/activations.py` + tests
- **Acceptation** : gradcheck ≤ 1e-7 (en évitant les points $x = 0$)

### [x] T1.5 — Max pooling stride 2 et stride 1
- **Spec** : §4.4 · **Dépend de** : T1.1 · **Taille** : M
- **Livrables** : `python/yolo/layers/pool.py` + tests
- **Acceptation** : gradcheck ≤ 1e-7 ; stride 1 conserve 13×13 ; gradient routé uniquement
  vers l'argmax
- **Notes** : stride 1 = padding à droite et en bas par réplication du bord (couche 11,
  hors base) ; tester avec des valeurs distinctes pour éviter les égalités d'argmax

### [x] T1.6 — Upsample, route, sigmoïde
- **Spec** : §4.5 · **Dépend de** : T1.1 · **Taille** : S
- **Livrables** : `python/yolo/layers/{upsample,route}.py`, `sigmoid` stable dans
  `activations.py` + tests
- **Acceptation** : gradcheck ≤ 1e-7 ; sigmoïde sans overflow pour $t = \pm 1000$
- **Notes** : la route découpe $\delta y$ ; le cumul des gradients d'une carte réutilisée
  est géré par le graphe (T1.7)

### [x] T1.7 — Graphe de réseau et parser `.cfg`
- **Spec** : §3, §4.5 · **Dépend de** : T1.2 à T1.6 · **Taille** : M
- **Livrables** : `python/yolo/models/graph.py` (exécution avant, caches, arrière en ordre
  inverse, **cumul des gradients** pour les sorties consommées deux fois),
  `python/yolo/models/cfg.py` (parser Darknet)
- **Acceptation** : gradcheck du mini-réseau du §11
  (conv-BN-leaky-pool-conv-BN-leaky-pool(s1)-route-conv1×1) ≤ 1e-7
- **Notes** : dans Tiny-YOLOv3, la couche 13 alimente la 14 et la route 17

### [x] T1.8 — Tiny-YOLOv2 et Tiny-YOLOv3
- **Spec** : §3.1, §3.2 · **Dépend de** : T1.7, T0.5 · **Taille** : S
- **Livrables** : `python/yolo/models/tiny_yolo.py`, fichiers `.cfg` dans `python/yolo/models/cfg/`
- **Acceptation** : formes de sortie et nombre de paramètres de chaque couche == tableaux du
  §3 ; passe avant 416×416 sans erreur ; initialisation He (§7.1)
- **Notes** : les `.cfg` Darknet sont hors base : les citer comme tels ; vérifier le réseau
  construit contre `yolo.models.specs` (T0.5), déjà testé contre les tableaux du §3

### [~] T1.9 — Chargeur de poids Darknet
- **Spec** : §7.1 (pré-entraînement), §10.5 étape 1 · **Dépend de** : T1.8 · **Taille** : M
- **Livrables** : `python/yolo/io/darknet_weights.py`
- **Acceptation** : chargement de `yolov3-tiny.weights` (COCO) et de `yolov2-tiny-voc.weights` ;
  détection visuellement correcte sur une image test (avec T3.1-T3.2)
- **Notes** : ordre Darknet par couche : [biais ou β], [γ, μ, σ²] si BN, puis poids ;
  en-tête de 4 ou 5 mots selon la version
- **Fait** : `tools/get_weights.sh` ; les deux fichiers se chargent en consommant exactement
  tous les octets (`yolov2-tiny-voc.weights` : en-tête de 4 mots ; `yolov3-tiny.weights` :
  5 mots, avec `cfg/yolov3-tiny.cfg` COCO). Contrôle rapide sans NMS sur les 12 premières
  images du test VOC 2007 : Tiny-YOLOv2 trouve au moins une classe vérité terrain sur 9.
  **Reste** : la vérification visuelle des boîtes, après T3.1-T3.2.
