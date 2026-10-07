# M17 — Article (optionnel)

Objectif : un **article de type papier, en français**, qui rend compte du travail du
dépôt : un détecteur Tiny-YOLO réécrit de zéro en NumPy, quantifié en INT8, vérifié
bit-exact en C++ puis sur un accélérateur HLS piloté par l'ARM d'un SoC, avec ses
extensions (formats basse précision, post-traitement matériel, streaming, autres jeux de
données). L'article doit **suivre le projet sans réécriture** : chiffres, tables, figures
et état des jalons y sont **générés dans des blocs balisés**, seul le récit est écrit à la
main.

Constat de départ :
- la [spécification](../../yolo-embarque-de-zero.md) est un document de **méthode**
  (ce qu'il faut implémenter et pourquoi, sourcé dans la base d'articles), pas un compte
  rendu de ce qui a été fait ni de ce que cela donne ;
- les résultats sont dispersés dans `results/` :
  [map_stades.md](../../results/map_stades.md) (VOC2007 test, flottant 56,30, entier
  55,66, C-sim 55,66, 4 952 images identiques),
  [map_int8.md](../../results/map_int8.md), [rapport.md](../../results/rapport.md)
  (projection 37,2 ms, 187,3 GOPS sur KV260), [quant_4bits.md](../../results/quant_4bits.md),
  [req_yolo.md](../../results/req_yolo.md), [postproc_hw.md](../../results/postproc_hw.md),
  [streaming.md](../../results/streaming.md), [benchmarks.csv](../../results/benchmarks.csv),
  et dans [resultats-balayages.md](resultats-balayages.md) pour les jeux hors VOC ;
- environ 80 figures sont publiées dans `results/figures/` par le registre de
  `tools/figures` ([M13](M13-figures.md)), chacune avec sa légende, sa source et sa
  commande ;
- une partie des chiffres sont des **projections** ou des **C-sim** : la KV260 et Vitis
  ne sont pas encore disponibles (M6-M8). L'article doit le dire chiffre par chiffre, et
  basculer quand les mesures arriveront ;
- M15 et M16 sont ouverts : la section « au-delà de VOC » changera encore.

Comme en [M14](M14-notebooks.md) et [M16](M16-presentation-jeux.md), l'article n'apporte
**aucun calcul propre** : tout nombre vient d'un outil du dépôt et se régénère en ligne de
commande.

**État** : 0 tâche faite sur 21.

## Conventions communes

### Fichiers

- **Source** : `docs/article/article.md`, un seul fichier Markdown (LaTeX en ligne
  `$…$` comme la spécification).
- **Chiffres** : `docs/article/chiffres.json`, instantané **versionné** de tous les nombres
  et tables de l'article. Une entrée par clé :
  `{"valeur": …, "statut": …, "source": …, "commande": …, "rev": …, "date": …}`.
- **Références** : `docs/article/refs.md`, reprise du §12 de la spécification ; citations
  au format de la base d'articles (`[2024-zhang#013.1]`, vérifiables avec `bin/pdb verify`, outil de la base, hors dépôt).
- **Export** (optionnel) : `build/article/article.pdf` et `.html` par pandoc ; rien
  d'exporté n'est versionné.
- **Outil** : `tools/article.py` (`python -m tools.article`), en Python pur, hors du paquet
  `yolo` ([ADR 0001](../adr/0001-numpy-pur.md)).

### Blocs générés

Un bloc est délimité par deux commentaires HTML, invisibles au rendu :

```markdown
La mAP entière vaut <!-- article:n:map.voc.int8 -->55,66<!-- /article --> points.

<!-- article:table:map.stades -->
| Stade | mAP | Statut |
|---|---|---|
| … | … | … |
<!-- /article -->

<!-- article:fig:chaine_verif -->
![…](../../results/figures/materiel/chaine_verif.png)
*Figure — …* <!-- python -m tools.figures chaine_verif -->
<!-- /article -->
```

| Type | Contenu | Origine |
|---|---|---|
| `n` | un nombre, formaté à la française (virgule décimale, espace fine des milliers) | clé de `chiffres.json` |
| `table` | une table Markdown | clé de `chiffres.json` (liste de lignes) |
| `fig` | image, légende `Figure.caption`, commande en commentaire ; `fig:<nom>` ou `fig:<nom>/<fichier>` quand le générateur produit plusieurs images | registre `FIGURES` de [tools/figures/__init__.py](../../tools/figures/__init__.py) |
| `etat` | avancement des jalons | table **Suivi** de [README.md](README.md) |
| `rev` | révision git et date du dernier `render` | `git rev-parse --short HEAD` |

Règles :
- le contenu d'un bloc **n'est jamais édité à la main** : il est réécrit par `render` ;
- tout ce qui est hors bloc (le récit) n'est **jamais touché** par l'outil ;
- un nombre cité dans le récit hors bloc est une faute : `--check` signale les nombres
  décimaux hors bloc dans les sections de résultats (liste blanche pour les constantes du
  réseau : 416, 13×13, 0,1 de la leaky, etc.) ;
- les figures sont citées **par nom de registre**, jamais par chemin : renommer une figure
  dans `tools/figures` casse `--check`, pas l'article en silence.

### Collecte en deux temps

1. **`collect`** : un registre `CLES` dans `tools/article.py` associe chaque clé à un
   lecteur, une source et un statut. Le lecteur lit la source (JSON de `build/`, CSV ou
   table Markdown de `results/`). Si la source manque, la valeur de `chiffres.json` est
   **gardée** et signalée, comme `MissingSource` des figures. `collect` réécrit
   `chiffres.json` avec la révision courante pour chaque valeur qui a changé.
2. **`render`** : réécrit les blocs de `article.md` depuis `chiffres.json` et le registre
   des figures, sans lire `build/`.

L'article se régénère donc partout (CI, Colab, machine sans `build/`), et le diff de
`chiffres.json` montre exactement quels résultats ont bougé entre deux versions.

### Statut de chaque chiffre

Chaque clé porte un statut, rendu en exposant ou en colonne « Statut » des tables :

| Statut | Sens | Exemple |
|---|---|---|
| `mesure` | mesuré sur la carte ou sur le jeu complet | mAP entière VOC2007 test |
| `csim` | calculé par le noyau HLS en C-sim | mAP FPGA 55,66 |
| `projection` | modèle de cycles ou de ressources, sans synthèse | 37,2 ms, 187,3 GOPS |
| `estimation` | modèle analytique hors synthèse | ressources du streaming (M9.4) |
| `palier-R` | sous-ensemble (50 images, règle M12) | balayages M15 |
| `publie` | chiffre d'un article de la base | lignes de `benchmarks.csv` hors « ce travail » |

L'article **ne présente jamais une projection comme une mesure** : le résumé et les
conclusions citent le statut à côté de chaque chiffre clé.

### Règles

- `make article` = `collect` puis `render` ; `make article-pdf` exporte en plus.
- `python -m tools.article --check` (dans `make ci`) échoue si :
  - un bloc diffère de ce que produirait `render` ;
  - un bloc cite une clé absente de `chiffres.json` ou une figure absente de `FIGURES` ;
  - une image citée n'existe pas dans `results/figures/` ;
  - une section de résultats contient un nombre hors bloc (liste blanche à part).
- Les clés sont stables et hiérarchiques : `<jeu>.<grandeur>.<variante>`
  (`map.voc.int8`, `perf.kv260.m10.ms`, `balayage.visdrone.meilleur.map`). Renommer une
  clé se fait dans `CLES` et dans l'article dans le même commit.
- Langue et typographie : celles de [conventions.md](../conventions.md) et de la
  spécification (français, virgule décimale, « hors base » pour ce qui n'est pas sourcé).

### Mise à jour à chaque jalon

À la clôture d'une tâche qui produit un résultat (ou au moins à chaque jalon) :

1. `make article` ;
2. relire le diff de `chiffres.json` et des blocs (`git diff docs/article/`) ;
3. relire le récit des sections dont un chiffre ou un statut a changé (table **Sections**
   ci-dessous) et le corriger ; passer leur état à « à jour à rév. X » ;
4. ajouter une entrée au **journal des versions** (annexe de l'article) : révision, date,
   ce qui a changé, en une ou deux lignes ;
5. `--check` vert, commit `M17 : article à jour (<jalon>)`.

## Plan de l'article

Chaque section indique les jalons qu'elle raconte et ses blocs principaux. La table sert
aussi à savoir **quelle section relire** quand un jalon avance.

| § | Section | Jalons | Blocs principaux (clés, figures) |
|---|---|---|---|
| 0 | Résumé | tous | `n` : mAP flottante / entière / C-sim, ms et GOPS projetés, MACs ; `rev` |
| 1 | Introduction et contributions | — | liste des contributions ; `etat` résumé |
| 2 | Travaux liés | §10.4 de la spec | `table:benchmarks` (`benchmarks.csv`), `fig:etat_art` |
| 3 | Réseaux et briques NumPy | M0-M2 | `fig:graphe/graphe_tiny-yolov2-voc`, `fig:fiche/fiche_tiny-yolov2-voc`, `fig:convolution`, `fig:retropropagation`, `fig:perte` ; `n` : MACs, paramètres, gradcheck |
| 4 | Inférence et évaluation | M3 | `table:map.voc.float` (prétraitements), `fig:pr/pr_variantes`, `fig:nms_map` |
| 5 | Quantification INT8 | M4 | `table:map.voc.int8`, `fig:calibration/calibration_tiny-yolov2-voc`, `fig:erreur_couches/erreur_couches_tiny-yolov2-voc`, `fig:quantification`, `fig:entier` |
| 6 | Golden C++ et chaîne de vérification | M5 | `fig:chaine_verif` ; `n` : écarts (0), nombre de dumps comparés |
| 7 | Accélérateur HLS et intégration SoC | M6, M7, M10 | `fig:moteur`, `fig:tuilage`, `fig:soc`, `fig:roofline_couches`, `fig:cascade_m10`, `fig:cycles_couches` |
| 8 | Résultats | M8 | `table:map.stades`, `table:perf.kv260`, `fig:map_stades`, `fig:ressources` ; statuts `csim` / `projection` |
| 9 | Extensions | M9.1-M9.4 | `table:map.formats` (INT8, puissances de 2, 4 bits), `fig:map_formats`, `fig:hw_postproc`, `fig:streaming` |
| 10 | Au-delà de VOC | M11, M15, M16 | `table:balayages.meilleurs` (par jeu), `fig:balayages/balayages_*`, `fig:detections` ; statut `palier-R` |
| 11 | Limites et travaux futurs | tous | `etat` détaillé ; liste générée des chiffres encore `projection` / `estimation` |
| 12 | Reproductibilité | M0, M12-M14 | commandes `make`, notebooks, `fig:tests`, `fig:code` ; `rev` |
| — | Références | — | `refs.md` |
| A | Annexe : journal des versions | — | écrit à la main, une entrée par mise à jour |

---

## A. Infrastructure

### [ ] T17.0 — Squelette de l'article
- **Spec** : — · **Dépend de** : — · **Taille** : S
- **Livrables** : `docs/article/article.md` (titre, auteurs, sections du plan, un bloc
  vide par bloc prévu, annexe « journal des versions ») ; `docs/article/refs.md` ;
  `docs/article/chiffres.json` vide (`{}`).
- **Acceptation** : le squelette se lit en Markdown (GitHub et VS Code) ; chaque section
  du plan y est, avec ses blocs balisés et une ligne « à rédiger ».

### [ ] T17.1 — Moteur de blocs `tools/article.py`
- **Spec** : — · **Dépend de** : T17.0 · **Taille** : M
- **Livrables** : `tools/article.py` (`parse`, `render`, `main`) ;
  `python/tests/test_article.py`.
- **Comportement attendu** :
  - `parse` repère les blocs `<!-- article:<type>:<clé> -->…<!-- /article -->`, en ligne
    ou sur plusieurs lignes ; un bloc mal fermé est une erreur avec numéro de ligne ;
  - `render` remplace le contenu de chaque bloc et laisse le reste du fichier octet pour
    octet ;
  - formatage des nombres à la française (décimales fixées par clé) ;
  - `--check` : voir **Règles** ; sortie lisible (ligne, clé, attendu, trouvé).
- **Acceptation** :
  - tests : idempotence (`render` deux fois = une fois), texte hors bloc inchangé, bloc
    inconnu ou mal fermé refusé, `--check` rouge sur un bloc modifié à la main, nombre hors
    bloc détecté ;
  - aucune dépendance hors bibliothèque standard.
- **Notes** : reprendre le schéma `--check` de `tools/notebooks/__main__.py`.

### [ ] T17.2 — Collecte des chiffres (`collect`)
- **Spec** : — · **Dépend de** : T17.1 · **Taille** : M
- **Livrables** : registre `CLES` et lecteurs dans `tools/article.py` ; premier
  `docs/article/chiffres.json`.
- **Lecteurs** (un par famille de sources, réutiliser les lecteurs existants plutôt que
  relire les fichiers) :
  - mAP aux stades : `build/m8/<net>/map_stades.json` (`tools/map_stades.py --json`),
    repli sur la table de `results/map_stades.md` ;
  - flottant / entier : `build/m8/<net>/eval_float_int.json`, tables de `map_float.md` et
    `map_int8.md` ;
  - formats : `build/m9/map_*.json`, tables de `quant_4bits.md` et `req_yolo.md` ;
  - performance : `tools/perf_model.py` (projection) puis, quand ils existeront, mesures
    de `results/mesures.md` ;
  - état de l'art : `results/benchmarks.csv` ;
  - balayages : `tools/notebooks/balayages.py` (sorties des notebooks `_sweep` et `_infer`) ;
  - modèle : MACs et paramètres de `make count-macs`.
- **Acceptation** :
  - chaque clé a une source, une commande et un statut ;
  - sans `build/`, `collect` garde les valeurs existantes et liste les clés non
    rafraîchies, sans échouer ;
  - test : une source modifiée change la valeur et la révision de sa seule clé.

### [ ] T17.3 — Blocs `fig`, `etat` et `rev`
- **Spec** : — · **Dépend de** : T17.1 · **Taille** : S
- **Livrables** : rendu des trois types dans `tools/article.py`.
- **Acceptation** :
  - `fig` : chemin relatif depuis `docs/article/`, légende = `Figure.caption`, commande
    en commentaire ; une figure `subset=True` (dans `build/`) est refusée, l'article ne
    cite que des figures publiées ;
  - `etat` : table jalon / tâches / faites lue dans la table **Suivi** de
    [README.md](README.md), variantes `etat` (complète) et `etat:resume` (une phrase) ;
  - `rev` : révision courte et date, suffixe `+modifs` si l'arbre de travail est sale.

### [ ] T17.4 — Cibles `make` et CI
- **Spec** : — · **Dépend de** : T17.2, T17.3 · **Taille** : S
- **Livrables** : cibles `article` et `article-pdf` du `Makefile` ; `--check` ajouté à
  `ci` ; ligne dans `make help`.
- **Acceptation** : `make article` puis `make ci` passent ; `make article-pdf` produit
  `build/article/article.pdf` si pandoc est installé, sinon un message clair (pas
  d'échec de `ci`).

## B. Rédaction

Acceptation commune des tâches de rédaction :
- la section est rédigée, au ton d'un article (récit de ce qui a été fait, choix et
  résultats), pas d'une documentation ;
- tous ses nombres de résultats sont dans des blocs, toutes ses figures sont citées par le
  registre ;
- les affirmations sur la méthode renvoient aux citations de la base, ou sont marquées
  « hors base » comme dans la spécification ;
- `--check` passe ; l'état de la section passe à « brouillon » dans la table **Sections**.

### [ ] T17.5 — §0 Résumé
- **Dépend de** : T17.6-T17.16 · **Taille** : S
- **Notes** : écrit en dernier ; 200 mots ; chaque chiffre avec son statut.

### [ ] T17.6 — §1 Introduction et contributions
- **Dépend de** : T17.0 · **Taille** : S
- **Notes** : contributions candidates : chaîne complète sans bibliothèque
  d'apprentissage ; bit-exact du flottant NumPy au noyau HLS (4 952 images identiques) ;
  moteur unique paramétré pour v2 et v3 ; comparaison de formats (INT8, puissances de 2,
  4 bits) sur le même modèle ; outillage reproductible (profils, figures, notebooks).

### [ ] T17.7 — §2 Travaux liés
- **Dépend de** : T17.2 · **Taille** : M
- **Notes** : s'appuyer sur §10.4 de la spécification et sur la colonne « ce qui empêche
  la comparaison directe » de [rapport.md](../../results/rapport.md).

### [ ] T17.8 — §3 Réseaux et briques NumPy
- **Dépend de** : T17.3 · **Taille** : M
- **Notes** : Tiny-YOLOv2/v3 couche par couche (spec §3), passes avant et arrière, gradcheck
  ([M1](M1-briques-numpy.md)), perte et entraînement ([M2](M2-cibles-perte-entrainement.md)).

### [ ] T17.9 — §4 Inférence et évaluation
- **Dépend de** : T17.2 · **Taille** : S
- **Notes** : effet du prétraitement (stretch, letterbox, interpolation Darknet) sur la
  mAP de référence ([M3](M3-inference-evaluation.md)).

### [ ] T17.10 — §5 Quantification INT8
- **Dépend de** : T17.2 · **Taille** : M
- **Notes** : fusion BN, calibration, M0/décalage, leaky 13/128, LUT des têtes ; écart
  sous 1 point, QAT non nécessaire ([M4](M4-quantification.md)).

### [ ] T17.11 — §6 Golden C++ et chaîne de vérification
- **Dépend de** : T17.3 · **Taille** : S
- **Notes** : comment l'égalité est vérifiée à chaque passage ([M5](M5-golden-cpp.md),
  `make golden-check`).

### [ ] T17.12 — §7 Accélérateur HLS et intégration SoC
- **Dépend de** : T17.3 · **Taille** : L
- **Notes** : choix de la carte ([ADR 0003](../adr/0003-choix-carte.md)), roofline,
  tuilage, moteur unique, carte de registres, driver ARM ([M6](M6-accelerateur-hls.md),
  [M7](M7-integration-soc.md)) ; cascade d'optimisations de [M10](M10-ameliorations.md).

### [ ] T17.13 — §8 Résultats
- **Dépend de** : T17.2 · **Taille** : M
- **Notes** : la section la plus exposée aux mises à jour : les phrases ne reprennent
  aucun chiffre, elles commentent les tables. Prévoir les deux formulations (avant et
  après mesure carte) dans T17.19.

### [ ] T17.14 — §9 Extensions
- **Dépend de** : T17.2 · **Taille** : M
- **Notes** : post-traitement matériel ([M9.1](M9.1-postproc-materiel.md)), REQ-YOLO et
  ADMM non convergé ([M9.2](M9.2-req-yolo.md)), 4 bits ([M9.3](M9.3-quant-4bits.md)),
  streaming en estimation ([M9.4](M9.4-streaming.md)). Les résultats négatifs ou partiels
  sont rapportés comme tels.

### [ ] T17.15 — §10 Au-delà de VOC
- **Dépend de** : T17.2 · **Taille** : M
- **Notes** : jeux de [M11](M11-jeux-de-donnees.md), balayages et inférences de
  [M15](M15-campagne-entrainement.md) ([resultats-balayages.md](resultats-balayages.md)),
  statistiques de [M16](M16-presentation-jeux.md) quand elles existeront. Section la plus
  susceptible de grandir : une sous-section par jeu, dans l'ordre de M15.

### [ ] T17.16 — §11 Limites et §12 Reproductibilité
- **Dépend de** : T17.3 · **Taille** : S
- **Notes** : §11 liste générée des clés encore `projection`, `estimation` ou `palier-R` ;
  §12 : commandes `make`, profils R/M/N ([M12](M12-profils-pc.md)), notebooks
  ([M14](M14-notebooks.md)), révision.

### [ ] T17.17 — Références et relecture d'ensemble
- **Dépend de** : T17.5-T17.16 · **Taille** : M
- **Livrables** : `refs.md` complet ; relecture (cohérence des notations avec la
  spécification, une seule définition par sigle, figures appelées dans le texte).
- **Acceptation** : chaque citation `[chunk_id]` passe `bin/pdb verify` ; toutes les
  sections à l'état « relu ».

## C. Maintenance

### [ ] T17.18 — Première version et journal
- **Dépend de** : T17.17 · **Taille** : S
- **Livrables** : entrée « v0 » du journal des versions ; procédure **Mise à jour à
  chaque jalon** appliquée une fois de bout en bout.
- **Acceptation** : un jalon qui avance (par exemple une ligne de M15) se répercute dans
  l'article par `make article` et une relecture de la seule section concernée.

### [ ] T17.19 — Bascule « projection → mesure » (carte)
- **Dépend de** : T7.4, T8.1-T8.3 sur carte · **Taille** : M
- **Livrables** : statuts des clés de performance et de mAP FPGA passés à `mesure` ;
  récit de §0, §8 et §11 réécrit ; comparaison projection / mesure ajoutée en §8.
- **Acceptation** : plus aucune clé `projection` citée dans le résumé.

### [ ] T17.20 — Version figée pour soumission
- **Dépend de** : T17.18 · **Taille** : S
- **Livrables** : tag git `article-vN`, `build/article/article.pdf`, entrée du journal.
- **Acceptation** : depuis le tag, `make article` ne change rien (`--check` vert) et
  l'export PDF se reproduit.

## Sections

État de rédaction de chaque section, mis à jour par la procédure de mise à jour :
*à rédiger*, *brouillon*, *relu*, *à jour à rév. X*.

| § | Section | État |
|---|---|---|
| 0 | Résumé | à rédiger |
| 1 | Introduction et contributions | à rédiger |
| 2 | Travaux liés | à rédiger |
| 3 | Réseaux et briques NumPy | à rédiger |
| 4 | Inférence et évaluation | à rédiger |
| 5 | Quantification INT8 | à rédiger |
| 6 | Golden C++ et chaîne de vérification | à rédiger |
| 7 | Accélérateur HLS et intégration SoC | à rédiger |
| 8 | Résultats | à rédiger |
| 9 | Extensions | à rédiger |
| 10 | Au-delà de VOC | à rédiger |
| 11 | Limites et travaux futurs | à rédiger |
| 12 | Reproductibilité | à rédiger |
