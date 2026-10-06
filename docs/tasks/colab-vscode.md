# Entraîner sur un kernel Colab depuis VS Code

Les notebooks `_train` et `_sweep` (M14) s'ouvrent dans VS Code et s'exécutent sur un
runtime Colab à GPU grâce à l'extension Google Colab (`google.colab`). Le notebook reste
le fichier du dépôt ; seul le kernel est distant. Les sorties partent sur Google Drive
(`colab.sync_outputs`) et se récupèrent sur le PC par rclone
([donnees-drive.md](donnees-drive.md)).

## Qui fait quoi

| Étape | Qui | Comment |
|---|---|---|
| connecter le kernel Colab, autoriser Google et Drive | utilisateur | VS Code (UI de l'extension) |
| régler les paramètres d'un notebook | utilisateur ou Claude | cellule « Paramètres » (le fichier est généré : voir plus bas) |
| lancer les cellules (« Run All ») | **utilisateur** | VS Code |
| suivre la perte, voir les runs finis | utilisateur ou Claude | rclone sur `gdrive:EmbeddedCV/runs` |
| récupérer les runs, évaluer, comparer | utilisateur ou Claude | `rclone copy`, notebooks d'inférence en local |

Claude (Claude Code) ne peut pas exécuter de cellules sur un kernel attaché à VS Code.
L'extension Colab n'expose que des commandes d'interface (`colab.mountServer`,
`colab.mountDrive`, `colab.openTerminal`…) et garde ses jetons dans le stockage secret de
VS Code. Le PC n'a pas de GPU, donc pas d'exécution locale non plus (palier N : des heures
sur CPU). Claude prépare les paramètres et suit les runs par rclone, et c'est l'utilisateur
qui clique sur « Run All ».

## Mise en place (une fois par session Colab)

1. Ouvrir le notebook dans VS Code, par exemple
   [notebooks/kitti/tiny-yolov3-kitti_train.ipynb](../../notebooks/kitti/tiny-yolov3-kitti_train.ipynb).
2. « Select Kernel » → « Colab » → **nouveau** serveur, type **GPU** (T4 ; L4 ou A100
   avec Colab Pro), puis connexion Google. Un serveur créé en CPU ne change pas de type :
   en créer un autre. Contrôle : `!nvidia-smi` dans une cellule affiche le GPU. Sur un
   runtime CPU, la cellule « Environnement » s'arrête avec « runtime Colab sans GPU » (avant
   ce contrôle, CuPy disait `cudaErrorInsufficientDriver`). La roue CuPy suit la version
   CUDA du pilote (`cupy-cuda13x` à partir de 13, sinon `cupy-cuda12x`).
3. Secrets Colab `KAGGLE_USERNAME` et `KAGGLE_KEY` si le jeu se télécharge depuis Kaggle
   (crowdhuman, visdrone sans archive sur Drive).

La cellule « Environnement » détecte Colab (`"google.colab" in sys.modules`). Le kernel
tourne dans `/content`, hors du dépôt : elle clone donc
`REPO_URL` dans `/content/EmbeddedComputervision`, le remet à `REV` (défaut `main`) à chaque
exécution, installe `python[data,plots]` et CuPy (`DEVICE = 'gpu'`). **Le code exécuté
est celui de `REV` sur GitHub, pas celui du PC** : pousser d'abord ce qui doit tourner.

Les données passent par `colab.prepare_data(DATASET, DRIVE_DIR)` (ordre : témoin, archive
Drive, rclone, téléchargement direct ; [M14](M14-notebooks.md#données-sur-colab)). Ce qui
est sur Drive aujourd'hui :

| Jeu | Source sur Colab |
|---|---|
| voc | `MyDrive/EmbeddedCV/data/voc.tar` |
| visdrone | `MyDrive/EmbeddedCV/data/visdrone.tar` |
| kitti | téléchargement direct (`get_datasets.sh kitti`) |
| flir | **manquant** : faire `tools/get_datasets.sh push flir` sur le PC avant |

À vérifier au premier essai : `drive.mount()` (`colab.mount_drive`) demande une
autorisation dans le notebook. Si elle ne s'affiche pas sous VS Code, monter Drive par la
commande de l'extension « Colab: Mount Google Drive to Server… », puis relancer la cellule.

## Lancer

`DEVICE` vaut `'gpu'` par défaut dans `_train` et `_sweep` (`'cpu'` pour les lancer sur
un PC sans GPU). Paramètres à régler dans la cellule « Paramètres », avant « Run All » :

| Notebook | Paramètres |
|---|---|
| `_train` | `ITERS` (4000), `BATCH` (16), `SUBSET` (50 ; 0 : split complet) |
| `_sweep` | `BATCHES` (`[8, 16, 32]`), `TRAIN_SUBSETS` (`[500, 0]`), `ITERS` (600 par run), `SKIP_DONE` |

Ces changements ne se commitent pas : les notebooks sont générés depuis
`tools/notebooks/gabarits.py` (`make notebooks`) et `--check` refuse une cellule
retouchée. Pour changer une valeur par défaut, modifier le gabarit.

Ordre conseillé, un notebook à la fois par runtime :

1. `voc/tiny-yolov3-voc_train` (archive sur Drive, référence de M2) ;
2. `kitti/tiny-yolov3-kitti_train`, puis `_sweep` ;
3. `visdrone/tiny-yolov3-visdrone_train`, puis `_sweep` ;
4. `flir/tiny-yolov3-flir_train` une fois `flir.tar` sur Drive.

## Coupure de session

Les sorties sont copiées dans `MyDrive/EmbeddedCV/runs/build/notebooks/<jeu>/<modèle>/`
toutes les 10 min et en fin de cellule. Après une coupure : reconnecter un kernel Colab
et relancer « Run All ». `colab.restore_outputs` recopie le dossier, l'entraînement
reprend au dernier checkpoint (`--resume`), et `_sweep` saute les runs finis
(`SKIP_DONE`). La reprise exacte suppose le même backend (T12.11).

## Suivre depuis le PC

```bash
rclone lsd gdrive:EmbeddedCV/runs/build/notebooks/kitti/tiny-yolov3-kitti/runs
rclone cat gdrive:EmbeddedCV/runs/build/notebooks/kitti/tiny-yolov3-kitti/loss.csv | tail
rclone lsl gdrive:EmbeddedCV/runs/build/notebooks/kitti/tiny-yolov3-kitti/ | grep weights
```

Un run est fini quand son `final.weights` existe. Le `loss.csv` sur Drive date au plus de
la dernière copie (10 min).

## Récupérer et comparer

```bash
rclone copy gdrive:EmbeddedCV/runs/build/notebooks/kitti build/notebooks/kitti
```

Ensuite, en local, le notebook d'inférence du modèle
([kitti/tiny-yolov3-kitti_infer.ipynb](../../notebooks/kitti/tiny-yolov3-kitti_infer.ipynb))
prend le run `RUN` (None : le plus récent), et `COMPARE = True` met les runs dans une même
table (T14.11). `run.json` donne le backend (`device`). Une mAP sur `SUBSET` images classe
les runs, elle n'est pas publiable (M12).

## Sorties des cellules et GitHub

Sous VS Code, les sorties d'un kernel Colab (texte, figures) s'enregistrent dans le
`.ipynb` du PC. GitHub les affiche si le fichier est poussé avec, les images comprises
(PNG en base64), mais pas le JavaScript ni les widgets, et l'aperçu échoue sur les gros
notebooks. Règle du dépôt ([M14](M14-notebooks.md#règles)) :

- notebook **exécuté en entier sans erreur**, cellules du gabarit intactes : il se commite
  avec ses sorties, et `make notebooks` le garde ;
- exécution partielle ou en erreur (ex. jeu absent) : `make ci` la refuse ; vider les
  sorties (« Clear All Outputs » dans VS Code, ou `make notebooks`) avant le commit, ou
  relancer jusqu'au bout ;
- les paramètres se changent sans commit, et une cellule ajoutée (ex. `!nvidia-smi`) se
  retire avant le commit.

Les fichiers de `build/notebooks/<jeu>/<modèle>/` (copiés sur Drive) restent la trace des
runs, et les chiffres publiables vont dans `results/` par les scripts de M12.
