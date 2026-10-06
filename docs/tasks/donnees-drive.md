# Jeux de données sur Google Drive

Les jeux ne passent pas par git (`data/` fait 33 Go). Pour les retrouver sur une autre
machine ou sur Colab, chaque jeu prêt est archivé en un tar et copié sur Google Drive, dans
`MyDrive/EmbeddedCV/data/`. Les notebooks et `tools/notebooks/colab.py:prepare_data` savent
le reprendre de là (section « Accès » ci-dessous).

```
MyDrive/EmbeddedCV/
  data/   <jeu>.tar   un tar non compressé par jeu, liens suivis (arborescence de data/)
  runs/               sorties et checkpoints des notebooks (colab.sync_outputs ; M14, T14.8)
```

Les runs d'entraînement suivent le même chemin que dans le dépôt (section « Runs
d'entraînement » plus bas).

## Mise en place de rclone (une fois par PC)

```bash
sudo apt install rclone
rclone config create gdrive drive scope=drive   # ouvre le navigateur : connexion Google
rclone lsd gdrive:                              # liste les dossiers du Drive
rclone about gdrive:                            # place libre
```

Le remote s'appelle `gdrive`, et les archives vont dans `gdrive:EmbeddedCV/data`. Pour en
changer, définir la variable d'environnement `DRIVE_REMOTE` (par exemple
`DRIVE_REMOTE=autre:Dossier/data`), lue par `tools/get_datasets.sh` et `colab.prepare_data`.

## Exporter un jeu

Sur le PC qui a le jeu (témoin présent, `tools/get_datasets.sh check`) :

```bash
tools/get_datasets.sh push voc visdrone crowdhuman exdark
```

`push` lance `pack` (`data/<racine>` → `data_archives/<jeu>.tar`, liens symboliques suivis,
`*.zip`, `*.tar` et `*.part` exclus), puis `rclone copy` vers `gdrive:EmbeddedCV/data`. L'archive
reste dans `data_archives/`, qui est ignoré par git, et peut être supprimée une fois
l'envoi vérifié :

```bash
rclone ls gdrive:EmbeddedCV/data     # tailles sur le Drive
ls -l data_archives/                 # doivent être identiques
```

| Jeu | Taille | Témoin (`data/…`) | Sur le Drive | Remarque |
|---|---|---|---|---|
| voc | 2,9 Go | `VOCdevkit/VOC2007/ImageSets/Main/test.txt` | ✓ (6 oct. 2026) | aussi par `tools/get_voc.sh` |
| exdark | 1,5 Go | `exdark/imageclasslist.txt` | ✓ (6 oct. 2026) | seule voie : annotations absentes de Kaggle |
| visdrone | 1,9 Go | `visdrone/VisDrone2019-DET-val/annotations` | ✓ (6 oct. 2026) | aussi par l'API Kaggle |
| crowdhuman | 11 Go | `crowdhuman/annotation_val.odgt` | ✓ (6 oct. 2026) | aussi par l'API Kaggle ; le plus long à envoyer |
| coco | ≈ 1,3 Go | `coco/annotations/instances_val2017.json` | — | absent du PC ; se retélécharge (`get_datasets.sh coco`) |
| kitti | ≈ 12 Go | `kitti/training/image_2/000000.png` | — | absent du PC ; se retélécharge (`get_datasets.sh kitti`) |
| flir | — | `flir/images_thermal_val/coco.json` | — | inscription, ou l'API Kaggle (miroir `samdazel/teledyne-flir-adas-thermal-dataset-v2`) |

Les tailles sont celles de `du -shL` sur le PC. Ensemble, les quatre premiers jeux font environ 17 Go.
C'est plus que le Drive gratuit (15 Go), mais le compte du projet a 100 Go. Vérifier la place
avec `rclone about gdrive:` avant un gros envoi. Après un nouvel envoi, mettre à jour la
colonne « Sur le Drive ».

## Accès

La règle est la même partout : **un jeu présent est utilisé tel quel ; sinon, il est repris du
Drive** ; en dernier recours, il est téléchargé, mais seulement sur Colab.

| Où | Jeu présent | Jeu absent |
|---|---|---|
| PC, ligne de commande | rien à faire | `tools/get_datasets.sh pull <jeu>` |
| PC, notebook | utilisé tel quel | `env.prepare` lance `pull` (rclone) ; à défaut, `Prerequis` avec les commandes |
| Colab | utilisé tel quel | Drive monté + `unpack` ; à défaut, téléchargement direct (VOC, COCO, KITTI, Kaggle : CrowdHuman, VisDrone, FLIR) |

`pull` ne fait rien si le témoin est présent. Sinon, il copie `gdrive:EmbeddedCV/data/<jeu>.tar`
dans `data_archives/`, puis l'extrait dans `data/` (`unpack`). Le code de retour est différent de 0 si rclone,
le remote ou l'archive manque, et le message dit quoi faire.

Dans le code, `colab.prepare_data` essaie les voies dans l'ordre et rend celle qui a abouti :

```python
from tools.notebooks import colab

colab.prepare_data("exdark")   # "ready" | "drive" | "rclone" | "download"
```

1. `ready` : le témoin est présent ;
2. `drive` : `<drive_dir>/data/<jeu>.tar` sur le Drive monté (Colab, `/content/drive`) ;
3. `rclone` : la même archive par `get_datasets.sh pull` (PC avec rclone ; `remote=None`
   désactive cette voie) ;
4. `download` : le téléchargement direct (`download=False` le désactive).

Si aucune voie n'aboutit, la fonction lève `RuntimeError` avec la commande `push` à lancer sur
le PC qui a le jeu. En local, les notebooks appellent `prepare_data` sans téléchargement
(`env.prepare`) : un téléchargement de plusieurs Go reste une commande explicite.

## Runs d'entraînement

Sur Colab, avec `DRIVE_DIR` (défaut `/content/drive/MyDrive/EmbeddedCV`), les notebooks
`_train` et `_sweep` copient leur dossier toutes les 10 min, puis en fin de cellule
(`colab.sync_outputs`). La copie va dans `runs/` sous le même chemin relatif que dans le
dépôt :

```
MyDrive/EmbeddedCV/runs/build/notebooks/<jeu>/<modèle>/
  final.weights, loss.csv, run.json …   run « train » (notebook _train)
  runs/b16-sall/, runs/b8-s500/ …       balayage lot × sous-ensemble (notebook _sweep)
  eval/<run>/                           évaluations des notebooks d'inférence
```

En début de session, `colab.restore_outputs` les recopie dans `build/notebooks/`. Un
entraînement coupé reprend ainsi au dernier checkpoint copié (`--resume`), et le notebook
d'inférence retrouve les runs (`RUN`, `COMPARE` ; M14, T14.11). Pour récupérer les runs
entraînés sur Colab sur le PC :

```bash
rclone copy gdrive:EmbeddedCV/runs/build/notebooks build/notebooks   # tous les runs
rclone copy gdrive:EmbeddedCV/runs/build/notebooks/kitti build/notebooks/kitti
```

Les notebooks d'inférence en local lisent ensuite ces runs comme ceux entraînés sur le PC.
Un run entraîné sur GPU l'indique dans `run.json` (`device`).

## Pièges

- Seul le tar est lu sur le Drive, d'une traite. Les images ne sont jamais lues à
  travers le montage de Drive, qui est lent pour des milliers de petits fichiers.
- `pack` écrit d'abord `<jeu>.tar.part`, puis le renomme : un tar présent est complet. Un
  `.part` qui reste après une interruption peut être supprimé.
- ExDark : le lien `data/exdark/ExDark` pointe vers tout `data/ExDark Dataset/`. L'archive
  contient donc aussi les annotations sous `exdark/ExDark/ExDark_Annno/`. Elles font
  quelques Mo et ne sont pas lues.
- Un jeu modifié en local (nouvelles annotations, sous-ensemble) doit être renvoyé
  avec `push`. `pull` ne remplace jamais un jeu déjà présent.
