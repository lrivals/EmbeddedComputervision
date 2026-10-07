# Statistiques des jeux de données (M16)

Synthèse des notebooks `notebooks/<jeu>/<jeu>_stats.ipynb`
([M16](M16-presentation-jeux.md)). Les blocs entre marqueurs `data_stats` sont générés
depuis `build/notebooks/<jeu>/stats/stats.json` par
`python tools/data_stats.py --report docs/tasks/stats-jeux.md` ; le reste (réponses aux
questions propres) est écrit à la main. Ce document sert de référence aux tâches de M11 et
M15.

Conventions : grandeurs « vues par le réseau » en letterbox 416×416 ; « petits » : aire
< 32² px à l'entrée ; « plus petits qu'une cellule » : plus grand côté sous le pas de la
tête (32 px pour 13×13, 16 px pour 26×26) ; « cibles perdues » : objets non-difficult qui
tombent sur la même cellule et la même ancre qu'un autre (`build_targets` n'en garde qu'un),
avec les ancres de la cfg d'affinage du jeu. Pixels sur 200 images par partie (graine 0).

## Comparatif

<!-- data_stats:comparatif -->
| jeu | classes | images | objets | objets/image (méd.) | petits à l'entrée | plus petits qu'une cellule fine | cibles perdues | images > 256 | luminance moy. | IoU ancres de la cfg |
|---|---|---|---|---|---|---|---|---|---|---|
| voc | 20 | 21 503 | 62 199 | 2 | 8,4 % | 0,3 % | 0,69 % | 0 | 112 | 0,603 |
| visdrone | 10 | 7 019 | 381 963 | 43 | 97,0 % | 75,3 % | 32,40 % | 24 | 97 | 0,601 |
| crowdhuman | 1 | 19 370 | 566 493 | 20 | 47,5 % | 10,0 % | 19,82 % | 32 | 113 | 0,679 |
| exdark | 12 | 5 563 | 17 498 | 2 | 18,1 % | 1,0 % | 0,74 % | 0 | 33 | 0,615 |
| flir | 15 | 11 886 | 191 728 | 15 | 85,6 % | 50,7 % | 17,70 % | 0 | 133 | 0,564 |
<!-- /data_stats:comparatif -->

## VOC

<!-- data_stats:voc -->
Révision `214f269`, entrée 416×416 (letterbox), cfg `tiny-yolov3-voc`, pixels sur 200 images (graine 0). Commande : `python tools/data_stats.py --dataset voc --split 2007:trainval,2012:trainval,2007:test --sample 200`.

| partie | images | objets | difficult | images vides | objets/image (méd. / max) | petits < 32² à 416×416 | plus petits qu'une cellule | cibles perdues | images > 256 / > 1 024 |
|---|---|---|---|---|---|---|---|---|---|
| 2007:trainval | 5 011 | 15 662 | 3 054 | 0 | 2 / 42 | 7,7 % | 13x13 3,2 % · 26x26 0,1 % | 0,62 % | 0 / 0 |
| 2012:trainval | 11 540 | 31 561 | 4 111 | 0 | 2 / 56 | 9,1 % | 13x13 4,7 % · 26x26 0,4 % | 0,73 % | 0 / 0 |
| 2007:test | 4 952 | 14 976 | 2 944 | 0 | 2 / 41 | 7,3 % | 13x13 3,5 % · 26x26 0,1 % | 0,66 % | 0 / 0 |
| groupe train | 16 551 | 47 223 | 7 165 | 0 | 2 / 56 | 8,7 % | 13x13 4,2 % · 26x26 0,3 % | 0,69 % | 0 / 0 |
| ensemble | 21 503 | 62 199 | 10 109 | 0 | 2 / 56 | 8,4 % | 13x13 4,0 % · 26x26 0,3 % | 0,69 % | 0 / 0 |

| split | images | objets | officiel |  |
|---|---|---|---|---|
| 2007:test | 4 952 | 12 032 | 4 952 / 12 032 | ok |
| 2007:trainval | 5 011 | 12 608 | 5 011 / 12 608 | ok |
| 2012:trainval | 11 540 | 27 450 | 11 540 / 27 450 | ok |
<!-- /data_stats:voc -->

**Question (T13.7, T13.13) : les classes à faible AP (bottle, pottedplant) sont-elles
rares ou petites ?** Ni l'une ni l'autre ne sont rares : sur VOC2007 test, bottle (469 objets
non-difficult, 3,9 %) et pottedplant (480, 4,0 %) sont plus fréquentes que 12 autres classes.
Elles sont **petites** : bottle a le plus petit côté médian à l'entrée (55 px en letterbox 416,
15 % de petits) et pottedplant le quatrième (80 px). La taille n'explique pas tout : car a autant
de petits objets (11 %, côté 93 px) pour une AP de 67,0 contre 21,6 (bottle) et 27,5
(pottedplant), et chair (35,3) n'a que 2 % de petits. Restent la forme (objets fins ou
irréguliers) et l'occultation. Les collisions sont négligeables partout (0,69 % des cibles).

## COCO val2017

Non exécuté : `data/coco` absent du PC (T16.13 reste ouverte).

## KITTI

Non exécuté : seules les images `testing/` (sans annotations) sont sur le PC,
`data_object_image_2.zip` est incomplet (T16.14 reste ouverte).

## VisDrone

<!-- data_stats:visdrone -->
Révision `214f269`, entrée 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-visdrone.cfg`, pixels sur 200 images (graine 0). Commande : `python tools/data_stats.py --dataset visdrone --split train,val --sample 200`.

| partie | images | objets | difficult | images vides | objets/image (méd. / max) | petits < 32² à 416×416 | plus petits qu'une cellule | cibles perdues | images > 256 / > 1 024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 6 471 | 343 204 | 0 | 0 | 42 / 902 | 96,9 % | 13x13 93,8 % · 26x26 75,1 % | 32,33 % | 21 / 0 |
| val | 548 | 38 759 | 0 | 0 | 65 / 317 | 97,9 % | 13x13 95,2 % · 26x26 77,6 % | 33,04 % | 3 / 0 |
| ensemble | 7 019 | 381 963 | 0 | 0 | 43 / 902 | 97,0 % | 13x13 93,9 % · 26x26 75,3 % | 32,40 % | 24 / 0 |

| split | images | objets | officiel |  |
|---|---|---|---|---|
| train | 6 471 | 343 204 | 6 471 / — | ok |
| val | 548 | 38 759 | 548 / — | ok |
<!-- /data_stats:visdrone -->

**Question (T11.5, T15) : part des objets plus petits qu'une cellule 26×26 ; collisions
de cibles.** En letterbox 416, **75 %** des objets sont plus petits qu'une cellule 26×26
(16 px) et 94 % qu'une cellule 13×13 ; 97 % sont « petits » au sens COCO. Un tiers des cibles
(**32,4 %**) se perd par collision : Tiny-YOLOv3 ne voit qu'un objet par cellule et par ancre.
Sur val, la classe car (36 % des objets) a un côté médian de 10 px et perd 28 % de ses cibles ;
pedestrian et people (5 px, 100 % de petits) en perdent 40 et 43 %, motor (6 px) 33 %. Cela
explique que seule car décolle dans les balayages de M15. Il faut une entrée plus grande ou des
tuiles, pas plus d'itérations. Le chargeur retire 10 191 régions ignorées et 1 564 « others » ;
32 959 boîtes font moins de 2 px de côté à l'entrée.

## FLIR ADAS v2

<!-- data_stats:flir -->
Révision `214f269`, entrée 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-flir-c1.cfg`, pixels sur 200 images (graine 0). Commande : `python tools/data_stats.py --dataset flir --split train,val --sample 200`.

| partie | images | objets | difficult | images vides | objets/image (méd. / max) | petits < 32² à 416×416 | plus petits qu'une cellule | cibles perdues | images > 256 / > 1 024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 10 742 | 175 032 | 0 | 264 | 15 / 93 | 85,6 % | 13x13 77,6 % · 26x26 50,6 % | 18,06 % | 0 / 0 |
| val | 1 144 | 16 696 | 0 | 16 | 14 / 63 | 86,3 % | 13x13 79,4 % · 26x26 51,2 % | 13,87 % | 0 / 0 |
| ensemble | 11 886 | 191 728 | 0 | 280 | 15 / 93 | 85,6 % | 13x13 77,8 % · 26x26 50,7 % | 17,70 % | 0 / 0 |

| split | images | objets | officiel |  |
|---|---|---|---|---|
| train | 10 742 | 175 032 | 10 742 / — | ok |
| val | 1 144 | 16 696 | 1 144 / — | ok |
<!-- /data_stats:flir -->

**Question (T11.7) : distribution des niveaux thermiques, et ce qu'en garde l'INT8 de L00.**
Les images thermiques 8 bits du jeu sont à un canal et plus claires que VOC (luminance
moyenne 133 contre 108 sur VOC2007 test). Les 256 niveaux sont occupés, et 99 % des pixels
tiennent sur 236 d'entre eux. L'entrée INT8 de L00 (échelle 1/127, `yolo.quant`) n'en garde que
**128** : deux niveaux thermiques voisins donnent le même entier, sur toute la plage. L'objet
dominant est petit (86 % de petits à 416, 51 % sous la cellule 26×26) et 18 % des cibles se
perdent par collision. La quantification de l'entrée pèse donc sans doute moins que la taille
des objets.

## ExDark

<!-- data_stats:exdark -->
Révision `214f269`, entrée 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-exdark.cfg`, pixels sur 200 images (graine 0). Commande : `python tools/data_stats.py --dataset exdark --split train,test --sample 200`.

| partie | images | objets | difficult | images vides | objets/image (méd. / max) | petits < 32² à 416×416 | plus petits qu'une cellule | cibles perdues | images > 256 / > 1 024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 3 000 | 10 381 | 0 | 1 | 2 / 33 | 17,8 % | 13x13 10,4 % · 26x26 0,6 % | 0,56 % | 0 / 0 |
| test | 2 563 | 7 117 | 0 | 0 | 2 / 30 | 18,5 % | 13x13 11,8 % · 26x26 1,7 % | 1,00 % | 0 / 0 |
| ensemble | 5 563 | 17 498 | 0 | 1 | 2 / 33 | 18,1 % | 13x13 11,0 % · 26x26 1,0 % | 0,74 % | 0 / 0 |

| split | images | objets | officiel |  |
|---|---|---|---|---|
| train | 3 000 | 10 381 | 3 000 / — | ok |
| test | 2 563 | 7 117 | 2 563 / — | ok |
<!-- /data_stats:exdark -->

**Question (T11.3) : luminosité par type d'éclairage, et écart aux images de calibration
VOC.** Luminance moyenne **33** sur 200 images d'ExDark, contre **108** sur VOC2007 test (le
split de calibration des échelles INT8 de M4) : 3,2 fois plus sombre. Par type d'éclairage
(imageclasslist.txt) : Low 6,5, Weak 20, Ambient 25, Single 32, Window 35, Strong 38, Object
40, Screen 59, Twilight 78 (3 à 63 images par type, à lire avec prudence). À l'échelle
d'entrée 1/127, une image moyenne d'ExDark n'occupe que les 16 premiers niveaux entiers environ
sur 127. Une calibration sur VOC fixe les échelles des activations pour une dynamique d'entrée
que ces images n'atteignent pas : c'est l'enjeu de T11.3. Côté géométrie, rien de
particulier (18 % de petits, 0,7 % de cibles perdues).

## CrowdHuman

<!-- data_stats:crowdhuman -->
Révision `214f269`, entrée 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-crowdhuman.cfg`, pixels sur 200 images (graine 0). Commande : `python tools/data_stats.py --dataset crowdhuman --split train,val --sample 200`.

| partie | images | objets | difficult | images vides | objets/image (méd. / max) | petits < 32² à 416×416 | plus petits qu'une cellule | cibles perdues | images > 256 / > 1 024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 15 000 | 438 783 | 99 218 | 0 | 20 / 468 | 47,6 % | 13x13 30,4 % · 26x26 10,1 % | 19,91 % | 28 / 0 |
| val | 4 370 | 127 710 | 28 229 | 0 | 20 / 365 | 47,2 % | 13x13 29,4 % · 26x26 9,8 % | 19,55 % | 4 / 0 |
| ensemble | 19 370 | 566 493 | 127 447 | 0 | 20 / 468 | 47,5 % | 13x13 30,2 % · 26x26 10,0 % | 19,82 % | 32 / 0 |

| split | images | objets | officiel |  |
|---|---|---|---|---|
| train | 15 000 | 339 565 | 15 000 / 339 565 | ok |
| val | 4 370 | 99 481 | 4 370 / 99 481 | ok |
<!-- /data_stats:crowdhuman -->

**Question (T11.6) : objets par image face aux 256 emplacements ; part d'objets occultés.**
Médiane de 20 personnes par image, maximum 468 ; **32 images** sur 19 370 (0,17 %) dépassent
256 boîtes annotées, aucune ne dépasse 1 024. Les emplacements de `yolo_post` suffisent donc
presque toujours pour la vérité terrain. Le goulot est la grille : **19,8 %** des cibles se
perdent par collision. **22,5 %** des boîtes sont en `difficult` (`mask` et `ignore`).
77 738 boîtes `fbox` débordent de l'image : elles sont rognées par le chargeur, 15 deviennent
vides et sont retirées.
