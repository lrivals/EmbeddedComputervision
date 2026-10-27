# Dataset statistics (M16)

Summary of the notebooks `notebooks/<dataset>/<dataset>_stats.ipynb`
([M16](M16-presentation-jeux.md)). The blocks between `data_stats` markers are generated
from `build/notebooks/<dataset>/stats/stats.json` by
`python tools/data_stats.py --report docs/tasks/stats-jeux.md`, and stay in French until
their generator is translated (M19, T19.6); the rest (answers to the specific questions) is
written by hand. This document serves as the reference for the tasks of M11 and M15.

Conventions: quantities "seen by the network" in a 416×416 letterbox; "small": area
< 32² px at the input; "smaller than a cell": longest side under the stride of the head
(32 px for 13×13, 16 px for 26×26); "lost targets": non-difficult objects that fall on the
same cell and the same anchor as another one (`build_targets` keeps only one), with the
anchors of the dataset's fine-tuning cfg. Pixels on 200 images per split (seed 0).

## Comparison

<!-- data_stats:comparatif -->
| dataset | classes | images | objects | objects/image (med.) | small at the input | smaller than a fine cell | lost targets | images > 256 | mean luminance | IoU of the cfg anchors |
|---|---|---|---|---|---|---|---|---|---|---|
| voc | 20 | 21,503 | 62,199 | 2 | 8.4 % | 0.3 % | 0.69 % | 0 | 112 | 0.603 |
| coco | 80 | 123,287 | 896,782 | 4 | 44.3 % | 14.6 % | 25.82 % | 0 | 111 | 0.550 |
| visdrone | 10 | 7,019 | 381,963 | 43 | 97.0 % | 75.3 % | 32.40 % | 24 | 97 | 0.601 |
| crowdhuman | 1 | 19,370 | 566,493 | 20 | 47.5 % | 10.0 % | 19.82 % | 32 | 113 | 0.679 |
| exdark | 12 | 5,563 | 17,498 | 2 | 18.1 % | 1.0 % | 0.74 % | 0 | 33 | 0.615 |
| flir | 15 | 11,886 | 191,728 | 15 | 85.6 % | 50.7 % | 17.70 % | 0 | 133 | 0.564 |
| auair | 8 | 32,823 | 131,977 | 3 | 79.8 % | 34.7 % | 8.88 % | 0 | 123 | 0.668 |
| dronevehicle | 2 | 14,726 | 235,858 | 12 | 65.4 % | 0.4 % | 2.88 % | 0 | 80 | 0.735 |
| hituav | 4 | 2,579 | 22,298 | 6 | 84.2 % | 44.6 % | 7.76 % | 0 | 125 | 0.698 |
| uavdt | 3 | 40,735 | 798,795 | 15 | 96.9 % | 56.9 % | 14.68 % | 0 | 97 | 0.711 |
| asl | 28 | 76,522 | 73,522 | 1 | 0.0 % | 0.0 % | 0.00 % | 0 | 132 | 0.768 |
| pets | 37 | 6,964 | 6,964 | 1 | 0.0 % | 0.0 % | 0.00 % | 0 | 114 | 0.824 |
| prewid | 1 | 3,413 | 1,924 | 1 | 38.5 % | 0.4 % | 0.16 % | 0 | 138 | 0.777 |
<!-- /data_stats:comparatif -->

## VOC

<!-- data_stats:voc -->
Revision `214f269`, input 416×416 (letterbox), cfg `tiny-yolov3-voc`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset voc --split 2007:trainval,2012:trainval,2007:test --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| 2007:trainval | 5,011 | 15,662 | 3,054 | 0 | 2 / 42 | 7.7 % | 13x13 3.2 % · 26x26 0.1 % | 0.62 % | 0 / 0 |
| 2012:trainval | 11,540 | 31,561 | 4,111 | 0 | 2 / 56 | 9.1 % | 13x13 4.7 % · 26x26 0.4 % | 0.73 % | 0 / 0 |
| 2007:test | 4,952 | 14,976 | 2,944 | 0 | 2 / 41 | 7.3 % | 13x13 3.5 % · 26x26 0.1 % | 0.66 % | 0 / 0 |
| train group | 16,551 | 47,223 | 7,165 | 0 | 2 / 56 | 8.7 % | 13x13 4.2 % · 26x26 0.3 % | 0.69 % | 0 / 0 |
| all | 21,503 | 62,199 | 10,109 | 0 | 2 / 56 | 8.4 % | 13x13 4.0 % · 26x26 0.3 % | 0.69 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| 2007:test | 4,952 | 12,032 | 4,952 / 12,032 | ok |
| 2007:trainval | 5,011 | 12,608 | 5,011 / 12,608 | ok |
| 2012:trainval | 11,540 | 27,450 | 11,540 / 27,450 | ok |
<!-- /data_stats:voc -->

**Question (T13.7, T13.13): are the classes with a low AP (bottle, pottedplant) rare or
small?** Neither is rare: on VOC2007 test, bottle (469 non-difficult objects, 3.9 %) and
pottedplant (480, 4.0 %) are more frequent than 12 other classes. They are **small**:
bottle has the smallest median side at the input (55 px in a 416 letterbox, 15 % small)
and pottedplant the fourth (80 px). Size does not explain everything: car has as many
small objects (11 %, side 93 px) for an AP of 67.0 against 21.6 (bottle) and 27.5
(pottedplant), and chair (35.3) has only 2 % small objects. What remains is the shape
(thin or irregular objects) and occlusion. Collisions are negligible everywhere (0.69 % of
the targets).

## COCO val2017

Not run: `data/coco` missing from the PC (T16.13 stays open).

## KITTI

Not run: only the `testing/` images (without annotations) are on the PC,
`data_object_image_2.zip` is incomplete (T16.14 stays open).

## VisDrone

<!-- data_stats:visdrone -->
Revision `214f269`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-visdrone.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset visdrone --split train,val --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 6,471 | 343,204 | 0 | 0 | 42 / 902 | 96.9 % | 13x13 93.8 % · 26x26 75.1 % | 32.33 % | 21 / 0 |
| val | 548 | 38,759 | 0 | 0 | 65 / 317 | 97.9 % | 13x13 95.2 % · 26x26 77.6 % | 33.04 % | 3 / 0 |
| all | 7,019 | 381,963 | 0 | 0 | 43 / 902 | 97.0 % | 13x13 93.9 % · 26x26 75.3 % | 32.40 % | 24 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 6,471 | 343,204 | 6,471 / — | ok |
| val | 548 | 38,759 | 548 / — | ok |
<!-- /data_stats:visdrone -->

**Question (T11.5, T15): share of objects smaller than a 26×26 cell; target collisions.**
In a 416 letterbox, **75 %** of the objects are smaller than a 26×26 cell (16 px) and 94 %
smaller than a 13×13 cell; 97 % are "small" in the COCO sense. A third of the targets
(**32.4 %**) are lost by collision: Tiny-YOLOv3 only sees one object per cell and per
anchor. On val, the car class (36 % of the objects) has a median side of 10 px and loses
28 % of its targets; pedestrian and people (5 px, 100 % small) lose 40 and 43 %, motor
(6 px) 33 %. This explains why only car takes off in the sweeps of M15. What is needed is
a larger input or tiles, not more iterations. The loader removes 10,191 ignored regions
and 1,564 "others"; 32,959 boxes are less than 2 px wide at the input.

## FLIR ADAS v2

<!-- data_stats:flir -->
Revision `3263136`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-flir-c1.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset flir --split train,val --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 10,742 | 175,032 | 0 | 264 | 15 / 93 | 85.6 % | 13x13 77.6 % · 26x26 50.6 % | 18.06 % | 0 / 0 |
| val | 1,144 | 16,696 | 0 | 16 | 14 / 63 | 86.3 % | 13x13 79.4 % · 26x26 51.2 % | 13.87 % | 0 / 0 |
| all | 11,886 | 191,728 | 0 | 280 | 15 / 93 | 85.6 % | 13x13 77.8 % · 26x26 50.7 % | 17.70 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 10,742 | 175,032 | 10,742 / — | ok |
| val | 1,144 | 16,696 | 1,144 / — | ok |
<!-- /data_stats:flir -->

**Question (T11.7): distribution of the thermal levels, and what the INT8 of L00 keeps of
it.** The 8-bit thermal images of the dataset are single-channel and brighter than VOC
(mean luminance 133 against 108 on VOC2007 test). The 256 levels are all used, and 99 % of
the pixels fit in 236 of them. The INT8 input of L00 (scale 1/127, `yolo.quant`) only keeps
**128**: two neighbouring thermal levels give the same integer, over the whole range. The
dominant object is small (86 % small at 416, 51 % under the 26×26 cell) and 18 % of the
targets are lost by collision. The quantization of the input therefore probably weighs
less than the size of the objects.

## ExDark

<!-- data_stats:exdark -->
Revision `214f269`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-exdark.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset exdark --split train,test --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 3,000 | 10,381 | 0 | 1 | 2 / 33 | 17.8 % | 13x13 10.4 % · 26x26 0.6 % | 0.56 % | 0 / 0 |
| test | 2,563 | 7,117 | 0 | 0 | 2 / 30 | 18.5 % | 13x13 11.8 % · 26x26 1.7 % | 1.00 % | 0 / 0 |
| all | 5,563 | 17,498 | 0 | 1 | 2 / 33 | 18.1 % | 13x13 11.0 % · 26x26 1.0 % | 0.74 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 3,000 | 10,381 | 3,000 / — | ok |
| test | 2,563 | 7,117 | 2,563 / — | ok |
<!-- /data_stats:exdark -->

**Question (T11.3): luminance per lighting type, and gap to the VOC calibration images.**
Mean luminance **33** on 200 ExDark images, against **108** on VOC2007 test (the
calibration split of the INT8 scales of M4): 3.2 times darker. Per lighting type
(imageclasslist.txt): Low 6.5, Weak 20, Ambient 25, Single 32, Window 35, Strong 38, Object
40, Screen 59, Twilight 78 (3 to 63 images per type, to be read with care). At the input
scale 1/127, an average ExDark image only uses about the first 16 integer levels out of
127. A calibration on VOC sets the activation scales for an input range that these images
do not reach: that is what T11.3 is about. On the geometry side, nothing special (18 %
small, 0.7 % of targets lost).

## CrowdHuman

<!-- data_stats:crowdhuman -->
Revision `214f269`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-crowdhuman.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset crowdhuman --split train,val --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 15,000 | 438,783 | 99,218 | 0 | 20 / 468 | 47.6 % | 13x13 30.4 % · 26x26 10.1 % | 19.91 % | 28 / 0 |
| val | 4,370 | 127,710 | 28,229 | 0 | 20 / 365 | 47.2 % | 13x13 29.4 % · 26x26 9.8 % | 19.55 % | 4 / 0 |
| all | 19,370 | 566,493 | 127,447 | 0 | 20 / 468 | 47.5 % | 13x13 30.2 % · 26x26 10.0 % | 19.82 % | 32 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 15,000 | 339,565 | 15,000 / 339,565 | ok |
| val | 4,370 | 99,481 | 4,370 / 99,481 | ok |
<!-- /data_stats:crowdhuman -->

**Question (T11.6): objects per image against the 256 slots; share of occluded objects.**
Median of 20 people per image, maximum 468; **32 images** out of 19,370 (0.17 %) exceed
256 annotated boxes, none exceeds 1,024. The slots of `yolo_post` are therefore almost
always enough for the ground truth. The bottleneck is the grid: **19.8 %** of the targets
are lost by collision. **22.5 %** of the boxes are `difficult` (`mask` and `ignore`).
77,738 `fbox` boxes overflow the image: they are clipped by the loader, 15 become empty
and are removed.

## Oxford-IIIT Pet

<!-- data_stats:pets -->
Revision `145dc7c`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-pets.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset pets --split train,test --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 3,302 | 3,302 | 0 | 0 | 1 / 1 | 0.0 % | 13x13 0.0 % · 26x26 0.0 % | 0.00 % | 0 / 0 |
| test | 3,662 | 3,662 | 0 | 0 | 1 / 1 | 0.0 % | 13x13 0.0 % · 26x26 0.0 % | 0.00 % | 0 / 0 |
| all | 6,964 | 6,964 | 0 | 0 | 1 / 1 | 0.0 % | 13x13 0.0 % · 26x26 0.0 % | 0.00 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 3,302 | 3,302 | — / — | ok |
| test | 3,662 | 3,662 | — / — | ok |
<!-- /data_stats:pets -->

**Question (T32.8): how much larger is the body box than the head box, and how many images
hold more than one animal?** On the 3 662 trainval images that have a head box, the body box
(min / max of the trimap mask) is a median **3.7 times** the head box in area (q10 1.8,
q90 9.3): close-ups of the face are frequent, so the body box often fills most of the photo.
The mask splits into several parts over 1 % of the image on **131 of 7 332** images
(1.8 %). On the 48 that were checked by eye, almost all show one animal cut apart by a coat,
a hand, a collar or a leash. Only about three show two animals, which then get one box over
both. So the body box is the union of the mask, not one box per part (T32.5).

## PREWID

<!-- data_stats:prewid -->
Revision `fba540c`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-prewid.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset prewid --split train,test --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 2,821 | 1,581 | 0 | 1,348 | 1 / 3 | 37.5 % | 13x13 5.1 % · 26x26 0.3 % | 0.00 % | 0 / 0 |
| test | 592 | 343 | 0 | 289 | 1 / 2 | 42.9 % | 13x13 1.7 % · 26x26 0.6 % | 0.87 % | 0 / 0 |
| all | 3,413 | 1,924 | 0 | 1,637 | 1 / 3 | 38.5 % | 13x13 4.5 % · 26x26 0.4 % | 0.16 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 2,821 | 1,581 | — / — | ok |
| test | 592 | 343 | — / — | ok |
<!-- /data_stats:prewid -->

**Question (T38.3): how far are the workers from each camera (box height per camera), and
what share of the frames is empty?** Box height in pixels of the native frame (480 rows for
the plant cameras, 1080 for the lab), with the share of empty frames per split:

| Camera | Frames | Empty (train / val / test / lab) | Boxes | Box height px: q10 / median / q90 | Under 32 px |
|---|---|---|---|---|---|
| CAM0 | 1 001 | 62 % / 63 % / 64 % / – | 391 | 60 / 89 / 179 | 0 |
| CAM1 | 1 000 | 30 % / 32 % / 31 % / – | 753 | 60 / 111 / 195 | 3 |
| CAM2 | 1 000 | 40 % / 41 % / 41 % / – | 657 | 48 / 71 / 107 | 37 |
| CAM3 | 1 002 | 59 % / 59 % / 60 % / – | 447 | 87 / 152 / 240 | 0 |
| CAMLAB01 | 3 000 | – / – / – / 48 % | 1 574 | 115 / 392 / 719 | 7 |

CAM2 sees the workers farthest away (median 71 px, 37 boxes under 32 px, at the back of the
aisle), CAM3 the closest (median 152 px). At the 416 input a plant frame shrinks by 0.65, so
a CAM2 worker is a median 46 px high and about 15 px wide: 38.5 % of the boxes are "small"
at the input although almost none is smaller than a fine cell (the boxes are tall and
narrow; k-means anchors 15×37 to 82×88, `results/anchors_prewid.md`). **Empty frames are
45 % of the plant frames and 48 % of the lab ones**; the split keeps the same share in train,
val and test for each camera (empty frames and worker frames are cut separately, T38.1), so
`test` has 289 empty frames to measure false alarms on.

<!-- data_stats:coco -->
Revision `3263136`, input 416×416 (letterbox), cfg `tiny-yolov3-coco`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset coco --split train2017,val2017 --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train2017 | 118,287 | 860,001 | 10,052 | 1,021 | 4 / 93 | 44.3 % | 13x13 33.8 % · 26x26 14.6 % | 25.83 % | 0 / 0 |
| val2017 | 5,000 | 36,781 | 446 | 48 | 4 / 63 | 44.5 % | 13x13 34.1 % · 26x26 14.2 % | 25.61 % | 0 / 0 |
| all | 123,287 | 896,782 | 10,498 | 1,069 | 4 / 93 | 44.3 % | 13x13 33.8 % · 26x26 14.6 % | 25.82 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| val2017 | 5,000 | 36,781 | 5,000 / 36,781 | ok |
| train2017 | 118,287 | 860,001 | 118,287 / 860,001 | ok |
<!-- /data_stats:coco -->

<!-- data_stats:auair -->
Revision `3263136`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-auair.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset auair --split train,val --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 27,880 | 114,237 | 0 | 2 | 3 / 52 | 81.7 % | 13x13 70.3 % · 26x26 35.1 % | 8.84 % | 0 / 0 |
| val | 4,943 | 17,740 | 0 | 1 | 3 / 56 | 67.9 % | 13x13 60.1 % · 26x26 31.5 % | 9.15 % | 0 / 0 |
| all | 32,823 | 131,977 | 0 | 3 | 3 / 56 | 79.8 % | 13x13 69.0 % · 26x26 34.7 % | 8.88 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 27,880 | 114,237 | 27,880 / — | ok |
| val | 4,943 | 17,740 | 4,943 / — | ok |
<!-- /data_stats:auair -->

<!-- data_stats:dronevehicle -->
Revision `3263136`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-dronevehicle.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset dronevehicle --split train,test --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 12,118 | 193,940 | 0 | 0 | 12 / 206 | 65.4 % | 13x13 23.7 % · 26x26 0.4 % | 2.90 % | 0 / 0 |
| test | 2,608 | 41,918 | 0 | 0 | 12 / 145 | 65.2 % | 13x13 25.1 % · 26x26 0.5 % | 2.80 % | 0 / 0 |
| all | 14,726 | 235,858 | 0 | 0 | 12 / 206 | 65.4 % | 13x13 23.9 % · 26x26 0.4 % | 2.88 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 12,118 | 193,940 | 12,118 / 193,940 | ok |
| test | 2,608 | 41,918 | 2,608 / 41,918 | ok |
<!-- /data_stats:dronevehicle -->

<!-- data_stats:hituav -->
Revision `3263136`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-hituav-c1.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset hituav --split train,test --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 2,008 | 17,518 | 0 | 0 | 6 / 42 | 83.9 % | 13x13 77.9 % · 26x26 43.7 % | 7.76 % | 0 / 0 |
| test | 571 | 4,780 | 0 | 0 | 6 / 32 | 85.2 % | 13x13 79.7 % · 26x26 48.2 % | 7.74 % | 0 / 0 |
| all | 2,579 | 22,298 | 0 | 0 | 6 / 42 | 84.2 % | 13x13 78.3 % · 26x26 44.6 % | 7.76 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 2,008 | 17,518 | 2,008 / 17,518 | ok |
| test | 571 | 4,780 | 571 / 4,780 | ok |
<!-- /data_stats:hituav -->

<!-- data_stats:uavdt -->
Revision `3263136`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-uavdt.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset uavdt --split train,test --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 24,143 | 422,911 | 0 | 314 | 15 / 66 | 97.5 % | 13x13 93.4 % · 26x26 45.6 % | 9.27 % | 0 / 0 |
| test | 16,592 | 375,884 | 0 | 12 | 15 / 107 | 96.3 % | 13x13 94.0 % · 26x26 69.6 % | 20.77 % | 0 / 0 |
| all | 40,735 | 798,795 | 0 | 326 | 15 / 107 | 96.9 % | 13x13 93.7 % · 26x26 56.9 % | 14.68 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 24,143 | 422,911 | 24,143 / 422,911 | ok |
| test | 16,592 | 375,884 | 16,592 / 375,884 | ok |
<!-- /data_stats:uavdt -->

<!-- data_stats:asl -->
Revision `3263136`, input 416×416 (letterbox), cfg `build/m11/cfg/tiny-yolov3-asl.cfg`, pixels on 200 images (seed 0). Command: `python tools/data_stats.py --dataset asl --split train,val --sample 200`.

| part | images | objects | difficult | empty images | objects/image (med. / max) | small < 32² at 416×416 | smaller than a cell | lost targets | images > 256 / > 1,024 |
|---|---|---|---|---|---|---|---|---|---|
| train | 69,984 | 67,434 | 0 | 2,550 | 1 / 1 | 0.0 % | 13x13 0.0 % · 26x26 0.0 % | 0.00 % | 0 / 0 |
| val | 6,538 | 6,088 | 0 | 450 | 1 / 1 | 0.0 % | 13x13 0.0 % · 26x26 0.0 % | 0.00 % | 0 / 0 |
| all | 76,522 | 73,522 | 0 | 3,000 | 1 / 1 | 0.0 % | 13x13 0.0 % · 26x26 0.0 % | 0.00 % | 0 / 0 |

| split | images | objects | official |  |
|---|---|---|---|---|
| train | 69,984 | 67,434 | — / — | ok |
| val | 6,538 | 6,088 | — / — | ok |
<!-- /data_stats:asl -->
