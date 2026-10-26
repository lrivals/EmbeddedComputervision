# M43 — Construction-PPE: persons and protective equipment (optional)

Goal: bring Construction-PPE (Ultralytics, YOLO format, `Person` and PPE classes) into the
M11 chain: first as a small **person** set for site footage, next to
[PREWID](M38-prewid.md) and [SODA](M42-soda.md); then, as a last optional task, as the
first **PPE** set of the project, which M38 lists as out of its scope. The boxes exist; the
[M22](M22-autolabel.md) labeller only checks that no person was left unboxed. The data
chain is the one of the other datasets.

| What it tests | Why |
|---|---|
| a ready YOLO set | the prep is a class remap: the cheapest industrial set to add |
| small training set | 1,416 images: fine-tuning from pre-trained weights with few images |
| PPE classes | helmet, vest and their "missing" classes: a worker without a helmet is a safety case PREWID cannot express |
| cross-domain transfer | Construction-PPE-trained → PREWID test |
| out-of-domain baseline | `Person` maps to VOC / COCO / CrowdHuman `person` |

**State**: not started (2026-10-26). Same dataset search as
[M41](M41-safe-unsafe-behaviours.md) and [M42](M42-soda.md). All tasks run on the PC
(Colab for the GPU runs).

## Summary

From the Ultralytics page (2026-10-26), to check on the local copy in T43.0.

| Property | Value |
|---|---|
| Name | Construction-PPE |
| Source | <https://docs.ultralytics.com/datasets/detect/construction-ppe/> (YAML with the download link) |
| Licence | **AGPL-3.0** (Ultralytics page): the images stay out of the repository; the Drive archive stays private; the datasheet carries the licence |
| Images | 1,416 |
| Classes | 11, among them `Person`, PPE worn and PPE missing; exact names and ids in T43.0 |
| Annotations | YOLO txt, one per image |
| Official split | train / val (/ test) of the YAML, to check |
| Pre-trained mapping | `Person` → VOC / COCO / CrowdHuman `person` |

## Preprocessing

```bash
tools/get_datasets.sh construction_ppe          # download from the YAML link, then the prep
python tools/prep_datasets.py construction_ppe --preview build/construction_ppe_preview.png
tools/get_datasets.sh check                     # "construction_ppe : prêt"
```

- **Classes**: `Person` → `person` (id 0) for the person loader; the PPE classes are kept
  in a second list, `CONSTRUCTION_PPE_CLASSES`, used by T43.6 only.
- **Split**: the official one.
- **Completeness**: the check of [T42.3](M42-soda.md) on 100 test images.
- **Output**: `data/construction_ppe/annotations_{train,val,test}.json` (person only) and
  `annotations_ppe_{train,val,test}.json` (all classes).

### Known limits

- **Photos, not CCTV**: many images are close-ups taken by hand; few far workers, few empty
  images. Its role is variety, not false alarms.
- **AGPL-3.0**: to keep in mind before any redistribution of derived weights or files.

## Conventions

Those of M11, M14, M16 and M38 apply. Files to touch: as in M38 with `construction_ppe`
in place of `prewid` (`CONSTRUCTION_PPE_PERSON_CLASSES`, `CONSTRUCTION_PPE_CLASSES`,
`load_construction_ppe`, `DATASETS["construction_ppe"]`, `MAPPINGS`; `prep_datasets.py`;
`get_datasets.sh`; `FICHES`; notebook registry; profiles `construction_ppe-prep`,
`construction_ppe-train`; `test_prep_construction_ppe`, `test_construction_ppe`).

### Stats question

| Dataset | Own question (`stats` notebook) | Task |
|---|---|---|
| construction_ppe | How tall are the persons at the network input, and what share of them lacks a helmet or a vest? | T43.3 |

---

## A. Data

### [ ] T43.0 — Inventory and `tools/get_datasets.sh`
- **Spec**: — · **Depends on**: T11.0 · **Runs on**: PC · **Size**: S
- **Deliverables**: `MARKER`, `HOWTO` (source, AGPL-3.0), `construction_ppe` command; the
  Summary checked on the local copy (class names and ids, split, image sizes)
- **Acceptance**: `tools/get_datasets.sh check` lists `construction_ppe`

### [ ] T43.1 — Remap, split and loader
- **Spec**: §5.1, §8.3 · **Depends on**: T43.0 · **Runs on**: PC · **Size**: S
- **Deliverables**: `tools/prep_datasets.py construction_ppe` (person and PPE files,
  preview); loaders, `DATASETS`, `MAPPINGS`; tests
- **Acceptance**: `load("construction_ppe")` gives the test counts of the Summary

### [ ] T43.2 — Annotation completeness
- **Spec**: §8.3 · **Depends on**: T43.1, T41.2 · **Runs on**: PC · **Size**: S
- **Deliverables**: the T42.3 check on 100 test images; rate in the Summary
- **Acceptance**: the rate is reported with the decision

## B. Registry and notebooks

### [ ] T43.3 — Datasheet, notebooks, stats run and Drive
- **Spec**: — · **Depends on**: T43.1 · **Runs on**: PC · **Size**: S
- **Deliverables**: `FICHES["construction_ppe"]` (AGPL-3.0); notebook registry entries;
  profiles; `construction_ppe_stats` run, answer in [stats-jeux.md](stats-jeux.md);
  private archive on the Drive
- **Acceptance**: `tools/data_stats.py --dataset construction_ppe --sample 8` and
  `python -m tools.notebooks all --check` pass

## C. Person detection

### [ ] T43.4 — Out-of-domain baseline
- **Spec**: §8.3 · **Depends on**: T43.1 · **Runs on**: PC · **Size**: S
- **Deliverables**: person AP50 on `test` of the pre-trained networks (settings of T38.6)
- **Acceptance**: the table is written

### [ ] T43.5 — Fine-tuning and cross-domain
- **Spec**: §7 · **Depends on**: T43.3, T43.4 · **Runs on**: PC (Colab for the GPU runs) ·
  **Size**: S
- **Deliverables**: `_train` notebook; person AP50 on `test`; on PREWID `test` of the
  result
- **Acceptance**: fine-tuned AP50 above the best baseline; the PREWID number is reported

## D. PPE (optional)

### [ ] T43.6 — PPE detection
- **Spec**: §7, §8.3 · **Depends on**: T43.5 · **Runs on**: PC (Colab) · **Size**: M
- **Deliverables**: a Tiny-YOLO fine-tuned on the 11 classes; per-class AP50; recall of the
  "missing PPE" classes at the demo threshold
- **Acceptance**: the table is written; the missing-PPE classes with too few boxes to
  measure are named
- **Notes**: the first non-person safety class of the project; the integer and FPGA stages
  follow the M4 chain as for any other dataset.

## Out of scope

- PPE from other sets (SODA PPE classes, if any, stay in M42's files).
- Pose, tracking.

## Recommended order

T43.0 → T43.1 → T43.4 at once, T43.2, T43.3 → T43.5; T43.6 last.
