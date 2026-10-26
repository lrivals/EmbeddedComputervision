# M42 — SODA: workers and machines on construction sites (optional)

Goal: bring SODA (Site Object Detection dAtaset, construction sites, boxed workers and
machines) into the M11 chain as a large **outdoor industrial** person set, next to
[PREWID](M38-prewid.md) (indoor plants, fixed cameras). The boxes exist: the
[M22](M22-autolabel.md) labeller is used only to check that no worker was left unboxed. The
data chain is the one of the other datasets (prep, loader in `DATASETS`, datasheet, M14 and
M16 notebooks, Drive, fine-tuning).

| What it tests | Why |
|---|---|
| many sites and conditions | about 20,000 images from several sites, weather and lighting: more variety than the four PREWID cameras |
| workers next to machines | excavators, cranes, scaffolds in the same images: the clutter a site camera sees |
| worker distance | workers from close-up to far away: miss rate per height of [M27](M27-safety-metrics.md) |
| cross-domain transfer | SODA-trained → PREWID test, and back: does an outdoor-site model work in a plant? |
| annotation completeness | the M22 labeller looks for unboxed workers before the boxes are trusted as ground truth |
| out-of-domain baseline | `person` maps to VOC / COCO / CrowdHuman `person` |

**State**: not started (2026-10-26). T42.0 holds a licence check: nothing goes into the
Drive archive before it. Same dataset search as [M41](M41-safe-unsafe-behaviours.md) and
[M43](M43-construction-ppe.md). All tasks run on the PC (Colab for the GPU runs).

## Summary

From the paper and the authors' page (2026-10-26), to check on the local copy in T42.0.

| Property | Value |
|---|---|
| Name | SODA, Site Object Detection dAtaset |
| Source | arXiv:2202.09554; data access from the authors' page <https://linjiarui.net/en/portfolio/2022-02-22-SODA-site-object-detection-dataset-for-deep-learning-in-construction> |
| Licence | **not confirmed** (T42.0, blocking) |
| Images | 19,846 |
| Objects | 286,201 boxes, 15 classes (workers, materials, machines, site elements); exact names in T42.0 |
| Camera | several sites and conditions; fixed cameras or not, and sequences or not, to check |
| Empty images | to check (needed for FPPI on empty scenes) |
| Official split | to check |
| Pre-trained mapping | worker class → VOC / COCO / CrowdHuman `person` |

## Preprocessing

```bash
tools/get_datasets.sh soda                     # checks the hand-copied folder, then the prep
python tools/prep_datasets.py soda --preview build/soda_preview.png
tools/get_datasets.sh check                    # "soda : prêt"
```

- **Classes**: the worker class becomes `person` (id 0); the other 14 classes are kept in
  the annotation files for a later multi-class use, and dropped by the `person` loader.
- **Split**: the official one if any. Otherwise by site when the images are sequences or
  share a site (to know in T42.0), random with a fixed seed only if they do not.
- **Checks**: boxes clipped to the image; under 4 px reported; worker boxes per image
  compared with the paper's counts.
- **Output**: `data/soda/annotations_{train,val,test}.json` (corners, labels, site if
  known).

### Annotation completeness (T42.3)

On 100 test images (fixed seed), the labeller of T41.2 (or `grounding-dino-tiny` at 0.55)
looks for `person`; a confident box with no ground-truth worker at IoU ≥ 0.5 is a
**candidate missing worker**, checked by eye in the editor. The rate of confirmed missing
workers says whether the test boxes can be taken as they are. If it is high (rule of
thumb: over 2 % of the workers), the test split is reviewed in the editor before the
safety tasks, and the reviewed copy is kept apart from the original.

### Known limits

- **Construction sites, not plants**: outdoor light, open space; the plant domain stays
  PREWID's.
- **Licence**: unknown until T42.0.
- **Empty images**: possibly few or none; FPPI on empty scenes may not be measurable here.

## Conventions

Those of M11, M14, M16 and M38 apply. Files to touch: as in M38 with `soda` in place of
`prewid` (`SODA_CLASSES`, `load_soda`, `DATASETS["soda"]`, `MAPPINGS[("soda", "voc")]`,
`MAPPINGS[("soda", "coco")]`; `prep_datasets.py soda`; `get_datasets.sh` `MARKER`,
`HOWTO`, `soda`, `check`; `FICHES["soda"]`; `matrice.py`, `gabarits.py`, `jeux.py`;
profiles `soda-prep`, `soda-train`; `test_prep_soda`, `test_soda`).

### Stats question

| Dataset | Own question (`stats` notebook) | Task |
|---|---|---|
| soda | How tall are the workers at the network input, how many per image, and how often does a machine overlap a worker? | T42.4 |

---

## A. Data

### [ ] T42.0 — Inventory and licence
- **Spec**: — · **Depends on**: T11.0 · **Runs on**: PC · **Size**: S
- **Deliverables**: licence, citation and access conditions in the Summary; `MARKER`,
  `HOWTO`, `soda` command; the Summary checked on the local copy (class names, image
  sizes, sequences or not, empty images, official split)
- **Acceptance**: the licence is written with its source; `tools/get_datasets.sh check`
  lists `soda`
- **Notes**: if the licence forbids keeping a copy or derived files, stop the milestone
  here and say so in the State.

### [ ] T42.1 — Boxes and split
- **Spec**: §5.1 · **Depends on**: T42.0 · **Runs on**: PC · **Size**: M
- **Deliverables**: `tools/prep_datasets.py soda` (format → corners, worker → `person`,
  split, preview); counts per split in the Summary; `test_prep_soda`
- **Acceptance**: image and box counts match the paper (or the difference is explained)

### [ ] T42.2 — Loader and mappings
- **Spec**: §8.3 · **Depends on**: T42.1 · **Runs on**: PC · **Size**: S
- **Deliverables**: `SODA_CLASSES`, `load_soda`, `DATASETS["soda"]`, `MAPPINGS`;
  `test_soda`
- **Acceptance**: `load("soda")` gives the test counts of the Summary

### [ ] T42.3 — Annotation completeness
- **Spec**: §8.3 · **Depends on**: T42.2, T41.2 · **Runs on**: PC · **Size**: S
- **Deliverables**: candidate missing workers on 100 test images (§ Annotation
  completeness), checked by eye; rate in the Summary
- **Acceptance**: the rate is reported with the decision (test kept as is, or reviewed)

## B. Registry and notebooks

### [ ] T42.4 — Datasheet, notebooks, stats run and Drive
- **Spec**: — · **Depends on**: T42.2 · **Runs on**: PC · **Size**: S
- **Deliverables**: `FICHES["soda"]`; notebook registry entries; profiles; `soda_stats`
  run with outputs, answer in [stats-jeux.md](stats-jeux.md); archive on the Drive if the
  licence allows it
- **Acceptance**: `tools/data_stats.py --dataset soda --sample 8` and
  `python -m tools.notebooks all --check` pass

## C. Person detection and safety

### [ ] T42.5 — Out-of-domain baseline
- **Spec**: §8.3 · **Depends on**: T42.2 · **Runs on**: PC · **Size**: S
- **Deliverables**: person AP50 on `test` of the pre-trained networks (settings of T38.6);
  table in this file
- **Acceptance**: the table is written

### [ ] T42.6 — Fine-tuning and cross-domain
- **Spec**: §7 · **Depends on**: T42.4, T42.5 · **Runs on**: PC (Colab for the GPU runs) ·
  **Size**: M
- **Deliverables**: `soda-prep`, `_train` notebook; AP50 on SODA `test`; AP50 on PREWID
  `test` of the SODA model, and on SODA `test` of the PREWID model (T38.7)
- **Acceptance**: fine-tuned AP50 above the best baseline; the two cross-domain numbers
  are reported

### [ ] T42.7 — M27 metrics on SODA
- **Spec**: §8.3 · **Depends on**: T42.6, T27.0, T27.1 · **Runs on**: PC (float, integer)
  · **Size**: S
- **Deliverables**: `tools/person_safety.py --dataset soda`; section in
  `results/person_safety.md`: miss rate per height, and at fixed FPPI if T42.0 found
  empty images (else at the operating point fixed on PREWID)
- **Acceptance**: the table is written, with the origin of the operating point stated

## Out of scope

- Multi-class detection of machines and materials (the classes stay in the files).
- PPE ([M43](M43-construction-ppe.md)).

## Recommended order

T42.0 (stop if the licence forbids it) → T42.1 → T42.2 → T42.5 at once, T42.3, then
T42.4 → T42.6 → T42.7 once T27.0 and T27.1 exist.
