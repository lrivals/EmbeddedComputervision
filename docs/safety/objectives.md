# Objectives → evidence (M31, T31.2)

One row per objective of the ML guidance, with the evidence in the repo (task, result file,
test) or **missing**. The machine-safety standards have not been read yet: their columns say
**to read** until they are.

The "Subject" column summarises the objective as the thesis state of the art cites it
(ARCYS, 2026-10; see the introductions of M28 and M29). Where the repo does not cite its
content, the column says "subject to check". It is a pointer, not the text of the objective.

Requirements SR-xx are in [requirements.md](requirements.md), ODD parameters Ox in
[odd.md](odd.md).

## EASA Concept Paper, Issue 02

| Objective | Subject | Evidence in the repo | Gap | IEC 61508 | ISO 13849 | IEC 61496 / TS 61496-4-3 | ISO/IEC TR 5469 |
|---|---|---|---|---|---|---|---|
| LM-02 | learning process captured; the MOC excludes online learning | offline training only (`python/yolo/train/`); training recipes [M15](../tasks/M15-campagne-entrainement.md) | the training set of the deployed detector is COCO (not the ODD); no fine-tune on industrial data yet | to read | to read | to read | to read |
| LM-06 | learning management (subject to check) | data statistics [stats-jeux.md](../tasks/stats-jeux.md); dataset loaders in `python/yolo/data/` | missing: no data requirement tied to the ODD | to read | to read | to read | to read |
| IMP-04-2 (MOC) | implementation optimizations without effect on the behaviour | integer → golden → C-sim pointwise ([results/equivalence.md](../../results/equivalence.md)); SMT proof of the kernels (T28.2) | float → integer is only bounded proximity (≤ 1 LSB per layer, −0.6 point R@0.1) | to read | to read | to read | to read |
| IMP-05 | implementation verification (subject to check) | three-stage metrics ([results/person_safety.md](../../results/person_safety.md)); exhaustive operator tests (T28.1) | co-sim and board rows missing (T28.5, deferred) | to read | to read | to read | to read |
| IMP-06 | properties preserved after conversion | equivalence matrix in `make ci` (`tools/equivalence.py --check`); traceability ([docs/traceability.md](../traceability.md)) | unsorted NMS depends on the cell order (SR-12) | to read | to read | to read | to read |
| EXP-04 | monitoring of the ODD | ODD axes ([odd.md](odd.md)); shift detection T29.0 ([results/ood_monitor.md](../../results/ood_monitor.md)) | monitor not measured on the six degradations (SR-13) | to read | to read | to read | to read |
| EXP-05 | monitoring of the input distribution | Mahalanobis / autoencoder / activation patterns (T25.5, T29.0) | AUROC 0.81-0.93, not a guard | to read | to read | to read | to read |
| EXP-06 | monitoring of the output confidence | max score and learned failure head ([results/failure_prediction.md](../../results/failure_prediction.md)) | 9-12 % of misses flagged at 5 % alarms (SR-05) | to read | to read | to read | to read |
| EXP-07 | indication to the user / operator (subject to check) | missing: no alert interface; the fallback states `CLEAR` / `PERSON` / `UNSAFE` are specified in T29.5 | missing | to read | to read | to read | to read |
| SRM-02 | fallback to a conventional system | missing: T29.5 state machine not written | missing (SR-08, SR-10, SR-11) | to read | to read | to read | to read |

## CoDANN II, equivalence levels (§3.6.1)

| Level | Stages where it is reached | Evidence | Gap | IEC 61508 | ISO 13849 | IEC 61496 / TS 61496-4-3 | ISO/IEC TR 5469 |
|---|---|---|---|---|---|---|---|
| Pointwise | integer → golden → C-sim (3 nets × 20 images); C-sim == integer on 4,952 VOC images (v2) | [results/equivalence.md](../../results/equivalence.md), [results/map_stades.md](../../results/map_stades.md) | co-sim, board | to read | to read | to read | to read |
| Bounded proximity | fake-quant → integer: ≤ 1 LSB per layer, measured not proved | [results/equivalence.md](../../results/equivalence.md) | bound measured on 20 images | to read | to read | to read | to read |
| Statistical | float → integer: −0.6 point R@0.1 on PREWID, −1.30 AP50 on COCO (v4) | [results/person_safety.md](../../results/person_safety.md) | | to read | to read | to read | to read |
| Monitor at the level of the function (§5.2.2) | failure head integer == C-sim (T39.5) | [results/failure_prediction.md](../../results/failure_prediction.md) | faults in the monitor (T25.9) missing | to read | to read | to read | to read |

## AMLAS (Hawkins 2021), stages 1 to 6

| Stage | Subject | Evidence in the repo | Gap | IEC 61508 | ISO 13849 | IEC 61496 / TS 61496-4-3 | ISO/IEC TR 5469 |
|---|---|---|---|---|---|---|---|
| 1. ML safety assurance scoping | system, ODD, hazard | [odd.md](odd.md), [requirements.md](requirements.md) | target standard to be given by ARCYS | to read | to read | to read | to read |
| 2. ML safety requirements | requirements on the ML component | SR-01, SR-03, SR-05, SR-13 ([requirements.md](requirements.md)) | numbers proposed, not derived from a standard | to read | to read | to read | to read |
| 3. Data management | relevance, completeness, balance, accuracy of the data | PREWID ([M38](../tasks/M38-prewid.md)), datasheets; industrial sets planned (M41-M43) | missing: no data requirement per ODD axis; posture and PPE not covered | to read | to read | to read | to read |
| 4. Model learning | training, internal tests | training code and recipes (M15, M24); operating point T27.2 | missing: no fine-tune on the ODD | to read | to read | to read | to read |
| 5. Model verification | independent test, degraded conditions | three-stage metrics T27.1, degradations T27.3, burst analysis T27.4 ([results/person_safety.md](../../results/person_safety.md)) | CrowdHuman / ExDark / ARCYS images pending | to read | to read | to read | to read |
| 6. Model deployment | integration, run-time monitors, fallback | equivalence (M28), latency bound ([results/latency_bound.md](../../results/latency_bound.md)), ABFT ([results/faults.md](../../results/faults.md)) | monitor in the chain (T29.4), fallback (T29.5), board (T28.5, T30.2) | to read | to read | to read | to read |

<!-- m36:explainability -->
## Explainability objectives

Filled by [T36.11](../tasks/M36-explanation.md): each explainability objective (EASA EXP,
AMLAS stages 4 and 5) to a result of T36.4 to T36.6, or "not covered".
<!-- /m36:explainability -->
