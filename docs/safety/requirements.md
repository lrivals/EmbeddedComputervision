# Hazard and safety requirements (M31, T31.1)

## Dreaded event

> **E1**: a person is inside the zone and is not reported during **N = 5 consecutive frames**
> (0.5 s at 10 frames/s), while at a distance where their image is **≥ 30 px high**
> ([odd.md](odd.md), O2-O3).

- N = 5 is the reaction budget used by T27.4 / T27.5. It must come from the stopping time of
  the machine and the approach speed of a person (ISO 13855 separation distance, to read).
  With a stopping time T and an approach speed K = 1.6 m/s, the zone must extend K·(T + 0.5 s)
  beyond the hazard.
- The opposite error (a person reported in an empty zone) is an unwanted stop: an
  availability cost, not a hazard. It is bounded by the 0.1 FPPI operating point.
- **Families** of the causes of E1, from [T25.10](../tasks/M25-dependability.md):
  - **P**: a perception error without fault (the model misses the person);
  - **X**: an execution anomaly (a hardware fault corrupts the computation);
  - **D**: a missed deadline (the result comes too late or never).

## Requirements

Status as of 2026-10-27: **met**, **not met** (measured, short of the number), or **open**
(not measured yet).

| Id | Requirement | Number | Verification method | Family | Evidence | Status |
|---|---|---|---|---|---|---|
| SR-01 | Recall of the detector at the operating point, inside the ODD | ≥ 95 % persons ≥ 30 px at ≤ 0.1 FPPI | R@0.1 on a held-out set, three stages (T27.1, T27.2) | P | [results/person_safety.md](../../results/person_safety.md) | **not met**: 90.7 / 90.1 % (float / integer) on PREWID test, 89.1 / 90.8 % on the check half |
| SR-02 | Integer chain identical to the hardware | byte for byte | equivalence matrix (T28.0) | X | [results/equivalence.md](../../results/equivalence.md) | **met** up to C-sim; co-sim and board pending |
| SR-03 | Recall in each ODD axis at its proposed limit | ≥ 85 % at severity 1 | degradations of T27.3 | P | [results/person_safety.md](../../results/person_safety.md) | **not met** for low light (61.5 %), occlusion (68.2 %), JPEG (80.5 %), motion blur and noise (84 %); met for fog (87.5 %) |
| SR-04 | Rate of E1 | below the target PFH (10⁻⁶ /h for SIL 2, to be confirmed by ARCYS) | failure budget (T31.3) | P, X, D | [results/failure_budget.md](../../results/failure_budget.md) | **not met** (orders of magnitude in the budget) |
| SR-05 | Share of the missed persons flagged by the run-time monitor | ≥ 90 % of the misses at ≤ 5 % alarms | monitor against the misses (T29.1, T27.7) | P | [results/failure_prediction.md](../../results/failure_prediction.md) | **not met**: 9-12 % at 5 % alarms (reduced runs) |
| SR-06 | Temporal vote | the E1 rate divided by ≥ 10 at ≤ 2 frames of added latency | k-of-n vote on video (T29.3) | P | — | **open** (T29.3 not done) |
| SR-07 | Detection of hardware faults in the accelerator | every fault that changes the output detected, or ≤ 1 % silent | fault injection in C-sim (T25.7), campaign (T25.1), selective protection (T25.4) | X | [results/faults.md](../../results/faults.md) | **not met**: `w` and `out` covered, `in` 13-18 % silent (L0, L2), `q` 67-81 % silent (after the check) |
| SR-08 | Frame watchdog | a frame later than the bound → `UNSAFE` | static bound (T30.0) and fallback logic (T29.5) | D | [results/latency_bound.md](../../results/latency_bound.md) | **open**: bound 92 ms / watchdog 116 ms for Tiny-YOLOv2; v3 / v4 dominated by the uncapped NMS (0.52 s); state machine T29.5 not written |
| SR-09 | Reaction time of the chain | camera → alert ≤ 0.5 s (N frames) | chain model (T30.4), board measurement (T30.2) | D | — | **open** |
| SR-10 | Missing frame | a missing or repeated frame → `UNSAFE` within one frame period | `sw-sim` tests of the fallback (T29.5) | D | — | **open** |
| SR-11 | Start-up | the output is `UNSAFE` until the first valid frame and a passing self-test (known image, known detections, ABFT silent) | `sw-sim` tests (T29.5) | X, D | — | **open** |
| SR-12 | Post-processing independent of the cell order | the NMS result does not depend on the order of the candidates | property test (T28.6) | X | [results/postproc_hw.md](../../results/postproc_hw.md) | **not met** by the unsorted hardware NMS: a permutation changes the result in 91 % (v2) / 74 % (v3) of cases; the sorted stream meets it, at −0.10 mAP |
| SR-13 | Out-of-ODD input flagged | shift AUROC ≥ 0.99 on each ODD axis beyond its limit | monitor comparison (T29.0, T29.1) | P | [results/ood_monitor.md](../../results/ood_monitor.md) | **not met**: 0.81 (ExDark) / 0.93 (FLIR); the six degradations not measured |

## Notes

- **SR-12** is the finding of [T28.6](../tasks/M28-equivalence.md). A result that depends on
  the order of the cells is not reproducible across runs when the order varies (e.g. a future
  parallel decode). The sorted stream should be the safety configuration.
- **SR-04** is the only requirement on the whole system. The others are the elements whose
  gains the budget combines.
- **The monitor, the vote and the watchdog** do not raise the recall. They turn a silent miss
  into a flagged one, and the fallback (T29.5) answers a flag with `UNSAFE`. The cost is an
  unwanted stop. It is counted with the alarm rate, never folded into the miss rate
  ([T27.7](../tasks/M27-safety-metrics.md)).
