# M40 — Conservative early alerts: deferred items

Parent: [M40-early-alerts.md](../M40-early-alerts.md). T40.0 to T40.3 run on the PC; only
the measurement waits for the board. Index: [README.md](README.md).

### [ ] T40.1 — Time to first suspicion on the board
- **Condition**: Board · **Needs first**: T40.1 (PC part), T30.2
- **To do**: the early heads in the bitstream; over 10⁵ frames, time from the frame start to
  each head flag and to the final result (mean, p99.9, maximum), next to the prefix bounds
  of T40.1 and T40.3.
- **Acceptance**: every measured first-suspicion time ≤ its static prefix bound.
- **Unlocks**: [T30.4](../M30-latency-bound.md) board row.
