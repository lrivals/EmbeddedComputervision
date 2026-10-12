# ADR 0001 — Reference model in Python + NumPy only

- Status: **accepted** (2026-10-04)

## Context

The spec aims at a rewrite "with no deep learning library": every forward and backward
pass must be written by hand and checked (§4, §11).

## Decision

`python/yolo/` depends on NumPy only. The gradients are validated by finite differences
(T1.1), not by comparison with PyTorch.

## Consequences

- Full control of the formulas, reusable as they are for the integer model.
- Slow training: the convolution goes through im2col (T1.2). The reference mAP can be
  obtained with pre-trained Darknet weights (T1.9), without a full training.
