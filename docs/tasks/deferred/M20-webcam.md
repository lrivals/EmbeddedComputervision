# M20 — Live webcam: deferred items

Parent: [M20-webcam.md](../M20-webcam.md). The plan and the projected latency budget of
T20.9 are written there. Index: [README.md](README.md).

### [ ] T20.9 — Webcam on the board
- **Condition**: Board (+ USB camera on the KV260) · **Needs first**: T7.4, T8.1
- **Done on the PC**: plan in 5 steps; projected budget ≈ 55-60 ms per frame (FPGA 37.2 ms
  from the C-sim projection).
- **To do**:
  - fill the rows marked "to measure" of the budget (decode + letterbox + int8 on one A53,
    post-processing, draw + display) from the T8.1 board measurements; this meets the
    acceptance of T20.9;
  - then, as an implementation: V4L2 capture in `sw/app/yolo_app.cpp`
    (`--camera /dev/video0`), display over DisplayPort or an MJPEG stream, three threads.
    The capture and display code can be written on the PC with the `sim` backend before.
