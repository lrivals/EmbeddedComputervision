# M0 — Infrastructure

Goal: a tooled repository where `make test` passes, and reference figures (parameters,
MACs) that confirm the reading of the architectures of §3.

### [x] T0.1 — Repository, layout, README
- **Spec**: — · **Depends on**: — · **Size**: S
- **Deliverables**: `git init`, layout of [architecture.md](../architecture.md),
  `README.md`, `.gitignore`, `Makefile`
- **Acceptance**: `git status` clean after the first commit; `data/`, `weights/`, `model/`
  and `build/` ignored

### [x] T0.2 — Python environment
- **Spec**: — · **Depends on**: T0.1 · **Size**: S
- **Deliverables**: `python/pyproject.toml` (numpy; dev: pytest, ruff), editable install
- **Acceptance**: `pip install -e "python[dev]"` then `make test-py` and `make lint` pass
- **Notes**: no deep learning library ([ADR 0001](../adr/0001-numpy-pur.md))

### [x] T0.3 — C++ toolchain
- **Spec**: — · **Depends on**: T0.1 · **Size**: S
- **Deliverables**: `cpp/golden/CMakeLists.txt`, Catch2 through FetchContent, smoke test
- **Acceptance**: `make test-cpp` builds and passes

### [x] T0.4 — Conventions and exchange format
- **Spec**: §0 (notation), §10.5 · **Depends on**: T0.1 · **Size**: M
- **Deliverables**: [conventions.md](../conventions.md) finalised;
  `model/example_manifest.json` (full Tiny-YOLOv3, dummy values); JSON schema
  `docs/manifest.schema.json`
- **Acceptance**: the example manifest describes the 24 layers of §3.2 and passes schema
  validation
- **Notes**: plan right away for the contiguous placement of route 20 (layers 19 and 8)
  and for the fused maxpool, so the format does not have to change later

### [x] T0.5 — Counting parameters and MACs
- **Spec**: §3.1, §3.2 · **Depends on**: T0.2 · **Size**: S
- **Deliverables**: `tools/count_macs.py` (reads a layer description, prints the table of
  §3)
- **Acceptance**: Tiny-YOLOv2 VOC = **15.86 M** parameters / **3.49 G** MACs;
  Tiny-YOLOv3 VOC = **8.71 M** / **2.74 G**; each row equal to the one in the tables
- **Notes**: BN = 2 parameters per channel, no conv bias before BN (§4.2). The "0–11" row
  of §3.2 showed 1,071.8 M (sum of rounded values); corrected to 1,071.6 M
  (1,071,562,752 MACs), the totals are unchanged.

### [x] T0.6 — PASCAL VOC dataset
- **Spec**: §1, §8.3 · **Depends on**: T0.2 · **Size**: M
- **Deliverables**: `python/yolo/data/voc.py` (XML parsing → normalised boxes),
  `tools/get_voc.sh` (VOC2007 + 2012 trainval, VOC2007 test)
- **Acceptance**: number of images and objects per split matching the official figures;
  10 randomly drawn images displayed with their correct boxes
- **Notes**: keep the `difficult` flag, ignored by the VOC evaluation
