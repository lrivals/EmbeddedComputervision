# References

Papers of the article base cited by the [article](article.md). Taken from §12 of the
[specification](../../yolo-embarque-de-zero.md#12-références-articles-de-la-base-utilisés).
Citations in the text use the base format `[short-id#chunk]` (for example
`[2024-zhang#013.1]`); each one can be checked with `bin/pdb verify`, a tool of the base
that lives outside this repository. Statements that no paper of the base supports are
marked "outside the base", as in the specification.

| Short id | Title | Authors | Year | Venue | Type | Link |
|---|---|---|---|---|---|---|
| `1506.02640` | You Only Look Once: Unified, Real-Time Object Detection | Redmon et al. | 2016 | CVPR 2016 | conference | https://doi.org/10.1109/CVPR.2016.91 |
| `1612.08242` | YOLO9000: Better, Faster, Stronger | Redmon & Farhadi | 2017 | CVPR 2017 | conference | https://arxiv.org/abs/1612.08242 |
| `1804.02767` | YOLOv3: An Incremental Improvement | Redmon & Farhadi | 2018 | arXiv | **preprint** | https://arxiv.org/abs/1804.02767 |
| `1904.04620` | Gaussian YOLOv3: An Accurate and Fast Object Detector Using Localization Uncertainty for Autonomous Driving | Choi et al. | 2019 | arXiv | **preprint** | https://arxiv.org/abs/1904.04620 |
| `2019-ding` | REQ-YOLO: A Resource-Aware, Efficient Quantization Framework for Object Detection on FPGAs | Ding et al. | 2019 | FPGA '19 (ACM/SIGDA) | conference | https://doi.org/10.1145/3289602.3293904 |
| `2023-zhai` | FPGA-Based Vehicle Detection and Tracking Accelerator | Zhai et al. | 2023 | Sensors 23(4) | journal | https://doi.org/10.3390/s23042208 |
| `2024-zhang` | End-to-end acceleration of the YOLO object detection framework on FPGA-only devices | Zhang et al. | 2024 | Neural Computing and Applications 36(3) | journal | https://doi.org/10.1007/s00521-023-09078-8 |
| `2025-kim` | Design and Implementation of a YOLOv2 Accelerator on a Zynq-7000 FPGA | Kim & Kim | 2025 | Sensors | journal | https://doi.org/10.3390/s25206359 |
| `2026-fata` | Low-Cost FPGA-Enhanced CNN Accelerator for Real-Time YOLO Object Detection and Classification | Fata & Elmannai | 2026 | IEEE Access 14 | journal | https://doi.org/10.1109/ACCESS.2026.3669451 |
| `2024-yan` | An FPGA-Based YOLOv5 Accelerator for Real-Time Industrial Vision Applications | Yan et al. | 2024 | Micromachines 15(9) | journal | https://doi.org/10.3390/mi15091164 |
| `2306.00001` | TinyissimoYOLO: A Quantized, Low-Memory Footprint, TinyML Object Detection Network for Low Power Microcontrollers | Moosmann et al. | 2023 | IEEE AICAS 2023 | conference | https://doi.org/10.1109/AICAS57966.2023.10168657 |
| `2023-montgomerie-corcoran` | SATAY: A Streaming Architecture Toolflow for Accelerating YOLO Models on FPGA Devices | Montgomerie-Corcoran et al. | 2023 | ICFPT 2023 | conference | https://doi.org/10.1109/ICFPT59805.2023.00025 |
| `2015-zhang` | Optimizing FPGA-based Accelerator Design for Deep Convolutional Neural Networks | Zhang et al. | 2015 | FPGA '15 (ACM/SIGDA) | conference | https://doi.org/10.1145/2684746.2689060 |
| `2018-venieris` | Toolflows for Mapping Convolutional Neural Networks on FPGAs: A Survey and Future Directions | Venieris et al. | 2018 | ACM Computing Surveys 51(3) | journal | https://doi.org/10.1145/3186332 |
| `2202.10935` | EF-Train: Enable Efficient On-device CNN Training on FPGA Through Data Reshaping for Online Adaptation or Personalization | Tang et al. | 2022 | arXiv | **preprint** | https://arxiv.org/abs/2202.10935 |
| `2024-campos` | End-to-end codesign of Hessian-aware quantized neural networks for FPGAs | Campos et al. | 2024 | ACM TRETS 17(3) | journal | https://doi.org/10.1145/3662000 |
| `2607.13106` | No Attention, No Problem: DPU-Aware Attention Approximation in Modern YOLO on FPGA | Karki et al. | 2026 | arXiv | **preprint** | https://arxiv.org/abs/2607.13106 |
| `2025-el-zeinaty` | Designing Object Detection Models for TinyML: Foundations, Comparative Analysis, Challenges, and Emerging Solutions | El Zeinaty et al. | 2025 | ACM Computing Surveys 58(2) | journal | https://doi.org/10.1145/3744339 |
| `2025-hozhabr` | A Survey on Real-Time Object Detection on FPGAs | Hozhabr & Giorgi | 2025 | IEEE Access 13 | journal | https://doi.org/10.1109/ACCESS.2025.3544515 |
| `2024-rech` | Artificial Neural Networks for Space and Safety-Critical Applications: Reliability Issues and Potential Solutions | Rech | 2024 | IEEE Trans. Nuclear Science 71(4) | journal | https://doi.org/10.1109/TNS.2024.3349956 |
| `2023-shuvo` | Efficient Acceleration of Deep Learning Inference on Resource-Constrained Edge Devices: A Review | Shuvo et al. | 2023 | Proceedings of the IEEE 111(1) | journal | https://doi.org/10.1109/JPROC.2022.3226481 |
| `2006.08357` | CoDeNet: Efficient Deployment of Input-Adaptive Object Detection on Embedded FPGAs | Dong et al. | 2021 | FPGA '21 | conference | https://doi.org/10.5281/zenodo.4341394 |

YOLOv3 is a technical report that was never published at a conference, but it is the de
facto reference. Papers marked **preprint** have not been peer reviewed.

Outside the base (no paper describes them; derived or read elsewhere): the hardware
handling of the 2×2 stride-1 max pooling of Tiny-YOLOv2/v3, the Darknet `.cfg` files
(exact channels, Tiny-YOLOv3 anchors), the integer requantization chain and the BN
derivatives.
