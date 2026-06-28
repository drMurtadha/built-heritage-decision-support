# Reproducibility Protocol: YOLOv11n with Lightweight Attention for Heritage Human Detection

Replication guide for *"Real-Time Human Detection for Heritage Monitoring Using YOLOv11n with Lightweight Attention Mechanisms"* (Junaidi, Mohamad, Othman, Mahdzar — UTM). This protocol follows the paper's own Methodology section (III) and Tables I–III, so a third party can re-run the comparison of baseline YOLOv11n vs. YOLOv11n+SimAM vs. YOLOv11n+ECA and check their numbers against the published ones.

---

## 1. What is being replicated

A controlled three-way comparison on one dataset, one training schedule, one hardware/software stack:

| Configuration | Attention module | Insertion point |
|---|---|---|
| Baseline | None | — |
| SimAM variant | SimAM (parameter-free, energy-function-based) | After each of the 3 backbone C3k2 blocks |
| ECA variant | Efficient Channel Attention (lightweight, parametric) | After each of the 3 backbone C3k2 blocks |

Target outcome: confirm that ECA improves detection (higher mAP@0.5/mAP@0.5:0.95) over baseline, while SimAM slightly degrades it — the paper's central finding.

---

## 2. Step-by-step procedure

### Step 1 — Acquire the dataset

- Source: Roboflow, **"People Detection," version 11**, Leo-Ueno workspace.
- URL: `https://universe.roboflow.com/leo-ueno/people-detection-o4rdr/dataset/11`
- 17,401 images, single class (`person`), varied lighting, occlusion, crowd density, background complexity.
- Confirmed split: 15,210 train / 1,431 validation / 760 test.
- **Risk to flag**: Roboflow dataset versions can be updated after publication; pull version 11 specifically, and record the download date and image count to confirm it matches 17,401 before training.

### Step 2 — Set up the environment

| Component | Specification used in the paper |
|---|---|
| GPU | NVIDIA GeForce RTX 3050, 8 GB VRAM |
| CPU | Intel Core i7-3770, 8 cores, 3.40 GHz |
| RAM | 16 GB DRAM |
| OS | Ubuntu 24.04.3 LTS |
| Deep learning framework | PyTorch 2.5.1 |
| GPU acceleration | CUDA 11.8 |
| Detection framework | Ultralytics YOLOv11, v8.3.0 |
| Language | Python |

Install: `pip install ultralytics==8.3.0 torch==2.5.1` (match CUDA build to the installed driver). A weaker GPU/VRAM than the RTX 3050's 8 GB will likely force smaller batch sizes than the adaptive 8/6/4 scheme below, which can shift results slightly — document any deviation.

### Step 3 — Implement the baseline model

- Load YOLOv11n (the smallest YOLOv11 variant: ≈2.59M parameters, 6.5 GFLOPs) directly from Ultralytics.
- This is the unmodified reference model — no code changes needed beyond standard Ultralytics training.

### Step 4 — Implement the attention variants

The paper does not publish a code repository, so the attention modules must be implemented from their original sources and spliced into the YOLOv11n backbone:

- **ECA**: implement per Wang et al., "ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks" (CVPR 2020) — a parameter-light 1D convolution over channel-wise global-average-pooled features, no dimensionality reduction.
- **SimAM**: implement per Yang et al., "SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks" (ICML 2021) — an energy-function-based per-neuron importance weighting with no added learnable parameters.
- **Insertion point (identical for both)**: directly after each of the three backbone C3k2 blocks (shallow/middle/deep feature scales), per Fig. 1 of the paper. Do not alter the neck (FPN+PAN) or head (multi-scale detect) — only the backbone changes.
- Verify parameter/FLOP parity against the paper's Table II before training: baseline and SimAM should both report ≈2.59M params / 6.5 GFLOPs; the ECA variant should report ≈2.60M params / 6.5 GFLOPs (negligible overhead). A large deviation signals the module was inserted at the wrong location or duplicated.

### Step 5 — Configure training (Table I, reproduced exactly)

| Parameter | Value |
|---|---|
| Epochs | 300 |
| Input image size | 640 × 640 |
| Optimizer | SGD |
| Initial learning rate (lr0) | 0.01 |
| Final learning rate factor (lrf) | 0.01 |
| Weight decay | 0.0005 |
| Warmup epochs | 5 |
| Warmup momentum | 0.8 |
| Batch size | 8, 6, 4 (adaptive, per GPU-memory pressure) |
| Workers | 4, 3, 2 (adaptive) |
| Data augmentation | Mosaic, MixUp, horizontal flip, HSV |
| Mosaic | 1.0 |
| MixUp | 0.15 |
| Copy-Paste | 0.0 |
| Close mosaic (epoch) | 30 |
| Horizontal flip (fliplr) | 0.5 |
| HSV (h, s, v) | 0.015, 0.7, 0.4 |
| Loss weights (box, cls, dfl) | 7.5, 0.5, 1.5 |
| Mixed precision (AMP) | Enabled |
| Random seed | 42 |
| Validation | Enabled during training |

Train all three configurations (baseline, +SimAM, +ECA) with these identical settings — only the attention module differs between runs.

**Risk to flag**: the paper does not specify the exact rule that triggers the 8→6→4 (or 4→3→2 workers) adaptive step-down. Treat it as "reduce on CUDA out-of-memory" and log when/why each step-down occurred, since this affects effective batch size and can shift results across replication attempts even with the same seed.

### Step 6 — Evaluate

Compute, on the 760-image test set:

- Precision (P)
- Recall (R)
- mAP@0.5 (IoU threshold 0.5)
- mAP@0.5:0.95 (averaged across IoU 0.5–0.95)

Also record parameter count and GFLOPs per configuration (Table II equivalent).

### Step 7 — Compare against the published benchmark

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 | Params (M) | GFLOPs |
|---|---|---|---|---|---|---|
| YOLOv11n (baseline) | 0.868 | 0.650 | 0.738 | 0.507 | 2.59 | 6.5 |
| YOLOv11n + SimAM | 0.860 | 0.642 | 0.730 | 0.500 | 2.59 | 6.5 |
| YOLOv11n + ECA | 0.870 | 0.658 | 0.794 | 0.528 | 2.60 | 6.5 |

Replication is considered successful if:

1. **Direction of effect matches**: ECA > baseline > SimAM on mAP@0.5 and mAP@0.5:0.95.
2. **Magnitude is close**: metrics fall within a reasonable tolerance band (e.g., ±0.02–0.03 absolute on mAP, allowing for GPU non-determinism, library-version drift, and the unspecified adaptive-batch-size trigger). Tight exact-match is not realistic given fixed-seed training still varies across hardware/cuDNN versions.
3. **Relative improvement holds**: ECA's mAP@0.5 should show roughly the same ~7–8% relative gain over baseline reported in the paper (0.794 vs. 0.738 ≈ +7.6%).

---

## 3. Known sources of replication variance

| Risk | Why it matters | Mitigation |
|---|---|---|
| No public code repository | Attention modules must be re-implemented from the original ECA/SimAM papers, not copied from the authors | Validate against published param/FLOP counts (Step 4) before trusting training results |
| GPU/CUDA/cuDNN version drift | Fixed random seed (42) does not guarantee bit-identical results across different GPU architectures or library versions | Report exact hardware/software stack used; treat results as "close, not identical" |
| Unspecified adaptive-batch-size trigger | Affects effective batch size and gradient noise | Log every batch-size step-down event and its cause (typically CUDA OOM) |
| Roboflow dataset versioning | Dataset "version 11" content could be edited/extended after publication | Record exact image count (17,401) and split sizes (15,210/1,431/760) post-download; flag mismatches |
| Single train/val/test split, no cross-validation reported | Results reflect one split, not an average — variance across splits is unknown | If resources allow, repeat training across 2–3 random splits to estimate variance |

---

## 4. Beyond replication: the paper's own stated next steps

The authors name three extensions, not yet completed in this study — useful to distinguish from the core replication above:

1. Multi-class detection (currently single-class: `person`).
2. Small-object detection performance.
3. Deployment on an edge device, specifically **NVIDIA Jetson** (the title references Jetson Nano, but on-device deployment and benchmarking are future work, not part of the reported results).

---

## Source

`Real-Time Human Detection for Heritage Monitoring Using YOLO on Jetson Nano REVISED.pdf` (uploaded), Sections III (Methodology), Table I (training configuration), Table II/III (results), and Section V (Conclusion).
