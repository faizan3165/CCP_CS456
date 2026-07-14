# CS-456 CCP — Technical Report
## Autonomous Visual Intelligence System for Scene Understanding
*CLOs 3, 4, 5 — classical + deep-learning pipeline*

## 1. Executive Summary
The four core perception capabilities of an outdoor navigation robot, each built and
**independently validated on a representative real, publicly-licensed dataset** (not one
shared feed): road edge/lane detection (T1, real dashcam roads), cross-view SIFT
correspondence between two real views of one outdoor landmark (T2), a from-scratch
RANSAC + multi-band **panorama** from real overlapping outdoor waterfront frames (T3), and a frozen
**ResNet-18** whose classification head is fine-tuned to label four outdoor scenes
(road, building, vegetation, pedestrian zone) at a best validation accuracy of **93.8%**
(target 70%). Task 5 evaluates all stages. No `cv2.Stitcher`; no synthetic or self-captured images.

## 2. Pipeline Architecture
Perception stack (see `report/technical_report.pdf` Figure 1): **T1** Gaussian→Canny→Hough ·
**T2** SIFT keypoints + Lowe-ratio matching · **T3** RANSAC homography → warp → multi-band
blend · **T4** frozen ResNet-18 + trained head → scene label. Each stage is validated on its
own real dataset (named under its box in Figure 1), not on a single shared frame.

## 3. Quantitative Results
| Task | Key metric | Value |
|---|---|---|
| T1 | edge-detection quality | balanced Canny edge density 1.11% of pixels; 86 lines at 'fine' Hough |
| T2 | keypoints (A/B), good matches | 2674/3506, 539 (ratio 0.202) |
| T3 | mean RANSAC inlier ratio | 86.0% (multiband blend, ref frame 2) |
| T4 | best / final val accuracy | 93.8% / 93.8% |

### T4 per-class metrics
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| building | 0.87 | 0.96 | 0.91 | 48 |
| pedestrian_zone | 0.96 | 0.98 | 0.97 | 48 |
| road | 0.93 | 0.83 | 0.88 | 48 |
| vegetation | 1.00 | 0.98 | 0.99 | 48 |

## 4. Failure Analysis (>= 2 modes per task, each with a visual)
**T1:** loose Canny → clutter & spurious Hough lines; strict Canny → faint lanes dropped. *(see `outputs/task1/straight_road_canny_sweep.png`)*
**T2:** ratio 0.9 → false positives, 0.5 → low recall; repetitive wall texture → ambiguous matches. *(see `outputs/task2/ratio_sweep.png`)*
**T3:** insufficient overlap → rank-deficient homography (edge case); multi-frame drift bends edges (mitigated by middle-frame anchoring). *(see `outputs/task3/pair_matches.png`)*
**T4:** visually-similar classes confused (confusion off-diagonals); Grad-CAM 'incorrect' attends to background. *(see `outputs/task4/gradcam.png`, `confusion_matrix.png`)*

## 5. Classical vs. Deep-Learning Trade-offs
| Dimension | Classical (T1–T3) | Deep learning (T4) |
|---|---|---|
| Speed | real-time on CPU | GPU/MPS to train, fast inference |
| Accuracy | exact when geometry holds | strong semantics, off-distribution drop |
| Interpretability | transparent maths | opaque; Grad-CAM needed |
| Data | none (tuned params) | >=100 images/class |
| Failure | low texture / overlap | similar-class confusion |

## 6. Limitations & Future Work
Per-capability datasets vs. one live robot feed (motion blur, exposure, temporal consistency);
frozen-backbone linear probe vs. deeper fine-tuning / full Places365; planar-homography assumption
vs. parallax (needs bundle adjustment / cylindrical warping); SIFT+FLANN are CPU-bound (SuperPoint /
GPU matching would help); fuse geometry + semantics into a temporal scene graph.

## 7. Data Sources & Citations
- T1 roads — Udacity CarND-LaneLines-P1 test images (MIT License).
- T2 pair — Oxford VGG affine 'graffiti' wall, views 1 & 3 (free for research; via opencv_extra).
- T3 frames — OpenCV stitching sample 'boat1-3', a St. Petersburg waterfront panorama (opencv_extra, BSD-3).
- T4 road/building/vegetation — Intel Image Classification photos (Kaggle).
- T4 pedestrian_zone — Places365 'crosswalk' (Zhou et al., IEEE TPAMI 2017).
- No images are self-captured; all inputs are real, publicly-licensed datasets.

---
*Reproducible via `python run_all.py` (fixed seed).*
