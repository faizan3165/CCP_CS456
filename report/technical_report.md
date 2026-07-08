# CS-456 CCP — Technical Report
## Autonomous Visual Intelligence System for Scene Understanding
*CLOs 3, 4, 5 — classical + deep-learning pipeline*

## 1. Executive Summary
An end-to-end perception pipeline for an outdoor navigation robot: structural edge/line
detection (T1), cross-view SIFT correspondence (T2), a manually-implemented RANSAC +
multi-band panorama (T3), and a fine-tuned ResNet-18 scene classifier (T4) reaching a
best validation accuracy of **93.8%** (target 70%). Task 5 evaluates all stages.

## 2. Pipeline Architecture
Input frames → **T1** Gaussian→Canny→Hough → **T2** SIFT keypoints + ratio matching →
**T3** RANSAC homography → warp → multi-band blend → **T4** ResNet-18 transfer learning
→ scene label. See `report/technical_report.pdf` Figure 1 for the block diagram.

## 3. Quantitative Results
| Task | Key metric | Value |
|---|---|---|
| T1 | Canny/Hough parameter sweeps | see `outputs/task1/task1_param_experiments_*.png` |
| T2 | keypoints (A/B), good matches | 604/969, 80 (ratio 0.132) |
| T3 | mean RANSAC inlier ratio | 95.2% (multiband blend, ref frame 3) |
| T4 | best / final val accuracy | 93.8% / 92.7% |

### T4 per-class metrics
| class | precision | recall | F1 | support |
|---|---|---|---|---|
| buildings | 0.93 | 0.83 | 0.88 | 48 |
| forest | 1.00 | 0.98 | 0.99 | 48 |
| mountain | 0.96 | 0.96 | 0.96 | 48 |
| street | 0.83 | 0.94 | 0.88 | 48 |

## 4. Failure Analysis (>= 2 modes per task)
**T1:** loose Canny → clutter & spurious Hough lines; large maxLineGap → merged lines.
**T2:** repetitive texture → ambiguous matches; ratio 0.9 → false positives, 0.5 → low recall.
**T3:** insufficient overlap → rank-deficient homography (handled as edge case); multi-frame
homography drift bends edges (mitigated by middle-frame anchoring).
**T4:** visually-similar classes confused (see confusion matrix off-diagonals); Grad-CAM
'incorrect' sample shows attention on background context. Visuals: `outputs/task4/gradcam.png`,
`outputs/task4/confusion_matrix.png`.

## 5. Classical vs. Deep-Learning Trade-offs
| Dimension | Classical (T1–T3) | Deep learning (T4) |
|---|---|---|
| Speed | real-time on CPU | GPU/MPS to train, fast inference |
| Accuracy | exact when geometry holds | strong semantics, off-distribution drop |
| Interpretability | transparent maths | opaque; Grad-CAM needed |
| Data | none (tuned params) | >=100 images/class |
| Failure | low texture / overlap | similar-class confusion |

## 6. Limitations & Future Work
Overlapping-crop panorama vs. true multi-camera capture; scale T4 to full Places365;
parallax needs bundle adjustment / cylindrical warping; SIFT+FLANN are CPU-bound (SuperPoint
/ GPU matching would help); fuse geometry + semantics into a temporal scene graph.

---
*Reproducible via `python run_all.py` (fixed seed). Data and code cited in README.*
