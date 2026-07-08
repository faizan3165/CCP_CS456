# CCP Implementation Plan — Autonomous Visual Intelligence System

**Course:** CS-456 Computer Vision · **Marks:** 10 · **CLOs:** 3, 4, 5

## Goal
A single-command, end-to-end computer-vision pipeline covering five tasks, plus a
critical technical report. Classical CV (Tasks 1–3) + deep learning (Task 4) +
evaluation (Task 5).

## Architecture
```
run_all.py                       # single entry point (--task all|1|2|3|4, --data auto|synthetic|real)
src/
  config.py                      # all paths + tunable parameters (one place)
  utils.py                       # io, plotting, seeding, progress
  data_setup.py                  # resilient data layer: download -> fallback to synthetic
  task1_edges.py                 # Gaussian -> Canny (3 configs) -> Prob. Hough -> overlay
  task2_sift.py                  # SIFT kp/desc -> FLANN+ratio -> top-50 matches -> stats
  task3_panorama.py              # SIFT -> RANSAC homography (manual) -> warp+blend
  task4_cnn.py                   # ResNet-18 transfer learning -> metrics -> Grad-CAM
outputs/task{1..4}/              # all PNG/JPG visualizations, named per task
report/technical_report.md       # Task 5 (convert to PDF)
```

## Data strategy (auto-source with fallback)
Every task's data is produced by `data_setup.py`:
- **auto** (default): try to fetch/derive real imagery; on any failure, generate a
  reproducible synthetic scene. Guarantees the repo runs on any machine offline.
- Tasks 2 & 3 derive a *second/overlapping* view by applying a **known homography**
  to a base image — this gives real geometric correspondence + a ground-truth check.
- Task 4 builds an `ImageFolder` of ≥4 classes × ≥100 imgs. Synthetic classes carry
  controlled noise/overlap so accuracy lands realistically (>70%, not 100%) and
  genuine misclassifications exist for failure analysis.
- README documents how to drop in real images (MIT Places365 subset) with citations.

## Key compliance constraints
- Python 3.8+; only OpenCV, NumPy, Matplotlib, PyTorch/torchvision, scikit-learn.
- Task 3 stitching is **manual** — no `cv2.Stitcher`.
- Reproducible: global seed. Runs end-to-end via `python run_all.py`.
- `LLM_DECLARATION.md` declares AI assistance (required by the brief).

## Verification
Run `python run_all.py` fresh; confirm every task writes its outputs and Task 4
reaches ≥70% val accuracy. Report references the generated artifacts.
