# CCP Implementation Plan — Autonomous Visual Intelligence System

**Course:** CS-456 Computer Vision · **Marks:** 10 · **CLOs:** 3, 4, 5

## Goal
A single-command, end-to-end computer-vision pipeline covering four perception capabilities
of an outdoor navigation robot, plus a critical technical report. Classical CV (Tasks 1–3) +
deep learning (Task 4) + evaluation (Task 5). Every input is a **real, publicly-licensed,
committed image** — nothing synthetic or self-captured.

## Architecture
```
run_all.py                       # single entry point (--task all|1|2|3|4, --no-report)
src/
  config.py                      # all paths + tunable parameters (one place)
  utils.py                       # io, plotting, seeding, progress
  data_setup.py                  # real-first bundled data layer (download-if-missing)
  task1_edges.py                 # Gaussian -> Canny (3 configs) -> Prob. Hough -> overlay + qual table
  task2_sift.py                  # SIFT kp/desc -> FLANN+ratio -> top-50 matches -> stats
  task3_panorama.py              # SIFT -> RANSAC homography (manual) -> warp + multi-band blend
  task4_cnn.py                   # frozen ResNet-18 + trained head -> metrics -> Grad-CAM
outputs/task{1..4}/              # all PNG/JPG visualizations, named per task
report/technical_report.{md,pdf} # Task 5
```

## Data strategy (real, bundled, cited)
Each task runs on real public imagery committed under `data/` and cited in the README:
- **T1** — Udacity CarND road images (lane markings), MIT.
- **T2** — Oxford VGG "graffiti" wall, two real views (viewpoint change), via opencv_extra.
- **T3** — OpenCV `boat1–boat3` overlapping outdoor waterfront frames (downscaled), via opencv_extra.
- **T4** — road/building/vegetation from Intel Image Classification (relabelled to the
  scenario terms) + `pedestrian_zone` from Places365 `crosswalk`; ≥100 images/class.

`data_setup.py` reads the committed files and re-fetches a missing one from its source once.
There is no synthetic fallback (removed): honest, real data only.

## Key compliance constraints
- Python 3.8+; only OpenCV, NumPy, Matplotlib, PyTorch/torchvision, scikit-learn.
- Task 3 stitching is **manual** — no `cv2.Stitcher`.
- Task 4 fine-tunes the **final classification head only** (frozen backbone).
- Reproducible: global seed. Runs end-to-end via `python run_all.py`.

## Verification
Run `python run_all.py` fresh; confirm every task writes its outputs, Task 4 reaches ≥70%
val accuracy, the panorama is genuinely wider than any input frame, and the report tabulates
a metric for every task (T1 included) with a failure visual per task. Report references the
generated artifacts and cites every image source.
