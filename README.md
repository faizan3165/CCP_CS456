# CS-456 Computer Vision — Complex Computing Problem (CCP)
## Autonomous Visual Intelligence System for Scene Understanding

An end-to-end computer-vision project covering the four perception capabilities an outdoor
navigation robot needs — edge/line detection, SIFT feature matching, panorama stitching, and
CNN scene classification — plus a critical technical report. Every input is a **real,
publicly-licensed image** bundled in the repo (nothing synthetic or self-captured).
**CLOs 3, 4, 5 · 10 marks.**

---

## 1. Quick start

```bash
# 1) create an isolated environment (Python 3.8+)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) install the allowed libraries
pip install -r requirements.txt

# 3) run the WHOLE pipeline (builds all outputs + report from the bundled real data)
python run_all.py
```

Useful variants:

```bash
python run_all.py --task 1          # run a single task (1|2|3|4)
python run_all.py --no-report       # skip Task-5 report generation
```

The input images are committed under `data/`, so the pipeline runs offline. If a `data/`
image is ever missing it is re-fetched once from its cited source. Everything is reproducible
(fixed global seed in `src/config.py`).

---

## 2. What each task does

| Task | Module | Summary | CLO |
|---|---|---|---|
| 1 | `src/task1_edges.py` | Gaussian → **Canny** (3 threshold configs) → **Probabilistic Hough** line detection with a parameter sweep + qualitative-outcome table, on real dashcam **road** images | 3 |
| 2 | `src/task2_sift.py` | **SIFT** keypoints/descriptors on two real views of one outdoor landmark → FLANN + **Lowe's ratio test (0.75)** → top-50 matches → precision/recall sweep | 3 |
| 3 | `src/task3_panorama.py` | **From-scratch RANSAC + normalized-DLT homography** → progressive warp → **multi-band blending** on real overlapping building frames (no `cv2.Stitcher`) | 4 |
| 4 | `src/task4_cnn.py` | **Frozen ResNet-18 + trained classification head** on 4 scenario classes (road/building/vegetation/pedestrian_zone) → curves, confusion matrix, per-class P/R, **Grad-CAM** | 4 |
| 5 | `src/task5_report.py` | Data-driven **technical report** (PDF + Markdown): summary, architecture, per-task metrics, failure analysis, trade-offs, citations | 5 |

Outputs are written per task to `outputs/task1..4/`, consolidated metrics to
`outputs/results.json`, and the report to `report/technical_report.{pdf,md}`.

---

## 3. Repository layout

```
run_all.py                 single-command entry point
requirements.txt           allowed libraries only
src/
  config.py                all paths + tunable parameters (one place)
  utils.py                 io, plotting, seeding
  data_setup.py            real-first bundled-image data layer (download-if-missing)
  task1_edges.py … task5_report.py
data/raw/                  committed real inputs for tasks 1-3
data/task4_dataset/        committed ImageFolder (train/val × 4 classes)
outputs/task1..4/          generated visualizations (PNG)
report/technical_report.*  Task-5 deliverable
docs/plans/                implementation plan
```

---

## 4. Data sourcing & citations

All input imagery is **real, publicly-licensed, and committed to the repository**. No image
is synthetic or self-captured. `src/data_setup.py` reads the committed files and, only if one
is missing, re-fetches it once from the original source below.

- **Task 1 — road images** — Udacity *CarND-LaneLines-P1* `test_images/`
  (`solidWhiteRight.jpg`, `solidYellowLeft.jpg`, `whiteCarLaneSwitch.jpg`), **MIT License**:
  <https://github.com/udacity/CarND-LaneLines-P1>. Real dashcam road scenes with lane markings.
- **Task 2 — two-view pair** — Oxford VGG affine-covariant **"graffiti"** wall, views 1 & 3
  (a genuine viewpoint change of one outdoor landmark), redistributed via `opencv_extra`
  (BSD-3): <https://www.robots.ox.ac.uk/~vgg/research/affine/> ·
  <https://github.com/opencv/opencv_extra> (`testdata/cv/.../graf`).
- **Task 3 — overlapping frames** — OpenCV stitching sample **`boat1`–`boat3`** (a real
  St. Petersburg waterfront shot from three overlapping camera positions; downscaled to
  1400 px), from `opencv_extra` (`testdata/stitching`, BSD-3):
  <https://github.com/opencv/opencv_extra>.
- **Task 4 dataset** —
  - `road`, `building`, `vegetation` — *Intel Image Classification* (street/buildings/forest,
    relabelled to the scenario terms), Kaggle:
    <https://www.kaggle.com/datasets/puneet6060/intel-image-classification>.
  - `pedestrian_zone` — *Places365* **`crosswalk`** category (Zhou et al., *Places: A 10M
    Image Database for Scene Recognition*, IEEE TPAMI 2017):
    <http://places2.csail.mit.edu/>, via the no-auth Hugging Face mirror
    `ljnlonoljpiljm/places365-256px`.
  - 160 train + 48 val images per class (≥100/class as required).

---

## 5. Notes on constraints

- Python 3.8+; libraries restricted to the allowed set (OpenCV, NumPy, Matplotlib,
  PyTorch/torchvision, scikit-learn). Any image (re)download uses only the Python standard
  library. (Pillow is pulled in transitively by torchvision for Task-4 image IO.)
- Panorama stitching is implemented manually — **no `cv2.Stitcher`** (from-scratch
  normalized-DLT + RANSAC homography).
- Runs end-to-end via a single command: `python run_all.py`.
