# CS-456 Computer Vision — Complex Computing Problem (CCP)
## Autonomous Visual Intelligence System for Scene Understanding

An end-to-end computer-vision pipeline for an outdoor navigation robot, covering five
tasks: edge/line detection, SIFT feature matching, panorama stitching, CNN scene
classification, and a critical technical report. **CLOs 3, 4, 5 · 10 marks.**

---

## 1. Quick start

```bash
# 1) create an isolated environment (Python 3.8+)
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2) install the allowed libraries
pip install -r requirements.txt

# 3) run the WHOLE pipeline (downloads real data, builds all outputs + report)
python run_all.py
```

Useful variants:

```bash
python run_all.py --task 1          # run a single task (1|2|3|4)
python run_all.py --data synthetic  # force fully-offline synthetic data
python run_all.py --data real       # require real data (fail instead of falling back)
python run_all.py --no-report       # skip Task-5 report generation
```

Everything is reproducible (fixed global seed in `src/config.py`).

---

## 2. What each task does

| Task | Module | Summary | CLO |
|---|---|---|---|
| 1 | `src/task1_edges.py` | Gaussian → **Canny** (3 threshold configs) → **Probabilistic Hough** line detection with parameter sweep | 3 |
| 2 | `src/task2_sift.py` | **SIFT** keypoints/descriptors → FLANN + **Lowe's ratio test (0.75)** → top-50 matches → precision/recall sweep | 3 |
| 3 | `src/task3_panorama.py` | **From-scratch RANSAC + normalized-DLT homography** → progressive warp → **multi-band blending** (no `cv2.Stitcher`) | 4 |
| 4 | `src/task4_cnn.py` | **ResNet-18 transfer learning** on 4 outdoor scene classes → curves, confusion matrix, per-class P/R, **Grad-CAM** | 4 |
| 5 | `src/task5_report.py` | Data-driven **technical report** (PDF + Markdown): summary, architecture, metrics, failure analysis, trade-offs | 5 |

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
  data_setup.py            real-first, synthetic-fallback data layer
  task1_edges.py … task5_report.py
outputs/task1..4/          generated visualizations (PNG)
report/technical_report.*  Task-5 deliverable
docs/plans/                implementation plan
```

---

## 4. Data sourcing & citations

All input imagery is obtained automatically (`src/data_setup.py`), real-first with a
reproducible synthetic fallback so the pipeline **always** runs end-to-end.

- **Tasks 1–3 base images** — OpenCV sample images (`building.jpg`, `box.png`,
  `box_in_scene.png`, `graf1.png`), from the OpenCV repository, **BSD-3-Clause**:
  <https://github.com/opencv/opencv> (`samples/data`). Task 2 uses OpenCV's canonical
  box/box-in-scene matching pair; Task 3 frames are overlapping crops of a wide real
  scene, simulating a camera pan.
- **Task 4 dataset** — *Intel Image Classification* (natural outdoor scenes:
  buildings, forest, glacier, mountain, sea, street), via the no-auth Hugging Face
  mirror `sfarrukhm/intel-image-classification`
  <https://huggingface.co/datasets/sfarrukhm/intel-image-classification>. We use 4
  categories (buildings, forest, mountain, street) with ≥100 images/class for training.

No images are self-captured. If the network is unavailable, run with `--data synthetic`.

---

## 5. Notes on constraints

- Libraries are restricted to the allowed set (OpenCV, NumPy, Matplotlib,
  PyTorch/torchvision, scikit-learn); downloads use only the Python standard library.
- Panorama stitching is implemented manually — **no `cv2.Stitcher`**.
