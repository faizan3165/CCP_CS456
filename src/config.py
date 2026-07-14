"""Central configuration for the CCP pipeline.

Every path and every tunable parameter lives here so experiments are reproducible
and the rest of the code stays declarative. Imported as ``from src import config``.

All task inputs are REAL, permissively-licensed images bundled under ``data/`` (see
README for full citations). Nothing is synthetic or self-captured.
"""

import math
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"                    # base/source images for tasks 1-3
TASK4_DATA_DIR = DATA_DIR / "task4_dataset"   # ImageFolder: <split>/<class>/*.jpg
OUTPUTS_DIR = ROOT / "outputs"
OUT_T1 = OUTPUTS_DIR / "task1"
OUT_T2 = OUTPUTS_DIR / "task2"
OUT_T3 = OUTPUTS_DIR / "task3"
OUT_T4 = OUTPUTS_DIR / "task4"
REPORT_DIR = ROOT / "report"

ALL_OUTPUT_DIRS = (RAW_DIR, TASK4_DATA_DIR, OUT_T1, OUT_T2, OUT_T3, OUT_T4, REPORT_DIR)

# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
SEED = 42

# --------------------------------------------------------------------------- #
# Task 1 — Edge Detection & Hough Transform
# --------------------------------------------------------------------------- #
# Real outdoor road images with lane markings (Udacity CarND-LaneLines-P1, MIT).
# (display name, filename in RAW_DIR, remote source for one-time populate-if-missing)
T1_IMAGES = [
    ("straight_road", "t1_road_a.jpg", "solidWhiteRight.jpg"),
    ("curved_road", "t1_road_b.jpg", "solidYellowLeft.jpg"),
    ("lane_switch", "t1_road_c.jpg", "whiteCarLaneSwitch.jpg"),
]

# Gaussian smoothing: 5x5 kernel suppresses sensor noise without erasing the thin
# lane markings we later want Canny to find; sigma=1.4 is the classic Canny value.
T1_GAUSSIAN_KERNEL = (5, 5)
T1_GAUSSIAN_SIGMA = 1.4

# Three deliberately different Canny (low, high) threshold configurations, tested on
# the SAME image to show the effect of hysteresis thresholds (required by the brief).
# ``qual`` is the qualitative outcome surfaced in the experiment table.
T1_CANNY_CONFIGS = [
    {"name": "loose", "low": 30, "high": 90,
     "qual": "many weak edges: foliage/road-texture clutter"},
    {"name": "balanced", "low": 50, "high": 150,
     "qual": "clean lane + structure edges (1:3 ratio)"},
    {"name": "strict", "low": 100, "high": 200,
     "qual": "only strong edges; faint/far lanes dropped"},
]

# Probabilistic Hough line experiments (rho, theta, threshold, minLineLength, maxLineGap).
T1_HOUGH_CONFIGS = [
    {"name": "fine", "rho": 1, "theta": math.pi / 180, "threshold": 40,
     "min_line_length": 40, "max_line_gap": 10,
     "qual": "many short, precise segments"},
    {"name": "coarse", "rho": 2, "theta": math.pi / 90, "threshold": 60,
     "min_line_length": 60, "max_line_gap": 20,
     "qual": "fewer, longer dominant lines"},
    {"name": "gappy", "rho": 1, "theta": math.pi / 180, "threshold": 30,
     "min_line_length": 30, "max_line_gap": 40,
     "qual": "bridges dashed lanes but merges separate lines"},
]
T1_HOUGH_PRIMARY = "fine"       # config used for the headline overlay
T1_LINE_COLOR = (0, 0, 255)     # RGB blue — distinct overlay colour for detected lines

# --------------------------------------------------------------------------- #
# Task 2 — SIFT Feature Extraction & Matching
# --------------------------------------------------------------------------- #
# Real same-scene / different-viewpoint pair: Oxford VGG "graffiti" wall, views 1 & 3
# (a genuine viewpoint change of one outdoor landmark), redistributed via opencv_extra.
T2_VIEW_A = ("t2_view_a.png", "img1.png")   # (local filename, remote name)
T2_VIEW_B = ("t2_view_b.png", "img3.png")
T2_SIFT_NFEATURES = 0           # 0 = keep all detected keypoints
T2_LOWE_RATIO = 0.75            # Lowe's ratio-test threshold (brief-specified)
T2_MATCHER = "flann"            # "flann" or "bf"
T2_TOP_MATCHES = 50             # number of good matches to draw
# Ratio-test sweep to discuss precision/recall trade-off.
T2_RATIO_SWEEP = [0.5, 0.6, 0.7, 0.75, 0.8, 0.9]

# --------------------------------------------------------------------------- #
# Task 3 — Image Stitching (manual, no cv2.Stitcher)
# --------------------------------------------------------------------------- #
# Real overlapping OUTDOOR panorama frames (opencv_extra stitching set boat1..boat3):
# a St. Petersburg waterfront shot from three overlapping camera positions, so the
# homographies carry real rotation + perspective and the RANSAC inlier ratio is
# meaningful (not the trivial ~100% of a single cropped image).
T3_FRAMES = [
    ("t3_frame_1.png", "boat1.jpg"),
    ("t3_frame_2.png", "boat2.jpg"),
    ("t3_frame_3.png", "boat3.jpg"),
]
T3_NUM_IMAGES = 3               # >= 3 overlapping frames required
T3_MAX_WIDTH = 1400             # downscale the 10MP source frames (keeps repo/runtime lean)
T3_RANSAC_REPROJ_THRESH = 5.0   # px reprojection tolerance for homography inliers
T3_RANSAC_MAX_ITERS = 2000
T3_MIN_MATCH_COUNT = 12         # below this, a pair is treated as "insufficient overlap"
T3_LOWE_RATIO = 0.75
T3_BLEND = "multiband"          # "alpha", "multiband", or "none"
T3_MULTIBAND_LEVELS = 4

# --------------------------------------------------------------------------- #
# Task 4 — CNN Scene Classification (PyTorch transfer learning)
# --------------------------------------------------------------------------- #
# Four outdoor scene categories exactly matching the CCP scenario (d):
#   road, building, vegetation, pedestrian_zone.
# road/building/vegetation are real Intel Image Classification photos (street/buildings/
# forest, relabelled); pedestrian_zone is Places365 "crosswalk" (the dataset the brief
# recommends). Both are bundled under data/task4_dataset (see README citations).
T4_CLASSES = ["road", "building", "vegetation", "pedestrian_zone"]  # scenario (d) set
T4_TRAIN_PER_CLASS = 160        # >= 100 required by the brief (per class)
T4_VAL_PER_CLASS = 48
# pedestrian_zone source (Places365, sorted-by-class; label 112 = crosswalk).
T4_PLACES_DATASET = "ljnlonoljpiljm/places365-256px"
T4_PLACES_CROSSWALK_LABEL = 112
T4_PLACES_CROSSWALK_OFFSET = 555000  # a confirmed offset inside the crosswalk block
T4_IMG_SIZE = 224
T4_BACKBONE = "resnet18"        # "resnet18" or "mobilenet_v2"
T4_BATCH_SIZE = 32
T4_EPOCHS = 8
T4_LR = 1e-3                    # classifier-head LR (backbone fully frozen)
T4_WEIGHT_DECAY = 1e-4
T4_TARGET_VAL_ACC = 0.70        # brief threshold
T4_GRADCAM_CORRECT = 2          # Grad-CAM on 2 correct ...
T4_GRADCAM_INCORRECT = 1        # ... and 1 incorrect sample
# ImageNet normalization (pretrained backbone expects this).
T4_MEAN = (0.485, 0.456, 0.406)
T4_STD = (0.229, 0.224, 0.225)
