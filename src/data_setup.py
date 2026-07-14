"""Data layer for all five tasks — REAL, permissively-licensed imagery only.

Every input is a real photograph bundled under ``data/`` and fully cited in the README.
Nothing is synthetic or self-captured. Each ``ensure_*`` function is *real-first*: it
reads the committed files, and if any are missing it performs a one-time download from
the original source (used only to (re)build the bundle). If a download is impossible it
raises a clear error rather than silently fabricating data.

Sources (all permissive, no authentication):
  * Task 1 — Udacity CarND-LaneLines-P1 road images (MIT)
    https://github.com/udacity/CarND-LaneLines-P1  (test_images/)
  * Task 2 — Oxford VGG affine "graffiti" wall, views 1 & 3 (free for research),
    redistributed via opencv_extra
    https://www.robots.ox.ac.uk/~vgg/research/affine/
  * Task 3 — OpenCV stitching sample "a1..a3" building panorama (opencv_extra)
    https://github.com/opencv/opencv_extra  (testdata/stitching/)
  * Task 4 — road/building/vegetation: Intel Image Classification photos (Kaggle);
    pedestrian_zone: Places365 "crosswalk" (Zhou et al., 2017). Both bundled.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.utils import log

# --------------------------------------------------------------------------- #
# Remote sources (used only to (re)build the committed bundle)
# --------------------------------------------------------------------------- #
_UDACITY = "https://raw.githubusercontent.com/udacity/CarND-LaneLines-P1/master/test_images"
_GRAF = ("https://raw.githubusercontent.com/opencv/opencv_extra/master/"
         "testdata/cv/detectors_descriptors_evaluation/images_datasets/graf")
_STITCH = "https://raw.githubusercontent.com/opencv/opencv_extra/master/testdata/stitching"
_HF = "https://datasets-server.huggingface.co"
_UA = {"User-Agent": "Mozilla/5.0 cs456-ccp/2.0"}


def _http_get(url, timeout=30, retries=6):
    """GET with retry/backoff for flaky mirrors."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # HTTPError / URLError / timeouts
            last = e
            time.sleep(min(12.0, 1.5 * (attempt + 1)))
    raise last


def _http_json(url, timeout=30):
    return json.loads(_http_get(url, timeout).decode("utf-8"))


def _ensure_file(local_name, remote_url, max_width=None):
    """Return the local path, downloading from ``remote_url`` only if missing.

    If ``max_width`` is set, the fetched image is decoded and downscaled (preserving
    aspect) so the committed bundle and any re-fetch stay identical in size.
    """
    dest = config.RAW_DIR / local_name
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = _http_get(remote_url)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"downloaded {remote_url} is not a decodable image")
    if max_width and img.shape[1] > max_width:
        h = int(round(img.shape[0] * max_width / img.shape[1]))
        img = cv2.resize(img, (max_width, h), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(dest), img)
    log(f"  fetched real image {local_name} <- {remote_url.rsplit('/', 1)[-1]}")
    return dest


# --------------------------------------------------------------------------- #
# Task 1 — outdoor road images (lane markings + structural lines)
# --------------------------------------------------------------------------- #
def ensure_task1_images():
    """Return a list of (name, path) real outdoor road images."""
    out = []
    for name, local, remote in config.T1_IMAGES:
        out.append((name, _ensure_file(local, f"{_UDACITY}/{remote}")))
    return out


# --------------------------------------------------------------------------- #
# Task 2 — two real views of the same outdoor scene (viewpoint change)
# --------------------------------------------------------------------------- #
def ensure_task2_pair():
    """Return (imgA_path, imgB_path, ground_truth_H) — H is None (unknown, real pair)."""
    a = _ensure_file(config.T2_VIEW_A[0], f"{_GRAF}/{config.T2_VIEW_A[1]}")
    b = _ensure_file(config.T2_VIEW_B[0], f"{_GRAF}/{config.T2_VIEW_B[1]}")
    return a, b, None


# --------------------------------------------------------------------------- #
# Task 3 — real sequence of overlapping frames for a panorama
# --------------------------------------------------------------------------- #
def ensure_task3_sequence(n=config.T3_NUM_IMAGES):
    """Return a list of >= n real overlapping frame paths (a genuine building pan)."""
    paths = []
    for local, remote in config.T3_FRAMES[:n]:
        paths.append(_ensure_file(local, f"{_STITCH}/{remote}", max_width=config.T3_MAX_WIDTH))
    return paths


# --------------------------------------------------------------------------- #
# Task 4 — scene-classification dataset (ImageFolder: <split>/<class>/*.jpg)
# --------------------------------------------------------------------------- #
def _rows_page(dataset, split, offset, length=100):
    url = (f"{_HF}/rows?dataset={urllib.parse.quote(dataset)}"
           f"&config=default&split={split}&offset={offset}&length={length}")
    return _http_json(url)


def _download_places365_crosswalk(train_dir, val_dir):
    """Block-extract Places365 'crosswalk' (label 112) into pedestrian_zone train/val.

    The datasets-server /filter index is partial, but the dataset is sorted by class at
    5000 imgs/class, so we page /rows from a confirmed in-block offset and keep only
    label-112 rows. Train/val are disjoint crosswalk images.
    """
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    n_tr, n_va = config.T4_TRAIN_PER_CLASS, config.T4_VAL_PER_CLASS
    label, offset, got = config.T4_PLACES_CROSSWALK_LABEL, config.T4_PLACES_CROSSWALK_OFFSET, 0
    while got < n_tr + n_va:
        page = _rows_page(config.T4_PLACES_DATASET, "train", offset, 100)
        rows = page.get("rows", [])
        if not rows:
            break
        walked_past = False
        for row in rows:
            r = row["row"]
            if r["label"] != label:
                walked_past = True
                break
            dest = (train_dir if got < n_tr else val_dir) / f"{(got if got < n_tr else got - n_tr):04d}.jpg"
            try:
                raw = _http_get(r["image"]["src"], retries=4)
                arr = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
                if arr is not None:
                    cv2.imwrite(str(dest), arr)
                    got += 1
            except Exception:
                continue
            if got >= n_tr + n_va:
                break
        offset += len(rows)
        if walked_past:
            break
    log(f"  [task4] pedestrian_zone: {len(list(train_dir.glob('*.jpg')))} train + "
        f"{len(list(val_dir.glob('*.jpg')))} val (Places365 crosswalk)")


def _class_ready(root, cls, need):
    return len(list((root / cls).glob("*.jpg"))) >= need


def ensure_task4_dataset():
    """Ensure the bundled 4-class ImageFolder is present; return (train, val, classes).

    road/building/vegetation are committed real Intel photos. pedestrian_zone is
    downloaded from Places365 only if the committed copy is missing.
    """
    train_dir = config.TASK4_DATA_DIR / "train"
    val_dir = config.TASK4_DATA_DIR / "val"

    # pedestrian_zone: populate-if-missing (the rest are bundled real photos).
    if not (_class_ready(train_dir, "pedestrian_zone", config.T4_TRAIN_PER_CLASS)
            and _class_ready(val_dir, "pedestrian_zone", config.T4_VAL_PER_CLASS)):
        log("  [task4] fetching pedestrian_zone from Places365 (crosswalk)...")
        _download_places365_crosswalk(train_dir / "pedestrian_zone", val_dir / "pedestrian_zone")

    missing = [c for c in config.T4_CLASSES
               if not (_class_ready(train_dir, c, config.T4_TRAIN_PER_CLASS)
                       and _class_ready(val_dir, c, config.T4_VAL_PER_CLASS))]
    if missing:
        raise RuntimeError(
            f"missing bundled Task-4 images for classes {missing}; expected committed data "
            f"under {config.TASK4_DATA_DIR} (>= {config.T4_TRAIN_PER_CLASS}/class train).")
    log(f"  [task4] dataset ready: {config.T4_CLASSES} "
        f"({config.T4_TRAIN_PER_CLASS} train + {config.T4_VAL_PER_CLASS} val / class)")
    return train_dir, val_dir, config.T4_CLASSES
