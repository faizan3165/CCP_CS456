"""Resilient data layer for all five tasks.

Design goal: the whole pipeline must run end-to-end with a single command on any
machine. So every task's input is obtained by a *real-first, synthetic-fallback*
strategy:

  * ``auto``      (default) try real imagery; on any network failure fall back to a
                  reproducible synthetic scene so the run never breaks.
  * ``real``      require real imagery (raise if download fails).
  * ``synthetic`` always generate synthetic imagery (fully offline).

Real sources (both permissively hosted, no authentication):
  * OpenCV sample images  -> tasks 1-3 base scenes
    https://github.com/opencv/opencv (samples/data, BSD-3-Clause)
  * Intel Image Classification (ground-level outdoor scenes) via a Hugging Face
    mirror -> task 4 dataset
    https://huggingface.co/datasets/sfarrukhm/intel-image-classification

See CITATIONS in the repo README for full attribution.
"""

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from src import config
from src.utils import log, save_bgr

# --------------------------------------------------------------------------- #
# Low-level HTTP (stdlib only — no third-party download library)
# --------------------------------------------------------------------------- #
_OPENCV_BASE = "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data"
_HF_BASE = "https://datasets-server.huggingface.co"
_UA = {"User-Agent": "cs456-ccp/1.0"}


def _http_get(url, timeout=30, retries=5):
    """GET with retry/backoff — the HF datasets-server occasionally returns 502/503."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=_UA)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:  # HTTPError (5xx), URLError, timeouts
            last = e
            time.sleep(min(8.0, 1.5 * (attempt + 1)))
    raise last


def _http_json(url, timeout=30):
    return json.loads(_http_get(url, timeout).decode("utf-8"))


def _download_to(url, dest, timeout=30):
    data = _http_get(url, timeout)
    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "wb") as f:
        f.write(data)
    return dest


def _fetch_opencv_sample(name, dest):
    """Download an OpenCV sample image (BSD-3), returning the local path."""
    if Path(dest).exists():
        return dest
    _download_to(f"{_OPENCV_BASE}/{name}", dest)
    log(f"  downloaded real image {name} -> {Path(dest).name}")
    return dest


# --------------------------------------------------------------------------- #
# Synthetic scene generators (deterministic fallbacks)
# --------------------------------------------------------------------------- #
def _synthetic_road(h=420, w=680):
    """A perspective road with lane markings — clean input for Canny + Hough."""
    rng = np.random.default_rng(config.SEED)
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (200, 170, 140)            # sky (BGR)
    img[h // 2 :] = (70, 75, 78)               # asphalt
    img += rng.integers(0, 12, img.shape, dtype=np.int16).astype(np.uint8)  # sensor noise
    cx = w // 2
    # left/right road edges converging to a vanishing point
    cv2.line(img, (int(w * 0.08), h), (cx - 12, h // 2), (255, 255, 255), 3)
    cv2.line(img, (int(w * 0.92), h), (cx + 12, h // 2), (255, 255, 255), 3)
    # dashed centre lane markings
    for i in range(0, 9):
        t0, t1 = i / 9.0, (i + 0.5) / 9.0
        y0, y1 = int(h - t0 * (h / 2)), int(h - t1 * (h / 2))
        x0 = int(cx + (0) * (1 - t0))
        cv2.line(img, (x0, y0), (x0, y1), (0, 240, 240), max(1, int(4 * (1 - t0))))
    return img


def _synthetic_textured(h=480, w=640, seed=config.SEED):
    """A feature-rich facade-like scene with corners for SIFT to latch onto."""
    rng = np.random.default_rng(seed)
    img = rng.integers(90, 150, (h, w, 3), dtype=np.uint8)
    for _ in range(90):
        x, y = rng.integers(0, w - 60), rng.integers(0, h - 60)
        bw, bh = rng.integers(18, 55), rng.integers(18, 55)
        color = tuple(int(c) for c in rng.integers(0, 255, 3))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, -1)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (20, 20, 20), 2)
    for _ in range(40):
        c = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        cv2.circle(img, c, int(rng.integers(4, 16)),
                   tuple(int(v) for v in rng.integers(0, 255, 3)), -1)
    return img


def _warp_view(img, shift_deg=12, tx=40, brightness=0.85):
    """Apply a known homography (rotation about centre + translation) + lighting change."""
    h, w = img.shape[:2]
    ang = np.deg2rad(shift_deg)
    cx, cy = w / 2, h / 2
    rot = np.array([[np.cos(ang), -np.sin(ang), cx - (np.cos(ang) * cx - np.sin(ang) * cy)],
                    [np.sin(ang), np.cos(ang), cy - (np.sin(ang) * cx + np.cos(ang) * cy)],
                    [0, 0, 1]], np.float64)
    persp = np.array([[1, 0, tx], [0, 1, 0], [3e-4, 1e-4, 1]], np.float64)
    H = persp @ rot
    warped = cv2.warpPerspective(img, H, (w, h), borderValue=(0, 0, 0))
    warped = np.clip(warped.astype(np.float32) * brightness, 0, 255).astype(np.uint8)
    return warped, H


# --------------------------------------------------------------------------- #
# Task 1 — road / structural images
# --------------------------------------------------------------------------- #
def ensure_task1_images(mode=None):
    """Return a list of (name, path) outdoor images with strong linear structure."""
    mode = mode or config.DATA_MODE
    out = []
    building = config.RAW_DIR / "t1_building.jpg"
    road = config.RAW_DIR / "t1_road.png"
    if mode in ("auto", "real"):
        try:
            _fetch_opencv_sample("building.jpg", building)
            out.append(("building", building))
        except Exception as e:
            if mode == "real":
                raise
            log(f"  [task1] real download failed ({e}); using synthetic road only")
    if not out or mode == "synthetic":
        save_bgr(road, _synthetic_road())
        out.append(("road", road))
    else:
        # add a synthetic road too, so we demo both structural + lane-marking lines
        save_bgr(road, _synthetic_road())
        out.append(("road", road))
    return out


# --------------------------------------------------------------------------- #
# Task 2 — two views of the same scene
# --------------------------------------------------------------------------- #
def ensure_task2_pair(mode=None):
    """Return (imgA_path, imgB_path, ground_truth_H_or_None).

    Real path uses OpenCV's canonical box / box_in_scene pair (genuine viewpoint
    change). Synthetic path derives a second view via a known homography.
    """
    mode = mode or config.DATA_MODE
    a = config.RAW_DIR / "t2_view_a.png"
    b = config.RAW_DIR / "t2_view_b.png"
    if mode in ("auto", "real"):
        try:
            _fetch_opencv_sample("box.png", a)
            _fetch_opencv_sample("box_in_scene.png", b)
            return a, b, None
        except Exception as e:
            if mode == "real":
                raise
            log(f"  [task2] real download failed ({e}); deriving synthetic views")
    base = _synthetic_textured()
    warped, H = _warp_view(base, shift_deg=14, tx=55, brightness=0.8)
    save_bgr(a, base)
    save_bgr(b, warped)
    return a, b, H


# --------------------------------------------------------------------------- #
# Task 3 — overlapping frame sequence for panorama
# --------------------------------------------------------------------------- #
def ensure_task3_sequence(mode=None, n=config.T3_NUM_IMAGES):
    """Return a list of >= n overlapping frame paths (left -> right pan)."""
    mode = mode or config.DATA_MODE
    src = config.RAW_DIR / "t3_wide.png"
    got_real = False
    if mode in ("auto", "real"):
        try:
            _fetch_opencv_sample("graf1.png", src)
            got_real = True
        except Exception as e:
            if mode == "real":
                raise
            log(f"  [task3] real download failed ({e}); using synthetic wide scene")
    if not got_real:
        wide = _synthetic_textured(h=420, w=1200, seed=7)
        save_bgr(src, wide)
    wide = cv2.imread(str(src), cv2.IMREAD_COLOR)
    h, w = wide.shape[:2]
    # slice n overlapping vertical windows (~45% overlap) simulating a camera pan
    win = int(w / (1 + (n - 1) * 0.55))
    step = int((w - win) / (n - 1))
    paths = []
    rng = np.random.default_rng(config.SEED)
    for i in range(n):
        x0 = min(i * step, w - win)
        frame = wide[:, x0 : x0 + win].copy()
        # mild per-frame brightness/gain variation so blending has to work
        gain = 0.9 + 0.06 * i
        frame = np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        p = config.RAW_DIR / f"t3_frame_{i + 1}.png"
        save_bgr(p, frame)
        paths.append(p)
    return paths


# --------------------------------------------------------------------------- #
# Task 4 — scene-classification dataset (ImageFolder: <split>/<class>/*.jpg)
# --------------------------------------------------------------------------- #
def _hf_filter_page(dataset, split, label_idx, offset, length=100):
    where = urllib.parse.quote(f'"label"={label_idx}')
    url = (f"{_HF_BASE}/filter?dataset={urllib.parse.quote(dataset)}"
           f"&config=default&split={split}&where={where}&offset={offset}&length={length}")
    return _http_json(url)


def _download_hf_class(dataset, split, label_idx, n, dest_dir):
    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(dest_dir.glob("*.jpg")))
    if existing >= n:
        return existing  # already downloaded on a previous run — resume
    got, offset = 0, 0
    while got < n:
        page = _hf_filter_page(dataset, split, label_idx, offset, 100)
        rows = page.get("rows", [])
        if not rows:
            break
        for r in rows:
            src = r["row"]["image"]["src"]
            try:
                _download_to(src, dest_dir / f"{got:04d}.jpg", timeout=30)
                got += 1
            except Exception:
                continue  # skip a transient image failure, keep going
            if got >= n:
                break
        offset += len(rows)
    return got


def _dataset_ready(train_dir, val_dir):
    if not train_dir.exists() or not val_dir.exists():
        return False
    for cls in config.T4_CLASSES:
        if len(list((train_dir / cls).glob("*.jpg"))) < config.T4_TRAIN_PER_CLASS:
            return False
        if len(list((val_dir / cls).glob("*.jpg"))) < config.T4_VAL_PER_CLASS:
            return False
    return True


def _build_real_task4(train_dir, val_dir):
    ds = config.T4_HF_DATASET
    for cls in config.T4_CLASSES:
        idx = config.T4_HF_LABELS.index(cls)
        # Retry the whole class a few times — resumes from images already on disk.
        n_tr = n_va = 0
        for attempt in range(3):
            n_tr = _download_hf_class(ds, "train", idx, config.T4_TRAIN_PER_CLASS, train_dir / cls)
            n_va = _download_hf_class(ds, "test", idx, config.T4_VAL_PER_CLASS, val_dir / cls)
            if n_tr >= config.T4_TRAIN_PER_CLASS and n_va >= config.T4_VAL_PER_CLASS:
                break
            log(f"  [task4] {cls}: got {n_tr}/{config.T4_TRAIN_PER_CLASS} train, "
                f"{n_va}/{config.T4_VAL_PER_CLASS} val — retry {attempt + 1}/3")
            time.sleep(3)
        log(f"  [task4] {cls:9s}: {n_tr} train + {n_va} val (real, Intel/HF)")
        if n_tr < config.T4_TRAIN_PER_CLASS or n_va < config.T4_VAL_PER_CLASS:
            raise RuntimeError(f"insufficient real images for '{cls}'")


def _build_synthetic_task4(train_dir, val_dir):
    """Four visually separable but noisy synthetic classes (fallback only)."""
    rng = np.random.default_rng(config.SEED)
    size = config.T4_IMG_SIZE

    def make(cls, seed):
        r = np.random.default_rng(seed)
        img = np.zeros((size, size, 3), np.uint8)
        if cls == "street":
            img[: size // 2] = (180, 160, 140)
            img[size // 2 :] = (70, 72, 75)
            cv2.line(img, (30, size), (size // 2, size // 2), (255, 255, 255), 3)
            cv2.line(img, (size - 30, size), (size // 2, size // 2), (255, 255, 255), 3)
        elif cls == "buildings":
            img[:] = (150, 150, 155)
            for _ in range(26):
                x, y = r.integers(0, size - 25), r.integers(0, size - 25)
                cv2.rectangle(img, (x, y), (x + 12, y + 18), (40, 40, 45), -1)
        elif cls == "forest":
            img[:] = (40, 110, 40)
            for _ in range(400):
                c = (int(r.integers(0, size)), int(r.integers(0, size)))
                cv2.circle(img, c, int(r.integers(2, 7)),
                           (int(r.integers(20, 70)), int(r.integers(90, 160)), int(r.integers(20, 70))), -1)
        else:  # mountain
            img[:] = (200, 180, 150)
            pts = np.array([[0, size], [size // 3, size // 3], [2 * size // 3, size // 2],
                            [size, size // 4], [size, size], [0, size]], np.int32)
            cv2.fillPoly(img, [pts], (120, 120, 125))
        img = np.clip(img.astype(np.int16) + r.integers(-18, 18, img.shape), 0, 255).astype(np.uint8)
        return img

    for split, per in (("train", config.T4_TRAIN_PER_CLASS), ("val", config.T4_VAL_PER_CLASS)):
        for cls in config.T4_CLASSES:
            d = (train_dir if split == "train" else val_dir) / cls
            d.mkdir(parents=True, exist_ok=True)
            for i in range(per):
                save_bgr(d / f"{i:04d}.jpg", make(cls, int(rng.integers(0, 1 << 30))))
    log("  [task4] built synthetic 4-class dataset (fallback)")


def ensure_task4_dataset(mode=None):
    """Populate an ImageFolder dataset and return (train_dir, val_dir, classes)."""
    mode = mode or config.DATA_MODE
    train_dir = config.TASK4_DATA_DIR / "train"
    val_dir = config.TASK4_DATA_DIR / "val"
    if _dataset_ready(train_dir, val_dir):
        log("  [task4] dataset already present — reusing")
        return train_dir, val_dir, config.T4_CLASSES
    if mode in ("auto", "real"):
        try:
            log("  [task4] downloading real Intel scene images (Hugging Face mirror)...")
            _build_real_task4(train_dir, val_dir)
            return train_dir, val_dir, config.T4_CLASSES
        except Exception as e:
            if mode == "real":
                raise
            log(f"  [task4] real download failed ({e}); falling back to synthetic")
    _build_synthetic_task4(train_dir, val_dir)
    return train_dir, val_dir, config.T4_CLASSES
