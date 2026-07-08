"""Shared helpers: reproducibility, image IO, and Matplotlib visualization.

Kept deliberately small and dependency-light (OpenCV, NumPy, Matplotlib only) so the
task modules stay focused on the computer-vision logic.
"""

import random
import time
from pathlib import Path

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from src import config

matplotlib.use("Agg")  # headless: render figures to files, never to a GUI window


# --------------------------------------------------------------------------- #
# Reproducibility & logging
# --------------------------------------------------------------------------- #
def set_seed(seed=config.SEED):
    """Seed Python and NumPy (and Torch if present) for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


_T0 = None


def log(msg):
    """Timestamped progress print (seconds since first log call)."""
    global _T0
    now = time.monotonic()
    if _T0 is None:
        _T0 = now
    print(f"[{now - _T0:7.1f}s] {msg}", flush=True)


def ensure_dirs():
    """Create every output directory declared in config."""
    for d in config.ALL_OUTPUT_DIRS:
        Path(d).mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Image IO  (internally BGR per OpenCV; helpers convert to/from RGB)
# --------------------------------------------------------------------------- #
def imread_bgr(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def imread_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def save_bgr(path, img_bgr):
    """Write a BGR uint8 image to disk as PNG/JPG."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img_bgr)


def save_rgb(path, img_rgb):
    save_bgr(path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------------- #
def show_row(images, titles, out_path, cmaps=None, suptitle=None, figsize_per=4.2):
    """Save a single row of images side-by-side (the brief asks for these repeatedly).

    ``images`` may be RGB (HxWx3) or grayscale (HxW). ``cmaps`` optionally overrides
    the colormap per panel (use "gray" for edge/mask panels).
    """
    n = len(images)
    if cmaps is None:
        cmaps = [None] * n
    fig, axes = plt.subplots(1, n, figsize=(figsize_per * n, figsize_per + 0.6))
    if n == 1:
        axes = [axes]
    for ax, im, title, cmap in zip(axes, images, titles, cmaps):
        if im.ndim == 2 and cmap is None:
            cmap = "gray"
        ax.imshow(im, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_table_figure(rows, col_labels, out_path, title=None, col_widths=None):
    """Render a small results table as a figure (used in outputs and the report)."""
    n_rows = len(rows)
    fig_h = 0.5 + 0.42 * (n_rows + 1)
    fig, ax = plt.subplots(figsize=(min(2.0 * len(col_labels), 12), fig_h))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    table = ax.table(
        cellText=[[str(c) for c in r] for r in rows],
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor("#2c3e6b")
        cell.set_text_props(color="white", fontweight="bold")
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
