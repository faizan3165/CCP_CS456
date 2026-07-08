"""Task 1 — Edge Detection & Hough Transform  [CLO-3].

Pipeline per image:
  1. Gaussian smoothing (kernel/sigma justified in config) to suppress noise.
  2. Canny edge detection swept over THREE (low, high) hysteresis configs on the
     SAME image to show the effect of the thresholds.
  3. Probabilistic Hough Line Transform (cv2.HoughLinesP) swept over three parameter
     sets; detected lines overlaid on the original in a distinct colour.

Outputs (outputs/task1/):
  * <name>_canny_sweep.png        original + 3 Canny threshold configs
  * <name>_pipeline.png           original | Canny (balanced) | Hough overlay
  * <name>_hough_<cfg>.png        overlay for each Hough config
  * task1_param_experiments.png   parameter-experiment table
"""

import cv2
import numpy as np

from src import config
from src.data_setup import ensure_task1_images
from src.utils import log, save_bgr, save_table_figure, show_row, bgr_to_rgb


def _canny(gray_blurred, low, high):
    return cv2.Canny(gray_blurred, low, high, L2gradient=True)


def _edge_density(edges):
    """Fraction of pixels flagged as edges — a simple 'clutter' proxy."""
    return float((edges > 0).mean())


def _hough_lines(edges, cfg):
    lines = cv2.HoughLinesP(
        edges, rho=cfg["rho"], theta=cfg["theta"], threshold=cfg["threshold"],
        minLineLength=cfg["min_line_length"], maxLineGap=cfg["max_line_gap"],
    )
    return lines if lines is not None else np.empty((0, 1, 4), np.int32)


def _line_stats(lines):
    if len(lines) == 0:
        return 0, 0.0
    seg = lines.reshape(-1, 4).astype(np.float64)
    lengths = np.hypot(seg[:, 2] - seg[:, 0], seg[:, 3] - seg[:, 1])
    return len(lengths), float(lengths.mean())


def _overlay(img_bgr, lines, color=config.T1_LINE_COLOR):
    out = img_bgr.copy()
    bgr = (int(color[2]), int(color[1]), int(color[0]))  # config colour is RGB
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), bgr, 2, cv2.LINE_AA)
    return out


def _process_image(name, path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, config.T1_GAUSSIAN_KERNEL, config.T1_GAUSSIAN_SIGMA)

    # --- Canny threshold sweep (3 configs on the SAME image) --------------- #
    canny_maps, canny_rows = {}, []
    for cfg in config.T1_CANNY_CONFIGS:
        edges = _canny(blurred, cfg["low"], cfg["high"])
        canny_maps[cfg["name"]] = edges
        canny_rows.append([name, cfg["name"], f'{cfg["low"]}/{cfg["high"]}',
                           f"{100 * _edge_density(edges):.2f}%"])
    show_row(
        [bgr_to_rgb(img)] + [canny_maps[c["name"]] for c in config.T1_CANNY_CONFIGS],
        ["original"] + [f'Canny {c["name"]} ({c["low"]},{c["high"]})' for c in config.T1_CANNY_CONFIGS],
        config.OUT_T1 / f"{name}_canny_sweep.png",
        suptitle=f"Task 1 — Canny threshold sweep ({name})",
    )

    # --- Hough parameter sweep on the balanced edge map -------------------- #
    balanced = canny_maps["balanced"]
    hough_rows, overlays = [], {}
    for cfg in config.T1_HOUGH_CONFIGS:
        lines = _hough_lines(balanced, cfg)
        n, mean_len = _line_stats(lines)
        overlays[cfg["name"]] = _overlay(img, lines)
        save_bgr(config.OUT_T1 / f"{name}_hough_{cfg['name']}.png", overlays[cfg["name"]])
        hough_rows.append([name, cfg["name"],
                           f'{cfg["threshold"]}/{cfg["min_line_length"]}/{cfg["max_line_gap"]}',
                           n, f"{mean_len:.0f}px"])

    primary = config.T1_HOUGH_PRIMARY
    show_row(
        [bgr_to_rgb(img), balanced, bgr_to_rgb(overlays[primary])],
        ["original", "Canny (balanced)", f"Hough overlay ({primary})"],
        config.OUT_T1 / f"{name}_pipeline.png",
        suptitle=f"Task 1 — Edge & line detection pipeline ({name})",
    )
    return canny_rows, hough_rows


def run():
    log("TASK 1 — Edge Detection & Hough Transform")
    images = ensure_task1_images()
    all_canny, all_hough = [], []
    for name, path in images:
        cr, hr = _process_image(name, path)
        all_canny += cr
        all_hough += hr

    # Combined parameter-experiment tables (required deliverable).
    save_table_figure(
        all_canny, ["image", "config", "low/high", "edge density"],
        config.OUT_T1 / "task1_param_experiments_canny.png",
        title="Task 1 — Canny threshold experiments",
    )
    save_table_figure(
        all_hough, ["image", "config", "thr/minLen/maxGap", "#lines", "mean len"],
        config.OUT_T1 / "task1_param_experiments_hough.png",
        title="Task 1 — Probabilistic Hough experiments",
    )
    log(f"  wrote Task 1 outputs to {config.OUT_T1}")
    return {"images": [n for n, _ in images], "canny": all_canny, "hough": all_hough}


if __name__ == "__main__":
    from src.utils import ensure_dirs, set_seed

    set_seed()
    ensure_dirs()
    run()
