"""Task 2 — SIFT Feature Extraction & Matching  [CLO-3].

Two real views of the same outdoor scene (Oxford VGG "graffiti" wall, views 1 & 3 —
a genuine viewpoint change of one landmark):
  1. Detect SIFT keypoints + descriptors in both images.
  2. Draw rich keypoints (size + orientation).
  3. Match descriptors (FLANN by default) and apply Lowe's ratio test (0.75).
  4. Draw the top-50 good matches; report total keypoints vs. good matches.
  5. Sweep the ratio threshold to illustrate the precision/recall trade-off.

Outputs (outputs/task2/):
  * keypoints_a.png / keypoints_b.png   rich keypoint visualizations
  * top_matches.png                     top-50 good matches (drawMatches)
  * ratio_sweep.png                     #good matches vs Lowe ratio (precision/recall)
  * task2_stats.png                     keypoint / match statistics table
"""

import cv2
import numpy as np

from src import config
from src.data_setup import ensure_task2_pair
from src.utils import bgr_to_rgb, log, save_rgb, save_table_figure


def _build_matcher():
    if config.T2_MATCHER == "bf":
        return cv2.BFMatcher(cv2.NORM_L2)  # L2 is correct for SIFT float descriptors
    # FLANN with a KD-tree index (float descriptors)
    index_params = dict(algorithm=1, trees=5)   # FLANN_INDEX_KDTREE = 1
    search_params = dict(checks=50)
    return cv2.FlannBasedMatcher(index_params, search_params)


def _ratio_test(knn_matches, ratio):
    """Lowe's ratio test: keep m only if it is clearly better than the 2nd-best n."""
    good = []
    for pair in knn_matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            good.append(m)
    return good


def _plot_ratio_sweep(ratios, counts, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(ratios, counts, "o-", color="#2c3e6b", linewidth=2)
    ax.axvline(config.T2_LOWE_RATIO, color="#c0392b", ls="--",
               label=f"chosen ratio = {config.T2_LOWE_RATIO}")
    ax.set_xlabel("Lowe's ratio threshold")
    ax.set_ylabel("# good matches retained")
    ax.set_title("Task 2 — Ratio threshold vs. match count\n(low ratio = high precision/low recall)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    log("TASK 2 — SIFT Feature Extraction & Matching")
    path_a, path_b, _ = ensure_task2_pair()
    img_a = cv2.imread(str(path_a), cv2.IMREAD_COLOR)
    img_b = cv2.imread(str(path_b), cv2.IMREAD_COLOR)
    gray_a = cv2.cvtColor(img_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(img_b, cv2.COLOR_BGR2GRAY)

    # --- 1. SIFT detection + description ----------------------------------- #
    sift = cv2.SIFT_create(nfeatures=config.T2_SIFT_NFEATURES)
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    log(f"  keypoints: A={len(kp_a)}  B={len(kp_b)}")

    # --- 2. Rich keypoint visualization (size + orientation) --------------- #
    flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    vis_a = cv2.drawKeypoints(img_a, kp_a, None, color=(0, 255, 0), flags=flags)
    vis_b = cv2.drawKeypoints(img_b, kp_b, None, color=(0, 255, 0), flags=flags)
    save_rgb(config.OUT_T2 / "keypoints_a.png", bgr_to_rgb(vis_a))
    save_rgb(config.OUT_T2 / "keypoints_b.png", bgr_to_rgb(vis_b))

    # --- 3. Match + Lowe ratio test ---------------------------------------- #
    matcher = _build_matcher()
    knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = _ratio_test(knn, config.T2_LOWE_RATIO)
    good = sorted(good, key=lambda m: m.distance)
    match_ratio = len(good) / max(1, min(len(kp_a), len(kp_b)))
    log(f"  good matches @ ratio {config.T2_LOWE_RATIO}: {len(good)}  "
        f"(match quality ratio = {match_ratio:.3f})")

    # --- 4. Draw the top-50 good matches ----------------------------------- #
    top = good[: config.T2_TOP_MATCHES]
    vis_match = cv2.drawMatches(
        img_a, kp_a, img_b, kp_b, top, None,
        matchColor=(0, 255, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    save_rgb(config.OUT_T2 / "top_matches.png", bgr_to_rgb(vis_match))

    # --- 5. Ratio-threshold sweep (precision/recall discussion) ------------ #
    ratios = config.T2_RATIO_SWEEP
    counts = [len(_ratio_test(knn, r)) for r in ratios]
    _plot_ratio_sweep(ratios, counts, config.OUT_T2 / "ratio_sweep.png")

    # --- stats table ------------------------------------------------------- #
    rows = [
        ["keypoints (A)", len(kp_a)],
        ["keypoints (B)", len(kp_b)],
        ["raw knn candidates", len(knn)],
        [f"good matches (ratio {config.T2_LOWE_RATIO})", len(good)],
        ["top matches drawn", len(top)],
        ["match quality ratio", f"{match_ratio:.3f}"],
    ]
    save_table_figure(rows, ["metric", "value"], config.OUT_T2 / "task2_stats.png",
                      title="Task 2 — SIFT matching statistics")
    log(f"  wrote Task 2 outputs to {config.OUT_T2}")
    return {
        "kp_a": len(kp_a), "kp_b": len(kp_b), "good": len(good),
        "match_ratio": match_ratio, "ratio_sweep": list(zip(ratios, counts)),
    }


if __name__ == "__main__":
    from src.utils import ensure_dirs, set_seed

    set_seed()
    ensure_dirs()
    run()
