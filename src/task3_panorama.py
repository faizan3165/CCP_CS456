"""Task 3 — Image Stitching / Panorama Construction  [CLO-4].

Fully MANUAL pipeline (no cv2.Stitcher):
  1. SIFT features + Lowe-ratio matches between each consecutive frame pair.
  2. Homography estimated with a FROM-SCRATCH RANSAC over a normalized-DLT solver
     (numpy/SVD) — we report the inlier ratio our own RANSAC finds.
  3. Consecutive homographies are chained to a middle reference frame; all frames
     are warped onto one canvas (cv2.warpPerspective is a primitive, not a stitcher).
  4. Seams removed with distance-transform feathering (alpha) or Burt-Adelson
     multi-band (Laplacian-pyramid) blending.

Edge cases handled: insufficient overlap (too few matches) and near-degenerate /
highly-distorting homographies are detected and reported.

Outputs (outputs/task3/):
  * frames.png            the source frames in order
  * pair_matches.png      inlier matches for the first pair (RANSAC result)
  * panorama.png          final stitched panorama
  * homographies.txt      the intermediate 3x3 homography matrices
  * task3_stats.png       inlier ratios + match counts per pair
"""

import cv2
import numpy as np

from src import config
from src.data_setup import ensure_task3_sequence
from src.utils import bgr_to_rgb, log, save_bgr, save_rgb, save_table_figure, show_row


# --------------------------------------------------------------------------- #
# Feature matching
# --------------------------------------------------------------------------- #
def _sift_match(gray_a, gray_b, ratio=config.T3_LOWE_RATIO):
    sift = cv2.SIFT_create()
    kp_a, desc_a = sift.detectAndCompute(gray_a, None)
    kp_b, desc_b = sift.detectAndCompute(gray_b, None)
    flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), dict(checks=50))
    knn = flann.knnMatch(desc_a, desc_b, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < ratio * n.distance]
    pts_a = np.float64([kp_a[m.queryIdx].pt for m in good])
    pts_b = np.float64([kp_b[m.trainIdx].pt for m in good])
    return pts_a, pts_b, kp_a, kp_b, good


# --------------------------------------------------------------------------- #
# From-scratch homography (normalized DLT + RANSAC)
# --------------------------------------------------------------------------- #
def _normalize(pts):
    """Hartley normalization: centre + isotropic scale so mean distance = sqrt(2)."""
    mean = pts.mean(axis=0)
    d = np.sqrt(((pts - mean) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2) / (d + 1e-12)
    T = np.array([[s, 0, -s * mean[0]], [0, s, -s * mean[1]], [0, 0, 1]], np.float64)
    ph = np.column_stack([pts, np.ones(len(pts))])
    return (T @ ph.T).T[:, :2], T


def _dlt(src, dst):
    """Direct Linear Transform: solve dst ~ H·src for >=4 correspondences."""
    src_n, ts = _normalize(src)
    dst_n, td = _normalize(dst)
    a = []
    for (x, y), (u, v) in zip(src_n, dst_n):
        a.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        a.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(a))
    h = vt[-1].reshape(3, 3)
    h = np.linalg.inv(td) @ h @ ts
    return h / h[2, 2]


def _reproj_err(H, src, dst):
    ph = np.column_stack([src, np.ones(len(src))])
    p = (H @ ph.T).T
    w = p[:, 2:3]
    w[np.abs(w) < 1e-12] = 1e-12
    return np.sqrt(((p[:, :2] / w - dst) ** 2).sum(axis=1))


def ransac_homography(src, dst, thresh=config.T3_RANSAC_REPROJ_THRESH,
                      iters=config.T3_RANSAC_MAX_ITERS, seed=config.SEED):
    """Manual RANSAC. Returns (H, inlier_mask, inlier_ratio)."""
    n = len(src)
    if n < 4:
        return None, None, 0.0
    rng = np.random.default_rng(seed)
    best_mask, best_count = None, -1
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            H = _dlt(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(H).all():
            continue
        inliers = _reproj_err(H, src, dst) < thresh
        if inliers.sum() > best_count:
            best_count, best_mask = int(inliers.sum()), inliers
    if best_mask is None or best_mask.sum() < 4:
        return None, best_mask, 0.0
    H = _dlt(src[best_mask], dst[best_mask])   # refit on all inliers
    return H, best_mask, float(best_mask.mean())


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def _corners(w, h):
    return np.float64([[0, 0], [w, 0], [w, h], [0, h]])


def _apply(H, pts):
    ph = np.column_stack([pts, np.ones(len(pts))])
    p = (H @ ph.T).T
    return p[:, :2] / p[:, 2:3]


def _perspective_distortion(H, w, h):
    """Ratio of warped to original area — flags extreme perspective stretch."""
    warped = _apply(H, _corners(w, h))
    def area(p):
        return 0.5 * abs(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                         np.dot(p[:, 1], np.roll(p[:, 0], -1)))
    return area(warped) / max(1.0, w * h)


# --------------------------------------------------------------------------- #
# Blending
# --------------------------------------------------------------------------- #
def _feather_weights(masks):
    weights = [cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) for m in masks]
    wsum = np.sum(weights, axis=0) + 1e-8
    return [w / wsum for w in weights]


def _alpha_blend(warped, weights):
    pano = np.zeros_like(warped[0], np.float32)
    for img, w in zip(warped, weights):
        pano += img.astype(np.float32) * w[..., None]
    return np.clip(pano, 0, 255).astype(np.uint8)


def _multiband_blend(warped, weights, levels=config.T3_MULTIBAND_LEVELS):
    """Burt-Adelson multi-band blend of N warped frames with feather weights."""
    def gauss_pyr(img):
        pyr = [img.astype(np.float32)]
        for _ in range(levels):
            pyr.append(cv2.pyrDown(pyr[-1]))
        return pyr

    def lap_pyr(img):
        gp = gauss_pyr(img)
        lp = []
        for i in range(levels):
            up = cv2.pyrUp(gp[i + 1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
            lp.append(gp[i] - up)
        lp.append(gp[-1])
        return lp

    blended = None
    for img, w in zip(warped, weights):
        lp = lap_pyr(img.astype(np.float32))
        wp = gauss_pyr(w)
        contrib = [lp[i] * wp[i][..., None] for i in range(len(lp))]
        blended = contrib if blended is None else [b + c for b, c in zip(blended, contrib)]
    out = blended[-1]
    for i in range(levels - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(blended[i].shape[1], blended[i].shape[0])) + blended[i]
    return np.clip(out, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run():
    log("TASK 3 — Manual Panorama Stitching")
    paths = ensure_task3_sequence()
    frames = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in paths]
    grays = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) for f in frames]
    n = len(frames)
    show_row([bgr_to_rgb(f) for f in frames], [f"frame {i+1}" for i in range(n)],
             config.OUT_T3 / "frames.png", suptitle="Task 3 — source frames (left → right pan)")

    # Pairwise L[i]: maps frame(i+1) coords -> frame(i) coords.
    pair_L, stats_rows, first_pair_vis = [], [], None
    for i in range(n - 1):
        pts_a, pts_b, kp_a, kp_b, good = _sift_match(grays[i], grays[i + 1])
        if len(good) < config.T3_MIN_MATCH_COUNT:
            log(f"  [pair {i+1}-{i+2}] EDGE CASE: only {len(good)} matches "
                f"(< {config.T3_MIN_MATCH_COUNT}) — insufficient overlap")
            stats_rows.append([f"{i+1}-{i+2}", len(good), "—", "insufficient overlap"])
            pair_L.append(None)
            continue
        # H maps frame(i+1) -> frame(i):  src = pts in i+1, dst = pts in i
        H, mask, inlier_ratio = ransac_homography(pts_b, pts_a)
        dist = _perspective_distortion(H, frames[i + 1].shape[1], frames[i + 1].shape[0])
        note = "ok" if 0.3 < dist < 3.0 else f"high distortion (area x{dist:.2f})"
        stats_rows.append([f"{i+1}-{i+2}", len(good),
                           f"{100*inlier_ratio:.1f}%  ({int(mask.sum())}/{len(good)})", note])
        log(f"  [pair {i+1}-{i+2}] matches={len(good)}  inlier ratio={100*inlier_ratio:.1f}%  {note}")
        pair_L.append(H)
        if first_pair_vis is None:
            inl = [g for g, mm in zip(good, mask) if mm][:60]
            first_pair_vis = cv2.drawMatches(frames[i], kp_a, frames[i + 1], kp_b, inl, None,
                                             matchColor=(0, 255, 0),
                                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if first_pair_vis is not None:
        save_rgb(config.OUT_T3 / "pair_matches.png", bgr_to_rgb(first_pair_vis))

    if any(L is None for L in pair_L):
        log("  aborting stitch: a pair had insufficient overlap. See task3_stats.png")
        save_table_figure(stats_rows, ["pair", "#matches", "RANSAC inliers", "note"],
                          config.OUT_T3 / "task3_stats.png", title="Task 3 — pairwise stitching stats")
        return {"stats": stats_rows, "panorama": None}

    # Chain homographies to a middle reference frame r.
    r = n // 2
    M = [None] * n
    M[r] = np.eye(3)
    for j in range(r - 1, -1, -1):        # frames left of ref
        M[j] = M[j + 1] @ np.linalg.inv(pair_L[j])
    for j in range(r + 1, n):             # frames right of ref
        M[j] = M[j - 1] @ pair_L[j - 1]

    # Canvas bounds from all warped corners.
    all_c = np.vstack([_apply(M[j], _corners(frames[j].shape[1], frames[j].shape[0]))
                       for j in range(n)])
    xmin, ymin = np.floor(all_c.min(axis=0)).astype(int)
    xmax, ymax = np.ceil(all_c.max(axis=0)).astype(int)
    Toff = np.array([[1, 0, -xmin], [0, 1, -ymin], [0, 0, 1]], np.float64)
    size = (int(xmax - xmin), int(ymax - ymin))

    warped, masks = [], []
    for j in range(n):
        Hf = Toff @ M[j]
        warped.append(cv2.warpPerspective(frames[j], Hf, size))
        m = cv2.warpPerspective(np.full(frames[j].shape[:2], 255, np.uint8), Hf, size)
        masks.append(m > 0)

    weights = _feather_weights(masks)
    try:
        pano = (_multiband_blend(warped, weights) if config.T3_BLEND == "multiband"
                else _alpha_blend(warped, weights))
    except Exception as e:
        log(f"  multi-band blend failed ({e}); using alpha feather blend")
        pano = _alpha_blend(warped, weights)

    # crop away fully-empty borders
    any_mask = np.any(masks, axis=0).astype(np.uint8)
    ys, xs = np.where(any_mask)
    pano = pano[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    save_bgr(config.OUT_T3 / "panorama.png", pano)

    # persist intermediate homography matrices
    with open(config.OUT_T3 / "homographies.txt", "w") as fh:
        fh.write("Intermediate homographies L[i] mapping frame(i+1) -> frame(i)\n")
        for i, H in enumerate(pair_L):
            fh.write(f"\nL[{i}] (frame {i+2} -> frame {i+1}):\n{np.array2string(H, precision=4)}\n")
        fh.write("\nCumulative M[j] mapping frame j -> reference frame "
                 f"{r+1}:\n")
        for j, H in enumerate(M):
            fh.write(f"\nM[{j}] (frame {j+1} -> ref):\n{np.array2string(H, precision=4)}\n")

    save_table_figure(stats_rows, ["pair", "#matches", "RANSAC inliers", "note"],
                      config.OUT_T3 / "task3_stats.png", title="Task 3 — pairwise stitching stats")
    log(f"  panorama {pano.shape[1]}x{pano.shape[0]} written to {config.OUT_T3}")
    mean_inlier = np.mean([float(row[2].split('%')[0]) for row in stats_rows if '%' in str(row[2])])
    return {"stats": stats_rows, "panorama": pano.shape, "mean_inlier_ratio": mean_inlier,
            "blend": config.T3_BLEND, "ref_frame": r + 1}


if __name__ == "__main__":
    from src.utils import ensure_dirs, set_seed

    set_seed()
    ensure_dirs()
    run()
