"""Task 5 — Critical Evaluation & Technical Report  [CLO-5].

Generates BOTH:
  * report/technical_report.md   — full prose source (editable), populated with the
    real metrics produced by tasks 1-4.
  * report/technical_report.pdf  — a structured 4-page PDF (matplotlib PdfPages, an
    allowed library) with: executive summary, pipeline architecture diagram,
    quantitative results (a metric per task, T1 included), failure analysis (>=2 modes
    per task with a visual per task), classical-vs-deep-learning trade-offs, limitations,
    and image citations.

The report is data-driven: numbers come from ``outputs/results.json`` (or the results
dict passed to ``generate``), so it always matches the latest run. Every input image is a
real, cited public dataset — the report says so explicitly (no integration is overstated:
each task is validated on its own representative dataset, not a single shared feed).
"""

import json
import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src import config
from src.utils import log

A4 = (8.27, 11.69)

# In-report image citations (kept in one place; also printed in the PDF + markdown).
CITATIONS = [
    "T1 roads — Udacity CarND-LaneLines-P1 test images (MIT License).",
    "T2 pair — Oxford VGG affine 'graffiti' wall, views 1 & 3 (free for research; via opencv_extra).",
    "T3 frames — OpenCV stitching sample 'boat1-3', a St. Petersburg waterfront panorama (opencv_extra, BSD-3).",
    "T4 road/building/vegetation — Intel Image Classification photos (Kaggle).",
    "T4 pedestrian_zone — Places365 'crosswalk' (Zhou et al., IEEE TPAMI 2017).",
    "No images are self-captured; all inputs are real, publicly-licensed datasets.",
]


# --------------------------------------------------------------------------- #
# small PDF layout helpers
# --------------------------------------------------------------------------- #
def _page():
    return plt.figure(figsize=A4)


def _text(fig, y, s, size=10, weight="normal", color="#111111", x=0.07, wrap=110, gap=0.006):
    if wrap:
        s = "\n".join(textwrap.fill(ln, wrap) if ln else "" for ln in s.split("\n"))
    fig.text(x, y, s, fontsize=size, fontweight=weight, color=color, va="top", ha="left")
    n = s.count("\n") + 1
    return y - (n * size * 1.7 / 792) - gap


def _rule(fig, y, x0=0.07, x1=0.93, color="#2c3e6b"):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, lw=1.3, transform=fig.transFigure))
    return y - 0.012


def _image(fig, rect, path, caption=None):
    try:
        img = plt.imread(str(path))
    except FileNotFoundError:
        return
    ax = fig.add_axes(rect)
    ax.imshow(img)
    ax.axis("off")
    if caption:
        ax.set_title(caption, fontsize=8, color="#333333")


def _t1_metric(R):
    """(balanced edge-density %, primary Hough line count, primary cfg name) or dashes."""
    eq = R.get("task1", {}).get("edge_quality", {}) or {}
    return (eq.get("canny_balanced_edge_density_pct", "-"),
            eq.get("hough_primary_lines_total", "-"),
            eq.get("hough_primary_cfg", "fine"))


# --------------------------------------------------------------------------- #
# page 1 — title, exec summary, architecture diagram
# --------------------------------------------------------------------------- #
def _architecture_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.text(5.0, 5.6, "Autonomous navigation robot — perception stack",
            ha="center", va="center", fontsize=8.5, fontweight="bold", color="#2c3e6b")
    boxes = [
        (0.3, "T1: edge / line\nGaussian->Canny\n->Hough", "#cfe0d8", "Udacity\nroad imgs"),
        (2.75, "T2: correspondence\nSIFT + Lowe\nratio match", "#cfe0d8", "VGG graffiti\n(2 views)"),
        (5.2, "T3: mapping\nRANSAC H ->\nwarp + blend", "#cfe0d8", "OpenCV boat1-3\nwaterfront pan"),
        (7.65, "T4: semantics\nResNet-18 head\n-> scene class", "#f3e0cf", "Intel + Places365\n(4 classes)"),
    ]
    for x, label, color, ds in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, 2.7), 2.0, 1.5,
                     boxstyle="round,pad=0.08", fc=color, ec="#2c3e6b", lw=1.2))
        ax.text(x + 1.0, 3.45, label, ha="center", va="center", fontsize=7.0)
        ax.text(x + 1.0, 2.15, ds, ha="center", va="center", fontsize=6.2, color="#555", style="italic")
    for x in (2.35, 4.8, 7.25):
        ax.annotate("", xy=(x + 0.35, 3.45), xytext=(x, 3.45),
                    arrowprops=dict(arrowstyle="->", color="#2c3e6b", lw=1.1))
    ax.add_patch(mpatches.FancyBboxPatch((3.3, 0.35), 3.4, 1.0,
                 boxstyle="round,pad=0.08", fc="#eee", ec="#c0392b", lw=1.2))
    ax.text(5.0, 0.85, "T5: critical evaluation\n(metrics · failures · trade-offs)",
            ha="center", va="center", fontsize=7.0)
    for x in (1.3, 3.75, 6.2, 8.65):
        ax.annotate("", xy=(5.0, 1.35), xytext=(x, 2.7),
                    arrowprops=dict(arrowstyle="->", color="#c0392b", lw=0.7, alpha=0.5))


def _page_overview(pdf, R):
    fig = _page()
    y = 0.96
    fig.text(0.07, y, "CS-456 Computer Vision — Complex Computing Problem",
             fontsize=14, fontweight="bold", color="#2c3e6b")
    y = _text(fig, 0.935, "Autonomous Visual Intelligence System for Scene Understanding",
              size=11, weight="bold")
    y = _text(fig, y, "Technical Report  |  CLOs 3, 4, 5  |  classical + deep-learning pipeline",
              size=9, color="#555")
    y = _rule(fig, y)

    y = _text(fig, y, "1. Executive Summary", size=12, weight="bold", color="#2c3e6b")
    acc = R.get("task4", {}).get("best_acc", 0.0)
    summary = (
        "This project implements the four core perception capabilities an outdoor navigation "
        "robot needs, each built and independently validated on a representative real, publicly "
        "licensed dataset (not a single shared feed). Task 1 detects road edges and lane lines "
        "(Gaussian -> Canny -> probabilistic Hough) on real dashcam road images. Task 2 establishes "
        "cross-view correspondence (SIFT + Lowe's ratio test) between two real views of the same "
        "outdoor landmark. Task 3 builds a real panorama from overlapping building frames using a "
        "from-scratch RANSAC homography estimator and multi-band blending on real overlapping "
        "outdoor waterfront frames (no cv2.Stitcher). Task 4 "
        "fine-tunes the classification head of a frozen ResNet-18 to label four outdoor scene "
        f"categories (road, building, vegetation, pedestrian zone), reaching {acc:.1%} validation "
        "accuracy (target 70%). Task 5 quantifies and critiques the accuracy, robustness and "
        "trade-offs of each stage."
    )
    y = _text(fig, y, summary, size=9.5)

    y = _text(fig, y - 0.005, "2. Pipeline Architecture", size=12, weight="bold", color="#2c3e6b")
    _architecture_axes(fig, [0.06, 0.40, 0.88, 0.22])
    cap = ("Figure 1 — The four perception capabilities of the robot's vision stack. Each is "
           "implemented and validated independently on the real dataset named beneath it (italic); "
           "Task 5 evaluates all four. They are not run on one shared frame.")
    fig.text(0.07, 0.375, textwrap.fill(cap, 108), fontsize=8, color="#333", va="top")

    d, lines, cfg = _t1_metric(R)
    t2, t3 = R.get("task2", {}), R.get("task3", {})
    strip = (
        f"Headline metrics:   T1 edge density (balanced) = {d}%, {lines} {cfg}-Hough lines   |   "
        f"T2 keypoints A/B = {t2.get('kp_a','-')}/{t2.get('kp_b','-')}, good = {t2.get('good','-')}   |   "
        f"T3 mean RANSAC inlier ratio = {t3.get('mean_inlier_ratio', float('nan')):.1f}%   |   "
        f"T4 best val acc = {acc:.1%}"
    )
    _text(fig, 0.30, strip, size=8.6, weight="bold", color="#2c3e6b", wrap=104)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# page 2 — quantitative results
# --------------------------------------------------------------------------- #
def _page_results(pdf, R):
    fig = _page()
    y = _text(fig, 0.955, "3. Quantitative Results", size=13, weight="bold", color="#2c3e6b")
    y = _rule(fig, y)
    d, lines, cfg = _t1_metric(R)
    t2, t3, t4 = R.get("task2", {}), R.get("task3", {}), R.get("task4", {})
    metrics = (
        f"Per-task metrics (from outputs/results.json):\n"
        f"  T1  edge-detection quality:  balanced Canny edge density = {d}% of pixels;  "
        f"{lines} lines at the '{cfg}' Hough setting.\n"
        f"  T2  match ratio:  {t2.get('good','-')} good matches / {t2.get('kp_a','-')} vs "
        f"{t2.get('kp_b','-')} keypoints  (ratio {t2.get('match_ratio',0):.3f}).\n"
        f"  T3  RANSAC inlier ratio:  mean {t3.get('mean_inlier_ratio', float('nan')):.1f}%  "
        f"({t3.get('blend','-')} blend, ref frame {t3.get('ref_frame','-')}).\n"
        f"  T4  CNN accuracy:  best {t4.get('best_acc',0):.1%} / final {t4.get('final_acc',0):.1%} "
        f"validation."
    )
    _text(fig, y, metrics, size=9, wrap=120)

    _image(fig, [0.07, 0.66, 0.40, 0.16], config.OUT_T1 / "task1_param_experiments_canny.png",
           "T1 — Canny experiments (with qualitative outcomes)")
    _image(fig, [0.53, 0.66, 0.40, 0.16], config.OUT_T2 / "task2_stats.png",
           "T2 — SIFT matching statistics")
    _image(fig, [0.07, 0.46, 0.40, 0.16], config.OUT_T3 / "task3_stats.png",
           "T3 — pairwise RANSAC inlier ratios")
    _image(fig, [0.53, 0.46, 0.40, 0.16], config.OUT_T4 / "task4_summary.png",
           "T4 — classifier summary")
    _image(fig, [0.08, 0.24, 0.40, 0.18], config.OUT_T4 / "training_curves.png",
           "T4 — training / validation curves")
    _image(fig, [0.54, 0.24, 0.38, 0.18], config.OUT_T4 / "confusion_matrix.png",
           "T4 — validation confusion matrix")
    _image(fig, [0.30, 0.03, 0.40, 0.17], config.OUT_T4 / "per_class_metrics.png",
           "T4 — per-class precision / recall / F1")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# page 3 — failure analysis
# --------------------------------------------------------------------------- #
def _page_failures(pdf, R):
    fig = _page()
    y = _text(fig, 0.955, "4. Failure Analysis (>= 2 modes per task, with visuals)", size=13,
              weight="bold", color="#2c3e6b")
    y = _rule(fig, y)

    # pull real numbers to make the analysis quantitative
    t1 = R.get("task1", {})
    can = t1.get("canny", [])
    dens = [float(r[3].rstrip("%")) for r in can if len(r) > 3 and str(r[3]).endswith("%")]
    dmin, dmax = (min(dens), max(dens)) if dens else (0, 0)
    sweep = dict((round(a, 2), b) for a, b in R.get("task2", {}).get("ratio_sweep", []))
    g05, g09 = sweep.get(0.5, "-"), sweep.get(0.9, "-")
    t3 = R.get("task3", {})
    inl = t3.get("mean_inlier_ratio", float("nan"))
    conf = R.get("task4", {}).get("confusion", [])
    classes = R.get("task4", {}).get("classes", config.T4_CLASSES)
    # largest off-diagonal confusion
    worst = ""
    if conf:
        best = (0, 0, 0)
        for i, row in enumerate(conf):
            for j, v in enumerate(row):
                if i != j and v > best[2]:
                    best = (i, j, v)
        if best[2] > 0:
            worst = f" (most confused: {classes[best[0]]}->{classes[best[1]]}, {best[2]} imgs)"

    text = (
        "Task 1 - Edge & line detection.\n"
        f"  (a) Loose Canny thresholds (30/90) flood the map with road-texture/foliage clutter: edge "
        f"density ranges ~{dmin:.1f}-{dmax:.1f}% across configs, so Hough returns many spurious short "
        "segments. (b) Strict thresholds (100/200) instead drop faint far-field lane markings. "
        "Mitigation: the 1:3 'balanced' threshold + a moderate maxLineGap.\n"
        "Task 2 - SIFT matching.\n"
        f"  (a) A high ratio (0.9) admits many false positives ({g09} matches) while a low ratio (0.5) "
        f"starves recall ({g05} matches) - the sweep quantifies the precision/recall trade-off. "
        "(b) Repetitive wall texture yields ambiguous descriptors and geometrically inconsistent matches.\n"
        "Task 3 - Panorama.\n"
        f"  (a) Insufficient overlap (< min matches) makes the homography rank-deficient - detected and "
        f"reported as an edge case. (b) Accumulated homography error across frames bends lines near the "
        f"panorama edges (mean inlier ratio {inl:.1f}%); middle-frame anchoring limits this drift.\n"
        "Task 4 - Scene classification.\n"
        f"  (a) Visually similar classes are confused{worst} - see the confusion-matrix off-diagonals. "
        "(b) The Grad-CAM 'incorrect' example shows the network attending to background context rather "
        "than the discriminative structure, explaining the misclassification."
    )
    y = _text(fig, y, text, size=8.6, wrap=118)

    # one visual example PER task (2x2 grid)
    _image(fig, [0.06, 0.20, 0.42, 0.15], config.OUT_T1 / "straight_road_canny_sweep.png",
           "T1 - Canny threshold sweep")
    _image(fig, [0.53, 0.20, 0.42, 0.15], config.OUT_T2 / "ratio_sweep.png",
           "T2 - matches vs Lowe ratio")
    _image(fig, [0.06, 0.02, 0.42, 0.15], config.OUT_T3 / "pair_matches.png",
           "T3 - RANSAC inlier matches (pair 1)")
    _image(fig, [0.53, 0.02, 0.42, 0.15], config.OUT_T4 / "gradcam.png",
           "T4 - Grad-CAM (2 correct + 1 incorrect)")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# page 4 — trade-offs + limitations + citations
# --------------------------------------------------------------------------- #
def _page_tradeoffs(pdf, R):
    fig = _page()
    y = _text(fig, 0.955, "5. Classical vs. Deep-Learning Trade-offs", size=13,
              weight="bold", color="#2c3e6b")
    y = _rule(fig, y)

    ax = fig.add_axes([0.07, 0.62, 0.86, 0.26])
    ax.axis("off")
    cols = ["Dimension", "Classical (T1-T3)", "Deep learning (T4)"]
    rows = [
        ["Speed", "Real-time on CPU", "Needs GPU/MPS to train; fast at inference"],
        ["Accuracy", "Exact when geometry holds", "High on semantics, degrades off-distribution"],
        ["Interpretability", "Fully transparent maths", "Opaque; needs Grad-CAM to explain"],
        ["Data requirement", "None (hand-tuned params)", ">=100 labelled images/class"],
        ["Failure mode", "Breaks on low texture/overlap", "Confuses visually similar classes"],
    ]
    table = ax.table(cellText=rows, colLabels=cols, cellLoc="left", loc="center",
                     colWidths=[0.20, 0.38, 0.42])
    table.auto_set_font_size(False)
    table.set_fontsize(8.6)
    table.scale(1, 1.55)
    for j in range(len(cols)):
        c = table[0, j]
        c.set_facecolor("#2c3e6b")
        c.set_text_props(color="white", fontweight="bold")

    y = _text(fig, 0.585, "6. Limitations & Future Work", size=13, weight="bold", color="#2c3e6b")
    y = _rule(fig, y)
    lim = (
        "- Each capability is validated on its own public dataset; a true robot would run all four on one "
        "live camera stream, adding motion blur, exposure change and temporal consistency needs.\n"
        "- Task 4 is a frozen-backbone linear probe on four classes; unfreezing deeper blocks and scaling "
        "to the full Places365 taxonomy would test generalisation and long-tail robustness.\n"
        "- Homographies assume a largely planar / rotational scene; strong parallax from a moving ground "
        "robot would need bundle adjustment or cylindrical warping.\n"
        "- SIFT + FLANN are CPU-bound; a learned detector (SuperPoint) or GPU matcher would raise throughput.\n"
        "- Future work: fuse the geometric map (T1-T3) with the semantic labels (T4) into a temporal scene graph."
    )
    y = _text(fig, y, lim, size=9, wrap=112)

    y = _text(fig, y - 0.006, "7. Data Sources & Citations", size=13, weight="bold", color="#2c3e6b")
    y = _rule(fig, y)
    y = _text(fig, y, "\n".join(f"- {c}" for c in CITATIONS), size=8.6, wrap=118)
    _text(fig, y - 0.004, "Reproducibility: fixed global seed; every figure regenerated by "
          "'python run_all.py'.", size=8.6, color="#555")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# markdown source
# --------------------------------------------------------------------------- #
def _write_markdown(R):
    t2, t3, t4 = R.get("task2", {}), R.get("task3", {}), R.get("task4", {})
    acc = t4.get("best_acc", 0.0)
    per_class = t4.get("per_class", [])
    pc_lines = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in per_class)
    d, lines, cfg = _t1_metric(R)
    cites = "\n".join(f"- {c}" for c in CITATIONS)
    md = f"""# CS-456 CCP — Technical Report
## Autonomous Visual Intelligence System for Scene Understanding
*CLOs 3, 4, 5 — classical + deep-learning pipeline*

## 1. Executive Summary
The four core perception capabilities of an outdoor navigation robot, each built and
**independently validated on a representative real, publicly-licensed dataset** (not one
shared feed): road edge/lane detection (T1, real dashcam roads), cross-view SIFT
correspondence between two real views of one outdoor landmark (T2), a from-scratch
RANSAC + multi-band **panorama** from real overlapping outdoor waterfront frames (T3), and a frozen
**ResNet-18** whose classification head is fine-tuned to label four outdoor scenes
(road, building, vegetation, pedestrian zone) at a best validation accuracy of **{acc:.1%}**
(target 70%). Task 5 evaluates all stages. No `cv2.Stitcher`; no synthetic or self-captured images.

## 2. Pipeline Architecture
Perception stack (see `report/technical_report.pdf` Figure 1): **T1** Gaussian→Canny→Hough ·
**T2** SIFT keypoints + Lowe-ratio matching · **T3** RANSAC homography → warp → multi-band
blend · **T4** frozen ResNet-18 + trained head → scene label. Each stage is validated on its
own real dataset (named under its box in Figure 1), not on a single shared frame.

## 3. Quantitative Results
| Task | Key metric | Value |
|---|---|---|
| T1 | edge-detection quality | balanced Canny edge density {d}% of pixels; {lines} lines at '{cfg}' Hough |
| T2 | keypoints (A/B), good matches | {t2.get('kp_a','-')}/{t2.get('kp_b','-')}, {t2.get('good','-')} (ratio {t2.get('match_ratio',0):.3f}) |
| T3 | mean RANSAC inlier ratio | {t3.get('mean_inlier_ratio', float('nan')):.1f}% ({t3.get('blend','-')} blend, ref frame {t3.get('ref_frame','-')}) |
| T4 | best / final val accuracy | {acc:.1%} / {t4.get('final_acc',0):.1%} |

### T4 per-class metrics
| class | precision | recall | F1 | support |
|---|---|---|---|---|
{pc_lines}

## 4. Failure Analysis (>= 2 modes per task, each with a visual)
**T1:** loose Canny → clutter & spurious Hough lines; strict Canny → faint lanes dropped. *(see `outputs/task1/straight_road_canny_sweep.png`)*
**T2:** ratio 0.9 → false positives, 0.5 → low recall; repetitive wall texture → ambiguous matches. *(see `outputs/task2/ratio_sweep.png`)*
**T3:** insufficient overlap → rank-deficient homography (edge case); multi-frame drift bends edges (mitigated by middle-frame anchoring). *(see `outputs/task3/pair_matches.png`)*
**T4:** visually-similar classes confused (confusion off-diagonals); Grad-CAM 'incorrect' attends to background. *(see `outputs/task4/gradcam.png`, `confusion_matrix.png`)*

## 5. Classical vs. Deep-Learning Trade-offs
| Dimension | Classical (T1–T3) | Deep learning (T4) |
|---|---|---|
| Speed | real-time on CPU | GPU/MPS to train, fast inference |
| Accuracy | exact when geometry holds | strong semantics, off-distribution drop |
| Interpretability | transparent maths | opaque; Grad-CAM needed |
| Data | none (tuned params) | >=100 images/class |
| Failure | low texture / overlap | similar-class confusion |

## 6. Limitations & Future Work
Per-capability datasets vs. one live robot feed (motion blur, exposure, temporal consistency);
frozen-backbone linear probe vs. deeper fine-tuning / full Places365; planar-homography assumption
vs. parallax (needs bundle adjustment / cylindrical warping); SIFT+FLANN are CPU-bound (SuperPoint /
GPU matching would help); fuse geometry + semantics into a temporal scene graph.

## 7. Data Sources & Citations
{cites}

---
*Reproducible via `python run_all.py` (fixed seed).*
"""
    (config.REPORT_DIR / "technical_report.md").write_text(md)


def generate(results=None):
    log("TASK 5 — generating technical report")
    if results is None:
        with open(config.OUTPUTS_DIR / "results.json") as f:
            results = json.load(f)
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    _write_markdown(results)
    pdf_path = config.REPORT_DIR / "technical_report.pdf"
    with PdfPages(str(pdf_path)) as pdf:
        _page_overview(pdf, results)
        _page_results(pdf, results)
        _page_failures(pdf, results)
        _page_tradeoffs(pdf, results)
    log(f"  wrote {pdf_path} and technical_report.md")


if __name__ == "__main__":
    generate()
