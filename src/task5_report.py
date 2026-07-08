"""Task 5 — Critical Evaluation & Technical Report  [CLO-5].

Generates BOTH:
  * report/technical_report.md   — full prose source (editable), populated with the
    real metrics produced by tasks 1-4.
  * report/technical_report.pdf  — a structured 4-page PDF (matplotlib PdfPages, an
    allowed library) with: executive summary, pipeline architecture diagram,
    quantitative results, failure analysis (>=2 modes/task with visuals),
    classical-vs-deep-learning trade-offs, and limitations / future work.

The report is data-driven: numbers come from ``outputs/results.json`` (or the results
dict passed to ``generate``), so it always matches the latest run.
"""

import json
import textwrap

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src import config
from src.utils import log

A4 = (8.27, 11.69)


# --------------------------------------------------------------------------- #
# small PDF layout helpers
# --------------------------------------------------------------------------- #
def _page():
    fig = plt.figure(figsize=A4)
    return fig


def _text(fig, y, s, size=10, weight="normal", color="#111111", x=0.07, wrap=110, gap=0.006):
    if wrap:
        s = "\n".join(textwrap.fill(ln, wrap) if ln else "" for ln in s.split("\n"))
    fig.text(x, y, s, fontsize=size, fontweight=weight, color=color, va="top", ha="left")
    n = s.count("\n") + 1
    return y - (n * size * 1.7 / 792) - gap


def _rule(fig, y, x0=0.07, x1=0.93, color="#2c3e6b"):
    fig.add_artist(plt.Line2D([x0, x1], [y, y], color=color, lw=1.3,
                              transform=fig.transFigure))
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


# --------------------------------------------------------------------------- #
# page 1 — title, exec summary, architecture diagram
# --------------------------------------------------------------------------- #
def _architecture_axes(fig, rect):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    boxes = [
        (0.4, "Input frames\n(real: OpenCV /\nIntel imagery)", "#dfe7f3"),
        (2.4, "T1: Gaussian ->\nCanny -> Hough\nlines", "#cfe0d8"),
        (4.4, "T2: SIFT\nkeypoints +\nratio matching", "#cfe0d8"),
        (6.4, "T3: RANSAC\nhomography ->\nwarp + blend", "#cfe0d8"),
        (8.4, "T4: ResNet-18\ntransfer ->\nscene class", "#f3e0cf"),
    ]
    for x, label, color in boxes:
        ax.add_patch(mpatches.FancyBboxPatch((x, 2.4), 1.5, 1.4,
                     boxstyle="round,pad=0.08", fc=color, ec="#2c3e6b", lw=1.2))
        ax.text(x + 0.75, 3.1, label, ha="center", va="center", fontsize=7.2)
    for x in (2.0, 4.0, 6.0, 8.0):
        ax.annotate("", xy=(x + 0.35, 3.1), xytext=(x, 3.1),
                    arrowprops=dict(arrowstyle="->", color="#2c3e6b", lw=1.3))
    ax.add_patch(mpatches.FancyBboxPatch((3.4, 0.3), 3.2, 1.1,
                 boxstyle="round,pad=0.08", fc="#eee", ec="#c0392b", lw=1.2))
    ax.text(5.0, 0.85, "T5: Critical evaluation\n(metrics · failures · trade-offs)",
            ha="center", va="center", fontsize=7.2)
    ax.annotate("", xy=(5.0, 1.4), xytext=(5.0, 2.4),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.3))


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
    t4 = R.get("task4", {})
    acc = t4.get("best_acc", 0.0)
    summary = (
        "This project implements an end-to-end visual pipeline for an outdoor "
        "navigation robot. Four perception stages are integrated and then critically "
        "evaluated. Task 1 extracts structural edges and lines (Gaussian smoothing -> "
        "Canny -> probabilistic Hough). Task 2 establishes cross-view correspondence "
        "with SIFT features and Lowe's ratio test. Task 3 builds a seamless panorama "
        "from overlapping frames using a from-scratch RANSAC homography estimator and "
        "multi-band blending (no cv2.Stitcher). Task 4 fine-tunes a pretrained "
        "ResNet-18 to classify outdoor scene categories, reaching a best validation "
        f"accuracy of {acc:.1%} (target 70%). The system therefore combines three "
        "classical geometric stages with one learned semantic stage, and the report "
        "quantifies the accuracy, robustness and trade-offs of each."
    )
    y = _text(fig, y, summary, size=9.5)

    y = _text(fig, y - 0.005, "2. Pipeline Architecture", size=12, weight="bold", color="#2c3e6b")
    _architecture_axes(fig, [0.06, 0.40, 0.88, 0.20])
    fig.text(0.07, 0.385, "Figure 1 — Data flow across the five tasks. Geometry stages "
             "(green) feed the learned stage (orange); Task 5 evaluates all of them.",
             fontsize=8, color="#333")

    # headline numbers strip
    t2 = R.get("task2", {})
    t3 = R.get("task3", {})
    strip = (
        f"Headline metrics:   "
        f"T2 keypoints A/B = {t2.get('kp_a','-')}/{t2.get('kp_b','-')}, "
        f"good matches = {t2.get('good','-')}   |   "
        f"T3 mean RANSAC inlier ratio = {t3.get('mean_inlier_ratio', float('nan')):.1f}%   |   "
        f"T4 best val acc = {acc:.1%}"
    )
    _text(fig, 0.35, strip, size=9, weight="bold", color="#2c3e6b")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# page 2 — quantitative results
# --------------------------------------------------------------------------- #
def _page_results(pdf, R):
    fig = _page()
    y = _text(fig, 0.955, "3. Quantitative Results", size=13, weight="bold", color="#2c3e6b")
    y = _rule(fig, y)
    _text(fig, y, "Metrics below are produced directly by the pipeline (outputs/results.json). "
          "Each task writes its own tables/plots under outputs/taskN/.", size=9)

    _image(fig, [0.07, 0.70, 0.40, 0.17], config.OUT_T1 / "task1_param_experiments_hough.png",
           "T1 — Hough parameter experiments")
    _image(fig, [0.53, 0.70, 0.40, 0.17], config.OUT_T2 / "task2_stats.png",
           "T2 — SIFT matching statistics")
    _image(fig, [0.07, 0.48, 0.40, 0.18], config.OUT_T3 / "task3_stats.png",
           "T3 — pairwise RANSAC inlier ratios")
    _image(fig, [0.53, 0.48, 0.40, 0.18], config.OUT_T4 / "task4_summary.png",
           "T4 — classifier summary")
    _image(fig, [0.09, 0.24, 0.40, 0.20], config.OUT_T4 / "training_curves.png",
           "T4 — training / validation curves")
    _image(fig, [0.54, 0.24, 0.38, 0.20], config.OUT_T4 / "confusion_matrix.png",
           "T4 — validation confusion matrix")
    _image(fig, [0.30, 0.03, 0.40, 0.18], config.OUT_T4 / "per_class_metrics.png",
           "T4 — per-class precision / recall / F1")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# page 3 — failure analysis
# --------------------------------------------------------------------------- #
def _page_failures(pdf, R):
    fig = _page()
    y = _text(fig, 0.955, "4. Failure Analysis (>= 2 modes per task)", size=13,
              weight="bold", color="#2c3e6b")
    y = _rule(fig, y)
    text = (
        "Task 1 — Edge & line detection.\n"
        "  (a) Loose Canny thresholds (30/90) flood the edge map with foliage/texture "
        "clutter, so Hough returns many spurious short segments; strict thresholds "
        "(100/200) instead drop faint but real lane/structure edges. (b) Probabilistic "
        "Hough with a large maxLineGap merges collinear-but-separate structures into one "
        "line. Mitigation: the 1:3 'balanced' threshold and a moderate gap.\n"
        "Task 2 — SIFT matching.\n"
        "  (a) Repetitive texture (identical windows/patterns) produces ambiguous "
        "descriptors and geometrically inconsistent matches. (b) A high ratio (0.9) "
        "admits many false positives (precision drops); a low ratio (0.5) starves recall. "
        "The ratio sweep quantifies this precision/recall trade-off.\n"
        "Task 3 — Panorama.\n"
        "  (a) Insufficient overlap (< min matches) makes the homography rank-deficient — "
        "detected and reported as an edge case rather than producing a broken warp. "
        "(b) Accumulated homography error across many frames bends straight lines near the "
        "panorama edges; anchoring at the middle frame limits this drift.\n"
        "Task 4 — Scene classification.\n"
        "  (a) Visually similar classes (e.g. mountain vs. buildings on the skyline) are "
        "confused — see the off-diagonal cells of the confusion matrix. (b) The Grad-CAM "
        "'incorrect' example shows the network attending to background context rather than "
        "the discriminative object, explaining the misclassification."
    )
    y = _text(fig, y, text, size=8.8, wrap=118)
    _image(fig, [0.07, 0.20, 0.40, 0.16], config.OUT_T1 / "building_canny_sweep.png",
           "T1 failure modes — Canny threshold sweep")
    _image(fig, [0.53, 0.20, 0.40, 0.16], config.OUT_T2 / "ratio_sweep.png",
           "T2 — precision/recall vs. ratio")
    _image(fig, [0.20, 0.02, 0.60, 0.16], config.OUT_T4 / "gradcam.png",
           "T4 — Grad-CAM (2 correct + 1 incorrect)")
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# page 4 — trade-offs + limitations
# --------------------------------------------------------------------------- #
def _page_tradeoffs(pdf, R):
    fig = _page()
    y = _text(fig, 0.955, "5. Classical vs. Deep-Learning Trade-offs", size=13,
              weight="bold", color="#2c3e6b")
    y = _rule(fig, y)

    ax = fig.add_axes([0.07, 0.60, 0.86, 0.28])
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
    table.scale(1, 1.6)
    for j in range(len(cols)):
        c = table[0, j]
        c.set_facecolor("#2c3e6b")
        c.set_text_props(color="white", fontweight="bold")

    y = _text(fig, 0.57, "6. Limitations & Future Work", size=13, weight="bold", color="#2c3e6b")
    y = _rule(fig, y)
    lim = (
        "- Panorama frames are drawn from a wide real scene by overlapping crops; genuine "
        "multi-camera capture with exposure differences would stress blending further.\n"
        "- The classifier is fine-tuned on four Intel scene categories; scaling to the full "
        "MIT Places365 taxonomy would test generalisation and long-tail robustness.\n"
        "- Homographies assume a largely planar / rotational scene; strong parallax from a "
        "moving ground robot would need bundle adjustment or cylindrical warping.\n"
        "- SIFT + FLANN are CPU-bound; a learned detector (SuperPoint) or GPU matcher would "
        "raise throughput for real-time navigation.\n"
        "- Future work: fuse the geometric map (T1-T3) with the semantic labels (T4) into a "
        "single scene graph, and add temporal consistency across video frames.\n\n"
        "Reproducibility: fixed global seed; every figure regenerated by 'python run_all.py'. "
        "Data and code are cited in README (OpenCV samples, BSD-3; Intel Image Classification "
        "via Hugging Face)."
    )
    _text(fig, y, lim, size=9, wrap=112)
    pdf.savefig(fig)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# markdown source
# --------------------------------------------------------------------------- #
def _write_markdown(R):
    t2, t3, t4 = R.get("task2", {}), R.get("task3", {}), R.get("task4", {})
    acc = t4.get("best_acc", 0.0)
    classes = t4.get("classes", config.T4_CLASSES)
    per_class = t4.get("per_class", [])
    pc_lines = "\n".join(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} |" for r in per_class)
    md = f"""# CS-456 CCP — Technical Report
## Autonomous Visual Intelligence System for Scene Understanding
*CLOs 3, 4, 5 — classical + deep-learning pipeline*

## 1. Executive Summary
An end-to-end perception pipeline for an outdoor navigation robot: structural edge/line
detection (T1), cross-view SIFT correspondence (T2), a manually-implemented RANSAC +
multi-band panorama (T3), and a fine-tuned ResNet-18 scene classifier (T4) reaching a
best validation accuracy of **{acc:.1%}** (target 70%). Task 5 evaluates all stages.

## 2. Pipeline Architecture
Input frames → **T1** Gaussian→Canny→Hough → **T2** SIFT keypoints + ratio matching →
**T3** RANSAC homography → warp → multi-band blend → **T4** ResNet-18 transfer learning
→ scene label. See `report/technical_report.pdf` Figure 1 for the block diagram.

## 3. Quantitative Results
| Task | Key metric | Value |
|---|---|---|
| T1 | Canny/Hough parameter sweeps | see `outputs/task1/task1_param_experiments_*.png` |
| T2 | keypoints (A/B), good matches | {t2.get('kp_a','-')}/{t2.get('kp_b','-')}, {t2.get('good','-')} (ratio {t2.get('match_ratio',0):.3f}) |
| T3 | mean RANSAC inlier ratio | {t3.get('mean_inlier_ratio', float('nan')):.1f}% ({t3.get('blend','-')} blend, ref frame {t3.get('ref_frame','-')}) |
| T4 | best / final val accuracy | {acc:.1%} / {t4.get('final_acc',0):.1%} |

### T4 per-class metrics
| class | precision | recall | F1 | support |
|---|---|---|---|---|
{pc_lines}

## 4. Failure Analysis (>= 2 modes per task)
**T1:** loose Canny → clutter & spurious Hough lines; large maxLineGap → merged lines.
**T2:** repetitive texture → ambiguous matches; ratio 0.9 → false positives, 0.5 → low recall.
**T3:** insufficient overlap → rank-deficient homography (handled as edge case); multi-frame
homography drift bends edges (mitigated by middle-frame anchoring).
**T4:** visually-similar classes confused (see confusion matrix off-diagonals); Grad-CAM
'incorrect' sample shows attention on background context. Visuals: `outputs/task4/gradcam.png`,
`outputs/task4/confusion_matrix.png`.

## 5. Classical vs. Deep-Learning Trade-offs
| Dimension | Classical (T1–T3) | Deep learning (T4) |
|---|---|---|
| Speed | real-time on CPU | GPU/MPS to train, fast inference |
| Accuracy | exact when geometry holds | strong semantics, off-distribution drop |
| Interpretability | transparent maths | opaque; Grad-CAM needed |
| Data | none (tuned params) | >=100 images/class |
| Failure | low texture / overlap | similar-class confusion |

## 6. Limitations & Future Work
Overlapping-crop panorama vs. true multi-camera capture; scale T4 to full Places365;
parallax needs bundle adjustment / cylindrical warping; SIFT+FLANN are CPU-bound (SuperPoint
/ GPU matching would help); fuse geometry + semantics into a temporal scene graph.

---
*Reproducible via `python run_all.py` (fixed seed). Data and code cited in README.*
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
