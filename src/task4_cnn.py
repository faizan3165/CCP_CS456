"""Task 4 — CNN-Based Scene Classification (PyTorch transfer learning)  [CLO-4].

  * Backbone: pretrained ResNet-18 (torchvision), fully frozen. Only the new final
    classification head is fine-tuned on our 4 outdoor scene classes (a linear probe),
    matching the brief's "fine-tune the final classification layers".
  * Data augmentation: random horizontal flip, rotation, colour jitter.
  * Reports: training/validation curves, confusion matrix, per-class precision/recall.
  * Interpretability: Grad-CAM heatmaps for 2 correctly and 1 incorrectly classified
    validation samples.

Outputs (outputs/task4/):
  * training_curves.png     loss + accuracy vs. epoch
  * confusion_matrix.png    validation confusion matrix
  * per_class_metrics.png   precision / recall / F1 per class
  * gradcam.png             Grad-CAM overlays (2 correct + 1 incorrect)
  * task4_summary.png       headline metrics table
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.models import ResNet18_Weights, resnet18

from src import config
from src.data_setup import ensure_task4_dataset
from src.utils import log, save_table_figure


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _loaders():
    mean, std = config.T4_MEAN, config.T4_STD
    train_tf = transforms.Compose([
        transforms.Resize((config.T4_IMG_SIZE, config.T4_IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((config.T4_IMG_SIZE, config.T4_IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    train_dir, val_dir, _ = ensure_task4_dataset()
    train_ds = ImageFolder(str(train_dir), transform=train_tf)
    val_ds = ImageFolder(str(val_dir), transform=eval_tf)
    train_ld = DataLoader(train_ds, batch_size=config.T4_BATCH_SIZE, shuffle=True, num_workers=0)
    val_ld = DataLoader(val_ds, batch_size=config.T4_BATCH_SIZE, shuffle=False, num_workers=0)
    return train_ld, val_ld, train_ds.classes


def _build_model(num_classes):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    for p in model.parameters():          # freeze the ENTIRE pretrained backbone
        p.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # only this new head trains
    return model


def _run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


# --------------------------------------------------------------------------- #
# Grad-CAM
# --------------------------------------------------------------------------- #
class GradCAM:
    """Grad-CAM on a target conv layer (last block of ResNet-18)."""

    def __init__(self, model, target_layer):
        self.model = model
        self.acts = None
        self.grads = None
        target_layer.register_forward_hook(self._fwd)
        target_layer.register_full_backward_hook(self._bwd)

    def _fwd(self, _m, _i, out):
        self.acts = out.detach()

    def _bwd(self, _m, _gi, gout):
        self.grads = gout[0].detach()

    def __call__(self, x, class_idx):
        # Backbone is frozen, so make the INPUT require grad; otherwise the target
        # conv activations are not in the autograd graph and Grad-CAM would be empty.
        x = x.clone().detach().requires_grad_(True)
        self.model.zero_grad()
        out = self.model(x)
        out[0, class_idx].backward()
        weights = self.grads.mean(dim=(2, 3), keepdim=True)      # GAP over spatial
        cam = torch.relu((weights * self.acts).sum(dim=1)).squeeze(0)
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        return cam.cpu().numpy()


def _denorm(x):
    mean = np.array(config.T4_MEAN)
    std = np.array(config.T4_STD)
    img = x.cpu().numpy().transpose(1, 2, 0) * std + mean
    return np.clip(img, 0, 1)


def _gradcam_panel(model, val_ld, classes, device, out_path):
    import cv2
    import matplotlib.pyplot as plt

    cam = GradCAM(model, model.layer4[-1])
    correct_samples, incorrect_samples = [], []
    model.eval()
    for x, y in val_ld:
        for i in range(x.size(0)):
            xi = x[i : i + 1].to(device)
            pred = model(xi).argmax(1).item()
            rec = (xi, int(y[i]), pred)
            if pred == y[i].item() and len(correct_samples) < config.T4_GRADCAM_CORRECT:
                correct_samples.append(rec)
            elif pred != y[i].item() and len(incorrect_samples) < config.T4_GRADCAM_INCORRECT:
                incorrect_samples.append(rec)
        if (len(correct_samples) >= config.T4_GRADCAM_CORRECT and
                len(incorrect_samples) >= config.T4_GRADCAM_INCORRECT):
            break

    samples = [("correct", s) for s in correct_samples] + \
              [("incorrect", s) for s in incorrect_samples]
    if not samples:
        return 0
    fig, axes = plt.subplots(1, len(samples), figsize=(4.0 * len(samples), 4.4))
    if len(samples) == 1:
        axes = [axes]
    for ax, (kind, (xi, true_i, pred_i)) in zip(axes, samples):
        heat = cam(xi, pred_i)
        img = _denorm(xi[0])
        heat = cv2.resize(heat, (img.shape[1], img.shape[0]))
        heat_c = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1] / 255.0
        overlay = np.clip(0.55 * img + 0.45 * heat_c, 0, 1)
        ax.imshow(overlay)
        color = "#1e8449" if kind == "correct" else "#c0392b"
        ax.set_title(f"{kind.upper()}\ntrue={classes[true_i]} | pred={classes[pred_i]}",
                     color=color, fontsize=10)
        ax.axis("off")
    fig.suptitle("Task 4 — Grad-CAM (2 correct + 1 incorrect)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return len(samples)


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #
def _plot_curves(history, out_path):
    import matplotlib.pyplot as plt

    ep = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    ax1.plot(ep, history["train_loss"], "o-", label="train")
    ax1.plot(ep, history["val_loss"], "s-", label="val")
    ax1.set_title("Loss"); ax1.set_xlabel("epoch"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(ep, history["train_acc"], "o-", label="train")
    ax2.plot(ep, history["val_acc"], "s-", label="val")
    ax2.axhline(config.T4_TARGET_VAL_ACC, color="#c0392b", ls="--", label="target 70%")
    ax2.set_title("Accuracy"); ax2.set_xlabel("epoch"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.suptitle("Task 4 — Training / validation curves", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _plot_confusion(cm, classes, out_path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Task 4 — Confusion matrix (validation)", fontweight="bold")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def run():
    log("TASK 4 — CNN Scene Classification (PyTorch transfer learning)")
    device = _device()
    log(f"  device = {device}")
    train_ld, val_ld, classes = _loaders()
    log(f"  classes = {classes}  (train batches={len(train_ld)}, val batches={len(val_ld)})")

    model = _build_model(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=config.T4_LR, weight_decay=config.T4_WEIGHT_DECAY)

    history = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc")}
    best_acc = 0.0
    for epoch in range(1, config.T4_EPOCHS + 1):
        tl, ta = _run_epoch(model, train_ld, criterion, device, optimizer)
        vl, va = _run_epoch(model, val_ld, criterion, device)
        history["train_loss"].append(tl); history["train_acc"].append(ta)
        history["val_loss"].append(vl); history["val_acc"].append(va)
        best_acc = max(best_acc, va)
        log(f"  epoch {epoch:2d}/{config.T4_EPOCHS}  "
            f"train_acc={ta:.3f}  val_acc={va:.3f}  val_loss={vl:.3f}")

    # --- evaluation artifacts --------------------------------------------- #
    _plot_curves(history, config.OUT_T4 / "training_curves.png")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in val_ld:
            p = model(x.to(device)).argmax(1).cpu().numpy()
            y_pred.extend(p.tolist()); y_true.extend(y.numpy().tolist())
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    _plot_confusion(cm, classes, config.OUT_T4 / "confusion_matrix.png")

    rep = classification_report(y_true, y_pred, labels=list(range(len(classes))),
                                target_names=classes, output_dict=True, zero_division=0)
    rows = [[c, f"{rep[c]['precision']:.2f}", f"{rep[c]['recall']:.2f}",
             f"{rep[c]['f1-score']:.2f}", int(rep[c]['support'])] for c in classes]
    save_table_figure(rows, ["class", "precision", "recall", "F1", "support"],
                      config.OUT_T4 / "per_class_metrics.png",
                      title="Task 4 — Per-class precision / recall / F1")

    n_cam = _gradcam_panel(model, val_ld, classes, device, config.OUT_T4 / "gradcam.png")

    final_acc = history["val_acc"][-1]
    summary = [
        ["backbone", config.T4_BACKBONE],
        ["classes", ", ".join(classes)],
        ["images/class (train)", config.T4_TRAIN_PER_CLASS],
        ["epochs", config.T4_EPOCHS],
        ["final val accuracy", f"{final_acc:.3f}"],
        ["best val accuracy", f"{best_acc:.3f}"],
        [f"target ({config.T4_TARGET_VAL_ACC:.0%}) met?", "YES" if best_acc >= config.T4_TARGET_VAL_ACC else "NO"],
        ["Grad-CAM samples", n_cam],
    ]
    save_table_figure(summary, ["metric", "value"], config.OUT_T4 / "task4_summary.png",
                      title="Task 4 — Summary")
    torch.save(model.state_dict(), config.OUT_T4 / "scene_classifier.pth")
    log(f"  best val accuracy={best_acc:.3f} (target {config.T4_TARGET_VAL_ACC:.0%}) — "
        f"outputs in {config.OUT_T4}")
    return {"history": history, "best_acc": best_acc, "final_acc": final_acc,
            "confusion": cm.tolist(), "per_class": rows, "classes": classes,
            "target_met": bool(best_acc >= config.T4_TARGET_VAL_ACC)}


if __name__ == "__main__":
    from src.utils import ensure_dirs, set_seed

    set_seed()
    ensure_dirs()
    run()
