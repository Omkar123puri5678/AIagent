"""Draw the results as images in results/sleep_staging/:  python -m sleep_staging.plots [recording]

Makes training_curves.png, confusion_matrix.png and hypnogram_<recording>.png, and writes
the test scores to test_report.txt. The optional argument picks which test recording to
draw the hypnogram for, e.g. `python -m sleep_staging.plots SC4131E`; the default is the first one.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap

from common.style import BLUE, BLUE_RAMP, INK, INK_SECONDARY, ORANGE, SURFACE, plt
from sleep_staging import config
from sleep_staging.dataset import load_recordings, split_recordings
from sleep_staging.evaluate import metrics, predict, report
from sleep_staging.model import SleepStager

RESULTS_DIR = Path("results/sleep_staging")
HISTORY_FILE = RESULTS_DIR / "history.csv"


def plot_training_curves() -> None:
    with open(HISTORY_FILE, newline="") as file:
        rows = list(csv.DictReader(file))
    epochs = [int(row["epoch"]) for row in rows]
    loss = [float(row["loss"]) for row in rows]
    accuracy = [float(row["val_accuracy"]) for row in rows]
    macro_f1 = [float(row["val_macro_f1"]) for row in rows]
    best = int(np.argmax(macro_f1))

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True, layout="constrained")
    figure.suptitle("Training progress", x=0.01, ha="left", fontsize=14, fontweight="bold")

    top.plot(epochs, loss, color=BLUE, linewidth=2)
    top.set_title("Training loss")
    top.grid(axis="y")

    bottom.plot(epochs, accuracy, color=BLUE, linewidth=2, label="Accuracy")
    bottom.plot(epochs, macro_f1, color=ORANGE, linewidth=2, label="Macro F1")
    bottom.plot(epochs[best], macro_f1[best], "o", color=ORANGE, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    bottom.set_title(f"Validation scores (dot = saved model: epoch {epochs[best]}, macro F1 {macro_f1[best]:.3f})")
    bottom.set_xlabel("Epoch")
    bottom.grid(axis="y")
    bottom.legend(frameon=False, loc="lower right", labelcolor=INK_SECONDARY)
    bottom.set_ylim(min(macro_f1 + accuracy) - 0.04, max(macro_f1 + accuracy) + 0.02)
    bottom.xaxis.get_major_locator().set_params(integer=True)

    figure.savefig(RESULTS_DIR / "training_curves.png", dpi=150)
    plt.close(figure)


def plot_confusion_matrix(
    matrix: np.ndarray,
    title: str = "Test set: how each true stage was scored",
    filename: str = "confusion_matrix.png",
) -> None:
    scores = metrics(matrix)
    share = matrix / matrix.sum(axis=1, keepdims=True)
    n = len(config.STAGE_NAMES)

    figure, axis = plt.subplots(figsize=(6.6, 6.2), layout="constrained")
    axis.imshow(share, cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP), vmin=0, vmax=1)
    for row in range(n):
        for column in range(n):
            # Text stays in ink colours; it only flips to light on the dark cells.
            color = SURFACE if share[row, column] > 0.55 else INK
            axis.text(column, row - 0.08, f"{100 * share[row, column]:.0f}%", ha="center", va="center",
                      color=color, fontsize=12, fontweight="bold")
            axis.text(column, row + 0.22, f"{matrix[row, column]:,}", ha="center", va="center",
                      color=color, fontsize=8, alpha=0.85)

    axis.set_xticks(range(n), config.STAGE_NAMES)
    axis.set_yticks(range(n), config.STAGE_NAMES)
    axis.set_xlabel("Predicted stage")
    axis.set_ylabel("True stage")
    axis.tick_params(length=0)
    # White gaps between cells.
    axis.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    axis.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    axis.grid(which="minor", color=SURFACE, linewidth=2)
    axis.tick_params(which="minor", length=0)
    for spine in axis.spines.values():
        spine.set_visible(False)

    figure.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold")
    axis.set_title(
        f"Share of each true stage (count below). Accuracy {scores['accuracy']:.1%}, "
        f"macro F1 {scores['macro_f1']:.3f}, kappa {scores['kappa']:.3f}",
        fontsize=9, fontweight="normal", color=INK_SECONDARY, pad=10,
    )
    figure.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(figure)


def plot_hypnogram(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    # Conventional hypnogram order, top to bottom: W, REM, N1, N2, N3.
    order = ["W", "REM", "N1", "N2", "N3"]
    height = np.array([len(order) - 1 - order.index(stage) for stage in config.STAGE_NAMES])
    hours = np.arange(len(y_true)) * 30 / 3600
    agreement = (y_true == y_pred).mean()

    figure, (top, middle, bottom) = plt.subplots(
        3, 1, figsize=(11, 5.6), sharex=True, layout="constrained", height_ratios=[4, 4, 0.7]
    )
    figure.suptitle(f"One night, scored by an expert and by the model ({name})",
                    x=0.01, ha="left", fontsize=14, fontweight="bold")

    for axis, labels, color, title in (
        (top, y_true, BLUE, "Expert scoring"),
        (middle, y_pred, ORANGE, f"Model prediction (agrees on {agreement:.1%} of epochs)"),
    ):
        axis.step(hours, height[labels], where="post", color=color, linewidth=1.2)
        axis.set_yticks(range(len(order)), order[::-1])
        axis.set_ylim(-0.4, len(order) - 0.6)
        axis.set_title(title)
        axis.grid(axis="y")

    wrong = hours[y_true != y_pred]
    bottom.vlines(wrong, 0, 1, color=INK_SECONDARY, linewidth=0.6)
    bottom.set_title("Epochs where they disagree")
    bottom.set_yticks([])
    bottom.set_ylim(0, 1)
    bottom.spines["left"].set_visible(False)
    bottom.set_xlabel("Hours from start of recording")
    bottom.set_xlim(0, hours[-1])

    figure.savefig(RESULTS_DIR / f"hypnogram_{name}.png", dpi=150)
    plt.close(figure)


def main() -> None:
    if not config.CHECKPOINT_FILE.exists():
        sys.exit("No trained model found. Run `python -m sleep_staging.train` first.")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SleepStager().to(device)
    model.load_state_dict(torch.load(config.CHECKPOINT_FILE, map_location=device)["model"])

    names = split_recordings()["test"]
    chosen = sys.argv[1] if len(sys.argv) > 1 else names[0]
    if chosen not in names:
        sys.exit(f"{chosen} is not a test recording. Choose one of: {', '.join(names)}")

    n = len(config.STAGE_NAMES)
    matrix = np.zeros((n, n), dtype=np.int64)
    for name, (x, y) in zip(names, load_recordings(names)):
        predicted = predict(model, x, device)
        np.add.at(matrix, (y, predicted), 1)
        if name == chosen:
            plot_hypnogram(name, y, predicted)

    plot_confusion_matrix(matrix)
    (RESULTS_DIR / "test_report.txt").write_text(report(matrix) + "\n")
    if HISTORY_FILE.exists():
        plot_training_curves()
    else:
        print(f"No {HISTORY_FILE}, so no training curves (train.py writes it).")
    print(f"Saved to {RESULTS_DIR.resolve()}:")
    for path in sorted(RESULTS_DIR.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
