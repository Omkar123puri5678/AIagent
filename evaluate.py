"""Score the trained model on the held-out test subjects:  python evaluate.py"""

import sys

import numpy as np
import torch

import config
from dataset import Recording, load_recordings, split_recordings
from model import SleepStager

EVAL_CHUNKS = 8  # sequences per forward pass when scoring a whole recording


@torch.no_grad()
def predict(model: SleepStager, x: np.ndarray, device: str) -> np.ndarray:
    """Predict a stage for every epoch of one recording, in windows of SEQ_LEN epochs."""
    model.eval()
    predictions = []
    step = config.SEQ_LEN * EVAL_CHUNKS
    for start in range(0, len(x), step):
        chunk = torch.from_numpy(x[start : start + step]).to(device).float()
        # Split into full windows, plus one shorter window for any leftover epochs.
        full = len(chunk) // config.SEQ_LEN * config.SEQ_LEN
        windows = [chunk[:full].view(-1, config.SEQ_LEN, *chunk.shape[1:])] if full else []
        if full < len(chunk):
            windows.append(chunk[full:].unsqueeze(0))
        for window in windows:
            predictions.append(model(window).argmax(dim=-1).flatten().cpu().numpy())
    return np.concatenate(predictions)


def confusion_matrix(model: SleepStager, recordings: list[Recording], device: str) -> np.ndarray:
    """Rows are the true stage, columns the predicted stage."""
    n = len(config.STAGE_NAMES)
    matrix = np.zeros((n, n), dtype=np.int64)
    for x, y in recordings:
        np.add.at(matrix, (y, predict(model, x, device)), 1)
    return matrix


def metrics(matrix: np.ndarray) -> dict:
    total = matrix.sum()
    correct = np.diag(matrix)
    true_counts, predicted_counts = matrix.sum(axis=1), matrix.sum(axis=0)
    f1 = 2 * correct / np.maximum(true_counts + predicted_counts, 1)
    accuracy = correct.sum() / total
    chance = (true_counts * predicted_counts).sum() / total**2
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(f1.mean()),
        "kappa": float((accuracy - chance) / (1 - chance)),  # Cohen's kappa: agreement beyond chance
        "f1": f1,
    }


def report(matrix: np.ndarray) -> str:
    scores = metrics(matrix)
    lines = [
        f"accuracy {scores['accuracy']:.3f} | macro F1 {scores['macro_f1']:.3f} | kappa {scores['kappa']:.3f}",
        "per-stage F1: " + "  ".join(f"{name} {f1:.3f}" for name, f1 in zip(config.STAGE_NAMES, scores["f1"])),
        "",
        "confusion matrix (rows = true stage, columns = predicted stage)",
        "      " + "".join(f"{name:>8}" for name in config.STAGE_NAMES),
    ]
    for name, row in zip(config.STAGE_NAMES, matrix):
        lines.append(f"{name:>6}" + "".join(f"{count:>8,}" for count in row))
    return "\n".join(lines)


def main() -> None:
    if not config.CHECKPOINT_FILE.exists():
        sys.exit("No trained model found. Run `python train.py` first.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(config.CHECKPOINT_FILE, map_location=device)
    model = SleepStager().to(device)
    model.load_state_dict(checkpoint["model"])

    names = split_recordings()["test"]
    print(f"Scoring {len(names)} test recordings (model from epoch {checkpoint['epoch']}) ...\n")
    print(report(confusion_matrix(model, load_recordings(names), device)))


if __name__ == "__main__":
    main()
