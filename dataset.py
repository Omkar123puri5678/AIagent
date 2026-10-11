"""Loading the preprocessed recordings and drawing training batches from them."""

import csv
import random

import numpy as np
import torch

import config

Recording = tuple[np.ndarray, np.ndarray]  # x: (epochs, channels, samples) float16, y: (epochs,) int64


def split_recordings() -> dict[str, list[str]]:
    """Assign every recording to train, val or test, keeping each subject in a single set."""
    with open(config.DATA_DIR / "index.csv", newline="") as file:
        rows = list(csv.DictReader(file))

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    # Shuffle within each study so both studies are represented in every set.
    for study in sorted({row["study"] for row in rows}):
        subjects = sorted({row["subject"] for row in rows if row["study"] == study})
        random.Random(config.SEED).shuffle(subjects)
        n_val = max(1, round(config.VAL_FRACTION * len(subjects)))
        n_test = max(1, round(config.TEST_FRACTION * len(subjects)))
        assignment = {subject: "val" for subject in subjects[:n_val]}
        assignment |= {subject: "test" for subject in subjects[n_val : n_val + n_test]}
        for row in rows:
            if row["study"] == study:
                splits[assignment.get(row["subject"], "train")].append(row["recording"])
    return splits


def load_recordings(names: list[str]) -> list[Recording]:
    recordings = []
    for name in names:
        with np.load(config.DATA_DIR / f"{name}.npz") as data:
            recordings.append((data["x"], data["y"].astype(np.int64)))
    return recordings


class BatchSampler:
    """Draws random windows of SEQ_LEN consecutive epochs from the training recordings."""

    def __init__(self, recordings: list[Recording], seed: int) -> None:
        self.recordings = [r for r in recordings if len(r[1]) >= config.SEQ_LEN]
        self.rng = np.random.default_rng(seed)
        # Longer recordings contain more possible windows, so pick them more often.
        windows = np.array([len(y) - config.SEQ_LEN + 1 for _, y in self.recordings], dtype=np.float64)
        self.probabilities = windows / windows.sum()

    def sample(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        xs, ys = [], []
        for index in self.rng.choice(len(self.recordings), size=config.BATCH_SIZE, p=self.probabilities):
            x, y = self.recordings[index]
            start = self.rng.integers(len(y) - config.SEQ_LEN + 1)
            xs.append(x[start : start + config.SEQ_LEN])
            ys.append(y[start : start + config.SEQ_LEN])
        x = torch.from_numpy(np.stack(xs)).to(device).float()
        y = torch.from_numpy(np.stack(ys)).to(device)
        return x, y


def class_weights(recordings: list[Recording]) -> torch.Tensor:
    """Loss weights that count rare stages more: the square root of inverse frequency."""
    counts = np.bincount(np.concatenate([y for _, y in recordings]), minlength=len(config.STAGE_NAMES))
    weights = np.sqrt(counts.sum() / (len(counts) * counts))
    return torch.tensor(weights, dtype=torch.float32)
