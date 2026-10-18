"""A deep model that reads a whole night to predict later cognitive impairment:

    python -m cognitive.deep [data folder]

Input: the epoch sequences from cognitive/epochs.py (in <data folder>/epochs) and the same people,
labels and cleaning as cognitive/baseline.py. Each night becomes a grid of 30-second epochs (up to
MAX_HOURS) by about 80 values per epoch:
  - the EEG spectrum as its shape (log power at each frequency minus the epoch's total), so
    differences in amplifier gain between hospitals do not matter, plus the total,
  - heart rate, SpO2, breathing and chin tone, standardised with training statistics,
  - the CAISR stage as one-hot columns and the share of each epoch in an event,
  - a 0/1 column per signal saying whether it was measured, so "missing" is not read as "average".

The network is a 1-D convolutional network over time: four blocks, each halving the time
resolution (from 30 s to 8 minutes), then attention pooling, which learns how much each part
of the night should count, and a small head that also sees age and sex. It is trained with
the impaired class weighted up, AdamW and early stopping on a held-out tenth of the training
people.

It is scored exactly like cognitive/baseline.py (5-fold cross-validation and leave-one-hospital-out,
age-conditioned AUROC), next to the logistic regression on the 49 features and an ensemble
of the two (the average of their rank-scaled risks). Saves <folder name>_deep_report.txt to
results/cognitive/.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold, train_test_split

from cognitive import RESULTS_DIR
from cognitive.baseline import FOLDS, SEED, fit_predict, load_table, make_models, scores
from cognitive.epochs import CHANNELS, FREQS

MAX_HOURS = 10
MAX_EPOCHS_PER_NIGHT = MAX_HOURS * 120
STAGES = [1, 2, 3, 4, 5]  # N3, N2, N1, REM, wake; anything else counts as unscored

WIDTH = 64
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-2
MAX_TRAINING_EPOCHS = 40
PATIENCE = 8
VAL_SHARE = 0.1
SEEDS = 3  # networks trained per split, averaged, since one network is noisy on 500 positives


def load_epochs(data_dir: Path, records: list[str]) -> list[np.ndarray]:
    packs = sorted((data_dir / "epochs").glob("epochs_*.npz"))
    if not packs:
        sys.exit(f"No epoch packs in {data_dir / 'epochs'}. Run cognitive/epochs.py on Kaggle and download its output.")
    found = {}
    wanted = set(records)
    for pack in packs:
        with np.load(pack) as nights:
            found |= {key: nights[key] for key in nights.files if key in wanted}
    return [found.get(record) for record in records]


def to_inputs(night: np.ndarray) -> np.ndarray:
    """(epochs, raw channels) -> (input channels, MAX_EPOCHS_PER_NIGHT) float16, before standardising."""
    night = night.astype(np.float32)
    column = {name: i for i, name in enumerate(CHANNELS)}
    stage = night[:, column["stage"]]
    scored = np.flatnonzero(np.isin(np.round(stage), STAGES))
    start = scored[0] if len(scored) else 0  # from lights out, not from when the recorder was switched on
    night = night[start : start + MAX_EPOCHS_PER_NIGHT]
    stage = night[:, column["stage"]]

    eeg = night[:, : len(FREQS)]
    total = np.log10(np.sum(10.0 ** eeg, axis=1, keepdims=True))
    shape = eeg - total
    others = night[:, column["hr"] : column["chin_log_tone"] + 1]
    one_hot = np.stack([np.round(stage) == code for code in STAGES], axis=1).astype(np.float32)
    events = night[:, column["resp_share"] :]
    measured = np.stack([np.isfinite(total[:, 0]), np.isfinite(others[:, 0]), np.isfinite(others[:, 2]),
                         np.isfinite(others[:, 4]), np.isfinite(others[:, 6])], axis=1).astype(np.float32)
    x = np.hstack([shape, total, others, one_hot, events, measured])
    out = np.full((MAX_EPOCHS_PER_NIGHT, x.shape[1]), np.nan, dtype=np.float32)
    out[: len(x)] = x
    return out.T.astype(np.float16)


N_CONTINUOUS = len(FREQS) + 1 + 7  # the columns that are standardised; one-hot, events and flags are not


class Standardise:
    """Per-channel median and spread from the training nights; NaN becomes 0 (the median) afterwards."""

    def fit(self, x: np.ndarray, rows: np.ndarray) -> "Standardise":
        sample = x[rows][:, :N_CONTINUOUS, ::10].astype(np.float32)
        sample = np.moveaxis(sample, 1, 0).reshape(N_CONTINUOUS, -1)
        self.centre = np.nanmedian(sample, axis=1)
        low, high = np.nanpercentile(sample, [25, 75], axis=1)
        self.spread = np.maximum(high - low, 1e-3) / 1.35
        return self

    def __call__(self, batch: np.ndarray) -> torch.Tensor:
        batch = batch.astype(np.float32)
        batch[:, :N_CONTINUOUS] = (batch[:, :N_CONTINUOUS] - self.centre[:, None]) / self.spread[:, None]
        batch = np.clip(np.nan_to_num(batch, nan=0.0), -5, 5)
        return torch.from_numpy(batch)


class NightNet(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        layers, width = [], channels
        for out in (WIDTH, WIDTH, 2 * WIDTH, 2 * WIDTH):
            layers += [nn.Conv1d(width, out, 5, padding=2, bias=False), nn.BatchNorm1d(out), nn.GELU(),
                       nn.Conv1d(out, out, 5, padding=2, bias=False), nn.BatchNorm1d(out), nn.GELU(),
                       nn.MaxPool1d(2), nn.Dropout(DROPOUT)]
            width = out
        self.body = nn.Sequential(*layers)
        self.attention = nn.Conv1d(width, 1, 1)
        self.head = nn.Sequential(nn.Linear(width + 2, 32), nn.GELU(), nn.Dropout(DROPOUT), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor, present: torch.Tensor, extra: torch.Tensor) -> torch.Tensor:
        h = self.body(x)
        # Recorded epochs only: the padding after the end of the night gets no attention.
        present = F.max_pool1d(present[:, None], 16)[:, 0, : h.shape[-1]]
        weights = self.attention(h)[:, 0].masked_fill(present < 0.5, -1e4).softmax(dim=-1)
        pooled = (h * weights[:, None]).sum(dim=-1)
        return self.head(torch.cat([pooled, extra], dim=1))[:, 0]


def batches(rows: np.ndarray, size: int, shuffle: bool, rng: np.random.Generator):
    rows = rng.permutation(rows) if shuffle else rows
    for i in range(0, len(rows), size):
        yield rows[i : i + size]


def train_network(x, present, extra, labels, train_rows, seed, device) -> tuple[NightNet, Standardise]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    fit_rows, val_rows = train_test_split(train_rows, test_size=VAL_SHARE, stratify=labels[train_rows], random_state=seed)
    standardise = Standardise().fit(x, fit_rows)
    model = NightNet(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    positive = labels[fit_rows].mean()
    pos_weight = torch.tensor((1 - positive) / positive, device=device)
    best, best_state, waited = np.inf, None, 0

    def run(rows, learn):
        model.train(learn)
        total = 0.0
        with torch.set_grad_enabled(learn):
            for chosen in batches(rows, BATCH_SIZE, learn, rng):
                logits = model(standardise(x[chosen]).to(device), torch.from_numpy(present[chosen]).to(device),
                               torch.from_numpy(extra[chosen]).to(device))
                loss = F.binary_cross_entropy_with_logits(logits, torch.from_numpy(labels[chosen]).float().to(device),
                                                          pos_weight=pos_weight)
                if learn:
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                total += loss.item() * len(chosen)
        return total / len(rows)

    for _ in range(MAX_TRAINING_EPOCHS):
        run(fit_rows, True)
        val_loss = run(val_rows, False)
        if val_loss < best:
            best, waited = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return model.eval(), standardise


@torch.no_grad()
def predict(models, x, present, extra, rows, device) -> np.ndarray:
    risk = np.zeros(len(rows))
    for model, standardise in models:
        out = []
        for chosen in batches(rows, 128, False, None):
            logits = model(standardise(x[chosen]).to(device), torch.from_numpy(present[chosen]).to(device),
                           torch.from_numpy(extra[chosen]).to(device))
            out.append(torch.sigmoid(logits).cpu().numpy())
        risk += rankdata(np.concatenate(out)) / len(rows)
    return risk / len(models)


def deep_risk(x, present, extra, labels, train_rows, test_rows, device) -> np.ndarray:
    models = [train_network(x, present, extra, labels, train_rows, SEED + s, device) for s in range(SEEDS)]
    return predict(models, x, present, extra, test_rows, device)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/hsp_large")
    table, names = load_table(data_dir)
    table["record"] = table["BidsFolder"] + "_ses-" + table["SessionID"].astype(str)
    nights = load_epochs(data_dir, table["record"].tolist())
    keep = np.array([n is not None for n in nights])
    print(f"{keep.sum()} of {len(table)} people have an epoch sequence; the rest are left out")
    table = table[keep].reset_index(drop=True)
    nights = [n for n in nights if n is not None]

    x = np.stack([to_inputs(n) for n in nights])
    present = np.isfinite(x[:, len(FREQS) + 1 + 7]).astype(np.float32)  # the N3 one-hot column is never NaN on a real epoch
    del nights
    labels = table["label"].to_numpy()
    ages, sites = table["Age"].to_numpy(), table["SiteID"].to_numpy()
    extra = np.stack([(table["Age"] - table["Age"].mean()) / table["Age"].std(), table["male"]], axis=1).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"inputs {x.shape} ({x.nbytes / 1e9:.1f} GB), device {device}")
    linear, _ = make_models(labels.mean())["logistic regression"]

    def score_all(train_rows, test_rows):
        deep = deep_risk(x, present, extra, labels, train_rows, test_rows, device)
        lr = fit_predict(linear, table.iloc[train_rows], table.iloc[test_rows], names)
        both = (rankdata(deep) + rankdata(lr)) / (2 * len(test_rows))
        return {"deep network": deep, "logistic regression": lr, "ensemble": both}

    started = time.time()
    risk = {name: np.zeros(len(table)) for name in ("deep network", "logistic regression", "ensemble")}
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for fold, (train_rows, test_rows) in enumerate(folds.split(table, table["label"].astype(str) + table["SiteID"]), 1):
        for name, r in score_all(train_rows, test_rows).items():
            risk[name][test_rows] = r
        print(f"  fold {fold}/{FOLDS} done ({(time.time() - started) / 60:.0f} min)", flush=True)

    lines = [f"{len(table)} people, {labels.sum()} later impaired, {x.shape[1]} values per epoch, "
             f"up to {MAX_EPOCHS_PER_NIGHT} epochs; {SEEDS} networks averaged per split", "",
             f"{FOLDS}-fold cross-validation (one repeat)",
             f"{'model':<21}{'age-cond. AUROC':>17}{'same hospital':>15}{'AUROC':>8}{'AUPRC':>8}"]
    for name, r in risk.items():
        s = scores(labels, r, ages, sites)
        lines.append(f"{name:<21}{s['age_auroc']:>17.3f}{s['same_site_auroc']:>15.3f}{s['auroc']:>8.3f}{s['auprc']:>8.3f}")
        print(lines[-1], flush=True)

    lines += ["", "leave-one-hospital-out: age-conditioned AUROC on the hospital left out of training",
              f"{'model':<21}" + "".join(f"{site:>10}" for site in sorted(set(sites)))]
    by_site = {name: [] for name in risk}
    for site in sorted(set(sites)):
        train_rows, test_rows = np.flatnonzero(sites != site), np.flatnonzero(sites == site)
        for name, r in score_all(train_rows, test_rows).items():
            by_site[name].append(scores(labels[test_rows], r, ages[test_rows], sites[test_rows])["age_auroc"])
        print(f"  held out {site} ({(time.time() - started) / 60:.0f} min)", flush=True)
    for name, values in by_site.items():
        lines.append(f"{name:<21}" + "".join(f"{v:>10.3f}" for v in values))

    text = "\n".join(lines)
    print("\n" + text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{data_dir.name}_deep_report.txt").write_text(text + "\n")
    print(f"\nSaved {data_dir.name}_deep_report.txt to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
