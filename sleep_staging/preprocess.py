"""Turn the raw Sleep-EDF recordings into labelled 30-second epochs:  python -m sleep_staging.preprocess

For every recording this
  1. reads the two EEG channels and the EOG channel (all sampled at 100 Hz),
  2. band-pass filters them to 0.3-35 Hz,
  3. cuts the signal into 30-second epochs and labels each from the hypnogram,
     merging stages 3 and 4 into N3 (AASM scoring),
  4. keeps only the sleep period plus 30 minutes of wake on either side,
  5. drops epochs scored as movement or unknown,
  6. standardises each channel to zero mean and unit variance within the recording,
and saves the result to data/sleep_edf/<recording>.npz, plus an index.csv summary.
"""

import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import mne
import numpy as np

RAW_DIR = Path("Dataset/archive/sleep-edf-database-expanded-1.0.0")
OUT_DIR = Path("data/sleep_edf")

CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz", "EOG horizontal"]
SAMPLE_RATE = 100
EPOCH_SECONDS = 30
EPOCH_SAMPLES = SAMPLE_RATE * EPOCH_SECONDS
FILTER_LOW_HZ, FILTER_HIGH_HZ = 0.3, 35.0
WAKE_MARGIN_EPOCHS = 30 * 60 // EPOCH_SECONDS  # 30 minutes
CLIP_STD = 20.0  # cap extreme artifacts after standardising
WORKERS = 4

STAGE_NAMES = ["W", "N1", "N2", "N3", "REM"]
STAGE_LABELS = {
    "Sleep stage W": 0,
    "Sleep stage 1": 1,
    "Sleep stage 2": 2,
    "Sleep stage 3": 3,
    "Sleep stage 4": 3,
    "Sleep stage R": 4,
}
UNSCORED = -1  # "Movement time", "Sleep stage ?" and anything not covered by the hypnogram


def find_recordings() -> list[tuple[Path, Path]]:
    """Pair every PSG file with its hypnogram (they share the first six characters)."""
    pairs = []
    for psg in sorted(RAW_DIR.glob("sleep-*/*-PSG.edf")):
        hypnograms = list(psg.parent.glob(f"{psg.name[:6]}*-Hypnogram.edf"))
        if len(hypnograms) != 1:
            print(f"Skipping {psg.name}: found {len(hypnograms)} hypnogram files")
            continue
        pairs.append((psg, hypnograms[0]))
    return pairs


def epoch_labels(hypnogram: Path, n_epochs: int) -> np.ndarray:
    labels = np.full(n_epochs, UNSCORED, dtype=np.int8)
    for annotation in mne.read_annotations(hypnogram):
        label = STAGE_LABELS.get(annotation["description"], UNSCORED)
        start = int(round(annotation["onset"] / EPOCH_SECONDS))
        stop = start + int(round(annotation["duration"] / EPOCH_SECONDS))
        labels[start:stop] = label  # slices past the end of the recording are ignored
    return labels


def process(pair: tuple[Path, Path]) -> dict:
    psg, hypnogram = pair
    name = psg.name[:7]  # e.g. SC4001E or ST7011J

    raw = mne.io.read_raw_edf(psg, include=CHANNELS, preload=True, verbose="ERROR")
    raw.reorder_channels(CHANNELS)
    if raw.info["sfreq"] != SAMPLE_RATE:
        raise ValueError(f"{psg.name}: expected {SAMPLE_RATE} Hz, got {raw.info['sfreq']}")
    raw.filter(FILTER_LOW_HZ, FILTER_HIGH_HZ, verbose="ERROR")

    signal = raw.get_data()  # (channels, samples), in volts
    n_epochs = signal.shape[1] // EPOCH_SAMPLES
    x = signal[:, : n_epochs * EPOCH_SAMPLES].reshape(len(CHANNELS), n_epochs, EPOCH_SAMPLES)
    x = x.transpose(1, 0, 2)  # (epochs, channels, samples)
    y = epoch_labels(hypnogram, n_epochs)

    asleep = np.flatnonzero(y > 0)
    if len(asleep) == 0:
        raise ValueError(f"{psg.name}: no sleep epochs found")
    first = max(0, asleep[0] - WAKE_MARGIN_EPOCHS)
    last = min(n_epochs, asleep[-1] + 1 + WAKE_MARGIN_EPOCHS)
    x, y = x[first:last], y[first:last]

    scored = y != UNSCORED
    n_dropped = int((~scored).sum())
    x, y = x[scored], y[scored]

    mean = x.mean(axis=(0, 2), keepdims=True)
    std = x.std(axis=(0, 2), keepdims=True)
    x = np.clip((x - mean) / std, -CLIP_STD, CLIP_STD).astype(np.float16)

    np.savez(OUT_DIR / f"{name}.npz", x=x, y=y)

    counts = np.bincount(y, minlength=len(STAGE_NAMES))
    return {
        "recording": name,
        "study": "cassette" if name.startswith("SC") else "telemetry",
        # Characters 4-5 identify the person; split train/test by this, not by recording.
        "subject": f"{name[:2]}{name[3:5]}",
        "night": name[5],
        "epochs": len(y),
        "dropped_unscored": n_dropped,
        **dict(zip(STAGE_NAMES, counts.tolist())),
    }


def main() -> None:
    pairs = find_recordings()
    if not pairs:
        sys.exit(f"No recordings found under {RAW_DIR}")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Processing {len(pairs)} recordings into {OUT_DIR} ...")

    rows, failures = [], []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = [(pair[0].name, pool.submit(process, pair)) for pair in pairs]
        for done, (psg_name, future) in enumerate(futures, start=1):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append((psg_name, error))
                print(f"FAILED {psg_name}: {error}")
            if done % 10 == 0 or done == len(futures):
                print(f"  {done}/{len(futures)}")

    with open(OUT_DIR / "index.csv", "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total = sum(row["epochs"] for row in rows)
    subjects = len({row["subject"] for row in rows})
    print(f"\n{len(rows)} recordings, {subjects} subjects, {total:,} epochs")
    for stage in STAGE_NAMES:
        count = sum(row[stage] for row in rows)
        print(f"  {stage:<3} {count:>7,}  {100 * count / total:4.1f}%")
    print(f"  dropped as unscored: {sum(row['dropped_unscored'] for row in rows):,}")
    if failures:
        print(f"\n{len(failures)} recordings failed:")
        for psg_name, error in failures:
            print(f"  {psg_name}: {error}")


if __name__ == "__main__":
    main()
