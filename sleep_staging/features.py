"""Compute hand-made features for every 30-second epoch:  python -m sleep_staging.features

For each channel this measures
  - band power in the delta, theta, alpha and beta bands (area under the Welch power
    spectrum, on a log scale), and
  - the three Hjorth parameters: activity, mobility and complexity,
and saves one row per epoch, with its sleep stage, to data/features.csv.

It reads the output of preprocess.py. Those signals are standardised within each
recording, so the powers are relative to that recording's overall level, not in microvolts.
"""

import sys

import numpy as np
import pandas as pd
from scipy.integrate import simpson
from scipy.signal import welch

from sleep_staging import config
from sleep_staging.preprocess import CHANNELS, SAMPLE_RATE

FEATURES_FILE = config.DATA_DIR.parent / "features.csv"

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13), "beta": (13, 30)}
WELCH_SECONDS = 2  # length of each Welch segment
TINY = 1e-12  # avoids log(0) and division by zero on a flat signal


def band_powers(x: np.ndarray) -> dict[str, np.ndarray]:
    """x: (epochs, samples) -> log10 band power per epoch, for each band."""
    freqs, power = welch(x, fs=SAMPLE_RATE, nperseg=SAMPLE_RATE * WELCH_SECONDS, axis=-1)
    result = {}
    for band, (low, high) in BANDS.items():
        mask = (freqs >= low) & (freqs <= high)
        result[band] = np.log10(simpson(y=power[:, mask], x=freqs[mask], axis=-1) + TINY)
    return result


def hjorth_parameters(x: np.ndarray) -> dict[str, np.ndarray]:
    """x: (epochs, samples) -> Hjorth activity, mobility and complexity per epoch."""
    dx = np.diff(x, axis=-1)
    ddx = np.diff(dx, axis=-1)
    var_x, var_dx, var_ddx = x.var(axis=-1) + TINY, dx.var(axis=-1) + TINY, ddx.var(axis=-1)
    mobility = np.sqrt(var_dx / var_x)
    return {
        "activity": np.log10(var_x),  # signal power; log scale because it spans orders of magnitude
        "mobility": mobility,  # roughly the mean frequency
        "complexity": np.sqrt(var_ddx / var_dx) / mobility,  # how far from a pure sine wave
    }


def main() -> None:
    index_file = config.DATA_DIR / "index.csv"
    if not index_file.exists():
        sys.exit(f"No preprocessed data in {config.DATA_DIR}. Run `python -m sleep_staging.preprocess` first.")
    index = pd.read_csv(index_file)

    tables = []
    for done, row in enumerate(index.itertuples(), start=1):
        with np.load(config.DATA_DIR / f"{row.recording}.npz") as data:
            x, y = data["x"].astype(np.float32), data["y"]
        columns = {"recording": row.recording, "subject": row.subject, "epoch": np.arange(len(y)), "stage": y}
        for channel_index, channel in enumerate(CHANNELS):
            short = channel.split()[1]  # "EEG Fpz-Cz" -> "Fpz-Cz"
            signal = x[:, channel_index]
            for name, values in (band_powers(signal) | hjorth_parameters(signal)).items():
                columns[f"{short}_{name}"] = values
        tables.append(pd.DataFrame(columns))
        if done % 20 == 0 or done == len(index):
            print(f"  {done}/{len(index)}")

    features = pd.concat(tables, ignore_index=True)
    features.to_csv(FEATURES_FILE, index=False, float_format="%.5f")
    print(f"\n{len(features):,} epochs x {features.shape[1] - 4} features saved to {FEATURES_FILE}\n")
    print(features.head().to_string())


if __name__ == "__main__":
    main()
