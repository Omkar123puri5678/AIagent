"""Turn every night of the Challenge 2026 data into a sequence of 30-second epochs:

    python -m cognitive.epochs <path to the Challenge 2026 data> --out <folder>

cognitive/features.py squeezes a night into one row of 49 numbers. The deep model in
cognitive/deep.py instead reads the whole night, so this keeps one row of numbers per epoch
(about 900 per night), small enough to train on a laptop:
  - EEG: log power from 0.5 to 30 Hz in 0.5 Hz steps (60 values), from the same
    derivation features.py uses,
  - heart rate and its beat-to-beat spread, from the ECG,
  - mean and lowest SpO2,
  - breathing rate and the log amplitude of the effort belt,
  - log chin muscle tone,
  - the CAISR stage, and the share of the epoch inside a breathing event, an arousal and
    a limb movement.
A signal the recording does not have, or an epoch where it is unusable, is NaN.

Nights are packed into files of PACK_SIZE nights (epochs_000.npz, ...; one array per
night, keyed by record name, stored as float16), so a crash loses at most one pack and a
rerun skips the nights already packed.
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from cognitive.features import (BREATHING_RATE, CHIN_NAMES, ECG_NAMES, ECG_RATE, EEG_RATE, EFFORT_NAMES, EPOCH_SECONDS,
                          SPO2_NAMES, TINY, bandpass, cut_epochs, epoch_stages, find, find_eeg, load, r_peaks,
                          resample)

FREQS = np.arange(0.5, 30.01, 0.5)
CHANNELS = ([f"eeg_{f:g}hz" for f in FREQS] + ["hr", "rr_spread_ms", "spo2_mean", "spo2_min", "breaths_per_min",
            "effort_log_amplitude", "chin_log_tone", "stage", "resp_share", "arousal_share", "limb_share"])
PACK_SIZE = 500


def eeg_spectrum(signals: dict, n: int) -> np.ndarray:
    found = find_eeg(signals)
    out = np.full((n, len(FREQS)), np.nan)
    if found is None:
        return out
    x = bandpass(resample(*found, EEG_RATE), EEG_RATE, 0.3, 35.0)
    x, _ = cut_epochs(x, EEG_RATE, np.zeros(n))
    freqs, power = welch(x, fs=EEG_RATE, nperseg=EEG_RATE * 2, axis=-1)
    chosen = np.isin(np.round(freqs, 2), np.round(FREQS, 2))
    spread = x.std(axis=1)
    flat = spread < 0.1 * np.median(spread)  # a disconnected electrode, not sleep
    out[: len(x)] = np.where(flat[:, None], np.nan, np.log10(power[:, chosen] + TINY))
    return out


def heart(signals: dict, n: int) -> np.ndarray:
    out = np.full((n, 2), np.nan)
    found = find(signals, ECG_NAMES)
    if found is None:
        return out
    peaks = r_peaks(resample(*found, ECG_RATE))
    times, rr = peaks[:-1] / ECG_RATE, np.diff(peaks) / ECG_RATE
    good = (rr > 0.3) & (rr < 2.0)
    if good.sum() < 300:
        return out
    typical = np.median(rr[good])
    good &= np.abs(rr - typical) < 0.4 * typical
    epoch = (times // EPOCH_SECONDS).astype(int)
    for e in range(n):
        inside = good & (epoch == e)
        if inside.sum() >= 10:  # most of an epoch's 25-50 beats were found
            out[e] = 60 / rr[inside].mean(), rr[inside].std() * 1000
    return out


def oxygen(signals: dict, n: int) -> np.ndarray:
    out = np.full((n, 2), np.nan)
    found = find(signals, SPO2_NAMES)
    if found is None or found[1] < 1:
        return out
    x, rate = found[0], int(round(found[1]))
    x = x[: len(x) // rate * rate].reshape(-1, rate).mean(axis=1)
    if np.median(x) <= 1.5:
        x = x * 100
    x = np.where((x >= 50) & (x <= 100), x, np.nan)
    m = min(n, len(x) // EPOCH_SECONDS)
    epochs = x[: m * EPOCH_SECONDS].reshape(m, EPOCH_SECONDS)
    usable = np.isfinite(epochs).sum(axis=1) >= EPOCH_SECONDS // 2
    with np.errstate(all="ignore"):
        out[:m, 0] = np.where(usable, np.nanmean(epochs, axis=1), np.nan)
        out[:m, 1] = np.where(usable, np.nanmin(epochs, axis=1), np.nan)
    return out


def breathing(signals: dict, n: int) -> np.ndarray:
    out = np.full((n, 2), np.nan)
    found = find(signals, EFFORT_NAMES)
    if found is None or found[1] < BREATHING_RATE:
        return out
    x, _ = cut_epochs(resample(*found, BREATHING_RATE), BREATHING_RATE, np.zeros(n))
    freqs, power = welch(x, fs=BREATHING_RATE, nperseg=x.shape[1], axis=-1)
    band = (freqs >= 0.1) & (freqs <= 0.6)
    spread = x.std(axis=1)
    moving = spread > 0
    out[: len(x), 0] = np.where(moving, 60 * freqs[band][power[:, band].argmax(axis=1)], np.nan)
    out[: len(x), 1] = np.where(moving, np.log10(spread + TINY), np.nan)
    return out


def chin(signals: dict, n: int) -> np.ndarray:
    out = np.full((n, 1), np.nan)
    found = find(signals, CHIN_NAMES)
    if found is None or found[1] < 100:
        return out
    x, _ = cut_epochs(bandpass(resample(*found, 100), 100, 10.0, 40.0), 100, np.zeros(n))
    out[: len(x), 0] = np.log10(np.sqrt((x**2).mean(axis=1)) + TINY)
    return out


def events(caisr: dict, n: int) -> np.ndarray:
    out = np.full((n, 3), np.nan)
    for column, name in enumerate(("resp", "arousal", "limb")):
        found = find(caisr, [f"{name}_caisr"])
        if found is None:
            continue
        x, rate = found
        samples = int(round(rate * EPOCH_SECONDS))
        if samples < 1:
            continue
        active = ((x > 0) & (x < 9)).astype(float)
        m = min(n, len(active) // samples)
        out[:m, column] = active[: m * samples].reshape(m, samples).mean(axis=1)
    return out


def process(job: tuple[Path, str, str, str]) -> tuple[str, np.ndarray, str]:
    root, site, patient, session = job
    name = f"{patient}_ses-{session}"
    caisr = load(root / "algorithmic_annotations" / site / f"{name}_caisr_annotations.edf")
    stages = epoch_stages(caisr)
    n = len(stages)
    signals = load(root / "physiological_data" / site / f"{name}.edf")
    parts, problems = [], []
    for extract, width in ((eeg_spectrum, len(FREQS)), (heart, 2), (oxygen, 2), (breathing, 2), (chin, 1)):
        try:  # one bad channel should not cost the rest of the night
            parts.append(extract(signals, n))
        except Exception as error:
            parts.append(np.full((n, width), np.nan))
            problems.append(f"{extract.__name__}: {error}")
    parts += [stages[:, None].astype(float), events(caisr, n)]
    return name, np.hstack(parts).astype(np.float16), "; ".join(problems)


def main() -> None:
    parser = argparse.ArgumentParser(description="One row of numbers per 30-second epoch, for every night.")
    parser.add_argument("data", type=Path, help="folder holding demographics.csv, or any folder above it")
    parser.add_argument("--out", type=Path, default=Path("epochs"))
    parser.add_argument("--limit", type=int, help="stop after this many recordings (for a quick try)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    demographics = next(iter(sorted(args.data.rglob("demographics.csv"))), None)
    if demographics is None:
        sys.exit(f"No demographics.csv found under {args.data}")
    root = demographics.parent
    nights = pd.read_csv(demographics)[["SiteID", "BidsFolder", "SessionID"]].drop_duplicates()
    jobs = [(root, str(row.SiteID), str(row.BidsFolder), str(row.SessionID)) for row in nights.itertuples()]

    args.out.mkdir(parents=True, exist_ok=True)
    packs = sorted(args.out.glob("epochs_*.npz"))
    done = {key for pack in packs for key in np.load(pack).files}
    jobs = [job for job in jobs if f"{job[2]}_ses-{job[3]}" not in done][: args.limit]
    print(f"{len(jobs)} recordings to process from {root} ({len(done)} already packed in {args.out})", flush=True)

    pending, problems, failures, next_pack = {}, [], [], len(packs)

    def flush() -> None:
        nonlocal next_pack
        if pending:
            np.savez(args.out / f"epochs_{next_pack:03d}.npz", **pending)
            pending.clear()
            next_pack += 1

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): f"{job[2]}_ses-{job[3]}" for job in jobs}
        for finished, future in enumerate(as_completed(futures), start=1):
            try:
                name, epochs, problem = future.result()
                pending[name] = epochs
                problems.append({"record": name, "epochs": len(epochs), "problems": problem})
            except Exception as error:
                failures.append(futures[future])
                print(f"FAILED {futures[future]}: {error}", flush=True)
            if len(pending) >= PACK_SIZE:
                flush()
            if finished % 50 == 0 or finished == len(futures):
                print(f"  {finished}/{len(futures)}", flush=True)
    flush()

    if problems:
        pd.DataFrame(problems).to_csv(args.out / f"problems_{next_pack:03d}.csv", index=False)
    (args.out / "channels.txt").write_text("\n".join(CHANNELS) + "\n")
    print(f"\n{len(problems)} nights packed into {args.out}, {len(failures)} failed (rerun to retry them)")


if __name__ == "__main__":
    main()
