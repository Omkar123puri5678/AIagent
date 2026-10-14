"""Turn the PhysioNet Challenge 2026 recordings into one row of features per night:

    python hsp_features.py <path to the Challenge 2026 data>

The recordings (a subset of the Human Sleep Project) are about 1.2 TB, so they are streamed
one night at a time and only a small table, hsp_features.csv, is kept. Join it to the dataset's demographics.csv (age, sex,
BMI and the Cognitive_Impairment label) on BidsFolder and SessionID.

For every night this reads the CAISR annotations and the raw signals and measures
  - sleep architecture: time asleep, efficiency, latencies, WASO, share of each stage,
  - events per hour of sleep: breathing events, arousals, limb movements,
  - EEG: relative band power in N2, N3 and REM, REM slowing, Hjorth parameters,
  - ECG: heart rate and heart rate variability during sleep,
  - SpO2: mean, minimum, time below 90% and desaturations per hour,
  - breathing rate and its variability, and chin muscle tone in REM relative to NREM.
A signal the recording does not have leaves its columns empty.

The table is saved every few recordings and a rerun skips the nights already in it, so a
long run can be split with --shard (e.g. --shard 1/3, 2/3, 3/3 on three machines).
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from fractions import Fraction
from pathlib import Path

import edfio
import numpy as np
import pandas as pd
from scipy.ndimage import maximum_filter1d, uniform_filter1d
from scipy.signal import butter, find_peaks, resample_poly, sosfiltfilt, welch

EPOCH_SECONDS = 30
# CAISR stage codes, as used by the organisers' example code; anything else is unscored.
N3, N2, N1, REM, WAKE = 1, 2, 3, 4, 5
STAGE_CODES = {"n1": N1, "n2": N2, "n3": N3, "rem": REM}

EEG_RATE = 100
EEG_BANDS = {"delta": (0.5, 4), "theta": (4, 8), "alpha": (8, 12), "sigma": (12, 15), "beta": (15, 30)}
# Scalp site and its mastoid reference, in order of preference.
EEG_DERIVATIONS = [("c4", "m1"), ("c3", "m2"), ("f4", "m1"), ("f3", "m2")]
ECG_RATE = 200
BREATHING_RATE = 10
TINY = 1e-12

# Channel names differ between hospitals; these are the organisers' aliases (channel_table.csv).
ECG_NAMES = ["ecg", "ekg", "ecg-la", "ecg-v1", "ecg-v2", "ecg-ll", "ecg-ra", "ecg l", "ecgl", "ecg i",
             "ecg ii", "ekg-r", "ekg-l", "ecg1", "ecg2", "ecg ref-ecg1", "ecg ref-ecg2", "ecg ref-ecg3", "ekg-e1"]
SPO2_NAMES = ["spo2", "sao2", "osat", "o2sat", "o2 sat", "o2-sat", "o2-saturation", "gnd-spo2"]
EFFORT_NAMES = ["abd", "abdomen", "abdominal", "abd res", "abdo res", "abdo", "effort abd", "abdominal effort",
                "abdom", "respitrace abdom", "abd-e1", "chest", "thorax", "thoracic", "thor res", "thoracic res",
                "chest res", "thor", "effort tho", "thoracic effort", "respitrace chest", "chest-e1"]
CHIN_NAMES = ["chin1-chin2", "chin1chin2", "chin", "china", "chin_emg", "emg.subm", "emg", "chin emg",
              "emg chin", "emg-chin", "chin1-chin3", "chin-a", "emg1-emg2"]


def load(path: Path) -> dict:
    """Channel name -> signal. The samples are only read from disk when a signal is used."""
    edf = edfio.read_edf(path, lazy_load_data=True)
    clean = lambda label: label.lower().replace("_pds", "").replace("_eg", "").replace(":", "-").strip()
    return {clean(signal.label): signal for signal in edf.signals}


def find(signals: dict, names: list[str]) -> tuple[np.ndarray, float] | None:
    """Samples and sampling rate of the first of these channels that the recording has."""
    for name in names:
        if name in signals:
            signal = signals[name]
            return np.nan_to_num(np.asarray(signal.data, dtype=np.float64)), float(signal.sampling_frequency)
    return None


def find_eeg(signals: dict) -> tuple[np.ndarray, float] | None:
    for site, mastoid in EEG_DERIVATIONS:
        references = [mastoid, mastoid.replace("m", "a")]
        names = [f"{prefix}{site}{separator}{reference}" for reference in references
                 for prefix in ("", "eeg ") for separator in ("-", "", "/")]
        found = find(signals, names + [f"{site}-avg"])
        if found:
            return found
        active, reference = find(signals, [site]), find(signals, references)
        if active and reference and active[1] == reference[1] and len(active[0]) == len(reference[0]):
            return active[0] - reference[0], active[1]
    return None


def resample(x: np.ndarray, rate: float, new_rate: float) -> np.ndarray:
    ratio = Fraction(new_rate / rate).limit_denominator(1000)
    return resample_poly(x, ratio.numerator, ratio.denominator)


def bandpass(x: np.ndarray, rate: float, low: float, high: float) -> np.ndarray:
    return sosfiltfilt(butter(4, [low, high], btype="band", fs=rate, output="sos"), x)


def cut_epochs(x: np.ndarray, rate: int, stages: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(epochs, samples) and the stage of each, trimmed to whichever of the two ends first."""
    samples = rate * EPOCH_SECONDS
    n = min(len(stages), len(x) // samples)
    return x[: n * samples].reshape(n, samples), stages[:n]


def is_sleep(stages: np.ndarray) -> np.ndarray:
    return (stages >= N3) & (stages <= REM)


def epoch_stages(caisr: dict) -> np.ndarray:
    """The CAISR stage at the start of every 30-second epoch."""
    found = find(caisr, ["stage_caisr"])
    if found is None:
        raise ValueError("no stage_caisr channel")
    stage, rate = found
    # CAISR stores one value per epoch (1/30 Hz); the small offset keeps rounding from picking the previous one.
    n_epochs = int((len(stage) / rate + 1e-6) // EPOCH_SECONDS)
    return stage[(np.arange(n_epochs) * EPOCH_SECONDS * rate + 1e-6).astype(int)].round().astype(int)


def architecture_features(stages: np.ndarray) -> dict:
    asleep = is_sleep(stages)
    scored = asleep | (stages == WAKE)
    if not asleep.any():
        raise ValueError("no sleep epochs")
    minutes = EPOCH_SECONDS / 60
    onset, final = np.flatnonzero(asleep)[[0, -1]]
    rem = np.flatnonzero(stages == REM)
    during = stages[onset : final + 1]
    during = during[scored[onset : final + 1]]
    hours = asleep.sum() * minutes / 60
    features = {
        "tst_min": asleep.sum() * minutes,
        "sleep_efficiency": asleep.sum() / scored.sum(),
        "sleep_latency_min": onset * minutes,
        "rem_latency_min": (rem[0] - onset) * minutes if len(rem) else np.nan,
        "waso_min": (during == WAKE).sum() * minutes,
        "stage_changes_per_hour": np.count_nonzero(np.diff(during)) / hours,
        "awakenings_per_hour": np.count_nonzero((during[1:] == WAKE) & (during[:-1] != WAKE)) / hours,
    }
    for name, code in STAGE_CODES.items():
        features[f"{name}_fraction"] = (stages == code).sum() / asleep.sum()
    return features


def event_features(caisr: dict, stages: np.ndarray) -> dict:
    """Breathing events, arousals and limb movements per hour of sleep."""
    hours = is_sleep(stages).sum() * EPOCH_SECONDS / 3600
    features = {}
    for name in ("resp", "arousal", "limb"):
        found = find(caisr, [f"{name}_caisr"])
        if found:
            active = ((found[0] > 0) & (found[0] < 9)).astype(np.int8)  # 9 marks unscored samples
            features[f"{name}_events_per_hour"] = np.count_nonzero(np.diff(active, prepend=0) == 1) / hours
    found = find(caisr, ["resp_caisr"])
    if found:  # breathing events carry a type code from 1 to 5 (apnea types, hypopnea, ...)
        kind = found[0].round().astype(int)
        starts = kind[np.flatnonzero(np.diff(kind, prepend=0) != 0)]
        for code in range(1, 6):
            features[f"resp_type{code}_per_hour"] = np.count_nonzero(starts == code) / hours
    return features


def eeg_features(signals: dict, stages: np.ndarray) -> dict:
    found = find_eeg(signals)
    if found is None:
        return {}
    x = bandpass(resample(*found, EEG_RATE), EEG_RATE, 0.3, 35.0)
    x, stages = cut_epochs(x, EEG_RATE, stages)
    spread = x.std(axis=1)
    clean = (spread > 0.1 * np.median(spread)) & (spread < 5 * np.median(spread))  # drop flat and artifact epochs

    freqs, power = welch(x, fs=EEG_RATE, nperseg=EEG_RATE * 2, axis=-1)
    total = power[:, (freqs >= 0.5) & (freqs <= 30)].sum(axis=1) + TINY
    relative = {band: power[:, (freqs >= low) & (freqs < high)].sum(axis=1) / total
                for band, (low, high) in EEG_BANDS.items()}

    features = {}
    for stage in ("n2", "n3", "rem"):
        chosen = clean & (stages == STAGE_CODES[stage])
        if chosen.sum() >= 10:  # at least 5 minutes
            for band in EEG_BANDS:
                features[f"eeg_{stage}_{band}"] = np.log10(relative[band][chosen].mean() + TINY)
    chosen = clean & (stages == REM)
    if chosen.sum() >= 10:
        slow = (relative["delta"] + relative["theta"])[chosen].mean()
        fast = (relative["alpha"] + relative["beta"])[chosen].mean()
        features["eeg_rem_slowing"] = np.log10(slow / (fast + TINY) + TINY)
    chosen = clean & is_sleep(stages)
    if chosen.sum() >= 10:
        dx = np.diff(x[chosen], axis=-1)
        mobility = np.sqrt(dx.var(axis=-1) / (x[chosen].var(axis=-1) + TINY))
        complexity = np.sqrt(np.diff(dx, axis=-1).var(axis=-1) / (dx.var(axis=-1) + TINY)) / (mobility + TINY)
        features |= {"eeg_mobility": mobility.mean(), "eeg_complexity": complexity.mean()}
    return features


def r_peaks(ecg: np.ndarray) -> np.ndarray:
    """Sample positions of the heartbeats in an ECG sampled at ECG_RATE."""
    energy = uniform_filter1d(bandpass(ecg, ECG_RATE, 5.0, 25.0) ** 2, size=ECG_RATE // 10)
    window = ECG_RATE * 5
    typical_beat = np.median(energy[: len(energy) // window * window].reshape(-1, window).max(axis=1))
    peaks, _ = find_peaks(energy, height=0.3 * typical_beat, distance=int(0.3 * ECG_RATE))
    return peaks


def ecg_features(signals: dict, stages: np.ndarray) -> dict:
    found = find(signals, ECG_NAMES)
    if found is None:
        return {}
    peaks = r_peaks(resample(*found, ECG_RATE))
    times, rr = peaks[:-1] / ECG_RATE, np.diff(peaks) / ECG_RATE
    epoch = np.minimum((times // EPOCH_SECONDS).astype(int), len(stages) - 1)
    typical = np.median(rr[(rr > 0.3) & (rr < 2.0)])
    asleep = is_sleep(stages)[epoch]
    good = (rr > 0.3) & (rr < 2.0) & (np.abs(rr - typical) < 0.4 * typical) & asleep
    # With a working lead nearly every interval looks normal; if not, the peaks are noise, not heartbeats.
    if good.sum() < 300 or good.sum() < 0.8 * asleep.sum():
        return {}
    change = np.diff(rr)[good[1:] & good[:-1]] * 1000  # ms, between neighbouring normal beats only
    features = {
        "hr_mean": 60 / rr[good].mean(),
        "hrv_sdnn_ms": rr[good].std() * 1000,
        "hrv_rmssd_ms": np.sqrt(np.mean(change**2)),
        "hrv_pnn50": np.mean(np.abs(change) > 50),
    }

    # LF/HF balance: median over 5-minute stretches of sleep in which nearly every beat is usable.
    ratios = []
    for start in np.arange(0, times[-1] - 300, 300):
        inside = (times >= start) & (times < start + 300) & good
        if rr[inside].sum() < 0.9 * 300:
            continue
        even = np.interp(np.arange(start, start + 300, 0.25), times[inside], rr[inside])
        freqs, power = welch(even - even.mean(), fs=4, nperseg=512)
        low = power[(freqs >= 0.04) & (freqs < 0.15)].sum()
        high = power[(freqs >= 0.15) & (freqs < 0.4)].sum()
        ratios.append(np.log10((low + TINY) / (high + TINY)))
    if ratios:
        features["hrv_log_lf_hf"] = np.median(ratios)
    return features


def spo2_features(signals: dict, stages: np.ndarray) -> dict:
    found = find(signals, SPO2_NAMES)
    if found is None or found[1] < 1:
        return {}
    x, rate = found[0], int(round(found[1]))
    x = x[: len(x) // rate * rate].reshape(-1, rate).mean(axis=1)  # one value per second
    if np.median(x) <= 1.5:  # stored as a fraction rather than a percentage
        x = x * 100
    n = min(len(x), len(stages) * EPOCH_SECONDS)
    x, asleep = x[:n], np.repeat(is_sleep(stages), EPOCH_SECONDS)[:n]
    valid = (x >= 50) & (x <= 100) & asleep
    if valid.sum() < 600:
        return {}
    # A desaturation: 3 points or more below the highest value of the previous two minutes.
    filled = np.where((x >= 50) & (x <= 100), x, np.median(x[valid]))
    baseline = maximum_filter1d(filled, size=120, origin=59)
    dipped = ((baseline - filled >= 3) & valid).astype(np.int8)
    return {
        "spo2_mean": x[valid].mean(),
        "spo2_min": np.percentile(x[valid], 1),
        "spo2_fraction_below_90": np.mean(x[valid] < 90),
        "spo2_desaturations_per_hour": np.count_nonzero(np.diff(dipped, prepend=0) == 1) / (valid.sum() / 3600),
    }


def breathing_features(signals: dict, stages: np.ndarray) -> dict:
    found = find(signals, EFFORT_NAMES)
    if found is None or found[1] < BREATHING_RATE:
        return {}
    x, stages = cut_epochs(resample(*found, BREATHING_RATE), BREATHING_RATE, stages)
    freqs, power = welch(x, fs=BREATHING_RATE, nperseg=x.shape[1], axis=-1)
    band = (freqs >= 0.1) & (freqs <= 0.6)
    breaths_per_min = 60 * freqs[band][power[:, band].argmax(axis=1)]
    chosen = is_sleep(stages) & (x.std(axis=1) > 0)
    if chosen.sum() < 10:
        return {}
    return {"breathing_rate_mean": breaths_per_min[chosen].mean(), "breathing_rate_std": breaths_per_min[chosen].std()}


def chin_features(signals: dict, stages: np.ndarray) -> dict:
    found = find(signals, CHIN_NAMES)
    if found is None or found[1] < 100:
        return {}
    x, stages = cut_epochs(bandpass(resample(*found, 100), 100, 10.0, 40.0), 100, stages)
    tone = np.sqrt((x**2).mean(axis=1))
    rem, nrem = stages == REM, is_sleep(stages) & (stages != REM)
    if rem.sum() < 10 or nrem.sum() < 10:
        return {}
    # Muscle tone normally drops in REM; a ratio near or above 0 means it did not.
    return {"chin_rem_vs_nrem_tone": np.log10((np.median(tone[rem]) + TINY) / (np.median(tone[nrem]) + TINY))}


def process(job: tuple[Path, str, str, str]) -> dict:
    root, site, patient, session = job
    name = f"{patient}_ses-{session}"
    row = {"record": name, "SiteID": site, "BidsFolder": patient, "SessionID": session}

    caisr = load(root / "algorithmic_annotations" / site / f"{name}_caisr_annotations.edf")
    stages = epoch_stages(caisr)
    row |= architecture_features(stages) | event_features(caisr, stages)

    signals = load(root / "physiological_data" / site / f"{name}.edf")
    problems = []
    for extract in (eeg_features, ecg_features, spo2_features, breathing_features, chin_features):
        try:  # one bad channel should not cost the rest of the night
            row |= extract(signals, stages)
        except Exception as error:
            problems.append(f"{extract.__name__}: {error}")
    row["problems"] = "; ".join(problems)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="One row of features per night of the Challenge 2026 data.")
    parser.add_argument("data", type=Path, help="folder holding demographics.csv, or any folder above it")
    parser.add_argument("--out", type=Path, default=Path("hsp_features.csv"))
    parser.add_argument("--shard", default="1/1", help="do part K of N, e.g. 2/3")
    parser.add_argument("--limit", type=int, help="stop after this many recordings (for a quick try)")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    demographics = next(iter(sorted(args.data.rglob("demographics.csv"))), None)
    if demographics is None:
        sys.exit(f"No demographics.csv found under {args.data}")
    root = demographics.parent
    nights = pd.read_csv(demographics)[["SiteID", "BidsFolder", "SessionID"]].drop_duplicates()
    jobs = [(root, str(row.SiteID), str(row.BidsFolder), str(row.SessionID)) for row in nights.itertuples()]

    part, parts = map(int, args.shard.split("/"))
    jobs = jobs[part - 1 :: parts]
    rows = pd.read_csv(args.out, dtype={"SessionID": str}).to_dict("records") if args.out.exists() else []
    done = {row["record"] for row in rows}
    jobs = [job for job in jobs if f"{job[2]}_ses-{job[3]}" not in done][: args.limit]
    print(f"{len(jobs)} recordings to process from {root} ({len(done)} already in {args.out})")

    failures = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process, job): f"{job[2]}_ses-{job[3]}" for job in jobs}
        for finished, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as error:
                failures.append((futures[future], error))
                print(f"FAILED {futures[future]}: {error}")
            if finished % 10 == 0 or finished == len(futures):
                pd.DataFrame(rows).to_csv(args.out, index=False, float_format="%.5g")
                print(f"  {finished}/{len(futures)}")

    if rows:
        table = pd.DataFrame(rows)
        filled = table.drop(columns=["record", "SiteID", "BidsFolder", "SessionID", "problems"]).notna().mean()
        print(f"\n{len(table)} recordings x {len(filled)} features saved to {args.out}")
        print("share of recordings with each feature:")
        print(filled.round(2).to_string())
    if failures:
        print(f"\n{len(failures)} recordings failed (rerun to retry them)")


if __name__ == "__main__":
    main()
