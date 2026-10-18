"""Look at one raw recording step by step:  python -m sleep_staging.explore [recording]

Prints the recording's details and saves these to results/sleep_staging/explore/<recording>/:
  01_raw_signals.png      every channel, first 10 seconds
  02_filtered.png         before and after a 1-40 Hz band-pass filter
  03_rereferenced.png     before and after average re-referencing
  04_power_spectrum.png   power spectral density, overall and per sleep stage
  05_time_frequency.png   Morlet wavelet power through the night, with the hypnogram
  signals_first_10_min.csv   the raw samples as a table

The default recording is SC4001E; pass another name such as ST7011J to change it.
"""

import sys

import mne
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.signal import welch

from common.style import BLUE, BLUE_RAMP, INK_MUTED, INK_SECONDARY, ORANGE, plt
from sleep_staging.plots import RESULTS_DIR
from sleep_staging.preprocess import EPOCH_SAMPLES, STAGE_NAMES, UNSCORED, WAKE_MARGIN_EPOCHS, epoch_labels, find_recordings

EEG_CHANNELS = ["EEG Fpz-Cz", "EEG Pz-Oz"]
FILTERED_CHANNELS = EEG_CHANNELS + ["EOG horizontal"]
MICROVOLT_CHANNELS = FILTERED_CHANNELS + ["EMG submental"]  # the rest are not voltages
WINDOW_SECONDS = 10
CSV_MINUTES = 10
STAGE_COLORS = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4"]
MORLET_FREQS = np.arange(1, 41)


def window(raw: mne.io.BaseRaw, channel: str) -> tuple[np.ndarray, np.ndarray]:
    """The first WINDOW_SECONDS of one channel, in microvolts where that applies."""
    data, times = raw.get_data(picks=[channel], tmax=WINDOW_SECONDS, return_times=True)
    return times, data[0] * (1e6 if channel in MICROVOLT_CHANNELS else 1)


def stacked_panels(channels: list[str], title: str, height: float = 1.5):
    figure, axes = plt.subplots(len(channels), 1, figsize=(10, height * len(channels) + 0.8),
                                sharex=True, layout="constrained", squeeze=False)
    figure.suptitle(title, x=0.01, ha="left", fontsize=14, fontweight="bold")
    for axis, channel in zip(axes[:, 0], channels):
        unit = "µV" if channel in MICROVOLT_CHANNELS else "recorded units"
        axis.set_title(f"{channel} ({unit})", fontsize=10)
        axis.grid(axis="y")
    axes[-1, 0].set_xlabel("Seconds from start of recording")
    return figure, axes[:, 0]


def plot_raw(raw: mne.io.BaseRaw, out_dir) -> None:
    figure, axes = stacked_panels(raw.ch_names, f"Raw signals, first {WINDOW_SECONDS} seconds", height=1.25)
    for axis, channel in zip(axes, raw.ch_names):
        axis.plot(*window(raw, channel), color=BLUE, linewidth=1)
    figure.savefig(out_dir / "01_raw_signals.png", dpi=150)
    plt.close(figure)


def plot_before_after(before, after, channels, title, labels, filename, out_dir) -> None:
    figure, axes = stacked_panels(channels, title, height=1.9)
    for axis, channel in zip(axes, channels):
        axis.plot(*window(before, channel), color=INK_MUTED, linewidth=1, label=labels[0])
        axis.plot(*window(after, channel), color=BLUE, linewidth=1.4, label=labels[1])
    axes[0].legend(frameon=False, loc="upper right", ncols=2, labelcolor=INK_SECONDARY)
    figure.savefig(out_dir / filename, dpi=150)
    plt.close(figure)


def to_db(power: np.ndarray) -> np.ndarray:
    return 10 * np.log10(power)


def plot_power_spectrum(eeg: mne.io.BaseRaw, labels: np.ndarray, out_dir) -> None:
    """Welch power spectrum: both EEG channels over the whole recording, then one channel per stage."""
    figure, (left, right) = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True, layout="constrained")
    figure.suptitle("Power spectral density (Welch)", x=0.01, ha="left", fontsize=14, fontweight="bold")

    spectrum = eeg.compute_psd(method="welch", fmin=0.5, fmax=40.0, verbose="ERROR")
    for power, channel, color in zip(spectrum.get_data() * 1e12, eeg.ch_names, (BLUE, ORANGE)):
        left.plot(spectrum.freqs, to_db(power), color=color, linewidth=2, label=channel)
    left.set_title("Whole recording, both EEG channels")
    left.set_ylabel("Power (dB, µV²/Hz)")

    # Per stage: cut the first channel into 30-second epochs and average the spectra within each stage.
    signal = eeg.get_data(picks=[0])[0] * 1e6
    epochs = signal[: len(labels) * EPOCH_SAMPLES].reshape(len(labels), EPOCH_SAMPLES)
    freqs, power = welch(epochs, fs=eeg.info["sfreq"], nperseg=400, axis=-1)
    keep = (freqs >= 0.5) & (freqs <= 40)
    for stage, (name, color) in enumerate(zip(STAGE_NAMES, STAGE_COLORS)):
        if (labels == stage).any():
            right.plot(freqs[keep], to_db(power[labels == stage].mean(axis=0)[keep]), color=color, linewidth=2, label=name)
    right.set_title(f"{eeg.ch_names[0]}, by sleep stage")

    for axis in (left, right):
        axis.set_xlabel("Frequency (Hz)")
        axis.grid(axis="y")
        axis.legend(frameon=False, loc="upper right", labelcolor=INK_SECONDARY)
    figure.savefig(out_dir / "04_power_spectrum.png", dpi=150)
    plt.close(figure)


def plot_time_frequency(eeg: mne.io.BaseRaw, labels: np.ndarray, out_dir) -> None:
    """Morlet wavelet power for every 30-second epoch of the sleep period, under the hypnogram."""
    asleep = np.flatnonzero(labels > 0)
    first = max(0, asleep[0] - WAKE_MARGIN_EPOCHS)
    last = min(len(labels), asleep[-1] + 1 + WAKE_MARGIN_EPOCHS)
    signal = eeg.get_data(picks=[0])[0] * 1e6
    epochs = signal[first * EPOCH_SAMPLES : last * EPOCH_SAMPLES].reshape(-1, 1, EPOCH_SAMPLES)
    labels = labels[first:last]

    power = []
    for start in range(0, len(epochs), 64):  # in chunks, to keep memory use small
        chunk = mne.time_frequency.tfr_array_morlet(
            epochs[start : start + 64], sfreq=eeg.info["sfreq"], freqs=MORLET_FREQS,
            n_cycles=MORLET_FREQS / 2.0, output="power", verbose="ERROR",
        )
        power.append(chunk[:, 0].mean(axis=-1))  # average over the 30 seconds
    power_db = to_db(np.concatenate(power)).T  # (frequencies, epochs)
    hours = len(labels) * 30 / 3600

    figure, (top, bottom) = plt.subplots(2, 1, figsize=(11, 6), sharex=True, layout="constrained", height_ratios=[1, 2.2])
    figure.suptitle(f"Time-frequency analysis of the sleep period ({eeg.ch_names[0]}, Morlet wavelets)",
                    x=0.01, ha="left", fontsize=14, fontweight="bold")

    order = ["W", "REM", "N1", "N2", "N3"]  # conventional hypnogram order, top to bottom
    height = np.array([len(order) - 1 - order.index(stage) for stage in STAGE_NAMES])
    scored = labels != UNSCORED
    top.step(np.flatnonzero(scored) * 30 / 3600, height[labels[scored]], where="post", color=BLUE, linewidth=1.2)
    top.set_yticks(range(len(order)), order[::-1])
    top.set_title("Sleep stage (expert scoring)")
    top.grid(axis="y")

    low, high = np.percentile(power_db, [2, 98])
    image = bottom.imshow(power_db, aspect="auto", origin="lower", vmin=low, vmax=high,
                          extent=(0, hours, MORLET_FREQS[0] - 0.5, MORLET_FREQS[-1] + 0.5),
                          cmap=LinearSegmentedColormap.from_list("blue", BLUE_RAMP))
    bottom.set_title("Power at each frequency")
    bottom.set_ylabel("Frequency (Hz)")
    bottom.set_xlabel("Hours from start of the sleep period")
    # Attached to both panels so their time axes stay lined up.
    figure.colorbar(image, ax=[top, bottom], label="Power (dB, µV²)", pad=0.01, shrink=0.66, anchor=(0, 0))
    figure.savefig(out_dir / "05_time_frequency.png", dpi=150)
    plt.close(figure)


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "SC4001E"
    pairs = {psg.name[:7]: (psg, hypnogram) for psg, hypnogram in find_recordings()}
    if name not in pairs:
        sys.exit(f"Unknown recording {name}. Examples: {', '.join(list(pairs)[:5])}")
    psg, hypnogram = pairs[name]
    out_dir = RESULTS_DIR / "explore" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load and inspect
    raw = mne.io.read_raw_edf(psg, preload=True, verbose="ERROR")
    labels = epoch_labels(hypnogram, raw.n_times // EPOCH_SAMPLES)
    print(f"File:           {psg.name}")
    print(f"Channels ({raw.info['nchan']}):   {', '.join(raw.ch_names)}")
    print(f"Sampling rate:  {raw.info['sfreq']:.0f} Hz")
    print(f"Duration:       {raw.times[-1]:,.2f} seconds ({raw.times[-1] / 3600:.1f} hours)")
    print(f"Samples:        {raw.n_times:,} per channel")
    print(f"Recorded:       {raw.info['meas_date']:%Y-%m-%d %H:%M}")
    counts = ", ".join(f"{stage} {(labels == i).sum()}" for i, stage in enumerate(STAGE_NAMES))
    print(f"30-s epochs:    {counts}, unscored {(labels == UNSCORED).sum()}")

    # 2. Raw signals
    plot_raw(raw, out_dir)

    # 3. Band-pass filter. Only the EEG and EOG channels: filtering temperature or the event marker means nothing.
    filtered = raw.copy().pick(FILTERED_CHANNELS).filter(l_freq=1.0, h_freq=40.0, verbose="ERROR")
    plot_before_after(raw, filtered, FILTERED_CHANNELS, "Band-pass filter, 1-40 Hz",
                      ("raw", "filtered"), "02_filtered.png", out_dir)

    # 4. Average re-reference across the two EEG channels
    eeg_filtered = filtered.copy().pick(EEG_CHANNELS)
    rereferenced = eeg_filtered.copy().set_eeg_reference("average", verbose="ERROR")
    plot_before_after(eeg_filtered, rereferenced, EEG_CHANNELS, "Average re-referencing of the EEG channels",
                      ("filtered", "re-referenced"), "03_rereferenced.png", out_dir)

    # 5. Export samples as a table
    data, times = raw.get_data(tmax=CSV_MINUTES * 60, return_times=True)
    table = pd.DataFrame(data.T, columns=raw.ch_names)
    table.insert(0, "Time_Seconds", times)
    table.to_csv(out_dir / f"signals_first_{CSV_MINUTES}_min.csv", index=False)
    print(f"\nFirst rows of the exported table:\n{table.head().to_string()}")

    # 6. Power spectrum and 7. time-frequency analysis, on 0.5-40 Hz EEG
    eeg = raw.copy().pick(EEG_CHANNELS).filter(l_freq=0.5, h_freq=40.0, verbose="ERROR")
    plot_power_spectrum(eeg, labels, out_dir)
    plot_time_frequency(eeg, labels, out_dir)

    print(f"\nSaved to {out_dir.resolve()}:")
    for path in sorted(out_dir.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
