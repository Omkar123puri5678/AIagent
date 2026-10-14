"""Use the cognitive impairment model on a new person.

    python predict.py train data/hsp_large
        Fits the logistic regression from hsp_baseline.py on everyone in the table and saves it
        to checkpoints/cognitive_model.joblib.

    python predict.py night <recording.edf> <caisr_annotations.edf> --age 67 --sex female
        Measures the night the same way hsp_features.py does and prints the person's risk.

    python predict.py table data/hsp_large/hsp_features.csv <record> --age 67 --sex female
        The same for a night already in a feature table (no raw recording needed).

The model is trained with the rare impaired class weighted up, so its raw probability is
not the chance of impairment. The output instead places the person among the training
people (overall and within 2 years of their age), gives their risk group with the share of
that group who were later impaired, and lists the features that moved their risk most.
The groups and their rates come from cross-validated predictions, so they are not flattered
by the model having seen those people.
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_val_predict

import hsp_features
from hsp_baseline import FOLDS, SEED, load_table, make_models
from hsp_explain import HIGH_SHARE, MODERATE_SHARE

MODEL_FILE = Path("checkpoints/cognitive_model.joblib")
QUANTILES = np.linspace(0, 1, 101)
AGE_BAND = 5  # years; same-age comparisons use risk percentiles stored per band, not individual people
REASONS = 5
SHORT_NIGHT_MINUTES = 120


def train(data_dir: Path) -> None:
    table, names = load_table(data_dir)
    labels = table["label"]
    model, _ = make_models(labels.mean())["logistic regression"]
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    risk = cross_val_predict(model, table[names], labels, cv=folds.split(table, labels.astype(str) + table["SiteID"]),
                             method="predict_proba")[:, 1]
    high, moderate = np.quantile(risk, [1 - HIGH_SHARE, 1 - HIGH_SHARE - MODERATE_SHARE])
    groups = np.where(risk >= high, "high", np.where(risk >= moderate, "moderate", "low"))

    # Only summaries of the training people are kept: percentiles, medians and group rates.
    bands = (table["Age"] // AGE_BAND * AGE_BAND).astype(int)
    saved = {
        "model": clone(model).fit(table[names], labels),
        "names": names,
        "medians": table[names].median(),
        "risk_quantiles": np.quantile(risk, QUANTILES),
        "band_quantiles": {band: np.quantile(risk[bands == band], QUANTILES)
                           for band in sorted(bands.unique()) if (bands == band).sum() >= 50},
        "thresholds": {"high": high, "moderate": moderate},
        "group_rates": {group: float(labels[groups == group].mean()) for group in ("low", "moderate", "high")},
        "overall_rate": float(labels.mean()),
        "people": len(table),
    }
    MODEL_FILE.parent.mkdir(exist_ok=True)
    joblib.dump(saved, MODEL_FILE)
    print(f"Trained on {len(table)} people ({labels.sum()} later impaired) with {len(names)} features; saved {MODEL_FILE}")
    print("later impaired by risk group (cross-validated): "
          + ", ".join(f"{g} {100 * r:.1f}%" for g, r in saved["group_rates"].items()))


def measure_night(recording: Path, caisr_file: Path) -> dict:
    """The per-night features of hsp_features.process, for files outside the Challenge folder layout."""
    stages = hsp_features.epoch_stages(caisr := hsp_features.load(caisr_file))
    row = hsp_features.architecture_features(stages) | hsp_features.event_features(caisr, stages)
    signals = hsp_features.load(recording)
    for extract in (hsp_features.eeg_features, hsp_features.ecg_features, hsp_features.spo2_features,
                    hsp_features.breathing_features, hsp_features.chin_features):
        try:
            row |= extract(signals, stages)
        except Exception as error:
            print(f"  skipped {extract.__name__}: {error}")
    return row


def percentile(quantiles: np.ndarray, risk: float) -> float:
    return float(100 * np.interp(risk, quantiles, QUANTILES))


def report(features: dict, age: float, sex: str) -> None:
    if not MODEL_FILE.exists():
        sys.exit(f"No {MODEL_FILE}. Run: python predict.py train data/hsp_large")
    saved = joblib.load(MODEL_FILE)
    names, model = saved["names"], saved["model"]
    person = pd.DataFrame([{**features, "Age": age, "male": float(sex.lower().startswith("m"))}]).reindex(columns=names)
    person = person.apply(pd.to_numeric, errors="coerce")
    risk = float(model.predict_proba(person)[0, 1])

    group = "high" if risk >= saved["thresholds"]["high"] else "moderate" if risk >= saved["thresholds"]["moderate"] else "low"
    band = int(age // AGE_BAND * AGE_BAND)
    lines = [
        f"age {age:.0f}, {sex.lower()} | {person.iloc[0, :-2].notna().sum()} of {len(names) - 2} sleep features measured",
        f"risk score {risk:.2f} (ranking only, not a probability)",
        f"higher than {percentile(saved['risk_quantiles'], risk):.0f}% of the {saved['people']} training people",
    ]
    if band in saved["band_quantiles"]:
        lines.append(f"higher than {percentile(saved['band_quantiles'][band], risk):.0f}% of training people aged {band}-{band + AGE_BAND - 1}")
    lines.append(f"risk group: {group.upper()} (in training, {100 * saved['group_rates'][group]:.1f}% of this group were later "
                 f"impaired, against {100 * saved['overall_rate']:.1f}% of everyone)")

    # A logistic regression's SHAP values: each feature's coefficient times its standardised value (log-odds).
    imputer, scaler, regression = (step for _, step in model.steps)
    contributions = regression.coef_[0] * scaler.transform(imputer.transform(person))[0]
    if features.get("tst_min", np.inf) < SHORT_NIGHT_MINUTES:
        lines.append(f"warning: only {features['tst_min']:.0f} minutes of sleep were scored, so this result is unreliable")
    lines.append("\nwhat moved the risk most:")
    for column in np.argsort(-np.abs(contributions))[:REASONS]:
        name, value = names[column], person.iloc[0, column]
        shown = "missing" if pd.isna(value) else f"{value:.3g}"
        effect = "raises" if contributions[column] > 0 else "lowers"
        lines.append(f"  {name:<30} {shown:>8}  (typical {saved['medians'][name]:.3g})  {effect} risk")
    lines.append("\nA research model (age-conditioned AUROC 0.667), not a diagnosis.")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final cognitive impairment model, or use it on one night.")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train").add_argument("data", type=Path, help="folder with hsp_features.csv and demographics.csv")
    night = commands.add_parser("night")
    night.add_argument("recording", type=Path)
    night.add_argument("caisr", type=Path, help="the night's CAISR annotations EDF")
    table = commands.add_parser("table")
    table.add_argument("features", type=Path, help="a feature table from hsp_features.py")
    table.add_argument("record", help="the night's record name, e.g. sub-I0002150028100_ses-1")
    for command in (night, table):
        command.add_argument("--age", type=float, required=True)
        command.add_argument("--sex", required=True, choices=["female", "male"])
    args = parser.parse_args()

    if args.command == "train":
        train(args.data)
    elif args.command == "night":
        report(measure_night(args.recording, args.caisr), args.age, args.sex)
    else:
        rows = pd.read_csv(args.features)
        match = rows[rows["record"] == args.record]
        if match.empty:
            sys.exit(f"{args.record} is not in {args.features}")
        report(match.iloc[0].to_dict(), args.age, args.sex)


if __name__ == "__main__":
    main()
