"""First models for predicting future cognitive impairment:  python -m cognitive.baseline [data folder]

Trains on the per-night feature table from cognitive/features.py, joined to the Challenge 2026
demographics (age, sex and the Cognitive_Impairment label), and compares
  - age alone (the reference: older people are at higher risk whatever their sleep),
  - logistic regression, a random forest and XGBoost on all features.

Each model is scored two ways:
  - 5-fold cross-validation, repeated, so every person is predicted by a model that never
    saw them, and
  - leave-one-hospital-out: train on two hospitals, test on the third, which is closer to
    the Challenge, whose test hospitals are not in the training data.
The headline score is the Challenge's age-conditioned AUROC: of all pairs of one person who
developed impairment and one who did not, at most two years apart in age, the share in
which the model gave the first the higher risk. 0.5 is guessing. Impairment is more common
at some hospitals than others, so a model can beat 0.5 on that score just by recognising
the hospital; the "same hospital" column only compares people from the same hospital and
cannot be gained that way.

For the yes/no scores each model flags the 20% of people it rates at highest risk.

Broken recordings are cleaned first: nights with under two hours of scored sleep are dropped,
EEG features from a flat channel are treated as missing, and the linear model caps every
feature at the 1st and 99th percentile of its training data, so a few impossible values
(10,000 limb movements an hour) cannot dominate it.

The data folder defaults to data/hsp (the small training set); data/hsp_large holds the
large one. Saves <folder name>_baseline_report.txt and <folder name>_baseline_curves.png
to results/cognitive/.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from cognitive import RESULTS_DIR
from common import style
from common.style import plt

DEFAULT_DATA_DIR = Path("data/hsp")
KEYS = ["SiteID", "BidsFolder", "SessionID"]

AGE_GAP = 2  # years; the Challenge compares people at most this far apart in age
FLAGGED_SHARE = 0.2  # the yes/no scores flag this share of people, the ones at highest risk
FOLDS, REPEATS = 5, 10
SEED = 0
MIN_SLEEP_MINUTES = 120  # shorter nights are too little sleep to measure; dropped
FLAT_EEG = -11.9  # features.py writes log10(1e-12) = -12 when an EEG channel carries no power
CLIP_QUANTILES = (0.01, 0.99)


class Clip(BaseEstimator, TransformerMixin):
    """Caps each feature at quantiles learned from the training data (missing values stay missing)."""

    def fit(self, X, y=None):
        self.low_, self.high_ = np.nanquantile(np.asarray(X, dtype=float), CLIP_QUANTILES, axis=0)
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.low_, self.high_)


def load_table(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, list[str]]:
    """One row per person with features, age, sex and the label, plus the feature names."""
    features_file, demographics_file = data_dir / "hsp_features.csv", data_dir / "demographics.csv"
    if not features_file.exists() or not demographics_file.exists():
        sys.exit(f"Need {features_file} and {demographics_file}. See the README section on the Challenge data.")
    features = pd.read_csv(features_file).drop(columns=["record", "problems"])
    demographics = pd.read_csv(demographics_file)
    table = features.merge(demographics[KEYS + ["Age", "Sex", "BMI", "Cognitive_Impairment"]], on=KEYS)
    table["male"] = table["Sex"].str.lower().str.startswith("m").astype(float)
    table["label"] = table["Cognitive_Impairment"].astype(int)
    table = table[table["tst_min"] >= MIN_SLEEP_MINUTES].reset_index(drop=True)
    eeg = [c for c in table.columns if c.startswith("eeg_")]
    table[eeg] = table[eeg].mask(table[eeg] <= FLAT_EEG)
    # BMI is left out on purpose: at the largest hospital it is missing for 94% of the people who stayed
    # healthy but only 35% of those later impaired, so a model scores well just from whether it was recorded.
    names = [c for c in features.columns if c not in KEYS] + ["Age", "male"]
    return table, names


def make_models(positive_share: float) -> dict:
    """Name -> (model, uses all features?). Every model weights the rare positive class up."""
    balanced = {"class_weight": "balanced"}
    return {
        "age only": (make_pipeline(StandardScaler(), LogisticRegression(**balanced)), False),
        "logistic regression": (
            make_pipeline(Clip(), SimpleImputer(strategy="median"), StandardScaler(),
                          LogisticRegression(C=0.1, max_iter=2000, **balanced)), True),
        "random forest": (
            make_pipeline(SimpleImputer(strategy="median"),
                          RandomForestClassifier(n_estimators=500, min_samples_leaf=5, n_jobs=-1,
                                                 random_state=SEED, class_weight="balanced_subsample")), True),
        "XGBoost": (
            XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=(1 - positive_share) / positive_share, random_state=SEED, n_jobs=-1), True),
    }


def age_conditioned_auroc(labels: np.ndarray, risk: np.ndarray, ages: np.ndarray, sites: np.ndarray | None = None) -> float:
    """The Challenge metric (compute_auroc_age in the organisers' evaluate_model.py).

    With sites, only pairs of people from the same hospital are compared.
    """
    positive, negative = labels == 1, labels == 0
    close = np.abs(ages[positive][:, None] - ages[negative][None, :]) <= AGE_GAP
    if sites is not None:
        close &= sites[positive][:, None] == sites[negative][None, :]
    ahead = risk[positive][:, None] > risk[negative][None, :]
    tied = risk[positive][:, None] == risk[negative][None, :]
    return float(((ahead + 0.5 * tied) * close).sum() / close.sum())


def scores(labels: np.ndarray, risk: np.ndarray, ages: np.ndarray, sites: np.ndarray) -> dict:
    flagged = risk >= np.quantile(risk, 1 - FLAGGED_SHARE)
    tp, fp = (flagged & (labels == 1)).sum(), (flagged & (labels == 0)).sum()
    fn, tn = (~flagged & (labels == 1)).sum(), (~flagged & (labels == 0)).sum()
    return {
        "age_auroc": age_conditioned_auroc(labels, risk, ages),
        "same_site_auroc": age_conditioned_auroc(labels, risk, ages, sites),
        "auroc": roc_auc_score(labels, risk),
        "auprc": average_precision_score(labels, risk),
        "accuracy": (tp + tn) / len(labels),
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "matrix": np.array([[tn, fp], [fn, tp]]),
    }


def fit_predict(model, train: pd.DataFrame, test: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return clone(model).fit(train[columns], train["label"]).predict_proba(test[columns])[:, 1]


def cross_validate(table: pd.DataFrame, model, columns: list[str]) -> tuple[list[dict], np.ndarray]:
    """Scores for every repeat, and the out-of-fold risks from the first repeat (for the plots)."""
    labels, ages, sites = table["label"].to_numpy(), table["Age"].to_numpy(), table["SiteID"].to_numpy()
    results, first = [], None
    for repeat in range(REPEATS):
        risk = np.zeros(len(table))
        folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED + repeat)
        for train, test in folds.split(table, table["label"].astype(str) + table["SiteID"]):
            risk[test] = fit_predict(model, table.iloc[train], table.iloc[test], columns)
        results.append(scores(labels, risk, ages, sites))
        first = risk if first is None else first
    return results, first


def leave_one_hospital_out(table: pd.DataFrame, model, columns: list[str]) -> dict:
    results = {}
    for site in sorted(table["SiteID"].unique()):
        train, test = table[table["SiteID"] != site], table[table["SiteID"] == site]
        risk = fit_predict(model, train, test, columns)
        results[site] = scores(test["label"].to_numpy(), risk, test["Age"].to_numpy(), test["SiteID"].to_numpy())
    return results


def plot_curves(labels: np.ndarray, risks: dict[str, np.ndarray], filename: str) -> None:
    colours = [style.INK_MUTED, style.BLUE, style.ORANGE, "#2f9e6e"]
    fig, (roc, pr) = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for (name, risk), colour in zip(risks.items(), colours):
        line = {"color": colour, "linewidth": 1.8, "linestyle": "--" if name == "age only" else "-"}
        false_positive, true_positive, _ = roc_curve(labels, risk)
        roc.plot(false_positive, true_positive, label=f"{name} ({roc_auc_score(labels, risk):.2f})", **line)
        precision, recall, _ = precision_recall_curve(labels, risk)
        pr.plot(recall, precision, label=f"{name} ({average_precision_score(labels, risk):.2f})", **line)
    roc.plot([0, 1], [0, 1], color=style.AXIS, linewidth=1)
    roc.set(title="ROC curve (area in brackets)", xlabel="False positive rate", ylabel="True positive rate")
    pr.axhline(labels.mean(), color=style.AXIS, linewidth=1)
    pr.set(title="Precision-recall curve (area in brackets)", xlabel="Recall", ylabel="Precision", ylim=(0, 1))
    for axis in (roc, pr):
        axis.grid(True)
        axis.legend(frameon=False, fontsize=9)
    fig.suptitle("Cross-validated predictions of future cognitive impairment", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    table, names = load_table(data_dir)
    labels = table["label"].to_numpy()
    sites = table.groupby("SiteID")["label"].agg(["size", "sum"])
    lines = [
        f"{len(table)} people, {labels.sum()} later impaired ({100 * labels.mean():.1f}%), {len(names)} features",
        "per hospital: " + ", ".join(f"{site} {row['size']} people / {row['sum']} impaired" for site, row in sites.iterrows()),
        "",
        f"{FOLDS}-fold cross-validation, mean of {REPEATS} repeats (+- spread between repeats)",
        f"{'model':<21}{'age-cond. AUROC':>17}{'same hospital':>17}{'AUROC':>8}{'AUPRC':>8}{'accuracy':>10}"
        f"{'precision':>11}{'recall':>8}{'specificity':>13}{'F1':>7}",
    ]

    risks, matrices, by_hospital = {}, {}, {}
    for name, (model, uses_all) in make_models(labels.mean()).items():
        columns = names if uses_all else ["Age"]
        results, risks[name] = cross_validate(table, model, columns)
        mean = {key: np.mean([r[key] for r in results]) for key in results[0] if key != "matrix"}
        spread = {key: np.std([r[key] for r in results]) for key in ("age_auroc", "same_site_auroc")}
        lines.append(
            f"{name:<21}{mean['age_auroc']:>10.3f} +-{spread['age_auroc']:.3f}"
            f"{mean['same_site_auroc']:>10.3f} +-{spread['same_site_auroc']:.3f}{mean['auroc']:>8.3f}{mean['auprc']:>8.3f}"
            f"{mean['accuracy']:>10.3f}{mean['precision']:>11.3f}{mean['recall']:>8.3f}{mean['specificity']:>13.3f}"
            f"{mean['f1']:>7.3f}"
        )
        matrices[name] = results[0]["matrix"]
        by_hospital[name] = leave_one_hospital_out(table, model, columns)
        print(lines[-1])

    lines += ["", "leave-one-hospital-out: age-conditioned AUROC (plain AUROC) on the hospital left out of training",
              f"{'model':<21}" + "".join(f"{site:>18}" for site in sites.index)]
    for name, results in by_hospital.items():
        lines.append(f"{name:<21}" + "".join(f"{r['age_auroc']:>10.3f} ({r['auroc']:.3f})" for r in results.values()))

    lines += ["", "confusion matrices, first cross-validation repeat (rows = truth, columns = not flagged / flagged)"]
    for name, matrix in matrices.items():
        lines += [f"{name}:", f"  not impaired {matrix[0, 0]:>6}{matrix[0, 1]:>6}", f"  impaired     {matrix[1, 0]:>6}{matrix[1, 1]:>6}"]

    text = "\n".join(lines)
    print("\n" + text)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"{data_dir.name}_baseline_report.txt").write_text(text + "\n")
    plot_curves(labels, risks, f"{data_dir.name}_baseline_curves.png")
    print(f"\nSaved {data_dir.name}_baseline_report.txt and {data_dir.name}_baseline_curves.png to {RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
