"""A simple baseline to compare the neural network against:  python baseline.py

Logistic regression on the hand-made features from features.py, trained and tested on
the same subject-wise split as the neural network. Saves its scores and confusion matrix
to results/.
"""

import sys

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
import plots
from dataset import split_recordings
from evaluate import report
from features import FEATURES_FILE


def main() -> None:
    if not FEATURES_FILE.exists():
        sys.exit(f"No {FEATURES_FILE}. Run `python features.py` first.")
    table = pd.read_csv(FEATURES_FILE)
    feature_names = [c for c in table.columns if c not in ("recording", "subject", "epoch", "stage")]

    # Same split as the neural network (train + its validation set, since nothing is tuned here).
    splits = split_recordings()
    train = table[table["recording"].isin(splits["train"] + splits["val"])]
    test = table[table["recording"].isin(splits["test"])]
    print(f"{len(feature_names)} features | {len(train):,} training epochs | {len(test):,} test epochs")

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[feature_names], train["stage"])
    predicted = model.predict(test[feature_names])

    matrix = confusion_matrix(test["stage"], predicted, labels=range(len(config.STAGE_NAMES)))
    text = report(matrix)
    print(f"\nTest set results for logistic regression:\n{text}")

    plots.RESULTS_DIR.mkdir(exist_ok=True)
    (plots.RESULTS_DIR / "baseline_report.txt").write_text(text + "\n")
    plots.plot_confusion_matrix(
        matrix, title="Test set: logistic regression baseline", filename="baseline_confusion_matrix.png"
    )
    print(f"\nSaved baseline_report.txt and baseline_confusion_matrix.png to {plots.RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
