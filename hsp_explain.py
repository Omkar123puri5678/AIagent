"""Explain the cognitive impairment predictions:  python hsp_explain.py [data folder]

Uses the random forest from hsp_baseline.py and SHAP values, which split each person's
predicted risk into a contribution from every feature. Everything is computed out of
fold: each person is scored and explained by a model that was trained without them.

Saves to results/ (prefixed with the data folder's name, data/hsp by default):
  - _importance.png: the features that move the predictions most, and in which direction,
  - _explanations.txt: that ranking, how many people in each risk group (low, moderate,
    high) went on to develop impairment, and worked examples for individual people.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold

import plots
from hsp_baseline import DEFAULT_DATA_DIR, FOLDS, SEED, load_table, make_models
from plots import plt

TOP_FEATURES = 15
# Risk groups by rank: the 10% of people at highest predicted risk, the next 20%, and the rest.
HIGH_SHARE, MODERATE_SHARE = 0.10, 0.20
EXAMPLES = 3  # people explained in detail per risk group shown


def out_of_fold(table: pd.DataFrame, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Predicted risk for every person, and each feature's contribution to it (people x features)."""
    model, _ = make_models(table["label"].mean())["random forest"]
    risk, contributions = np.zeros(len(table)), np.zeros((len(table), len(names)))
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=SEED)
    for train, test in folds.split(table, table["label"].astype(str) + table["SiteID"]):
        fitted = clone(model).fit(table.iloc[train][names], table.iloc[train]["label"])
        imputer, forest = fitted.steps[0][1], fitted.steps[-1][1]
        filled = imputer.transform(table.iloc[test][names])
        risk[test] = forest.predict_proba(filled)[:, 1]
        values = np.asarray(shap.TreeExplainer(forest).shap_values(filled, check_additivity=False))
        contributions[test] = values[..., 1] if values.ndim == 3 else values  # the "impaired" class
    return risk, contributions


def risk_groups(risk: np.ndarray) -> np.ndarray:
    high, moderate = np.quantile(risk, [1 - HIGH_SHARE, 1 - HIGH_SHARE - MODERATE_SHARE])
    return np.where(risk >= high, "high", np.where(risk >= moderate, "moderate", "low"))


def directions(table: pd.DataFrame, names: list[str], contributions: np.ndarray) -> np.ndarray:
    """+1 where a higher value of the feature raises the predicted risk, -1 where it lowers it."""
    signs = []
    for column, name in enumerate(names):
        known = table[name].notna().to_numpy()
        ranks = table[name][known].rank().to_numpy()
        signs.append(np.sign(np.corrcoef(ranks, contributions[known, column])[0, 1]) if known.sum() > 2 else 0)
    return np.nan_to_num(np.array(signs))


def plot_importance(names: list[str], importance: np.ndarray, signs: np.ndarray, filename: str) -> None:
    order = np.argsort(importance)[-TOP_FEATURES:]
    fig, axis = plt.subplots(figsize=(8, 5.2))
    colours = [plots.ORANGE if signs[i] > 0 else plots.BLUE for i in order]
    axis.barh([names[i] for i in order], importance[order], color=colours, height=0.7)
    axis.set(title="What moves the predicted risk of cognitive impairment",
             xlabel="Average change in predicted risk (mean |SHAP value|)")
    axis.grid(True, axis="x")
    axis.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color=plots.ORANGE), plt.Rectangle((0, 0), 1, 1, color=plots.BLUE)]
    axis.legend(handles, ["higher value, higher risk", "higher value, lower risk"], frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(plots.RESULTS_DIR / filename, dpi=150)
    plt.close(fig)


def explain_person(row: pd.Series, names: list[str], contributions: np.ndarray, medians: pd.Series) -> list[str]:
    lines = []
    for column in np.argsort(-np.abs(contributions))[:5]:
        name, value = names[column], row[names[column]]
        shown = "missing" if pd.isna(value) else f"{value:.3g}"
        effect = "raises" if contributions[column] > 0 else "lowers"
        lines.append(f"    {name:<30} {shown:>8}  (typical {medians[name]:.3g})  {effect} risk by {abs(contributions[column]):.3f}")
    return lines


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    table, names = load_table(data_dir)
    print(f"Explaining {len(table)} people with {len(names)} features (takes a few minutes)...")
    risk, contributions = out_of_fold(table, names)
    groups, labels = risk_groups(risk), table["label"].to_numpy()

    importance = np.abs(contributions).mean(axis=0)
    signs = directions(table, names, contributions)
    lines = ["features ranked by how much they move the predicted risk (mean |SHAP value|)"]
    for rank, column in enumerate(np.argsort(-importance)[:TOP_FEATURES], start=1):
        arrow = "higher value, higher risk" if signs[column] > 0 else "higher value, lower risk"
        lines.append(f"  {rank:>2}. {names[column]:<30} {importance[column]:.4f}   {arrow}")

    lines += ["", f"risk groups by predicted risk (high = top {HIGH_SHARE:.0%}, moderate = next {MODERATE_SHARE:.0%})",
              f"  {'group':<10}{'people':>8}{'later impaired':>16}{'rate':>8}"]
    for group in ("low", "moderate", "high"):
        inside = groups == group
        lines.append(f"  {group:<10}{inside.sum():>8}{labels[inside].sum():>16}{100 * labels[inside].mean():>7.1f}%")
    lines.append(f"  {'everyone':<10}{len(labels):>8}{labels.sum():>16}{100 * labels.mean():>7.1f}%")

    medians = table[names].median()
    lines += ["", "worked examples: the five features that moved each person's risk most"]
    for title, chosen in (("highest predicted risk", np.argsort(-risk)[:EXAMPLES]),
                          ("lowest predicted risk", np.argsort(risk)[:EXAMPLES])):
        lines.append(f"{title}:")
        for person in chosen:
            row = table.iloc[person]
            outcome = "later impaired" if labels[person] else "not impaired"
            lines.append(f"  {row['BidsFolder']} | age {row['Age']:.0f} | predicted risk {risk[person]:.2f} "
                         f"({groups[person]}) | outcome: {outcome}")
            lines += explain_person(row, names, contributions[person], medians)

    text = "\n".join(lines)
    print("\n" + text)
    plots.RESULTS_DIR.mkdir(exist_ok=True)
    (plots.RESULTS_DIR / f"{data_dir.name}_explanations.txt").write_text(text + "\n")
    plot_importance(names, importance, signs, f"{data_dir.name}_importance.png")
    print(f"\nSaved {data_dir.name}_explanations.txt and {data_dir.name}_importance.png to {plots.RESULTS_DIR.resolve()}")


if __name__ == "__main__":
    main()
