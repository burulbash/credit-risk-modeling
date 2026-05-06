from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import REPORTS_DIR, PLOTS_DIR, TARGET  # noqa: E402


SCORECARD_PREDICTIONS = REPORTS_DIR / "scorecard_oot_predictions.csv"

RATING_SCALE = [
    {"rating_grade": "A", "score_min": 750, "score_max": np.inf, "risk_level": "lowest"},
    {"rating_grade": "B", "score_min": 700, "score_max": 750, "risk_level": "low"},
    {"rating_grade": "C", "score_min": 650, "score_max": 700, "risk_level": "medium_low"},
    {"rating_grade": "D", "score_min": 600, "score_max": 650, "risk_level": "medium"},
    {"rating_grade": "E", "score_min": 550, "score_max": 600, "risk_level": "high"},
    {"rating_grade": "F", "score_min": -np.inf, "score_max": 550, "risk_level": "highest"},
]


def assign_rating(score: float) -> str:
    for row in RATING_SCALE:
        if row["score_min"] <= score < row["score_max"]:
            return row["rating_grade"]
    return "F"


def status_from_oe_ratio(oe_ratio: float) -> str:
    if pd.isna(oe_ratio):
        return "no_defaults_or_no_expected"
    if 0.80 <= oe_ratio <= 1.25:
        return "ok"
    if 0.60 <= oe_ratio < 0.80 or 1.25 < oe_ratio <= 1.50:
        return "watch"
    return "review"


def make_monotonicity_checks(summary: pd.DataFrame) -> pd.DataFrame:
    ordered = summary.sort_values("grade_order").copy()
    ordered = ordered[ordered["loans"] > 0].copy()

    checks = []

    for metric in ["avg_predicted_pd", "observed_default_rate"]:
        values = ordered[metric].to_numpy()
        violations = int(np.sum(np.diff(values) < -1e-12))

        worst_violation = 0.0
        if len(values) > 1:
            worst_violation = float(np.min(np.diff(values)))

        checks.append(
            {
                "metric": metric,
                "is_monotonic_increasing_with_risk": violations == 0,
                "number_of_violations": violations,
                "worst_step_change": worst_violation,
            }
        )

    return pd.DataFrame(checks)


def plot_rating_default_rate(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("grade_order")
    plot_df = plot_df[plot_df["loans"] > 0]

    plt.figure(figsize=(8, 5))
    plt.bar(plot_df["rating_grade"], plot_df["observed_default_rate"])
    plt.xlabel("Rating grade")
    plt.ylabel("Observed default rate")
    plt.title("Observed default rate by internal rating grade - OOT")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rating_grade_default_rate.png", dpi=140)
    plt.close()


def plot_rating_pd_vs_observed(summary: pd.DataFrame) -> None:
    plot_df = summary.sort_values("grade_order")
    plot_df = plot_df[plot_df["loans"] > 0]

    x = np.arange(len(plot_df))

    plt.figure(figsize=(8, 5))
    plt.plot(x, plot_df["avg_predicted_pd"], marker="o", label="Average predicted PD")
    plt.plot(x, plot_df["observed_default_rate"], marker="o", label="Observed default rate")
    plt.xticks(x, plot_df["rating_grade"])
    plt.xlabel("Rating grade")
    plt.ylabel("Rate")
    plt.title("Predicted PD vs observed default rate by rating grade - OOT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rating_grade_pd_vs_observed_dr.png", dpi=140)
    plt.close()


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    if not SCORECARD_PREDICTIONS.exists():
        raise FileNotFoundError("Run src/run_scorecard.py first.")

    df = pd.read_csv(SCORECARD_PREDICTIONS)

    required_cols = {"loan_id", TARGET, "pd_scorecard", "score"}
    missing = required_cols - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns in scorecard predictions: {sorted(missing)}")

    df["rating_grade"] = df["score"].apply(assign_rating)

    scale = pd.DataFrame(RATING_SCALE)
    scale["grade_order"] = range(1, len(scale) + 1)
    scale["score_max_display"] = scale["score_max"].replace(np.inf, np.nan)
    scale.to_csv(REPORTS_DIR / "internal_rating_scale.csv", index=False)

    summary = (
        df.groupby("rating_grade", as_index=False)
        .agg(
            loans=("loan_id", "size"),
            observed_defaults=(TARGET, "sum"),
            observed_default_rate=(TARGET, "mean"),
            avg_predicted_pd=("pd_scorecard", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
    )

    summary = scale[["rating_grade", "grade_order", "risk_level"]].merge(
        summary,
        on="rating_grade",
        how="left",
    )

    numeric_cols = [
        "loans",
        "observed_defaults",
        "observed_default_rate",
        "avg_predicted_pd",
        "min_score",
        "max_score",
    ]
    summary[numeric_cols] = summary[numeric_cols].fillna(0)

    summary["expected_defaults"] = summary["loans"] * summary["avg_predicted_pd"]
    summary["oe_ratio"] = np.where(
        summary["expected_defaults"] > 0,
        summary["observed_defaults"] / summary["expected_defaults"],
        np.nan,
    )
    summary["status"] = summary["oe_ratio"].apply(status_from_oe_ratio)

    monotonicity = make_monotonicity_checks(summary)

    summary.to_csv(REPORTS_DIR / "rating_grade_summary.csv", index=False)
    monotonicity.to_csv(REPORTS_DIR / "rating_monotonicity_checks.csv", index=False)

    plot_rating_default_rate(summary)
    plot_rating_pd_vs_observed(summary)

    print("Internal rating summary")
    print(summary)

    print()
    print("Monotonicity checks")
    print(monotonicity)

    print()
    print("Saved:")
    print(REPORTS_DIR / "internal_rating_scale.csv")
    print(REPORTS_DIR / "rating_grade_summary.csv")
    print(REPORTS_DIR / "rating_monotonicity_checks.csv")


if __name__ == "__main__":
    main()
