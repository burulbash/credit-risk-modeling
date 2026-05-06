from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATE_COL, REPORTS_DIR, TARGET  # noqa: E402
from src.db import read_csv_table, read_postgres_table  # noqa: E402
from src.splitting import make_time_split  # noqa: E402


CALIBRATED_PREDICTIONS = REPORTS_DIR / "oot_predictions_pd_models_calibrated.csv"
SCORECARD_PREDICTIONS = REPORTS_DIR / "scorecard_oot_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PD calibration by business segments.")
    parser.add_argument("--source", choices=["postgres", "csv"], default="postgres")
    parser.add_argument("--csv-path", default=None)

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--table", default="credit_risk_modeling_mart")

    parser.add_argument("--preferred-model", default="xgboost")

    return parser.parse_args()


def load_mart(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "csv":
        return read_csv_table(args.csv_path)

    return read_postgres_table(
        table_name=args.table,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
    )


def find_pd_column(predictions: pd.DataFrame, preferred_model: str) -> str:
    preferred = f"pd_{preferred_model}_calibrated"

    if preferred in predictions.columns:
        return preferred

    candidates = [
        col for col in predictions.columns
        if col.startswith("pd_") and col.endswith("_calibrated")
    ]

    if not candidates:
        candidates = [col for col in predictions.columns if col.startswith("pd_")]

    if not candidates:
        raise ValueError("No PD columns found in calibrated predictions.")

    return sorted(candidates)[0]


def status_from_oe_ratio(oe_ratio: float) -> str:
    if pd.isna(oe_ratio):
        return "no_expected_defaults"
    if 0.80 <= oe_ratio <= 1.25:
        return "ok"
    if 0.60 <= oe_ratio < 0.80 or 1.25 < oe_ratio <= 1.50:
        return "watch"
    return "review"


def summarize_segment(df: pd.DataFrame, segment_col: str, pd_col: str) -> pd.DataFrame:
    report = (
        df.groupby(segment_col, dropna=False)
        .agg(
            loans=("loan_id", "size"),
            observed_defaults=(TARGET, "sum"),
            observed_default_rate=(TARGET, "mean"),
            avg_predicted_pd=(pd_col, "mean"),
            expected_defaults=(pd_col, "sum"),
        )
        .reset_index()
        .rename(columns={segment_col: "segment"})
    )

    report["calibration_error"] = report["avg_predicted_pd"] - report["observed_default_rate"]
    report["oe_ratio"] = np.where(
        report["expected_defaults"] > 0,
        report["observed_defaults"] / report["expected_defaults"],
        np.nan,
    )
    report["status"] = report["oe_ratio"].apply(status_from_oe_ratio)
    report.insert(0, "segment_type", segment_col)

    return report.sort_values(["segment_type", "loans"], ascending=[True, False])


def add_bureau_band(df: pd.DataFrame) -> pd.DataFrame:
    if "bureau_score" not in df.columns:
        return df

    bins = [-np.inf, 550, 600, 650, 700, 750, np.inf]
    labels = ["<550", "550_599", "600_649", "650_699", "700_749", "750_plus"]

    df["bureau_score_band"] = pd.cut(df["bureau_score"], bins=bins, labels=labels)

    return df


def main() -> None:
    args = parse_args()

    if not CALIBRATED_PREDICTIONS.exists():
        raise FileNotFoundError("Run src/run_probability_calibration.py first.")

    predictions = pd.read_csv(CALIBRATED_PREDICTIONS)
    pd_col = find_pd_column(predictions, args.preferred_model)

    mart = load_mart(args)
    mart[DATE_COL] = pd.to_datetime(mart[DATE_COL], errors="coerce")

    _, _, oot_df, _ = make_time_split(mart)

    keep_cols = [
        "loan_id",
        TARGET,
        "product_type",
        "channel",
        "segment",
        "bureau_score",
        DATE_COL,
    ]
    keep_cols = [col for col in keep_cols if col in oot_df.columns]

    df = oot_df[keep_cols].merge(
        predictions[["loan_id", pd_col]],
        on="loan_id",
        how="inner",
    )

    if SCORECARD_PREDICTIONS.exists():
        scorecard = pd.read_csv(SCORECARD_PREDICTIONS)

        if {"loan_id", "score"}.issubset(scorecard.columns):
            def rating(score: float) -> str:
                if score >= 750:
                    return "A"
                if score >= 700:
                    return "B"
                if score >= 650:
                    return "C"
                if score >= 600:
                    return "D"
                if score >= 550:
                    return "E"
                return "F"

            scorecard["rating_grade"] = scorecard["score"].apply(rating)
            df = df.merge(scorecard[["loan_id", "rating_grade"]], on="loan_id", how="left")

    df = add_bureau_band(df)

    segment_cols = [
        col for col in [
            "rating_grade",
            "product_type",
            "channel",
            "segment",
            "bureau_score_band",
        ]
        if col in df.columns
    ]

    reports = [summarize_segment(df, col, pd_col) for col in segment_cols]
    combined = pd.concat(reports, ignore_index=True)

    combined.to_csv(REPORTS_DIR / "pd_segment_calibration_summary.csv", index=False)

    for segment_col in segment_cols:
        segment_report = combined[combined["segment_type"] == segment_col].copy()
        segment_report.to_csv(
            REPORTS_DIR / f"pd_calibration_by_{segment_col}.csv",
            index=False,
        )

    print(f"Using PD column: {pd_col}")
    print()
    print("PD segment calibration")
    print(combined.head(30))
    print()
    print("Saved:")
    print(REPORTS_DIR / "pd_segment_calibration_summary.csv")


if __name__ == "__main__":
    main()
