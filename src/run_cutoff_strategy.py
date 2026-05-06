from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATE_COL, PLOTS_DIR, REPORTS_DIR, TARGET  # noqa: E402
from src.db import read_csv_table, read_postgres_table  # noqa: E402
from src.splitting import make_time_split  # noqa: E402


SCORECARD_PREDICTIONS = REPORTS_DIR / "scorecard_oot_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simulate score cutoff strategy.")
    parser.add_argument("--source", choices=["postgres", "csv"], default="postgres")
    parser.add_argument("--csv-path", default=None)

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--table", default="credit_risk_modeling_mart")

    parser.add_argument("--funding-cost-rate", type=float, default=0.06)
    parser.add_argument("--operating-cost-rate", type=float, default=0.02)

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


def normalize_rate(series: pd.Series, fallback: float = 0.18) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").fillna(fallback)

    values = np.where(values > 1, values / 100, values)

    return pd.Series(values).clip(0, 1)


def prepare_strategy_dataset(args: argparse.Namespace) -> pd.DataFrame:
    if not SCORECARD_PREDICTIONS.exists():
        raise FileNotFoundError("Run src/run_scorecard.py first.")

    scorecard = pd.read_csv(SCORECARD_PREDICTIONS)

    mart = load_mart(args)
    mart[DATE_COL] = pd.to_datetime(mart[DATE_COL], errors="coerce")

    _, _, oot_df, _ = make_time_split(mart)

    keep_cols = [
        "loan_id",
        TARGET,
        "ead_proxy_12m",
        "lgd_proxy",
        "interest_rate",
        "term_months",
        "product_type",
    ]
    keep_cols = [col for col in keep_cols if col in oot_df.columns]

    df = scorecard.merge(oot_df[keep_cols], on=["loan_id", TARGET], how="left")

    if "ead_proxy_12m" not in df.columns:
        df["ead_proxy_12m"] = 1.0

    if "lgd_proxy" not in df.columns:
        df["lgd_proxy"] = 0.75

    if "interest_rate" not in df.columns:
        df["interest_rate"] = 0.18

    if "term_months" not in df.columns:
        df["term_months"] = 12

    df["ead"] = pd.to_numeric(df["ead_proxy_12m"], errors="coerce").fillna(0).clip(lower=0)
    df["lgd"] = pd.to_numeric(df["lgd_proxy"], errors="coerce").fillna(0.75).clip(0, 1)
    df["interest_rate_clean"] = normalize_rate(df["interest_rate"])
    df["term_years"] = pd.to_numeric(df["term_months"], errors="coerce").fillna(12).clip(lower=1) / 12

    return df


def simulate_cutoffs(df: pd.DataFrame, funding_cost_rate: float, operating_cost_rate: float) -> pd.DataFrame:
    min_score = int(np.floor(df["score"].min() / 25) * 25)
    max_score = int(np.ceil(df["score"].max() / 25) * 25)

    cutoffs = list(range(min_score, max_score + 1, 25))

    rows = []
    total_loans = len(df)

    for cutoff in cutoffs:
        approved = df[df["score"] >= cutoff].copy()

        if approved.empty:
            continue

        approved_loans = len(approved)
        approval_rate = approved_loans / total_loans

        total_ead = approved["ead"].sum()
        avg_pd = approved["pd_scorecard"].mean()
        observed_default_rate = approved[TARGET].mean()

        expected_loss = (approved["pd_scorecard"] * approved["lgd"] * approved["ead"]).sum()

        interest_income = (
            approved["ead"]
            * approved["interest_rate_clean"]
            * approved["term_years"].clip(upper=1)
        ).sum()

        funding_cost = total_ead * funding_cost_rate
        operating_cost = total_ead * operating_cost_rate
        risk_adjusted_profit = interest_income - expected_loss - funding_cost - operating_cost

        rows.append(
            {
                "score_cutoff": cutoff,
                "approved_loans": approved_loans,
                "approval_rate": approval_rate,
                "approved_ead": total_ead,
                "avg_pd": avg_pd,
                "observed_default_rate": observed_default_rate,
                "total_expected_loss": expected_loss,
                "expected_loss_rate": expected_loss / total_ead if total_ead > 0 else np.nan,
                "estimated_interest_income": interest_income,
                "estimated_funding_cost": funding_cost,
                "estimated_operating_cost": operating_cost,
                "risk_adjusted_profit": risk_adjusted_profit,
                "profit_rate_on_ead": risk_adjusted_profit / total_ead if total_ead > 0 else np.nan,
            }
        )

    return pd.DataFrame(rows)


def plot_approval_vs_bad_rate(strategy: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(strategy["approval_rate"], strategy["observed_default_rate"], marker="o")
    plt.xlabel("Approval rate")
    plt.ylabel("Observed default rate")
    plt.title("Cutoff strategy: approval rate vs observed default rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cutoff_approval_rate_vs_bad_rate.png", dpi=140)
    plt.close()


def plot_expected_loss_tradeoff(strategy: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(strategy["approval_rate"], strategy["expected_loss_rate"], marker="o")
    plt.xlabel("Approval rate")
    plt.ylabel("Expected loss rate")
    plt.title("Cutoff strategy: approval rate vs expected loss rate")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cutoff_expected_loss_tradeoff.png", dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = prepare_strategy_dataset(args)
    strategy = simulate_cutoffs(
        df,
        funding_cost_rate=args.funding_cost_rate,
        operating_cost_rate=args.operating_cost_rate,
    )

    strategy.to_csv(REPORTS_DIR / "cutoff_strategy_table.csv", index=False)

    plot_approval_vs_bad_rate(strategy)
    plot_expected_loss_tradeoff(strategy)

    print("Cutoff strategy")
    print(strategy.head(20))
    print()
    print("Saved:")
    print(REPORTS_DIR / "cutoff_strategy_table.csv")


if __name__ == "__main__":
    main()
