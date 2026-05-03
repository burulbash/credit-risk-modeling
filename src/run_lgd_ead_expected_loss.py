from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
from urllib.parse import quote_plus

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

TARGET = "target_default_90dpd_12m"
DATE_COL = "application_date"

PD_PREDICTIONS_PATH = REPORTS_DIR / "oot_predictions_pd_models_calibrated.csv"


def ensure_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def build_engine(args: argparse.Namespace):
    password = args.db_password or os.getenv("PGPASSWORD")
    if password is None:
        password = getpass.getpass(f"Password for PostgreSQL user {args.db_user}: ")

    url = (
        f"postgresql+psycopg2://{args.db_user}:{quote_plus(password)}"
        f"@{args.db_host}:{args.db_port}/{args.db_name}"
    )
    return create_engine(url)


def load_mart(args: argparse.Namespace) -> pd.DataFrame:
    engine = build_engine(args)

    query = f"""
        SELECT
            loan_id,
            application_id,
            client_id,
            application_date,
            product_type,
            channel,
            segment,
            ead_proxy_12m,
            lgd_proxy,
            total_recovered_amount,
            target_default_90dpd_12m
        FROM {args.table_name}
    """

    df = pd.read_sql(query, engine)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    return df


def choose_pd_column(preds: pd.DataFrame, preferred_model: str) -> str:
    preferred_col = f"pd_{preferred_model}_calibrated"
    if preferred_col in preds.columns:
        return preferred_col

    calibrated_cols = [c for c in preds.columns if c.startswith("pd_") and c.endswith("_calibrated")]
    if calibrated_cols:
        return calibrated_cols[0]

    raw_cols = [c for c in preds.columns if c.startswith("pd_")]
    if raw_cols:
        return raw_cols[0]

    raise ValueError("No PD prediction columns found.")


def make_score_bands(df: pd.DataFrame) -> pd.Series:
    # Risk bands based on predicted PD quantiles.
    # Band A = lowest risk, Band E = highest risk.
    labels = ["A_lowest_risk", "B", "C", "D", "E_highest_risk"]
    return pd.qcut(df["pd"], q=5, labels=labels, duplicates="drop")


def summarize_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    return (
        df.groupby(group_cols, observed=True)
        .agg(
            loans=("loan_id", "size"),
            observed_default_rate=(TARGET, "mean"),
            avg_pd=("pd", "mean"),
            avg_lgd=("lgd", "mean"),
            avg_ead=("ead", "mean"),
            total_ead=("ead", "sum"),
            total_expected_loss=("expected_loss", "sum"),
            expected_loss_rate=("expected_loss_rate", "mean"),
        )
        .reset_index()
        .sort_values("total_expected_loss", ascending=False)
    )


def plot_expected_loss_by_product(report: pd.DataFrame) -> None:
    data = report.sort_values("total_expected_loss", ascending=True)

    plt.figure(figsize=(8, 5))
    plt.barh(data["product_type"], data["total_expected_loss"])
    plt.xlabel("Total expected loss")
    plt.ylabel("Product type")
    plt.title("Expected loss by product type - OOT")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "expected_loss_by_product_oot.png", dpi=140)
    plt.close()


def plot_expected_loss_by_risk_band(report: pd.DataFrame) -> None:
    data = report.sort_values("risk_band")

    plt.figure(figsize=(8, 5))
    plt.bar(data["risk_band"].astype(str), data["expected_loss_rate"])
    plt.xlabel("PD risk band")
    plt.ylabel("Average expected loss rate")
    plt.title("Expected loss rate by risk band - OOT")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "expected_loss_rate_by_risk_band_oot.png", dpi=140)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default=None)
    parser.add_argument("--table-name", default="credit_risk_modeling_mart")
    parser.add_argument("--preferred-model", default="xgboost")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not PD_PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PD_PREDICTIONS_PATH}. Run src/run_probability_calibration.py first."
        )

    mart = load_mart(args)
    preds = pd.read_csv(PD_PREDICTIONS_PATH)
    preds[DATE_COL] = pd.to_datetime(preds[DATE_COL], errors="coerce")

    pd_col = choose_pd_column(preds, args.preferred_model)
    print(f"Using PD column: {pd_col}")

    preds = preds[["loan_id", pd_col]].rename(columns={pd_col: "pd"})

    df = mart.merge(preds, on="loan_id", how="inner")

    df["ead"] = pd.to_numeric(df["ead_proxy_12m"], errors="coerce")
    df["lgd"] = pd.to_numeric(df["lgd_proxy"], errors="coerce")

    # Conservative fallback for loans without observed recovery estimate.
    df["ead"] = df["ead"].fillna(0)
    df["lgd"] = df["lgd"].fillna(df["lgd"].median())
    df["lgd"] = df["lgd"].clip(0, 1)
    df["pd"] = df["pd"].clip(0, 1)

    df["expected_loss"] = df["pd"] * df["lgd"] * df["ead"]
    df["expected_loss_rate"] = np.where(
        df["ead"] > 0,
        df["expected_loss"] / df["ead"],
        np.nan,
    )

    df["risk_band"] = make_score_bands(df)
    df["application_month"] = df[DATE_COL].dt.to_period("M").astype(str)

    portfolio_summary = pd.DataFrame(
        [
            {
                "loans": len(df),
                "observed_default_rate": df[TARGET].mean(),
                "avg_pd": df["pd"].mean(),
                "avg_lgd": df["lgd"].mean(),
                "avg_ead": df["ead"].mean(),
                "total_ead": df["ead"].sum(),
                "total_expected_loss": df["expected_loss"].sum(),
                "portfolio_expected_loss_rate": df["expected_loss"].sum() / df["ead"].sum(),
            }
        ]
    )

    by_product = summarize_by(df, ["product_type"])
    by_channel = summarize_by(df, ["channel"])
    by_segment = summarize_by(df, ["segment"])
    by_risk_band = summarize_by(df, ["risk_band"])
    by_month = summarize_by(df, ["application_month"])

    df[
        [
            "loan_id",
            "application_id",
            "client_id",
            DATE_COL,
            "product_type",
            "channel",
            "segment",
            TARGET,
            "pd",
            "lgd",
            "ead",
            "expected_loss",
            "expected_loss_rate",
            "risk_band",
        ]
    ].to_csv(REPORTS_DIR / "expected_loss_oot_predictions.csv", index=False)

    portfolio_summary.to_csv(REPORTS_DIR / "expected_loss_portfolio_summary.csv", index=False)
    by_product.to_csv(REPORTS_DIR / "expected_loss_by_product.csv", index=False)
    by_channel.to_csv(REPORTS_DIR / "expected_loss_by_channel.csv", index=False)
    by_segment.to_csv(REPORTS_DIR / "expected_loss_by_segment.csv", index=False)
    by_risk_band.to_csv(REPORTS_DIR / "expected_loss_by_risk_band.csv", index=False)
    by_month.to_csv(REPORTS_DIR / "expected_loss_by_application_month.csv", index=False)

    plot_expected_loss_by_product(by_product)
    plot_expected_loss_by_risk_band(by_risk_band)

    print("\nExpected loss portfolio summary")
    print(portfolio_summary)

    print("\nExpected loss by product")
    print(by_product)

    print("\nExpected loss by risk band")
    print(by_risk_band)

    print("\nSaved expected loss reports to:", REPORTS_DIR)


if __name__ == "__main__":
    main()
