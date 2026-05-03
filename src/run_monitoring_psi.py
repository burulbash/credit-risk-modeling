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

FEATURES_FOR_PSI = [
    "bureau_score",
    "total_limit",
    "oldest_trade_months",
    "outstanding_debt",
    "max_dpd_24m",
    "active_loans_count",
    "bureau_debt_to_income",
    "utilization_rate",
    "verified_income",
    "debt_to_income_est",
    "requested_amount",
    "requested_amount_to_income",
    "device_risk_score",
    "age_at_application",
    "client_tenure_days",
    "segment",
    "product_type",
    "channel",
    "employment_type",
    "verification_status",
    "city_type",
    "education_level",
]


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


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    engine = build_engine(args)
    cols = ["loan_id", DATE_COL, TARGET] + FEATURES_FOR_PSI
    cols = list(dict.fromkeys(cols))
    query = f"SELECT {', '.join(cols)} FROM {args.table_name};"

    df = pd.read_sql(query, engine)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET] = df[TARGET].astype(int)

    print(f"Loaded monitoring dataset: {df.shape}")
    return df


def make_time_split(df: pd.DataFrame):
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:valid_end].copy(),
        df.iloc[valid_end:].copy(),
    )


def psi_status(psi: float) -> str:
    if psi < 0.10:
        return "stable"
    if psi < 0.25:
        return "moderate_shift"
    return "significant_shift"


def make_bins(expected: pd.Series, actual: pd.Series, bins: int = 10) -> tuple[pd.Series, pd.Series]:
    if pd.api.types.is_numeric_dtype(expected):
        exp_num = pd.to_numeric(expected, errors="coerce")
        act_num = pd.to_numeric(actual, errors="coerce")

        non_missing = exp_num.dropna()

        if non_missing.nunique() <= 2:
            exp_bin = exp_num.astype("Int64").astype(str).replace("<NA>", "MISSING")
            act_bin = act_num.astype("Int64").astype(str).replace("<NA>", "MISSING")
            return exp_bin, act_bin

        edges = np.unique(non_missing.quantile(np.linspace(0, 1, bins + 1)).to_numpy())

        if len(edges) < 3:
            exp_bin = exp_num.astype("Int64").astype(str).replace("<NA>", "MISSING")
            act_bin = act_num.astype("Int64").astype(str).replace("<NA>", "MISSING")
            return exp_bin, act_bin

        edges[0] = -np.inf
        edges[-1] = np.inf

        exp_bin = pd.cut(exp_num, bins=edges, include_lowest=True).astype(str)
        act_bin = pd.cut(act_num, bins=edges, include_lowest=True).astype(str)

        exp_bin = exp_bin.where(~exp_num.isna(), "MISSING")
        act_bin = act_bin.where(~act_num.isna(), "MISSING")

        return exp_bin, act_bin

    exp_cat = expected.astype("object").where(~expected.isna(), "MISSING").astype(str)
    act_cat = actual.astype("object").where(~actual.isna(), "MISSING").astype(str)

    top_categories = exp_cat.value_counts(normalize=True)
    top_categories = top_categories[top_categories >= 0.01].index.tolist()

    exp_bin = exp_cat.where(exp_cat.isin(top_categories), "OTHER")
    act_bin = act_cat.where(act_cat.isin(top_categories), "OTHER")

    return exp_bin, act_bin


def calculate_psi(expected: pd.Series, actual: pd.Series, feature: str, comparison: str) -> tuple[float, pd.DataFrame]:
    exp_bin, act_bin = make_bins(expected, actual)

    exp_dist = exp_bin.value_counts(normalize=True, dropna=False)
    act_dist = act_bin.value_counts(normalize=True, dropna=False)

    bins = sorted(set(exp_dist.index).union(set(act_dist.index)))

    rows = []
    psi_total = 0.0
    eps = 1e-6

    for bin_name in bins:
        exp_pct = float(exp_dist.get(bin_name, 0.0))
        act_pct = float(act_dist.get(bin_name, 0.0))

        exp_adj = max(exp_pct, eps)
        act_adj = max(act_pct, eps)

        psi_component = (act_adj - exp_adj) * np.log(act_adj / exp_adj)
        psi_total += psi_component

        rows.append(
            {
                "comparison": comparison,
                "feature": feature,
                "bin": bin_name,
                "expected_pct": exp_pct,
                "actual_pct": act_pct,
                "psi_component": psi_component,
            }
        )

    return psi_total, pd.DataFrame(rows)


def plot_top_psi(summary: pd.DataFrame, comparison: str) -> None:
    data = (
        summary[summary["comparison"] == comparison]
        .sort_values("psi", ascending=False)
        .head(15)
        .sort_values("psi", ascending=True)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(data["feature"], data["psi"])
    plt.xlabel("PSI")
    plt.ylabel("Feature")
    plt.title(f"Top PSI features - {comparison}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"top_psi_features_{comparison}.png", dpi=140)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default=None)
    parser.add_argument("--table-name", default="credit_risk_modeling_mart")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    df = load_data(args)
    train_df, valid_df, oot_df = make_time_split(df)

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "min_date": train_df[DATE_COL].min(),
                "max_date": train_df[DATE_COL].max(),
                "default_rate": train_df[TARGET].mean(),
            },
            {
                "split": "valid",
                "rows": len(valid_df),
                "min_date": valid_df[DATE_COL].min(),
                "max_date": valid_df[DATE_COL].max(),
                "default_rate": valid_df[TARGET].mean(),
            },
            {
                "split": "oot",
                "rows": len(oot_df),
                "min_date": oot_df[DATE_COL].min(),
                "max_date": oot_df[DATE_COL].max(),
                "default_rate": oot_df[TARGET].mean(),
            },
        ]
    )

    all_details = []
    summary_rows = []

    for comparison, actual_df in [
        ("train_vs_valid", valid_df),
        ("train_vs_oot", oot_df),
    ]:
        for feature in FEATURES_FOR_PSI:
            if feature not in train_df.columns:
                continue

            psi, detail = calculate_psi(
                expected=train_df[feature],
                actual=actual_df[feature],
                feature=feature,
                comparison=comparison,
            )

            all_details.append(detail)
            summary_rows.append(
                {
                    "comparison": comparison,
                    "feature": feature,
                    "psi": psi,
                    "status": psi_status(psi),
                }
            )

    psi_summary = pd.DataFrame(summary_rows).sort_values(
        ["comparison", "psi"],
        ascending=[True, False],
    )

    psi_details = pd.concat(all_details, ignore_index=True)

    psi_summary.to_csv(REPORTS_DIR / "psi_feature_summary.csv", index=False)
    psi_details.to_csv(REPORTS_DIR / "psi_feature_details.csv", index=False)
    split_summary.to_csv(REPORTS_DIR / "monitoring_split_summary.csv", index=False)

    plot_top_psi(psi_summary, "train_vs_valid")
    plot_top_psi(psi_summary, "train_vs_oot")

    print("\nMonitoring split summary")
    print(split_summary)

    print("\nTop PSI features")
    print(psi_summary.head(20))

    print("\nSaved reports:")
    print(REPORTS_DIR / "psi_feature_summary.csv")
    print(REPORTS_DIR / "psi_feature_details.csv")
    print(REPORTS_DIR / "monitoring_split_summary.csv")


if __name__ == "__main__":
    main()