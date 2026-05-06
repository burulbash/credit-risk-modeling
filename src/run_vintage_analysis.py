from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATE_COL, PLOTS_DIR, REPORTS_DIR, TARGET  # noqa: E402
from src.db import read_postgres_table  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build vintage default curves.")
    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--mart-table", default="credit_risk_modeling_mart")
    parser.add_argument("--snapshot-table", default="loan_monthly_snapshot")
    return parser.parse_args()


def load_table(args: argparse.Namespace, table_name: str) -> pd.DataFrame:
    return read_postgres_table(
        table_name=table_name,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
    )


def find_first_existing(columns: set[str], candidates: list[str]) -> str | None:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    return None


def build_snapshot_vintage(mart: pd.DataFrame, snapshots: pd.DataFrame) -> pd.DataFrame:
    cols = set(snapshots.columns)

    mob_col = find_first_existing(cols, ["mob", "month_on_book", "months_on_book"])
    dpd_col = find_first_existing(cols, ["dpd", "days_past_due", "current_dpd", "max_dpd"])

    if mob_col is None or dpd_col is None:
        raise ValueError("Could not find MOB/DPD columns in loan_monthly_snapshot.")

    base = mart[["loan_id", DATE_COL, TARGET]].copy()
    base[DATE_COL] = pd.to_datetime(base[DATE_COL], errors="coerce")
    base["origination_month"] = base[DATE_COL].dt.to_period("M").astype(str)

    snap = snapshots[["loan_id", mob_col, dpd_col]].copy()
    snap = snap.rename(columns={mob_col: "mob", dpd_col: "dpd"})
    snap["mob"] = pd.to_numeric(snap["mob"], errors="coerce").astype("Int64")
    snap["dpd"] = pd.to_numeric(snap["dpd"], errors="coerce").fillna(0)
    snap["default_90dpd"] = (snap["dpd"] >= 90).astype(int)

    df = snap.merge(base, on="loan_id", how="inner")
    df = df.dropna(subset=["mob"])

    loan_counts = (
        base.groupby("origination_month", as_index=False)
        .agg(loans_originated=("loan_id", "nunique"))
    )

    defaults_by_mob = (
        df.groupby(["origination_month", "mob"], as_index=False)
        .agg(defaults_90dpd=("default_90dpd", "sum"))
    )

    defaults_by_mob = defaults_by_mob.sort_values(["origination_month", "mob"])
    defaults_by_mob["cumulative_defaults_90dpd"] = (
        defaults_by_mob.groupby("origination_month")["defaults_90dpd"].cumsum()
    )

    result = defaults_by_mob.merge(loan_counts, on="origination_month", how="left")
    result["cumulative_default_rate"] = (
        result["cumulative_defaults_90dpd"] / result["loans_originated"]
    )

    return result


def build_mart_fallback_vintage(mart: pd.DataFrame) -> pd.DataFrame:
    df = mart[["loan_id", DATE_COL, TARGET]].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df["origination_month"] = df[DATE_COL].dt.to_period("M").astype(str)
    df["mob"] = 12

    result = (
        df.groupby(["origination_month", "mob"], as_index=False)
        .agg(
            loans_originated=("loan_id", "nunique"),
            cumulative_defaults_90dpd=(TARGET, "sum"),
        )
    )

    result["defaults_90dpd"] = result["cumulative_defaults_90dpd"]
    result["cumulative_default_rate"] = (
        result["cumulative_defaults_90dpd"] / result["loans_originated"]
    )

    return result


def plot_vintage_curves(vintage: pd.DataFrame) -> None:
    plot_df = vintage.copy()
    plot_df["origination_quarter"] = pd.PeriodIndex(
        plot_df["origination_month"],
        freq="M",
    ).asfreq("Q").astype(str)

    quarter_curves = (
        plot_df.groupby(["origination_quarter", "mob"], as_index=False)
        .agg(
            loans_originated=("loans_originated", "sum"),
            cumulative_defaults_90dpd=("cumulative_defaults_90dpd", "sum"),
        )
    )
    quarter_curves["cumulative_default_rate"] = (
        quarter_curves["cumulative_defaults_90dpd"] / quarter_curves["loans_originated"]
    )

    recent_quarters = sorted(quarter_curves["origination_quarter"].unique())[-8:]

    plt.figure(figsize=(9, 5))

    for quarter in recent_quarters:
        subset = quarter_curves[quarter_curves["origination_quarter"] == quarter]
        plt.plot(
            subset["mob"],
            subset["cumulative_default_rate"],
            marker="o",
            label=quarter,
        )

    plt.xlabel("Month on book")
    plt.ylabel("Cumulative 90+ DPD default rate")
    plt.title("Vintage cumulative default curves")
    plt.legend(title="Origination quarter", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "vintage_90dpd_cumulative_default_rate.png", dpi=140)
    plt.close()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    mart = load_table(args, args.mart_table)

    try:
        snapshots = load_table(args, args.snapshot_table)
        vintage = build_snapshot_vintage(mart, snapshots)
        source = args.snapshot_table
    except Exception as exc:
        print(f"Could not build vintage from monthly snapshots: {exc}")
        print("Using mart-level 12-month fallback vintage.")
        vintage = build_mart_fallback_vintage(mart)
        source = args.mart_table

    vintage.to_csv(REPORTS_DIR / "vintage_default_curves.csv", index=False)
    plot_vintage_curves(vintage)

    print(f"Vintage source: {source}")
    print()
    print(vintage.head(20))
    print()
    print("Saved:")
    print(REPORTS_DIR / "vintage_default_curves.csv")
    print(PLOTS_DIR / "vintage_90dpd_cumulative_default_rate.png")


if __name__ == "__main__":
    main()
