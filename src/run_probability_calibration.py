from __future__ import annotations

import argparse
import getpass
import os
from pathlib import Path
from urllib.parse import quote_plus

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"

TARGET = "target_default_90dpd_12m"
DATE_COL = "application_date"

ID_AND_DATE_COLS = [
    "loan_id",
    "application_id",
    "client_id",
    "application_date",
    "disbursement_date",
]

LEAKAGE_AND_HELPER_COLS = [
    "target_default_90dpd_12m",
    "target_ever30_6m",
    "max_observed_mob",
    "max_dpd_12m",
    "first_90dpd_mob",
    "ead_proxy_12m",
    "lgd_proxy",
    "total_recovered_amount",
    "collections_actions_count",
    "ever_contact_success_flag",
    "engine_scorecard_score",
    "engine_pd_estimate",
    "engine_risk_grade",
    "engine_policy_version",
    "engine_offered_amount",
    "engine_offered_term_months",
    "engine_offered_rate",
    "engine_rule_hits_count",
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


def load_mart(args: argparse.Namespace) -> pd.DataFrame:
    engine = build_engine(args)
    df = pd.read_sql(f"SELECT * FROM {args.table_name};", engine)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET] = df[TARGET].astype(int)
    print(f"Loaded mart shape: {df.shape}")
    return df


def make_time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:valid_end].copy(),
        df.iloc[valid_end:].copy(),
    )


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = set(ID_AND_DATE_COLS + LEAKAGE_AND_HELPER_COLS)

    suspicious_tokens = [
        "target",
        "future",
        "recovered",
        "collection",
        "dpd_12m",
        "first_90",
        "max_dpd",
        "observed_mob",
    ]

    feature_cols = []
    for col in df.columns:
        if col in drop_cols:
            continue
        low = col.lower()
        if any(token in low for token in suspicious_tokens):
            continue
        feature_cols.append(col)

    return feature_cols


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": float(np.max(tpr - fpr)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
        "mean_predicted_pd": float(np.mean(y_score)),
        "observed_default_rate": float(np.mean(y_true)),
    }


def calibration_table(
    df: pd.DataFrame,
    score_col: str,
    model_name: str,
    prediction_type: str,
    n_bins: int = 10,
) -> pd.DataFrame:
    data = df[[TARGET, score_col]].dropna().copy()
    data["pd_band"] = pd.qcut(data[score_col], q=n_bins, duplicates="drop")

    table = (
        data.groupby("pd_band", observed=True)
        .agg(
            loans=(TARGET, "size"),
            defaults=(TARGET, "sum"),
            avg_predicted_pd=(score_col, "mean"),
            observed_default_rate=(TARGET, "mean"),
            min_predicted_pd=(score_col, "min"),
            max_predicted_pd=(score_col, "max"),
        )
        .reset_index()
    )

    table.insert(0, "model", model_name)
    table.insert(1, "prediction_type", prediction_type)
    table["band_number"] = np.arange(1, len(table) + 1)
    table["calibration_error"] = table["avg_predicted_pd"] - table["observed_default_rate"]
    table["abs_calibration_error"] = table["calibration_error"].abs()

    return table


def plot_raw_vs_calibrated(
    table: pd.DataFrame,
    model_name: str,
) -> None:
    plt.figure(figsize=(7, 5))

    for prediction_type in ["raw", "calibrated"]:
        sub = table[table["prediction_type"] == prediction_type]
        plt.plot(
            sub["avg_predicted_pd"],
            sub["observed_default_rate"],
            marker="o",
            label=prediction_type,
        )

    max_value = max(
        table["avg_predicted_pd"].max(),
        table["observed_default_rate"].max(),
        0.01,
    )

    plt.plot([0, max_value], [0, max_value], linestyle="--", label="perfect")
    plt.xlabel("Average predicted PD")
    plt.ylabel("Observed default rate")
    plt.title(f"Raw vs calibrated PD - {model_name} - OOT")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"calibration_raw_vs_calibrated_{model_name}_oot.png", dpi=140)
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

    df = load_mart(args)
    train_df, valid_df, oot_df = make_time_split(df)
    feature_cols = get_feature_columns(df)

    y_valid = valid_df[TARGET].astype(int).to_numpy()
    y_oot = oot_df[TARGET].astype(int).to_numpy()

    model_paths = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError("No model artifacts found in outputs/models. Run src/train_pd.py first.")

    summary_rows = []
    all_band_tables = []

    oot_predictions = oot_df[
        ["loan_id", "application_id", "client_id", "application_date", TARGET]
    ].copy()

    for model_path in model_paths:
        model_name = model_path.stem
        print(f"Calibrating model: {model_name}")

        pipeline = joblib.load(model_path)

        raw_valid = pipeline.predict_proba(valid_df[feature_cols])[:, 1]
        raw_oot = pipeline.predict_proba(oot_df[feature_cols])[:, 1]

        calibrator = LogisticRegression(solver="lbfgs")
        calibrator.fit(logit(raw_valid).reshape(-1, 1), y_valid)

        calibrated_oot = calibrator.predict_proba(logit(raw_oot).reshape(-1, 1))[:, 1]

        for prediction_type, score in [
            ("raw", raw_oot),
            ("calibrated", calibrated_oot),
        ]:
            metrics = compute_metrics(y_oot, score)
            summary_rows.append(
                {
                    "model": model_name,
                    "prediction_type": prediction_type,
                    **metrics,
                }
            )

        temp = oot_df[[TARGET]].copy()
        temp[f"pd_{model_name}_raw"] = raw_oot
        temp[f"pd_{model_name}_calibrated"] = calibrated_oot

        raw_table = calibration_table(
            temp,
            score_col=f"pd_{model_name}_raw",
            model_name=model_name,
            prediction_type="raw",
        )
        calibrated_table = calibration_table(
            temp,
            score_col=f"pd_{model_name}_calibrated",
            model_name=model_name,
            prediction_type="calibrated",
        )

        combined_table = pd.concat([raw_table, calibrated_table], ignore_index=True)
        all_band_tables.append(combined_table)
        plot_raw_vs_calibrated(combined_table, model_name)

        oot_predictions[f"pd_{model_name}_raw"] = raw_oot
        oot_predictions[f"pd_{model_name}_calibrated"] = calibrated_oot

    summary = pd.DataFrame(summary_rows).sort_values(
        ["model", "prediction_type"]
    )
    band_report = pd.concat(all_band_tables, ignore_index=True)

    summary.to_csv(REPORTS_DIR / "probability_calibration_summary.csv", index=False)
    band_report.to_csv(REPORTS_DIR / "probability_calibration_by_band.csv", index=False)
    oot_predictions.to_csv(REPORTS_DIR / "oot_predictions_pd_models_calibrated.csv", index=False)

    print("\nProbability calibration summary")
    print(summary)
    print(f"\nSaved: {REPORTS_DIR / 'probability_calibration_summary.csv'}")
    print(f"Saved: {REPORTS_DIR / 'probability_calibration_by_band.csv'}")
    print(f"Saved: {REPORTS_DIR / 'oot_predictions_pd_models_calibrated.csv'}")


if __name__ == "__main__":
    main()
