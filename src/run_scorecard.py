from __future__ import annotations

import argparse
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATE_COL,
    PLOTS_DIR,
    REPORTS_DIR,
    SCORECARD_BAND_LABELS,
    SCORECARD_BANDS,
    SCORECARD_BASE_ODDS,
    SCORECARD_BASE_SCORE,
    SCORECARD_MODELS_DIR,
    SCORECARD_PDO,
    TARGET,
)
from src.db import read_csv_table, read_postgres_table  # noqa: E402
from src.metrics import compute_binary_metrics  # noqa: E402
from src.splitting import make_time_split  # noqa: E402


SCORECARD_FEATURES = [
    "bureau_score",
    "active_loans_count",
    "closed_loans_count",
    "delinquency_30_12m",
    "delinquency_60_12m",
    "delinquency_90_24m",
    "max_dpd_24m",
    "inquiries_30d",
    "inquiries_90d",
    "outstanding_debt",
    "total_limit",
    "utilization_rate",
    "oldest_trade_months",
    "bureau_file_thin_flag",
    "external_collections_flag",
    "verified_income",
    "declared_income",
    "debt_to_income_est",
    "expense_to_income_est",
    "income_stability_score",
    "job_tenure_months",
    "total_work_experience_months",
    "requested_amount",
    "requested_term_months",
    "principal_amount",
    "term_months",
    "interest_rate",
    "requested_amount_to_income",
    "principal_amount_to_income",
    "bureau_debt_to_income",
    "device_risk_score",
    "doc_mismatch_flag",
    "synthetic_identity_flag",
    "fraud_suspect_flag",
    "age_at_application",
    "client_tenure_days",
    "is_existing_customer",
    "repeat_application_flag",
    "product_type",
    "channel",
    "digital_channel",
    "purpose_code",
    "segment",
    "employment_type",
    "verification_status",
    "city_type",
    "education_level",
]


ID_COLS = ["loan_id", "application_id", "client_id", DATE_COL, TARGET]


def ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    SCORECARD_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "csv":
        df = read_csv_table(args.csv_path)
    else:
        df = read_postgres_table(
            table_name=args.table,
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
        )

    required_cols = ID_COLS + [col for col in SCORECARD_FEATURES if col in df.columns]
    missing_id_cols = [col for col in ID_COLS if col not in df.columns]

    if missing_id_cols:
        raise ValueError(f"Missing required columns: {missing_id_cols}")

    df = df[required_cols].copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df[TARGET] = df[TARGET].astype(int)

    print(f"Loaded scorecard dataset: {df.shape}")
    print(f"Default rate: {df[TARGET].mean():.4f}")

    return df


def compute_woe_table(
    train_df: pd.DataFrame,
    feature: str,
    target: str,
    max_bins: int = 5,
    min_category_share: float = 0.01,
) -> tuple[pd.DataFrame, dict]:
    x = train_df[feature]
    y = train_df[target].astype(int)

    is_numeric = pd.api.types.is_numeric_dtype(x)

    meta = {
        "feature": feature,
        "is_numeric": is_numeric,
        "edges": None,
        "top_categories": None,
        "woe_map": {},
        "default_woe": 0.0,
    }

    if is_numeric:
        numeric_x = pd.to_numeric(x, errors="coerce")
        non_missing = numeric_x.dropna()

        if non_missing.nunique() <= 2:
            binned = numeric_x.astype("Int64").astype(str)
            binned = binned.replace("", "MISSING")
            meta["is_numeric"] = False
        else:
            quantiles = np.linspace(0, 1, max_bins + 1)
            edges = np.unique(non_missing.quantile(quantiles).to_numpy())

            if len(edges) < 3:
                binned = numeric_x.astype("Int64").astype(str)
                binned = binned.replace("", "MISSING")
                meta["is_numeric"] = False
            else:
                edges[0] = -np.inf
                edges[-1] = np.inf

                binned = pd.cut(numeric_x, bins=edges, include_lowest=True).astype(str)
                binned = binned.where(~numeric_x.isna(), "MISSING")
                meta["edges"] = edges
    else:
        raw = x.astype("object").where(~x.isna(), "MISSING").astype(str)
        value_share = raw.value_counts(normalize=True)

        top_categories = value_share[value_share >= min_category_share].index.tolist()
        binned = raw.where(raw.isin(top_categories), "OTHER")

        meta["top_categories"] = top_categories

    temp = pd.DataFrame({"bin": binned, "target": y})

    grouped = (
        temp.groupby("bin", dropna=False)
        .agg(
            total=("target", "size"),
            bad=("target", "sum"),
        )
        .reset_index()
    )

    grouped["good"] = grouped["total"] - grouped["bad"]
    grouped["bad_rate"] = grouped["bad"] / grouped["total"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    smoothing = 0.5

    grouped["good_dist"] = (grouped["good"] + smoothing) / (
        total_good + smoothing * len(grouped)
    )
    grouped["bad_dist"] = (grouped["bad"] + smoothing) / (
        total_bad + smoothing * len(grouped)
    )

    grouped["woe"] = np.log(grouped["good_dist"] / grouped["bad_dist"])
    grouped["iv_component"] = (grouped["good_dist"] - grouped["bad_dist"]) * grouped["woe"]

    grouped["feature"] = feature
    grouped["iv"] = grouped["iv_component"].sum()

    meta["woe_map"] = dict(zip(grouped["bin"].astype(str), grouped["woe"]))
    meta["default_woe"] = 0.0

    return grouped[
        [
            "feature",
            "bin",
            "total",
            "good",
            "bad",
            "bad_rate",
            "good_dist",
            "bad_dist",
            "woe",
            "iv_component",
            "iv",
        ]
    ].sort_values(["feature", "bin"]), meta


def transform_feature_to_woe(df: pd.DataFrame, feature: str, meta: dict) -> pd.Series:
    x = df[feature]

    if meta["is_numeric"] and meta["edges"] is not None:
        numeric_x = pd.to_numeric(x, errors="coerce")
        binned = pd.cut(numeric_x, bins=meta["edges"], include_lowest=True).astype(str)
        binned = binned.where(~numeric_x.isna(), "MISSING")
    else:
        raw = x.astype("object").where(~x.isna(), "MISSING").astype(str)
        top_categories = meta.get("top_categories")

        if top_categories is not None:
            binned = raw.where(raw.isin(top_categories), "OTHER")
        else:
            binned = raw

    return binned.astype(str).map(meta["woe_map"]).fillna(meta["default_woe"]).astype(float)


def build_woe_dataset(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    features: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    woe_tables = []
    metas = {}

    for feature in features:
        table, meta = compute_woe_table(train_df, feature, TARGET)
        woe_tables.append(table)
        metas[feature] = meta

    woe_bins = pd.concat(woe_tables, ignore_index=True)

    iv_report = (
        woe_bins.groupby("feature", as_index=False)
        .agg(iv=("iv", "first"), bins=("bin", "nunique"))
        .sort_values("iv", ascending=False)
    )

    selected_features = iv_report[
        (iv_report["iv"] >= 0.01) & (iv_report["iv"] <= 1.0)
    ]["feature"].tolist()

    if len(selected_features) < 5:
        selected_features = iv_report.head(15)["feature"].tolist()
    else:
        selected_features = selected_features[:20]

    print()
    print("Top IV features")
    print(iv_report.head(15))
    print(f"\nSelected scorecard features: {len(selected_features)}")

    def transform(df: pd.DataFrame) -> pd.DataFrame:
        result = pd.DataFrame(index=df.index)

        for feature in selected_features:
            result[f"woe_{feature}"] = transform_feature_to_woe(df, feature, metas[feature])

        return result

    x_train = transform(train_df)
    x_valid = transform(valid_df)
    x_oot = transform(oot_df)

    return x_train, x_valid, x_oot, woe_bins, iv_report, metas


def pd_to_score(pd_values: np.ndarray) -> np.ndarray:
    pd_values = np.clip(pd_values, 1e-6, 1 - 1e-6)

    factor = SCORECARD_PDO / np.log(2)
    offset = SCORECARD_BASE_SCORE - factor * np.log(SCORECARD_BASE_ODDS)

    odds_good_bad = (1 - pd_values) / pd_values

    return offset + factor * np.log(odds_good_bad)


def make_score_band_report(oot_df: pd.DataFrame, y_score: np.ndarray) -> pd.DataFrame:
    report = oot_df[["loan_id", TARGET]].copy()
    report["pd_scorecard"] = y_score
    report["score"] = pd_to_score(y_score)

    report["score_band"] = pd.cut(
        report["score"],
        bins=SCORECARD_BANDS,
        labels=SCORECARD_BAND_LABELS,
    )

    band_report = (
        report.groupby("score_band", observed=True)
        .agg(
            loans=("loan_id", "size"),
            defaults=(TARGET, "sum"),
            observed_default_rate=(TARGET, "mean"),
            avg_predicted_pd=("pd_scorecard", "mean"),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
    )

    report.to_csv(REPORTS_DIR / "scorecard_oot_predictions.csv", index=False)

    return band_report


def plot_score_bands(band_report: pd.DataFrame) -> None:
    plt.figure(figsize=(8, 5))
    plt.bar(band_report["score_band"].astype(str), band_report["observed_default_rate"])
    plt.xlabel("Score band")
    plt.ylabel("Observed default rate")
    plt.title("Observed default rate by score band - OOT")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "scorecard_bad_rate_by_score_band_oot.png", dpi=140)
    plt.close()


def make_scorecard_points(
    model: LogisticRegression,
    woe_bins: pd.DataFrame,
    selected_woe_features: list[str],
) -> pd.DataFrame:
    factor = SCORECARD_PDO / np.log(2)
    offset = SCORECARD_BASE_SCORE - factor * np.log(SCORECARD_BASE_ODDS)

    coefficients = model.coef_.ravel()
    intercept = float(model.intercept_[0])

    coef_map = {
        feature.replace("woe_", ""): coef
        for feature, coef in zip(selected_woe_features, coefficients)
    }

    rows = []

    for _, row in woe_bins.iterrows():
        feature = row["feature"]

        if feature not in coef_map:
            continue

        coef = coef_map[feature]
        points = -factor * coef * row["woe"]

        rows.append(
            {
                "feature": feature,
                "bin": row["bin"],
                "woe": row["woe"],
                "coefficient": coef,
                "points": points,
                "bad_rate": row["bad_rate"],
                "iv": row["iv"],
            }
        )

    scorecard = pd.DataFrame(rows)

    base_points = offset - factor * intercept

    base_row = pd.DataFrame(
        [
            {
                "feature": "__BASE_SCORE__",
                "bin": "intercept",
                "woe": np.nan,
                "coefficient": intercept,
                "points": base_points,
                "bad_rate": np.nan,
                "iv": np.nan,
            }
        ]
    )

    return pd.concat([base_row, scorecard], ignore_index=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build classic WOE/IV scorecard.")
    parser.add_argument("--source", choices=["postgres", "csv"], default="postgres")
    parser.add_argument("--csv-path", default=None)

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--table", default="credit_risk_modeling_mart")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    df = load_data(args)

    train_df, valid_df, oot_df, _ = make_time_split(df)

    available_features = [feature for feature in SCORECARD_FEATURES if feature in df.columns]

    x_train, x_valid, x_oot, woe_bins, iv_report, metas = build_woe_dataset(
        train_df=train_df,
        valid_df=valid_df,
        oot_df=oot_df,
        features=available_features,
    )

    y_train = train_df[TARGET].astype(int).to_numpy()
    y_valid = valid_df[TARGET].astype(int).to_numpy()
    y_oot = oot_df[TARGET].astype(int).to_numpy()

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x_train, y_train)

    datasets = {
        "train": (x_train, y_train),
        "valid": (x_valid, y_valid),
        "oot": (x_oot, y_oot),
    }

    metrics_rows = []

    for split_name, (x_split, y_split) in datasets.items():
        y_score = model.predict_proba(x_split)[:, 1]

        metrics_rows.append(
            {
                "model": "woe_scorecard_logistic",
                "split": split_name,
                **compute_binary_metrics(y_split, y_score),
            }
        )

    metrics = pd.DataFrame(metrics_rows)

    oot_pd = model.predict_proba(x_oot)[:, 1]
    band_report = make_score_band_report(oot_df, oot_pd)
    scorecard_points = make_scorecard_points(model, woe_bins, list(x_train.columns))

    woe_bins.to_csv(REPORTS_DIR / "scorecard_woe_bins.csv", index=False)
    iv_report.to_csv(REPORTS_DIR / "scorecard_iv_report.csv", index=False)
    metrics.to_csv(REPORTS_DIR / "scorecard_model_metrics.csv", index=False)
    band_report.to_csv(REPORTS_DIR / "scorecard_bad_rate_by_score_band.csv", index=False)
    scorecard_points.to_csv(REPORTS_DIR / "scorecard_points.csv", index=False)

    artifact_path = SCORECARD_MODELS_DIR / "woe_scorecard_logistic.joblib"

    joblib.dump(
        {
            "model": model,
            "features": list(x_train.columns),
            "woe_metadata": metas,
            "scorecard_params": {
                "base_score": SCORECARD_BASE_SCORE,
                "pdo": SCORECARD_PDO,
                "base_odds": SCORECARD_BASE_ODDS,
                "score_bands": SCORECARD_BANDS,
                "score_band_labels": SCORECARD_BAND_LABELS,
            },
        },
        artifact_path,
    )

    plot_score_bands(band_report)

    print()
    print("Scorecard model metrics")
    print(metrics)

    print()
    print("Score band report")
    print(band_report)

    print()
    print("Saved reports:")
    print(REPORTS_DIR / "scorecard_iv_report.csv")
    print(REPORTS_DIR / "scorecard_woe_bins.csv")
    print(REPORTS_DIR / "scorecard_model_metrics.csv")
    print(REPORTS_DIR / "scorecard_bad_rate_by_score_band.csv")
    print(REPORTS_DIR / "scorecard_points.csv")
    print(f"Scorecard artifact saved to: {artifact_path}")


if __name__ == "__main__":
    main()
