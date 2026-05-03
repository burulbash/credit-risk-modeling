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

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False


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
    # targets / future performance
    "target_default_90dpd_12m",
    "target_ever30_6m",
    "max_observed_mob",
    "max_dpd_12m",
    "first_90dpd_mob",

    # LGD/EAD/outcome helpers, not allowed in PD features
    "ead_proxy_12m",
    "lgd_proxy",
    "total_recovered_amount",
    "collections_actions_count",
    "ever_contact_success_flag",

    # existing decision engine benchmark columns:
    # useful for comparison later, but excluded from challenger PD model
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
    for directory in [REPORTS_DIR, PLOTS_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def build_engine(args: argparse.Namespace):
    password = args.db_password or os.getenv("PGPASSWORD")

    if password is None:
        password = getpass.getpass(f"Password for PostgreSQL user {args.db_user}: ")

    password_quoted = quote_plus(password)

    url = (
        f"postgresql+psycopg2://{args.db_user}:{password_quoted}"
        f"@{args.db_host}:{args.db_port}/{args.db_name}"
    )
    return create_engine(url)


def load_from_postgres(args: argparse.Namespace) -> pd.DataFrame:
    engine = build_engine(args)
    query = f"SELECT * FROM {args.table_name};"
    print(f"Reading table from PostgreSQL: {args.table_name}")
    df = pd.read_sql(query, engine)
    print(f"Loaded shape: {df.shape}")
    return df


def load_from_csv(args: argparse.Namespace) -> pd.DataFrame:
    if args.csv_path is None:
        raise ValueError("--csv-path is required when --source csv")
    df = pd.read_csv(args.csv_path)
    print(f"Loaded CSV shape: {df.shape}")
    return df


def basic_data_checks(df: pd.DataFrame) -> None:
    if TARGET not in df.columns:
        raise ValueError(f"Target column not found: {TARGET}")

    if DATE_COL not in df.columns:
        raise ValueError(f"Date column not found: {DATE_COL}")

    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")

    if df[TARGET].isna().any():
        raise ValueError("Target contains missing values.")

    if df[TARGET].nunique() < 2:
        raise ValueError("Target has only one class. Model training is impossible.")

    print("\nBasic checks")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Default rate: {df[TARGET].mean():.4f}")
    print(f"Date range: {df[DATE_COL].min()} -> {df[DATE_COL].max()}")


def make_time_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    n = len(df)
    train_end = int(n * 0.70)
    valid_end = int(n * 0.85)

    train_df = df.iloc[:train_end].copy()
    valid_df = df.iloc[train_end:valid_end].copy()
    oot_df = df.iloc[valid_end:].copy()

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

    split_summary.to_csv(REPORTS_DIR / "pd_time_split_summary.csv", index=False)
    print("\nTime split summary")
    print(split_summary)

    return train_df, valid_df, oot_df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    drop_cols = set(ID_AND_DATE_COLS + LEAKAGE_AND_HELPER_COLS)
    features = [col for col in df.columns if col not in drop_cols]

    # Safety guard: exclude suspicious names from PD features.
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

    safe_features = []
    excluded_by_name = []

    for col in features:
        low = col.lower()
        if any(token in low for token in suspicious_tokens):
            excluded_by_name.append(col)
        else:
            safe_features.append(col)

    pd.DataFrame({"feature": safe_features}).to_csv(
        REPORTS_DIR / "pd_feature_columns.csv", index=False
    )

    if excluded_by_name:
        pd.DataFrame({"excluded_column": excluded_by_name}).to_csv(
            REPORTS_DIR / "pd_excluded_suspicious_columns.csv", index=False
        )

    print(f"\nFeature columns: {len(safe_features)}")
    if excluded_by_name:
        print(f"Excluded suspicious columns: {len(excluded_by_name)}")

    return safe_features


def build_preprocessor(train_df: pd.DataFrame, feature_cols: list[str]) -> ColumnTransformer:
    X = train_df[feature_cols]

    numeric_features = [
        col for col in feature_cols if pd.api.types.is_numeric_dtype(X[col])
    ]

    categorical_features = [
        col for col in feature_cols if col not in numeric_features
    ]

    pd.DataFrame(
        {
            "feature": numeric_features + categorical_features,
            "type": ["numeric"] * len(numeric_features)
            + ["categorical"] * len(categorical_features),
        }
    ).to_csv(REPORTS_DIR / "pd_feature_types.csv", index=False)

    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_features),
            ("cat", categorical_pipe, categorical_features),
        ],
        remainder="drop",
    )


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return {
            "roc_auc": np.nan,
            "gini": np.nan,
            "ks": np.nan,
            "average_precision": np.nan,
            "brier_score": np.nan,
        }

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)

    return {
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": float(np.max(tpr - fpr)),
        "average_precision": float(average_precision_score(y_true, y_score)),
        "brier_score": float(brier_score_loss(y_true, y_score)),
    }


def plot_roc(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f"ROC-AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def plot_score_distribution(
    y_score: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 5))
    plt.hist(y_score, bins=40)
    plt.xlabel("Predicted PD")
    plt.ylabel("Count")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def get_models(y_train: pd.Series) -> dict[str, object]:

    models: dict[str, object] = {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic_regression": LogisticRegression(
            max_iter=2000,
            solver="lbfgs",
            random_state=42,
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=200,
            random_state=42,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            min_samples_leaf=100,
            n_jobs=-1,
            random_state=42,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )

    return models


def save_model_importance(model_name: str, pipeline: Pipeline) -> None:
    try:
        preprocessor = pipeline.named_steps["preprocess"]
        estimator = pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()

        if hasattr(estimator, "coef_"):
            values = np.ravel(estimator.coef_)
            importance = np.abs(values)
            report = pd.DataFrame(
                {
                    "feature": feature_names,
                    "coefficient": values,
                    "abs_importance": importance,
                }
            ).sort_values("abs_importance", ascending=False)

        elif hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
            report = pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": values,
                }
            ).sort_values("importance", ascending=False)
        else:
            return

        report.head(100).to_csv(
            REPORTS_DIR / f"feature_importance_{model_name}.csv", index=False
        )
    except Exception as exc:
        print(f"Could not save feature importance for {model_name}: {exc}")


def train_and_evaluate(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    oot_df: pd.DataFrame,
    feature_cols: list[str],
) -> None:
    X_train = train_df[feature_cols]
    y_train = train_df[TARGET].astype(int)

    datasets = {
        "train": (train_df[feature_cols], train_df[TARGET].astype(int)),
        "valid": (valid_df[feature_cols], valid_df[TARGET].astype(int)),
        "oot": (oot_df[feature_cols], oot_df[TARGET].astype(int)),
    }

    preprocessor = build_preprocessor(train_df, feature_cols)
    models = get_models(y_train)

    metrics_rows = []
    oot_predictions = oot_df[
        ["loan_id", "application_id", "client_id", "application_date", TARGET]
    ].copy()

    for model_name, estimator in models.items():
        print(f"\nTraining model: {model_name}")

        pipeline = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)

        for split_name, (X_split, y_split) in datasets.items():
            y_score = pipeline.predict_proba(X_split)[:, 1]
            metrics = compute_metrics(y_split, y_score)

            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "rows": len(y_split),
                    "default_rate": float(np.mean(y_split)),
                    **metrics,
                }
            )

            if split_name == "oot":
                plot_roc(
                    y_split,
                    y_score,
                    PLOTS_DIR / f"roc_{model_name}_oot.png",
                    f"ROC Curve - {model_name} - OOT",
                )
                plot_score_distribution(
                    y_score,
                    PLOTS_DIR / f"score_distribution_{model_name}_oot.png",
                    f"Score Distribution - {model_name} - OOT",
                )
                oot_predictions[f"pd_{model_name}"] = y_score

        save_model_importance(model_name, pipeline)
        joblib.dump(pipeline, MODELS_DIR / f"{model_name}.joblib")

    metrics_df = pd.DataFrame(metrics_rows).sort_values(["split", "roc_auc"], ascending=[True, False])
    metrics_df.to_csv(REPORTS_DIR / "pd_model_metrics.csv", index=False)
    oot_predictions.to_csv(REPORTS_DIR / "oot_predictions_pd_models.csv", index=False)

    print("\nModel metrics")
    print(metrics_df)

    print(f"\nReports saved to: {REPORTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"Models saved to: {MODELS_DIR}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--source", choices=["postgres", "csv"], default="postgres")

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--db-password", default=None)
    parser.add_argument("--table-name", default="credit_risk_modeling_mart")

    parser.add_argument("--csv-path", default=None)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if args.source == "postgres":
        df = load_from_postgres(args)
    else:
        df = load_from_csv(args)

    basic_data_checks(df)
    train_df, valid_df, oot_df = make_time_split(df)
    feature_cols = get_feature_columns(df)

    train_and_evaluate(train_df, valid_df, oot_df, feature_cols)


if __name__ == "__main__":
    main()