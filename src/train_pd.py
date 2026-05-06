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

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import PrecisionRecallDisplay, RocCurveDisplay
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import (  # noqa: E402
    DATE_COL,
    MODEL_PARAMS,
    PD_MODELS_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    TARGET,
)
from src.db import read_csv_table, read_postgres_table  # noqa: E402
from src.features import get_pd_feature_columns, infer_feature_types  # noqa: E402
from src.metrics import compute_binary_metrics  # noqa: E402
from src.splitting import make_time_split  # noqa: E402


try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except ImportError:
    XGBClassifier = None
    XGBOOST_AVAILABLE = False


def ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PD_MODELS_DIR.mkdir(parents=True, exist_ok=True)


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> ColumnTransformer:
    if scale_numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
    else:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
            ]
        )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=50)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )


def build_model_pipeline(
    estimator,
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool,
) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features=numeric_features,
                    categorical_features=categorical_features,
                    scale_numeric=scale_numeric,
                ),
            ),
            ("model", estimator),
        ]
    )


def get_models(
    numeric_features: list[str],
    categorical_features: list[str],
    include_xgboost: bool = True,
) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {}

    models["dummy_prior"] = build_model_pipeline(
        estimator=DummyClassifier(strategy="prior"),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
    )

    models["logistic_regression"] = build_model_pipeline(
        estimator=LogisticRegression(**MODEL_PARAMS["logistic_regression"]),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=True,
    )

    models["decision_tree"] = build_model_pipeline(
        estimator=DecisionTreeClassifier(**MODEL_PARAMS["decision_tree"]),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
    )

    models["random_forest"] = build_model_pipeline(
        estimator=RandomForestClassifier(**MODEL_PARAMS["random_forest"]),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        scale_numeric=False,
    )

    if include_xgboost and XGBOOST_AVAILABLE:
        models["xgboost"] = build_model_pipeline(
            estimator=XGBClassifier(**MODEL_PARAMS["xgboost"]),
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scale_numeric=False,
        )
    elif include_xgboost and not XGBOOST_AVAILABLE:
        print("XGBoost is not installed. Skipping xgboost model.")

    return models


def predict_positive_class(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Model {type(model)} does not support predict_proba.")

    return model.predict_proba(x)[:, 1]


def save_roc_plot(
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
    split_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    RocCurveDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"ROC curve - {model_name} - {split_name}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"roc_{model_name}_{split_name}.png", dpi=140)
    plt.close(fig)


def save_score_distribution_plot(
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
    split_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.hist(y_score[y_true.to_numpy() == 0], bins=40, alpha=0.65, label="non-default")
    ax.hist(y_score[y_true.to_numpy() == 1], bins=40, alpha=0.65, label="default")

    ax.set_title(f"Score distribution - {model_name} - {split_name}")
    ax.set_xlabel("Predicted PD")
    ax.set_ylabel("Count")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"score_distribution_{model_name}_{split_name}.png", dpi=140)
    plt.close(fig)


def save_pr_plot(
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
    split_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_score, ax=ax)
    ax.set_title(f"Precision-Recall curve - {model_name} - {split_name}")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"pr_{model_name}_{split_name}.png", dpi=140)
    plt.close(fig)


def get_transformed_feature_names(model: Pipeline) -> list[str]:
    preprocessor = model.named_steps["preprocessor"]

    try:
        return list(preprocessor.get_feature_names_out())
    except AttributeError:
        return []


def save_feature_importance(model_name: str, model: Pipeline) -> None:
    estimator = model.named_steps["model"]
    feature_names = get_transformed_feature_names(model)

    if not feature_names:
        print(f"Feature names are unavailable for {model_name}.")
        return

    if hasattr(estimator, "feature_importances_"):
        values = estimator.feature_importances_
    elif hasattr(estimator, "coef_"):
        values = np.abs(estimator.coef_[0])
    else:
        return

    if len(values) != len(feature_names):
        print(
            f"Could not save feature importance for {model_name}: "
            f"{len(values)} values for {len(feature_names)} features."
        )
        return

    importance = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": values,
            }
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    importance.to_csv(REPORTS_DIR / f"feature_importance_{model_name}.csv", index=False)


def train_and_evaluate(
    df: pd.DataFrame,
    feature_columns: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    include_xgboost: bool = True,
) -> None:
    train_df, valid_df, oot_df, split_summary = make_time_split(df)

    split_summary.to_csv(REPORTS_DIR / "pd_time_split_summary.csv", index=False)

    print()
    print("Time split summary")
    print(split_summary)

    x_train = train_df[feature_columns]
    y_train = train_df[TARGET]

    splits = {
        "train": (train_df[feature_columns], train_df[TARGET]),
        "valid": (valid_df[feature_columns], valid_df[TARGET]),
        "oot": (oot_df[feature_columns], oot_df[TARGET]),
    }

    models = get_models(
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_xgboost=include_xgboost,
    )

    metrics_rows: list[dict[str, object]] = []
    oot_predictions = oot_df[["loan_id", "application_id", "client_id", DATE_COL, TARGET]].copy()

    for model_name, model in models.items():
        print()
        print(f"Training model: {model_name}")

        model.fit(x_train, y_train)
        joblib.dump(model, PD_MODELS_DIR / f"{model_name}.joblib")

        save_feature_importance(model_name=model_name, model=model)

        for split_name, (x_split, y_split) in splits.items():
            y_score = predict_positive_class(model, x_split)
            metric_values = compute_binary_metrics(y_split, y_score)

            metrics_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "rows": len(y_split),
                    "default_rate": float(y_split.mean()),
                    **metric_values,
                }
            )

            if split_name == "oot":
                oot_predictions[f"pd_{model_name}"] = y_score
                save_roc_plot(y_split, y_score, model_name, split_name)
                save_pr_plot(y_split, y_score, model_name, split_name)
                save_score_distribution_plot(y_split, y_score, model_name, split_name)

    metrics_df = (
        pd.DataFrame(metrics_rows)
        .sort_values(["split", "roc_auc"], ascending=[True, False])
        .reset_index(drop=True)
    )

    metrics_df.to_csv(REPORTS_DIR / "pd_model_metrics.csv", index=False)
    oot_predictions.to_csv(REPORTS_DIR / "oot_predictions_pd_models.csv", index=False)

    print()
    print("Model metrics")
    print(metrics_df.sort_values(["split", "roc_auc"], ascending=[True, False]))

    print()
    print(f"Reports saved to: {REPORTS_DIR}")
    print(f"Plots saved to: {PLOTS_DIR}")
    print(f"PD models saved to: {PD_MODELS_DIR}")


def load_data(args: argparse.Namespace) -> pd.DataFrame:
    if args.source == "csv":
        return read_csv_table(args.csv_path)

    return read_postgres_table(
        table_name=args.table,
        db_host=args.db_host,
        db_port=args.db_port,
        db_name=args.db_name,
        db_user=args.db_user,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PD models for credit risk.")
    parser.add_argument("--source", choices=["postgres", "csv"], default="postgres")
    parser.add_argument("--csv-path", default=None)

    parser.add_argument("--db-host", default="localhost")
    parser.add_argument("--db-port", default="5432")
    parser.add_argument("--db-name", default="credit_risk_synth")
    parser.add_argument("--db-user", default="postgres")
    parser.add_argument("--table", default="credit_risk_modeling_mart")
    parser.add_argument(
        "--skip-xgboost",
        action="store_true",
        help="Skip XGBoost for quick local smoke runs.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Use only the first N rows after sorting by application date.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_output_dirs()

    df = load_data(args)

    if TARGET not in df.columns:
        raise ValueError(f"Target column is missing: {TARGET}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET]).copy()

    if args.max_rows is not None:
        if args.max_rows <= 0:
            raise ValueError("--max-rows must be a positive integer.")
        df = df.sort_values(DATE_COL).head(args.max_rows).copy()

    print()
    print("Basic checks")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {df.shape[1]:,}")
    print(f"Default rate: {df[TARGET].mean():.4f}")
    print(f"Date range: {df[DATE_COL].min().date()} -> {df[DATE_COL].max().date()}")

    feature_columns, suspicious_columns = get_pd_feature_columns(df)
    numeric_features, categorical_features = infer_feature_types(df, feature_columns)

    pd.Series(feature_columns, name="feature").to_csv(
        REPORTS_DIR / "pd_feature_columns.csv",
        index=False,
    )

    pd.DataFrame(
        {
            "feature": numeric_features + categorical_features,
            "type": ["numeric"] * len(numeric_features) + ["categorical"] * len(categorical_features),
        }
    ).to_csv(REPORTS_DIR / "pd_feature_types.csv", index=False)

    pd.Series(suspicious_columns, name="excluded_suspicious_column").to_csv(
        REPORTS_DIR / "pd_excluded_suspicious_columns.csv",
        index=False,
    )

    print()
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Excluded suspicious columns: {len(suspicious_columns)}")
    print(f"Numeric features: {len(numeric_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    train_and_evaluate(
        df=df,
        feature_columns=feature_columns,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        include_xgboost=not args.skip_xgboost,
    )


if __name__ == "__main__":
    main()
