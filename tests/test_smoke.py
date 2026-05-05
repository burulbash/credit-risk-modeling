from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import (
    DATE_COL,
    PD_MODELS_DIR,
    SCORECARD_MODELS_DIR,
    TARGET,
)
from src.features import get_pd_feature_columns, infer_feature_types
from src.metrics import compute_binary_metrics
from src.splitting import make_time_split


def test_project_paths_are_separated() -> None:
    assert PD_MODELS_DIR.name == "pd"
    assert SCORECARD_MODELS_DIR.name == "scorecard"
    assert PD_MODELS_DIR != SCORECARD_MODELS_DIR


def test_compute_binary_metrics() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.05, 0.20, 0.70, 0.90])

    metrics = compute_binary_metrics(y_true, y_score)

    assert metrics["roc_auc"] == 1.0
    assert metrics["gini"] == 1.0
    assert metrics["ks"] == 1.0
    assert 0 <= metrics["brier_score"] <= 1


def test_time_split_and_feature_filtering() -> None:
    df = pd.DataFrame(
        {
            "loan_id": range(10),
            "application_id": range(100, 110),
            "client_id": range(200, 210),
            DATE_COL: pd.date_range("2024-01-01", periods=10, freq="D"),
            TARGET: [0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            "bureau_score": [650, 640, 700, 620, 610, 680, 590, 710, 660, 600],
            "segment": ["mass", "mass", "premium", "mass", "mass", "premium", "mass", "premium", "mass", "mass"],
            "future_bad_field": [1] * 10,
            "target_helper": [0] * 10,
        }
    )

    features, suspicious = get_pd_feature_columns(df)
    numeric_features, categorical_features = infer_feature_types(df, features)

    assert "bureau_score" in features
    assert "segment" in features
    assert "future_bad_field" not in features
    assert "target_helper" not in features
    assert "bureau_score" in numeric_features
    assert "segment" in categorical_features
    assert set(suspicious) == {"future_bad_field", "target_helper"}

    train_df, valid_df, oot_df, split_summary = make_time_split(df)

    assert len(train_df) == 7
    assert len(valid_df) == 1
    assert len(oot_df) == 2
    assert set(split_summary["split"]) == {"train", "valid", "oot"}
