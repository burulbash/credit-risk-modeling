from __future__ import annotations

import pandas as pd

from src.config import ID_AND_DATE_COLS, PD_EXCLUDED_COLUMNS, PD_SAFETY_EXCLUDE_TOKENS


def get_pd_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Return leakage-safe PD feature columns and excluded suspicious columns."""

    explicit_drop_cols = set(ID_AND_DATE_COLS + PD_EXCLUDED_COLUMNS)
    candidate_features = [col for col in df.columns if col not in explicit_drop_cols]

    suspicious_columns = [
        col
        for col in candidate_features
        if any(token in col.lower() for token in PD_SAFETY_EXCLUDE_TOKENS)
    ]

    feature_columns = [col for col in candidate_features if col not in suspicious_columns]

    return feature_columns, suspicious_columns


def infer_feature_types(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    numeric_features = [
        col
        for col in feature_columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_features = [
        col
        for col in feature_columns
        if col not in numeric_features
    ]

    return numeric_features, categorical_features
