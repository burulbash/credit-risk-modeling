from __future__ import annotations

import pandas as pd

from src.config import DATE_COL, TARGET, TRAIN_SIZE, VALID_SIZE


def make_time_split(
    df: pd.DataFrame,
    date_col: str = DATE_COL,
    target_col: str = TARGET,
    train_size: float = TRAIN_SIZE,
    valid_size: float = VALID_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = df.copy()
    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data = data.sort_values(date_col).reset_index(drop=True)

    n_rows = len(data)
    train_end = int(n_rows * train_size)
    valid_end = int(n_rows * (train_size + valid_size))

    train_df = data.iloc[:train_end].copy()
    valid_df = data.iloc[train_end:valid_end].copy()
    oot_df = data.iloc[valid_end:].copy()

    split_summary = pd.DataFrame(
        [
            {
                "split": "train",
                "rows": len(train_df),
                "min_date": train_df[date_col].min(),
                "max_date": train_df[date_col].max(),
                "default_rate": train_df[target_col].mean(),
            },
            {
                "split": "valid",
                "rows": len(valid_df),
                "min_date": valid_df[date_col].min(),
                "max_date": valid_df[date_col].max(),
                "default_rate": valid_df[target_col].mean(),
            },
            {
                "split": "oot",
                "rows": len(oot_df),
                "min_date": oot_df[date_col].min(),
                "max_date": oot_df[date_col].max(),
                "default_rate": oot_df[target_col].mean(),
            },
        ]
    )

    return train_df, valid_df, oot_df, split_summary
