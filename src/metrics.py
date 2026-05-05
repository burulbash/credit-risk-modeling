from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
    roc_curve,
)


def compute_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    include_probability_metrics: bool = True,
) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        metrics = {
            "roc_auc": np.nan,
            "gini": np.nan,
            "ks": np.nan,
            "average_precision": np.nan,
        }
    else:
        auc = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        metrics = {
            "roc_auc": float(auc),
            "gini": float(2 * auc - 1),
            "ks": float(np.max(tpr - fpr)),
            "average_precision": float(average_precision_score(y_true, y_score)),
        }

    if include_probability_metrics:
        metrics["brier_score"] = float(brier_score_loss(y_true, y_score))
        metrics["mean_predicted_pd"] = float(np.mean(y_score))
        metrics["observed_default_rate"] = float(np.mean(y_true))

    return metrics
