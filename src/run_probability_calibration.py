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
from sklearn.isotonic import IsotonicRegression

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import DATE_COL, PD_MODELS_DIR, PLOTS_DIR, REPORTS_DIR, TARGET  # noqa: E402
from src.db import read_csv_table, read_postgres_table  # noqa: E402
from src.metrics import compute_binary_metrics  # noqa: E402
from src.splitting import make_time_split  # noqa: E402


ID_COLS = ["loan_id", "application_id", "client_id", DATE_COL, TARGET]


def ensure_output_dirs() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def predict_positive_class(model, x: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(f"Loaded artifact {type(model)} does not support predict_proba.")

    return model.predict_proba(x)[:, 1]


def build_pd_bands(
    y_true: pd.Series,
    y_score: np.ndarray,
    model_name: str,
    prediction_type: str,
    n_bands: int = 10,
) -> pd.DataFrame:
    data = pd.DataFrame(
        {
            "target": y_true.to_numpy(),
            "score": y_score,
        }
    )

    data["band"] = pd.qcut(
        data["score"].rank(method="first"),
        q=n_bands,
        labels=False,
        duplicates="drop",
    )

    report = (
        data.groupby("band", as_index=False)
        .agg(
            rows=("target", "size"),
            defaults=("target", "sum"),
            observed_default_rate=("target", "mean"),
            mean_predicted_pd=("score", "mean"),
            min_predicted_pd=("score", "min"),
            max_predicted_pd=("score", "max"),
        )
        .sort_values("band")
    )

    report["model"] = model_name
    report["prediction_type"] = prediction_type

    return report[
        [
            "model",
            "prediction_type",
            "band",
            "rows",
            "defaults",
            "observed_default_rate",
            "mean_predicted_pd",
            "min_predicted_pd",
            "max_predicted_pd",
        ]
    ]


def plot_raw_vs_calibrated(
    bands: pd.DataFrame,
    model_name: str,
) -> None:
    model_bands = bands[bands["model"] == model_name].copy()

    if model_bands.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 5))

    for prediction_type, group in model_bands.groupby("prediction_type"):
        ax.plot(
            group["band"],
            group["mean_predicted_pd"],
            marker="o",
            label=f"{prediction_type}: predicted PD",
        )

    observed = (
        model_bands[model_bands["prediction_type"] == "raw"]
        .sort_values("band")
        [["band", "observed_default_rate"]]
    )

    if observed.empty:
        observed = (
            model_bands.sort_values("band")
            .drop_duplicates("band")
            [["band", "observed_default_rate"]]
        )

    ax.plot(
        observed["band"],
        observed["observed_default_rate"],
        marker="o",
        linestyle="--",
        label="observed default rate",
    )

    ax.set_title(f"Raw vs calibrated PD by band - {model_name}")
    ax.set_xlabel("PD band")
    ax.set_ylabel("Rate")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"calibration_raw_vs_calibrated_{model_name}_oot.png", dpi=140)
    plt.close(fig)


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
    parser = argparse.ArgumentParser(description="Calibrate PD model probabilities.")
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

    if TARGET not in df.columns:
        raise ValueError(f"Target column is missing: {TARGET}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL, TARGET]).copy()

    train_df, valid_df, oot_df, _ = make_time_split(df)

    model_paths = sorted(PD_MODELS_DIR.glob("*.joblib"))

    if not model_paths:
        raise FileNotFoundError(
            f"No PD model artifacts found in {PD_MODELS_DIR}. "
            "Run src/train_pd.py first."
        )

    print(f"Loaded mart shape: {df.shape}")
    print(f"Reading PD models from: {PD_MODELS_DIR}")

    summary_rows: list[dict[str, object]] = []
    band_reports: list[pd.DataFrame] = []

    oot_predictions = oot_df[[col for col in ID_COLS if col in oot_df.columns]].copy()

    for model_path in model_paths:
        model_name = model_path.stem
        print(f"Calibrating model: {model_name}")

        model = joblib.load(model_path)

        valid_raw = predict_positive_class(model, valid_df)
        oot_raw = predict_positive_class(model, oot_df)

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(valid_raw, valid_df[TARGET].to_numpy())

        oot_calibrated = calibrator.predict(oot_raw)
        oot_calibrated = np.clip(oot_calibrated, 0.0, 1.0)

        for prediction_type, scores in [
            ("raw", oot_raw),
            ("calibrated", oot_calibrated),
        ]:
            metrics = compute_binary_metrics(oot_df[TARGET], scores)

            summary_rows.append(
                {
                    "model": model_name,
                    "prediction_type": prediction_type,
                    **metrics,
                }
            )

            band_reports.append(
                build_pd_bands(
                    y_true=oot_df[TARGET],
                    y_score=scores,
                    model_name=model_name,
                    prediction_type=prediction_type,
                )
            )

        oot_predictions[f"pd_{model_name}_raw"] = oot_raw
        oot_predictions[f"pd_{model_name}_calibrated"] = oot_calibrated

    summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["model", "prediction_type"])
        .reset_index(drop=True)
    )

    bands = pd.concat(band_reports, ignore_index=True)

    summary.to_csv(REPORTS_DIR / "probability_calibration_summary.csv", index=False)
    bands.to_csv(REPORTS_DIR / "probability_calibration_by_band.csv", index=False)
    oot_predictions.to_csv(REPORTS_DIR / "oot_predictions_pd_models_calibrated.csv", index=False)

    for model_name in summary["model"].unique():
        plot_raw_vs_calibrated(bands, model_name)

    print()
    print("Probability calibration summary")
    print(summary)

    print()
    print(f"Saved: {REPORTS_DIR / 'probability_calibration_summary.csv'}")
    print(f"Saved: {REPORTS_DIR / 'probability_calibration_by_band.csv'}")
    print(f"Saved: {REPORTS_DIR / 'oot_predictions_pd_models_calibrated.csv'}")


if __name__ == "__main__":
    main()
