from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve


LEAKAGE_PATTERNS = ('future', 'next', 'target', 'post_', '_after', 'dpd_90')


@dataclass
class WOEFeatureSpec:
    feature: str
    kind: str
    mapping: dict[str, dict[str, float | str]]
    iv: float


class WOETransformer:
    def __init__(self, min_bin_size: float = 0.05, max_bins: int = 5):
        self.min_bin_size = min_bin_size
        self.max_bins = max_bins
        self.specs: dict[str, WOEFeatureSpec] = {}
        self.iv_table_: pd.DataFrame | None = None

    @staticmethod
    def _safe_woe(events: float, non_events: float, total_events: float, total_non_events: float) -> float:
        events = max(events, 0.5)
        non_events = max(non_events, 0.5)
        rate_event = events / max(total_events, 1.0)
        rate_non_event = non_events / max(total_non_events, 1.0)
        return float(np.log(rate_non_event / rate_event))

    def _fit_numeric(self, s: pd.Series, y: pd.Series, feature: str) -> WOEFeatureSpec:
        non_null = s.dropna()
        n_unique = non_null.nunique(dropna=True)
        q = min(self.max_bins, max(2, n_unique))
        try:
            _, bins = pd.qcut(non_null, q=q, duplicates='drop', retbins=True)
            bins = np.unique(bins)
            if len(bins) < 3:
                raise ValueError('Too few bins')
        except Exception:
            bins = np.array([-np.inf, s.min(skipna=True), s.max(skipna=True), np.inf], dtype=float)
            bins = np.unique(bins)
        if bins[0] != -np.inf:
            bins[0] = -np.inf
        if bins[-1] != np.inf:
            bins[-1] = np.inf

        binned = pd.cut(s, bins=bins, include_lowest=True)
        tmp = pd.DataFrame({'bin': binned.astype(str), 'y': y})
        grouped = tmp.groupby('bin', dropna=False)['y'].agg(['count', 'sum'])
        grouped['non_event'] = grouped['count'] - grouped['sum']
        total_events = float(grouped['sum'].sum())
        total_non_events = float(grouped['non_event'].sum())
        mapping: dict[str, dict[str, float | str]] = {}
        iv = 0.0
        for bin_name, row in grouped.iterrows():
            woe = self._safe_woe(float(row['sum']), float(row['non_event']), total_events, total_non_events)
            dist_event = max(float(row['sum']), 0.5) / max(total_events, 1.0)
            dist_non_event = max(float(row['non_event']), 0.5) / max(total_non_events, 1.0)
            iv_bin = (dist_non_event - dist_event) * np.log(dist_non_event / dist_event)
            mapping[str(bin_name)] = {'woe': woe, 'points_label': str(bin_name), 'iv_bin': float(iv_bin)}
            iv += float(iv_bin)
        mapping['__bins__'] = {'bins_json': json.dumps([float(x) if np.isfinite(x) else str(x) for x in bins])}
        if s.isna().any():
            miss = tmp['y'][s.isna()]
            miss_events = float(miss.sum())
            miss_non_events = float(miss.shape[0] - miss.sum())
            mapping['__MISSING__'] = {
                'woe': self._safe_woe(miss_events, miss_non_events, total_events, total_non_events),
                'points_label': 'MISSING',
                'iv_bin': 0.0,
            }
        return WOEFeatureSpec(feature=feature, kind='numeric', mapping=mapping, iv=iv)

    def _fit_categorical(self, s: pd.Series, y: pd.Series, feature: str) -> WOEFeatureSpec:
        tmp = pd.DataFrame({'cat': s.fillna('MISSING').astype(str), 'y': y})
        grouped = tmp.groupby('cat', dropna=False)['y'].agg(['count', 'sum'])
        grouped['non_event'] = grouped['count'] - grouped['sum']
        total_events = float(grouped['sum'].sum())
        total_non_events = float(grouped['non_event'].sum())
        mapping: dict[str, dict[str, float | str]] = {}
        iv = 0.0
        for cat, row in grouped.iterrows():
            woe = self._safe_woe(float(row['sum']), float(row['non_event']), total_events, total_non_events)
            dist_event = max(float(row['sum']), 0.5) / max(total_events, 1.0)
            dist_non_event = max(float(row['non_event']), 0.5) / max(total_non_events, 1.0)
            iv_bin = (dist_non_event - dist_event) * np.log(dist_non_event / dist_event)
            mapping[str(cat)] = {'woe': woe, 'points_label': str(cat), 'iv_bin': float(iv_bin)}
            iv += float(iv_bin)
        return WOEFeatureSpec(feature=feature, kind='categorical', mapping=mapping, iv=iv)

    def fit(self, df: pd.DataFrame, target: str, features: list[str]) -> 'WOETransformer':
        y = df[target].astype(int)
        iv_rows = []
        for feature in features:
            s = df[feature]
            if pd.api.types.is_numeric_dtype(s):
                spec = self._fit_numeric(s, y, feature)
            else:
                spec = self._fit_categorical(s, y, feature)
            self.specs[feature] = spec
            iv_rows.append({'feature': feature, 'iv': spec.iv, 'type': spec.kind})
        self.iv_table_ = pd.DataFrame(iv_rows).sort_values('iv', ascending=False).reset_index(drop=True)
        return self

    def _transform_series(self, s: pd.Series, spec: WOEFeatureSpec) -> pd.Series:
        if spec.kind == 'numeric':
            raw_bins = json.loads(spec.mapping['__bins__']['bins_json'])
            bins = [(-np.inf if x == '-inf' else np.inf if x == 'inf' else float(x)) for x in raw_bins]
            binned = pd.cut(s, bins=bins, include_lowest=True).astype(str)
            out = binned.map(lambda key: spec.mapping.get(key, {}).get('woe', 0.0)).astype(float)
            if '__MISSING__' in spec.mapping:
                out = out.where(~s.isna(), float(spec.mapping['__MISSING__']['woe']))
            return out.fillna(0.0)
        cat = s.fillna('MISSING').astype(str)
        return cat.map(lambda key: spec.mapping.get(key, {}).get('woe', 0.0)).astype(float).fillna(0.0)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = pd.DataFrame(index=df.index)
        for feature, spec in self.specs.items():
            transformed[f'woe_{feature}'] = self._transform_series(df[feature], spec)
        return transformed

    def to_scorecard_table(self, logistic_model: Any, base_score: int = 600, pdo: int = 50, base_odds: int = 20) -> pd.DataFrame:
        factor = pdo / np.log(2)
        offset = base_score - factor * np.log(base_odds)
        coefs = np.ravel(logistic_model.coef_)
        rows = []
        n_features = len(self.specs)
        base_points = offset - factor * float(logistic_model.intercept_[0]) / max(n_features, 1)
        for idx, (feature, spec) in enumerate(self.specs.items()):
            beta = float(coefs[idx])
            for bin_name, values in spec.mapping.items():
                if bin_name.startswith('__'):
                    continue
                woe = float(values['woe'])
                points = -factor * (beta * woe) + base_points / max(n_features, 1)
                rows.append(
                    {
                        'feature': feature,
                        'bin_or_category': values['points_label'],
                        'woe': round(woe, 6),
                        'coefficient': round(beta, 6),
                        'points': round(points, 2),
                    }
                )
        return pd.DataFrame(rows).sort_values(['feature', 'points'], ascending=[True, False]).reset_index(drop=True)


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    outputs = base_dir / 'outputs'
    reports = outputs / 'reports'
    plots = outputs / 'plots'
    models = outputs / 'models'
    for d in (outputs, reports, plots, models):
        d.mkdir(parents=True, exist_ok=True)
    return {'outputs': outputs, 'reports': reports, 'plots': plots, 'models': models}


def save_model(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path: Path) -> Any:
    return joblib.load(path)


def compute_metrics(y_true: pd.Series | np.ndarray, y_score: pd.Series | np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    if len(np.unique(y_true)) < 2:
        return {
            'roc_auc': np.nan,
            'gini': np.nan,
            'ks': np.nan,
            'average_precision': np.nan,
            'pr_auc': np.nan,
            'brier_score': np.nan,
        }

    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks = float(np.max(tpr - fpr))
    avg_precision = float(average_precision_score(y_true, y_score))
    brier = brier_score_loss(y_true, y_score)

    return {
        'roc_auc': float(auc),
        'gini': float(2 * auc - 1),
        'ks': ks,
        'average_precision': avg_precision,
        'pr_auc': avg_precision,
        'brier_score': float(brier),
    }


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def plot_calibration(y_true: np.ndarray, y_score: np.ndarray, output_path: Path, title: str) -> None:
    frac_pos, mean_pred = calibration_curve(y_true, y_score, n_bins=10, strategy='quantile')
    plt.figure(figsize=(7, 5))
    plt.plot(mean_pred, frac_pos, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', label='Perfect calibration')
    plt.xlabel('Mean predicted PD')
    plt.ylabel('Observed default rate')
    plt.title(title)
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def plot_feature_importance(importances: pd.Series, output_path: Path, title: str, top_n: int = 15) -> None:
    top = importances.sort_values(ascending=False).head(top_n).sort_values(ascending=True)
    plt.figure(figsize=(8, 6))
    plt.barh(top.index.astype(str), top.values)
    plt.xlabel('Importance')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()


def _psi_from_counts(expected: np.ndarray, actual: np.ndarray) -> float:
    expected = np.where(expected == 0, 1e-6, expected)
    actual = np.where(actual == 0, 1e-6, actual)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def compute_psi_numeric(expected: pd.Series, actual: pd.Series, bins: int = 10) -> float:
    quantiles = np.unique(np.nanpercentile(expected, np.linspace(0, 100, bins + 1)))
    if len(quantiles) < 3:
        return 0.0
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    exp_bins = pd.cut(expected, bins=quantiles, include_lowest=True)
    act_bins = pd.cut(actual, bins=quantiles, include_lowest=True)
    exp_dist = exp_bins.value_counts(normalize=True, sort=False)
    act_dist = act_bins.value_counts(normalize=True, sort=False)
    aligned = pd.concat([exp_dist, act_dist], axis=1).fillna(0.0)
    aligned.columns = ['expected', 'actual']
    return _psi_from_counts(aligned['expected'].values, aligned['actual'].values)


def compute_psi_categorical(expected: pd.Series, actual: pd.Series) -> float:
    exp_dist = expected.fillna('MISSING').astype(str).value_counts(normalize=True)
    act_dist = actual.fillna('MISSING').astype(str).value_counts(normalize=True)
    aligned = pd.concat([exp_dist, act_dist], axis=1).fillna(0.0)
    aligned.columns = ['expected', 'actual']
    return _psi_from_counts(aligned['expected'].values, aligned['actual'].values)


def psi_interpretation(psi: float) -> str:
    if psi < 0.10:
        return 'stable'
    if psi < 0.25:
        return 'minor_shift'
    return 'major_shift'


def make_data_quality_report(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append(
            {
                'column': col,
                'dtype': str(s.dtype),
                'missing_pct': round(float(s.isna().mean()), 4),
                'n_unique': int(s.nunique(dropna=True)),
                'sample_min': s.min() if pd.api.types.is_numeric_dtype(s) else None,
                'sample_max': s.max() if pd.api.types.is_numeric_dtype(s) else None,
            }
        )
    return pd.DataFrame(rows).sort_values(['missing_pct', 'n_unique'], ascending=[False, False]).reset_index(drop=True)


def detect_leakage_candidates(df: pd.DataFrame, target: str, ignore: list[str] | None = None) -> pd.DataFrame:
    ignore = set(ignore or [])
    y = df[target]
    rows = []
    for col in df.columns:
        if col == target or col in ignore:
            continue
        reasons = []
        lower = col.lower()
        if any(pattern in lower for pattern in LEAKAGE_PATTERNS):
            reasons.append('name_pattern')
        if pd.api.types.is_numeric_dtype(df[col]):
            series = df[col]
            if series.notna().sum() > 100 and series.nunique(dropna=True) > 5:
                try:
                    valid = series.notna() & y.notna()
                    auc = roc_auc_score(y[valid], series[valid])
                    auc = max(auc, 1 - auc)
                    if auc > 0.95:
                        reasons.append('single_feature_auc_gt_0.95')
                except Exception:
                    pass
        if reasons:
            rows.append({'column': col, 'reason': '|'.join(reasons)})
    return pd.DataFrame(rows).sort_values('column').reset_index(drop=True)


def month_split(df: pd.DataFrame, date_col: str = 'application_date') -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    train = d[d[date_col] < '2023-09-01'].copy()
    valid = d[(d[date_col] >= '2023-09-01') & (d[date_col] < '2023-11-01')].copy()
    oot = d[d[date_col] >= '2023-11-01'].copy()
    return train, valid, oot


def make_psi_report(train_df: pd.DataFrame, oot_df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for feature in features:
        if pd.api.types.is_numeric_dtype(train_df[feature]):
            psi = compute_psi_numeric(train_df[feature], oot_df[feature])
        else:
            psi = compute_psi_categorical(train_df[feature], oot_df[feature])
        rows.append({'feature': feature, 'psi': round(psi, 6), 'status': psi_interpretation(psi)})
    return pd.DataFrame(rows).sort_values('psi', ascending=False).reset_index(drop=True)


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)


def infer_feature_types(df: pd.DataFrame, target: str, ignore: list[str] | None = None) -> tuple[list[str], list[str]]:
    ignore = set(ignore or []) | {target}
    features = [c for c in df.columns if c not in ignore]
    num = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
    cat = [c for c in features if c not in num]
    return num, cat


def robust_predict_proba(model: Any, X: pd.DataFrame | np.ndarray) -> np.ndarray:
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, 'decision_function'):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    preds = model.predict(X)
    return np.asarray(preds).astype(float)


def explain_psi_thresholds() -> str:
    return 'PSI < 0.10 = stable, 0.10-0.25 = minor shift, > 0.25 = major shift.'
