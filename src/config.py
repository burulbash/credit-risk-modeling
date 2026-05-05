from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
PLOTS_DIR = PROJECT_ROOT / "outputs" / "plots"

MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
PD_MODELS_DIR = MODELS_DIR / "pd"
SCORECARD_MODELS_DIR = MODELS_DIR / "scorecard"

TARGET = "target_default_90dpd_12m"
DATE_COL = "application_date"

TRAIN_SIZE = 0.70
VALID_SIZE = 0.15
OOT_SIZE = 0.15

RANDOM_STATE = 42


ID_AND_DATE_COLS = [
    "loan_id",
    "application_id",
    "client_id",
    "application_date",
    "disbursement_date",
]


# These columns are not valid PD model features.
# They are either target columns, future performance fields, LGD/EAD outcome helpers,
# or existing decision engine outputs used only for comparison.
PD_EXCLUDED_COLUMNS = [
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


# Secondary safety check. The explicit list above is the main anti-leakage control.
# This name-based guard is kept only to catch accidental future/outcome columns.
PD_SAFETY_EXCLUDE_TOKENS = [
    "target",
    "future",
    "recovered",
    "collection",
    "dpd_12m",
    "first_90",
    "max_dpd",
    "observed_mob",
]


MODEL_PARAMS = {
    "logistic_regression": {
        "max_iter": 2000,
        "solver": "lbfgs",
        "random_state": RANDOM_STATE,
    },
    "decision_tree": {
        "max_depth": 5,
        "min_samples_leaf": 200,
        "random_state": RANDOM_STATE,
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 8,
        "min_samples_leaf": 100,
        "n_jobs": -1,
        "random_state": RANDOM_STATE,
    },
    "xgboost": {
        "n_estimators": 300,
        "max_depth": 3,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    },
}


# Scorecard convention:
# 600 points at 20:1 good/bad odds, with 50 points doubling the odds.
SCORECARD_BASE_SCORE = 600
SCORECARD_PDO = 50
SCORECARD_BASE_ODDS = 20
SCORECARD_BANDS = [-float("inf"), 550, 600, 650, 700, 750, float("inf")]
SCORECARD_BAND_LABELS = [
    "F_<550",
    "E_550_599",
    "D_600_649",
    "C_650_699",
    "B_700_749",
    "A_750_plus",
]
