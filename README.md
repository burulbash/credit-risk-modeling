# End-to-End Credit Risk Modeling on a Synthetic Lending Database

Portfolio project with a full credit risk modeling workflow on top of a synthetic retail lending database.

The idea was to avoid the typical “train a model on a ready-made CSV” format. Instead, the project starts from a normalized lending database, loads it into PostgreSQL, builds a leakage-safe modeling mart with SQL, and then trains and validates classic credit risk models in Python.

The project covers:

- synthetic lending database generation;
- PostgreSQL schema and data loading;
- SQL-based modeling mart;
- PD model training and out-of-time validation;
- model comparison: Logistic Regression, Decision Tree, Random Forest, XGBoost;
- probability calibration;
- WOE / IV / scorecard;
- PSI monitoring;
- LGD / EAD / Expected Loss analysis.

The data is fully synthetic and does not contain real customer information.

---

## Project structure

```text
CreditRiskModeling/
├── database/
│   ├── generate_data.py
│   ├── schema.sql
│   ├── load_postgres.sql
│   ├── data_dictionary.md
│   ├── qa_report.md
│   └── validation_note.md
│
├── sql/
│   ├── 01_build_modeling_mart.sql
│   └── 02_modeling_mart_quality_checks.sql
│
├── src/
│   ├── prepare_postgres_exports.py
│   ├── train_pd.py
│   ├── run_probability_calibration.py
│   ├── run_scorecard.py
│   ├── run_monitoring_psi.py
│   └── run_lgd_ead_expected_loss.py
│
├── data/
│   └── sample/
│       └── credit_risk_modeling_mart_sample.csv
│
├── outputs/
│   ├── reports/
│   └── plots/
│
├── README.md
├── requirements.txt
└── .gitignore
```

The full generated dataset is not stored in the repository. It can be reproduced with the generator using a fixed seed.

A small sample of the modeling mart is included under `data/sample/` for quick inspection and lightweight testing.

---

## Business context

The synthetic database represents a simplified retail lending lifecycle:

```text
client profile
    -> application
    -> bureau / income / fraud checks
    -> decision engine result
    -> booked loan
    -> payment schedule
    -> actual payments
    -> monthly loan performance
    -> collections and recoveries
```

The main modeling task is to estimate Probability of Default for booked loans.

The PD target is:

```text
target_default_90dpd_12m = 1 if the loan reaches 90+ DPD within the first 12 months on book
```

In SQL, the target is defined from monthly performance observations:

```sql
MAX(
    CASE
        WHEN mob BETWEEN 0 AND 11 AND dpd >= 90 THEN 1
        ELSE 0
    END
) AS target_default_90dpd_12m
```

Post-origination tables such as payments, monthly performance snapshots and collections are used only for target construction and LGD/EAD analysis. They are not used as PD model features.

---

## Database layer

The database is generated as a set of normalized tables:

```text
clients
client_contact_info
applications
bureau_snapshot
employment_income_snapshot
fraud_flags
decision_engine_results
loans
payment_schedule
payments
loan_monthly_snapshot
collections_actions
macro_monthly_factors
application_events
```

The generated `medium` dataset used in this project had the following scale:

| Object | Count |
|---|---:|
| Clients | 82,000 |
| Applications | 150,000 |
| Booked loans | 90,180 |
| Payment schedule rows | 1,390,455 |
| Payments rows | 1,119,347 |
| Loan monthly snapshot rows | 1,145,727 |
| Collections actions rows | 83,809 |

Only loans with at least 12 months of observed performance are kept in the final PD modeling mart.

The final modeling mart contains:

| Metric | Value |
|---|---:|
| Rows | 52,544 |
| Unique loans | 52,544 |
| Unique applications | 52,544 |
| Unique clients | 37,516 |
| Date range | 2022-01-01 to 2025-03-28 |
| 90+ DPD 12M default rate | 3.04% |
| 30+ DPD 6M rate | 8.43% |

---

## Leakage control

A major part of the project is preventing data leakage.

For the PD model, I used only features that would be available at application or origination time:

- client demographics;
- application parameters;
- bureau snapshot;
- income and employment snapshot;
- fraud and verification flags;
- macro indicators by application month.

I excluded future or outcome-related fields from PD training, including:

- future DPD values;
- monthly performance status;
- write-off outcomes;
- collections actions;
- recovery amounts;
- LGD/EAD helper columns;
- existing decision engine score and PD estimate.

The existing decision engine columns are kept in the mart for benchmarking and analysis, but they are not used in the main challenger PD model.

---

## Modeling approach

The data is split by application date:

| Split | Rows | Period | Default rate |
|---|---:|---|---:|
| Train | 36,780 | 2022-01-01 to 2024-03-02 | 2.90% |
| Validation | 7,882 | 2024-03-02 to 2024-08-14 | 3.44% |
| OOT | 7,882 | 2024-08-14 to 2025-03-28 | 3.32% |

The out-of-time split is used as the main model quality check.

Models trained:

- Dummy baseline;
- Logistic Regression;
- Decision Tree;
- Random Forest;
- XGBoost.

Metrics:

- ROC-AUC;
- Gini;
- KS;
- Average Precision;
- Brier Score.

---

## PD model results

OOT model comparison:

| Model | ROC-AUC | Gini | KS | Average Precision | Brier Score |
|---|---:|---:|---:|---:|---:|
| Random Forest | 0.694 | 0.388 | 0.292 | 0.073 | 0.0317 |
| XGBoost | 0.688 | 0.376 | 0.294 | 0.079 | 0.0315 |
| Logistic Regression | 0.674 | 0.349 | 0.280 | 0.069 | 0.0357 |
| Decision Tree | 0.647 | 0.293 | 0.257 | 0.059 | 0.0318 |
| Dummy baseline | 0.500 | 0.000 | 0.000 | 0.033 | 0.0322 |

Random Forest achieved the best OOT ROC-AUC. XGBoost had the best probability calibration after validation-based calibration.

---

## Probability calibration

For credit risk, ranking quality is not enough. If model output is used as PD, predicted probabilities need to be checked against observed default rates.

I applied post-hoc probability calibration using the validation split and evaluated the calibrated models on the OOT sample.

Selected OOT calibration results:

| Model | Prediction type | Mean predicted PD | Observed default rate | Brier Score |
|---|---|---:|---:|---:|
| XGBoost | calibrated | 3.28% | 3.32% | 0.0315 |
| Random Forest | calibrated | 3.43% | 3.32% | 0.0316 |
| Logistic Regression | calibrated | 4.57% | 3.32% | 0.0319 |

The calibration step improved the probability quality of the Logistic Regression model substantially. XGBoost gave the best calibrated PD estimates in this experiment.

---

## WOE / IV / Scorecard

I also built a classic credit risk scorecard pipeline:

- WOE binning;
- IV calculation;
- IV-based feature screening;
- Logistic Regression on WOE-transformed features;
- score conversion;
- score bands;
- observed bad rate by score band.

Top IV features:

| Feature | IV |
|---|---:|
| segment | 0.288 |
| bureau_file_thin_flag | 0.215 |
| bureau_score | 0.183 |
| total_limit | 0.181 |
| oldest_trade_months | 0.177 |
| closed_loans_count | 0.124 |
| outstanding_debt | 0.118 |
| max_dpd_24m | 0.107 |

Scorecard OOT quality:

| Metric | Value |
|---|---:|
| ROC-AUC | 0.677 |
| Gini | 0.355 |
| KS | 0.284 |
| Brier Score | 0.0317 |

Score bands on OOT sample:

| Score band | Loans | Observed default rate | Avg predicted PD |
|---|---:|---:|---:|
| F < 550 | 92 | 10.87% | 11.01% |
| E 550-599 | 1,041 | 6.92% | 6.23% |
| D 600-649 | 2,256 | 4.39% | 3.48% |
| C 650-699 | 2,946 | 2.00% | 1.76% |
| B 700-749 | 1,547 | 1.42% | 1.01% |

The scorecard produced monotonic risk bands on the OOT sample.

---

## Monitoring / PSI

I calculated Population Stability Index for selected features between train, validation and OOT periods.

Most core features were stable:

| Feature | Train vs OOT PSI | Status |
|---|---:|---|
| bureau_score | 0.0047 | stable |
| verified_income | 0.0027 | stable |
| outstanding_debt | 0.0022 | stable |
| product_type | 0.0016 | stable |
| channel | 0.0003 | stable |

`client_tenure_days` showed a large PSI shift. This is expected because the OOT period is later in time, so customer tenure naturally increases. I flagged it as a time-trend feature and monitor it separately from the core risk features.

---

## LGD / EAD / Expected Loss

The project also includes a simplified Expected Loss layer:

```text
Expected Loss = PD x LGD x EAD
```

Where:

- PD comes from the calibrated XGBoost model;
- EAD is proxied by outstanding principal during the first 12 months;
- LGD is estimated from synthetic recovery information.

OOT portfolio summary:

| Metric | Value |
|---|---:|
| Loans | 7,882 |
| Observed default rate | 3.32% |
| Average PD | 3.28% |
| Average LGD | 96.87% |
| Average EAD | 545,705 |
| Total EAD | 4.30B |
| Total Expected Loss | 130.4M |
| Portfolio EL rate | 3.03% |

Expected Loss by risk band:

| Risk band | Loans | Observed default rate | Avg PD | EL rate |
|---|---:|---:|---:|---:|
| A lowest risk | 1,577 | 1.20% | 1.12% | 1.11% |
| B | 1,576 | 1.78% | 1.74% | 1.71% |
| C | 1,576 | 2.79% | 2.48% | 2.42% |
| D | 1,576 | 3.49% | 3.73% | 3.60% |
| E highest risk | 1,577 | 7.36% | 7.33% | 6.80% |

The risk bands show a clear increase in observed default rate, predicted PD and Expected Loss rate.

---

## How to run

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate synthetic data:

```bash
python database/generate_data.py --size medium --seed 42 --output-dir data/raw/exports_medium
```

Prepare PostgreSQL-ready CSV exports:

```bash
python src/prepare_postgres_exports.py \
  --input-dir data/raw/exports_medium \
  --output-dir data/raw/exports_medium_pg
```

Create PostgreSQL database:

```bash
createdb -h localhost -p 5432 -U postgres credit_risk_synth
```

Create schema:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f database/schema.sql
```

Create a local load script with your absolute path.

For Git Bash on Windows:

```bash
DATA_DIR="$(pwd -W)/data/raw/exports_medium_pg"
sed "s#{{DATA_DIR}}#$DATA_DIR#g" database/load_postgres.sql > database/load_postgres_local.sql
```

Load data into PostgreSQL:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f database/load_postgres_local.sql
```

Build the modeling mart:

```bash
psql -v ON_ERROR_STOP=1 \
  -h localhost \
  -p 5432 \
  -U postgres \
  -d credit_risk_synth \
  -f sql/01_build_modeling_mart.sql
```

Run the PD model training:

```bash
python src/train_pd.py --source postgres --db-name credit_risk_synth --db-user postgres
```

Run probability calibration:

```bash
python src/run_probability_calibration.py --db-name credit_risk_synth --db-user postgres
```

Run the scorecard pipeline:

```bash
python src/run_scorecard.py --db-name credit_risk_synth --db-user postgres
```

Run PSI monitoring:

```bash
python src/run_monitoring_psi.py --db-name credit_risk_synth --db-user postgres
```

Run Expected Loss analysis:

```bash
python src/run_lgd_ead_expected_loss.py \
  --db-name credit_risk_synth \
  --db-user postgres \
  --preferred-model xgboost
```

---

## Lightweight sample mode

A small sample of the modeling mart is included for quick inspection:

```text
data/sample/credit_risk_modeling_mart_sample.csv
```

The full dataset is intentionally not stored in GitHub. It is reproducible from the generator.

You can run the PD training script on the sample CSV:

```bash
python src/train_pd.py \
  --source csv \
  --csv-path data/sample/credit_risk_modeling_mart_sample.csv
```

For the full project results, use the PostgreSQL flow.

---

## Main outputs

Reports are saved under:

```text
outputs/reports/
```

Key reports:

```text
pd_model_metrics.csv
probability_calibration_summary.csv
scorecard_iv_report.csv
scorecard_model_metrics.csv
scorecard_bad_rate_by_score_band.csv
psi_core_feature_summary.csv
expected_loss_portfolio_summary.csv
expected_loss_by_risk_band.csv
```

Plots are saved under:

```text
outputs/plots/
```

Key plots:

```text
roc_random_forest_oot.png
roc_xgboost_oot.png
calibration_raw_vs_calibrated_xgboost_oot.png
scorecard_bad_rate_by_score_band_oot.png
top_psi_features_train_vs_oot.png
expected_loss_rate_by_risk_band_oot.png
```

---

## Limitations

This is a synthetic portfolio, so the results should not be interpreted as real credit risk estimates.

Important limitations:

- synthetic data generation assumptions drive the final model performance;
- LGD and EAD are simplified proxy estimates;
- no regulatory validation is performed;
- no production feature store is implemented;
- no online scoring API is included in the current version.

The focus of this project is the end-to-end modeling workflow: database design, leakage-safe feature construction, model validation, calibration, scorecarding, monitoring and Expected Loss analysis.

---

## What this project demonstrates

This project shows a complete classical ML workflow for credit risk:

- working with normalized relational data instead of a ready-made CSV;
- building a SQL modeling mart;
- defining a time-dependent default target;
- preventing post-origination leakage;
- using out-of-time validation;
- comparing several ML models;
- checking probability calibration;
- building a WOE/IV scorecard;
- monitoring feature stability;
- calculating Expected Loss.