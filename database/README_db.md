# Synthetic Credit Risk / Lending Analytics DB

Synthetic PostgreSQL-ready relational dataset for retail lending / MFO / digital lending analytics.

## Architecture (short plan)

- **Entity layer**: `clients`, `client_contact_info`
- **Origination layer**: `applications`, `bureau_snapshot`, `employment_income_snapshot`, `fraud_flags`, `decision_engine_results`, `application_events`
- **Booked portfolio layer**: `loans`, `payment_schedule`
- **Behavior layer**: `payments`, `loan_monthly_snapshot`, `collections_actions`
- **External context layer**: `macro_monthly_factors`

The dataset is intentionally relational, not flat, so you can practice:
- joins and leakage control,
- application-date training base construction,
- funnel analytics,
- time-based drift,
- MoB / vintage / FPD / roll-rate,
- score / approval / bad-rate trade-off.

## What is included

Files in this package:
- `schema.sql`
- `generate_data.py`
- `load_postgres.sql`
- `data_dictionary.md`
- `qa_report.md`
- `sql_practice_tasks.md`

## Main design choices

- **Fully synthetic**: no real PII, no real bureau data, no real customers.
- **Kazakhstan-like retail lending flavor**: regions, products, channels, MFO / unsecured retail patterns.
- **Not toy-sized by default**: medium profile targets realistic practice volume.
- **Temporal drift**: policy changes, macro stress months, changing approval appetite, changing observed risk.
- **Useful but imperfect predictors**: bureau score, DTI, income stability, repeat status, channel, product, region, macro all matter, but with noise and interactions.
- **Moderate data dirt**: nulls, dirty phones/emails, employer spelling noise, thin-file missing bureau fields, rare status/timestamp inconsistencies, some duplicate-like contact identities.
- **Leakage is possible if joined incorrectly**: post-origination tables exist by design.

## Table set

Required tables:
- clients
- client_contact_info
- applications
- bureau_snapshot
- employment_income_snapshot
- decision_engine_results
- loans
- payment_schedule
- payments
- loan_monthly_snapshot
- collections_actions
- macro_monthly_factors

Optional but included:
- fraud_flags
- application_events

## Size profiles

Configured in `generate_data.py`:

- `small`: ~18k clients / 32k applications
- `medium` (default): ~82k clients / 150k applications
- `large`: ~180k clients / 340k applications

Expected realized volumes depend on approval / booking simulation. For `medium`, the script is designed to land roughly around:
- clients: ~82k
- applications: ~150k
- approved applications: typically ~45%–60%
- booked loans: typically ~60k–90k
- monthly snapshots: typically 1M+
- payments: typically 700k+
- collections actions: typically 100k+

## Requirements

- Python 3.11+
- pandas
- numpy
- faker (optional, generator works without it)
- PostgreSQL 13+ recommended

Install example:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy faker
```

## How to generate data

Medium by default:

```bash
python generate_data.py --size medium --seed 42 --output-dir ./exports_medium
```

Other sizes:

```bash
python generate_data.py --size small --seed 42 --output-dir ./exports_small
python generate_data.py --size large --seed 42 --output-dir ./exports_large
```

Notes:
- generation is reproducible by `--seed`
- CSVs are exported per table into `output-dir`
- `_generation_summary.csv` is also exported with headline metrics

## How to create schema in PostgreSQL

```bash
createdb synthetic_credit_risk
psql -d synthetic_credit_risk -f schema.sql
```

## How to load CSVs into PostgreSQL

Option A: use `load_postgres.sql`

1. Open `load_postgres.sql`
2. Replace `{{DATA_DIR}}` with the absolute path to your export folder
3. Run:

```bash
psql -d synthetic_credit_risk -f load_postgres.sql
```

Option B: run `\copy` commands manually

Example:

```bash
psql -d synthetic_credit_risk
\copy clients FROM '/absolute/path/exports_medium/clients.csv' WITH (FORMAT csv, HEADER true)
\copy client_contact_info FROM '/absolute/path/exports_medium/client_contact_info.csv' WITH (FORMAT csv, HEADER true)
\copy applications FROM '/absolute/path/exports_medium/applications.csv' WITH (FORMAT csv, HEADER true)
\copy bureau_snapshot FROM '/absolute/path/exports_medium/bureau_snapshot.csv' WITH (FORMAT csv, HEADER true)
\copy employment_income_snapshot FROM '/absolute/path/exports_medium/employment_income_snapshot.csv' WITH (FORMAT csv, HEADER true)
\copy fraud_flags FROM '/absolute/path/exports_medium/fraud_flags.csv' WITH (FORMAT csv, HEADER true)
\copy decision_engine_results FROM '/absolute/path/exports_medium/decision_engine_results.csv' WITH (FORMAT csv, HEADER true)
\copy loans FROM '/absolute/path/exports_medium/loans.csv' WITH (FORMAT csv, HEADER true)
\copy payment_schedule FROM '/absolute/path/exports_medium/payment_schedule.csv' WITH (FORMAT csv, HEADER true)
\copy payments FROM '/absolute/path/exports_medium/payments.csv' WITH (FORMAT csv, HEADER true)
\copy loan_monthly_snapshot FROM '/absolute/path/exports_medium/loan_monthly_snapshot.csv' WITH (FORMAT csv, HEADER true)
\copy collections_actions FROM '/absolute/path/exports_medium/collections_actions.csv' WITH (FORMAT csv, HEADER true)
\copy macro_monthly_factors FROM '/absolute/path/exports_medium/macro_monthly_factors.csv' WITH (FORMAT csv, HEADER true)
\copy application_events FROM '/absolute/path/exports_medium/application_events.csv' WITH (FORMAT csv, HEADER true)
```

## Recommended learning workflows

### SQL / BI / risk analytics
- build an application-date base table with only pre-decision info
- build an approval funnel
- compare policy versions before/after
- analyze approval vs bad rate by score band
- compute FPD / ever30 / ever60 / ever90
- compute vintages by disbursement month and MoB
- compute roll-rates between delinquency buckets
- monitor collections effectiveness by stage / action type

### pandas / EDA / ML
- build one-row-per-application dataset
- join only application-time features
- derive targets from `loan_monthly_snapshot`
- train logistic regression / trees / RF / gradient boosting baselines
- compute ROC-AUC / Gini / KS / PSI
- compare time-based split vs random split

## Leakage warning

Safe pre-origination modeling tables:
- applications
- bureau_snapshot
- employment_income_snapshot
- fraud_flags
- decision_engine_results (only if your modeling point is **after** engine score; otherwise exclude score and decisions)
- macro_monthly_factors as of application month

Leakage-prone tables for application-level PD model:
- loans
- payment_schedule
- payments
- loan_monthly_snapshot
- collections_actions
- application_events **after** final decision / disbursement events

## Practical targets you can derive

From booked loans:
- `target_default_30dpd_6m`
- `target_default_90dpd_12m`
- `fpd_flag`
- `ever_30 / ever_60 / ever_90`
- roll-rates between `current`, `1_29`, `30_59`, `60_89`, `90_plus`
- early closure / prepayment
- recovery effectiveness

## Notes on realism

This is not a “single Kaggle table”. The generator intentionally keeps the database normalized enough to train:
- FK joins
- one-to-many joins
- deduping and latest-record logic
- window functions
- time-window aggregations
- leakage-safe mart building

See `qa_report.md` for the embedded business logic and dirty-data inventory.
