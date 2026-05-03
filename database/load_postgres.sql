-- Replace {{DATA_DIR}} with the absolute path to your export directory before running.
-- Example:
--   sed "s#{{DATA_DIR}}#/Users/me/projects/synth_credit/exports_medium#g" load_postgres.sql | psql -d synthetic_credit_risk

\copy clients FROM '{{DATA_DIR}}/clients.csv' WITH (FORMAT csv, HEADER true)
\copy client_contact_info FROM '{{DATA_DIR}}/client_contact_info.csv' WITH (FORMAT csv, HEADER true)
\copy applications FROM '{{DATA_DIR}}/applications.csv' WITH (FORMAT csv, HEADER true)
\copy bureau_snapshot FROM '{{DATA_DIR}}/bureau_snapshot.csv' WITH (FORMAT csv, HEADER true)
\copy employment_income_snapshot FROM '{{DATA_DIR}}/employment_income_snapshot.csv' WITH (FORMAT csv, HEADER true)
\copy fraud_flags FROM '{{DATA_DIR}}/fraud_flags.csv' WITH (FORMAT csv, HEADER true)
\copy decision_engine_results FROM '{{DATA_DIR}}/decision_engine_results.csv' WITH (FORMAT csv, HEADER true)
\copy loans FROM '{{DATA_DIR}}/loans.csv' WITH (FORMAT csv, HEADER true)
\copy payment_schedule FROM '{{DATA_DIR}}/payment_schedule.csv' WITH (FORMAT csv, HEADER true)
\copy payments FROM '{{DATA_DIR}}/payments.csv' WITH (FORMAT csv, HEADER true)
\copy loan_monthly_snapshot FROM '{{DATA_DIR}}/loan_monthly_snapshot.csv' WITH (FORMAT csv, HEADER true)
\copy collections_actions FROM '{{DATA_DIR}}/collections_actions.csv' WITH (FORMAT csv, HEADER true)
\copy macro_monthly_factors FROM '{{DATA_DIR}}/macro_monthly_factors.csv' WITH (FORMAT csv, HEADER true)
\copy application_events FROM '{{DATA_DIR}}/application_events.csv' WITH (FORMAT csv, HEADER true)
