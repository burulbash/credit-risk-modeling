-- PostgreSQL-compatible DDL for synthetic retail credit risk / lending analytics database.

DROP TABLE IF EXISTS application_events CASCADE;
DROP TABLE IF EXISTS collections_actions CASCADE;
DROP TABLE IF EXISTS loan_monthly_snapshot CASCADE;
DROP TABLE IF EXISTS payments CASCADE;
DROP TABLE IF EXISTS payment_schedule CASCADE;
DROP TABLE IF EXISTS loans CASCADE;
DROP TABLE IF EXISTS decision_engine_results CASCADE;
DROP TABLE IF EXISTS fraud_flags CASCADE;
DROP TABLE IF EXISTS employment_income_snapshot CASCADE;
DROP TABLE IF EXISTS bureau_snapshot CASCADE;
DROP TABLE IF EXISTS applications CASCADE;
DROP TABLE IF EXISTS client_contact_info CASCADE;
DROP TABLE IF EXISTS clients CASCADE;
DROP TABLE IF EXISTS macro_monthly_factors CASCADE;

CREATE TABLE clients (
    client_id BIGINT PRIMARY KEY,
    registration_date DATE NOT NULL,
    birth_date DATE NOT NULL,
    gender TEXT,
    marital_status TEXT,
    education_level TEXT,
    region_code TEXT,
    city_type TEXT,
    residence_type TEXT,
    dependents_count INTEGER,
    citizenship_flag SMALLINT,
    segment TEXT,
    is_existing_customer SMALLINT,
    created_at TIMESTAMP NOT NULL
);

CREATE TABLE client_contact_info (
    client_id BIGINT PRIMARY KEY REFERENCES clients(client_id),
    phone_raw TEXT,
    email_raw TEXT,
    phone_normalized_flag SMALLINT,
    email_valid_flag SMALLINT,
    address_text TEXT,
    employer_name_raw TEXT,
    employer_industry TEXT,
    contact_update_date DATE
);

CREATE TABLE applications (
    application_id BIGINT PRIMARY KEY,
    client_id BIGINT NOT NULL REFERENCES clients(client_id),
    application_date TIMESTAMP NOT NULL,
    product_type TEXT NOT NULL,
    channel TEXT NOT NULL,
    branch_code TEXT,
    digital_channel TEXT,
    requested_amount NUMERIC(18,2) NOT NULL,
    requested_term_months INTEGER NOT NULL,
    purpose_code TEXT,
    promo_flag SMALLINT,
    refinance_flag SMALLINT,
    repeat_application_flag SMALLINT,
    application_status TEXT,
    application_region TEXT,
    device_type TEXT,
    utm_source TEXT,
    campaign_id TEXT
);

CREATE TABLE bureau_snapshot (
    application_id BIGINT PRIMARY KEY REFERENCES applications(application_id),
    bureau_pull_date TIMESTAMP,
    bureau_score INTEGER,
    active_loans_count INTEGER,
    closed_loans_count INTEGER,
    delinquency_30_12m INTEGER,
    delinquency_60_12m INTEGER,
    delinquency_90_24m INTEGER,
    max_dpd_24m INTEGER,
    inquiries_30d INTEGER,
    inquiries_90d INTEGER,
    outstanding_debt NUMERIC(18,2),
    total_limit NUMERIC(18,2),
    utilization_rate NUMERIC(10,4),
    oldest_trade_months INTEGER,
    bureau_file_thin_flag SMALLINT,
    external_collections_flag SMALLINT
);

CREATE TABLE employment_income_snapshot (
    application_id BIGINT PRIMARY KEY REFERENCES applications(application_id),
    employment_type TEXT,
    employer_industry TEXT,
    job_tenure_months INTEGER,
    total_work_experience_months INTEGER,
    declared_income NUMERIC(18,2),
    verified_income NUMERIC(18,2),
    other_income NUMERIC(18,2),
    income_stability_score INTEGER,
    salary_project_flag SMALLINT,
    pension_contrib_months_12m INTEGER,
    debt_to_income_est NUMERIC(10,4),
    expense_to_income_est NUMERIC(10,4)
);

CREATE TABLE fraud_flags (
    application_id BIGINT PRIMARY KEY REFERENCES applications(application_id),
    device_risk_score NUMERIC(10,2),
    doc_mismatch_flag SMALLINT,
    synthetic_identity_flag SMALLINT,
    fraud_suspect_flag SMALLINT,
    verification_status TEXT,
    aml_pep_flag SMALLINT,
    blacklist_hit_flag SMALLINT
);

CREATE TABLE decision_engine_results (
    application_id BIGINT PRIMARY KEY REFERENCES applications(application_id),
    scorecard_score INTEGER,
    pd_estimate NUMERIC(12,5),
    risk_grade TEXT,
    hard_decline_flag SMALLINT,
    hard_decline_reason TEXT,
    manual_review_flag SMALLINT,
    policy_version TEXT,
    cutoff_value INTEGER,
    offered_amount NUMERIC(18,2),
    offered_term_months INTEGER,
    offered_rate NUMERIC(10,4),
    decision_final TEXT,
    decision_timestamp TIMESTAMP,
    model_version TEXT,
    rule_hits_count INTEGER
);

CREATE TABLE loans (
    loan_id BIGINT PRIMARY KEY,
    application_id BIGINT UNIQUE NOT NULL REFERENCES applications(application_id),
    client_id BIGINT NOT NULL REFERENCES clients(client_id),
    disbursement_date DATE NOT NULL,
    product_type TEXT NOT NULL,
    principal_amount NUMERIC(18,2) NOT NULL,
    term_months INTEGER NOT NULL,
    interest_rate NUMERIC(10,4) NOT NULL,
    monthly_payment_amount NUMERIC(18,2) NOT NULL,
    origination_fee NUMERIC(18,2),
    insurance_fee NUMERIC(18,2),
    status_current TEXT,
    close_date DATE,
    restructuring_flag SMALLINT,
    writeoff_flag SMALLINT
);

CREATE TABLE payment_schedule (
    loan_id BIGINT NOT NULL REFERENCES loans(loan_id),
    installment_no INTEGER NOT NULL,
    due_date DATE NOT NULL,
    principal_due NUMERIC(18,2) NOT NULL,
    interest_due NUMERIC(18,2) NOT NULL,
    fee_due NUMERIC(18,2),
    total_due NUMERIC(18,2) NOT NULL,
    PRIMARY KEY (loan_id, installment_no)
);

CREATE TABLE payments (
    payment_id BIGINT PRIMARY KEY,
    loan_id BIGINT NOT NULL REFERENCES loans(loan_id),
    payment_date DATE NOT NULL,
    amount_paid NUMERIC(18,2) NOT NULL,
    principal_paid NUMERIC(18,2),
    interest_paid NUMERIC(18,2),
    fee_paid NUMERIC(18,2),
    payment_channel TEXT,
    is_partial_payment SMALLINT,
    days_from_due INTEGER,
    payment_source TEXT
);

CREATE TABLE loan_monthly_snapshot (
    loan_id BIGINT NOT NULL REFERENCES loans(loan_id),
    snapshot_month DATE NOT NULL,
    mob INTEGER NOT NULL,
    os_principal NUMERIC(18,2),
    accrued_interest NUMERIC(18,2),
    dpd INTEGER,
    delinquency_bucket TEXT,
    status_month_end TEXT,
    ever_30_flag SMALLINT,
    ever_60_flag SMALLINT,
    ever_90_flag SMALLINT,
    payment_ratio_month NUMERIC(10,4),
    utilization_like_metric NUMERIC(10,4),
    cure_flag SMALLINT,
    restructuring_flag SMALLINT,
    recovery_stage TEXT,
    PRIMARY KEY (loan_id, snapshot_month)
);

CREATE TABLE collections_actions (
    action_id BIGINT PRIMARY KEY,
    loan_id BIGINT NOT NULL REFERENCES loans(loan_id),
    action_date DATE NOT NULL,
    action_stage TEXT,
    action_type TEXT,
    agency_type TEXT,
    promise_to_pay_flag SMALLINT,
    outcome_code TEXT,
    recovered_amount NUMERIC(18,2),
    contact_success_flag SMALLINT
);

CREATE TABLE macro_monthly_factors (
    month DATE PRIMARY KEY,
    base_rate NUMERIC(10,2),
    inflation_index NUMERIC(12,2),
    unemployment_proxy NUMERIC(10,2),
    fx_stress_index NUMERIC(12,2),
    consumer_confidence_proxy NUMERIC(12,2),
    stress_regime_flag SMALLINT
);

CREATE TABLE application_events (
    event_id BIGINT PRIMARY KEY,
    application_id BIGINT NOT NULL REFERENCES applications(application_id),
    event_timestamp TIMESTAMP NOT NULL,
    event_type TEXT,
    channel TEXT,
    device_type TEXT,
    session_minutes NUMERIC(10,2),
    event_value NUMERIC(10,4)
);

-- Core indexes for analytics, joins, and time-based filtering
CREATE INDEX idx_clients_region_segment ON clients(region_code, segment);
CREATE INDEX idx_clients_registration_date ON clients(registration_date);

CREATE INDEX idx_applications_client_date ON applications(client_id, application_date);
CREATE INDEX idx_applications_date ON applications(application_date);
CREATE INDEX idx_applications_status ON applications(application_status);
CREATE INDEX idx_applications_product_channel ON applications(product_type, channel);
CREATE INDEX idx_applications_region ON applications(application_region);

CREATE INDEX idx_bureau_score ON bureau_snapshot(bureau_score);
CREATE INDEX idx_bureau_delinquency ON bureau_snapshot(delinquency_30_12m, delinquency_60_12m, delinquency_90_24m);

CREATE INDEX idx_employment_dti ON employment_income_snapshot(debt_to_income_est);
CREATE INDEX idx_employment_income ON employment_income_snapshot(verified_income, declared_income);

CREATE INDEX idx_fraud_suspect ON fraud_flags(fraud_suspect_flag, blacklist_hit_flag);

CREATE INDEX idx_decision_final_policy ON decision_engine_results(decision_final, policy_version);
CREATE INDEX idx_decision_score ON decision_engine_results(scorecard_score);
CREATE INDEX idx_decision_pd ON decision_engine_results(pd_estimate);
CREATE INDEX idx_decision_timestamp ON decision_engine_results(decision_timestamp);

CREATE INDEX idx_loans_client_id ON loans(client_id);
CREATE INDEX idx_loans_disbursement_date ON loans(disbursement_date);
CREATE INDEX idx_loans_status_current ON loans(status_current);
CREATE INDEX idx_loans_product_type ON loans(product_type);

CREATE INDEX idx_schedule_due_date ON payment_schedule(due_date);
CREATE INDEX idx_schedule_loan_due_date ON payment_schedule(loan_id, due_date);

CREATE INDEX idx_payments_loan_date ON payments(loan_id, payment_date);
CREATE INDEX idx_payments_date ON payments(payment_date);
CREATE INDEX idx_payments_channel ON payments(payment_channel);

CREATE INDEX idx_snapshot_month ON loan_monthly_snapshot(snapshot_month);
CREATE INDEX idx_snapshot_loan_mob ON loan_monthly_snapshot(loan_id, mob);
CREATE INDEX idx_snapshot_bucket ON loan_monthly_snapshot(delinquency_bucket);
CREATE INDEX idx_snapshot_status_month_end ON loan_monthly_snapshot(status_month_end);
CREATE INDEX idx_snapshot_dpd ON loan_monthly_snapshot(dpd);

CREATE INDEX idx_collections_loan_date ON collections_actions(loan_id, action_date);
CREATE INDEX idx_collections_stage_type ON collections_actions(action_stage, action_type);

CREATE INDEX idx_macro_stress_flag ON macro_monthly_factors(stress_regime_flag);

CREATE INDEX idx_app_events_app_ts ON application_events(application_id, event_timestamp);
CREATE INDEX idx_app_events_type ON application_events(event_type);
