DROP TABLE IF EXISTS credit_risk_modeling_mart;

CREATE TABLE credit_risk_modeling_mart AS
WITH loan_targets AS (
    SELECT
        loan_id,
        MAX(mob) AS max_observed_mob,

        MAX(
            CASE
                WHEN mob BETWEEN 0 AND 11 AND dpd >= 90 THEN 1
                ELSE 0
            END
        ) AS target_default_90dpd_12m,

        MAX(
            CASE
                WHEN mob BETWEEN 0 AND 5 AND dpd >= 30 THEN 1
                ELSE 0
            END
        ) AS target_ever30_6m,

        MAX(CASE WHEN mob BETWEEN 0 AND 11 THEN dpd ELSE NULL END) AS max_dpd_12m,

        MIN(
            CASE
                WHEN dpd >= 90 THEN mob
                ELSE NULL
            END
        ) AS first_90dpd_mob,

        MAX(
            CASE
                WHEN mob BETWEEN 0 AND 11 THEN os_principal
                ELSE NULL
            END
        ) AS ead_proxy_12m

    FROM loan_monthly_snapshot
    GROUP BY loan_id
),

collections AS (
    SELECT
        loan_id,
        COUNT(*) AS collections_actions_count,
        SUM(COALESCE(recovered_amount, 0)) AS total_recovered_amount,
        MAX(CASE WHEN contact_success_flag = 1 THEN 1 ELSE 0 END) AS ever_contact_success_flag
    FROM collections_actions
    GROUP BY loan_id
)

SELECT
    -- technical keys
    l.loan_id,
    l.application_id,
    l.client_id,

    -- dates for validation/split, not direct model features
    a.application_date::date AS application_date,
    l.disbursement_date,

    -- targets
    lt.target_default_90dpd_12m,
    lt.target_ever30_6m,
    lt.max_observed_mob,
    lt.max_dpd_12m,
    lt.first_90dpd_mob,

    -- client features available at application time
    EXTRACT(YEAR FROM AGE(a.application_date::date, c.birth_date))::integer AS age_at_application,
    (a.application_date::date - c.registration_date)::integer AS client_tenure_days,
    c.gender,
    c.marital_status,
    c.education_level,
    c.region_code,
    c.city_type,
    c.residence_type,
    c.dependents_count,
    c.citizenship_flag,
    c.segment,
    c.is_existing_customer,

    -- contact quality, no raw phone/email/address
    ci.phone_normalized_flag,
    ci.email_valid_flag,
    ci.employer_industry AS contact_employer_industry,

    -- application features
    a.product_type,
    a.channel,
    a.branch_code,
    a.digital_channel,
    a.requested_amount::double precision AS requested_amount,
    a.requested_term_months,
    a.purpose_code,
    a.promo_flag,
    a.refinance_flag,
    a.repeat_application_flag,
    a.application_region,
    a.device_type,
    a.utm_source,
    a.campaign_id,

    -- bureau features
    b.bureau_score,
    b.active_loans_count,
    b.closed_loans_count,
    b.delinquency_30_12m,
    b.delinquency_60_12m,
    b.delinquency_90_24m,
    b.max_dpd_24m,
    b.inquiries_30d,
    b.inquiries_90d,
    b.outstanding_debt::double precision AS outstanding_debt,
    b.total_limit::double precision AS total_limit,
    b.utilization_rate::double precision AS utilization_rate,
    b.oldest_trade_months,
    b.bureau_file_thin_flag,
    b.external_collections_flag,

    -- income / employment features
    ei.employment_type,
    ei.employer_industry AS income_employer_industry,
    ei.job_tenure_months,
    ei.total_work_experience_months,
    ei.declared_income::double precision AS declared_income,
    ei.verified_income::double precision AS verified_income,
    ei.other_income::double precision AS other_income,
    ei.income_stability_score,
    ei.salary_project_flag,
    ei.pension_contrib_months_12m,
    ei.debt_to_income_est::double precision AS debt_to_income_est,
    ei.expense_to_income_est::double precision AS expense_to_income_est,

    -- fraud / verification features
    f.device_risk_score::double precision AS device_risk_score,
    f.doc_mismatch_flag,
    f.synthetic_identity_flag,
    f.fraud_suspect_flag,
    f.verification_status,
    f.aml_pep_flag,
    f.blacklist_hit_flag,

    -- loan origination fields
    l.principal_amount::double precision AS principal_amount,
    l.term_months,
    l.interest_rate::double precision AS interest_rate,
    l.monthly_payment_amount::double precision AS monthly_payment_amount,
    l.origination_fee::double precision AS origination_fee,
    l.insurance_fee::double precision AS insurance_fee,

    -- macro features by application month
    m.base_rate::double precision AS base_rate,
    m.inflation_index::double precision AS inflation_index,
    m.unemployment_proxy::double precision AS unemployment_proxy,
    m.fx_stress_index::double precision AS fx_stress_index,
    m.consumer_confidence_proxy::double precision AS consumer_confidence_proxy,
    m.stress_regime_flag,

    -- existing decision engine benchmark columns
    -- not for the main challenger model, but useful for comparison
    der.scorecard_score AS engine_scorecard_score,
    der.pd_estimate::double precision AS engine_pd_estimate,
    der.risk_grade AS engine_risk_grade,
    der.policy_version AS engine_policy_version,
    der.offered_amount::double precision AS engine_offered_amount,
    der.offered_term_months AS engine_offered_term_months,
    der.offered_rate::double precision AS engine_offered_rate,
    der.rule_hits_count AS engine_rule_hits_count,

    -- derived features
    (ei.declared_income - ei.verified_income)::double precision AS income_verification_gap,

    CASE
        WHEN ei.declared_income > 0
        THEN (ei.verified_income / ei.declared_income)::double precision
        ELSE NULL
    END AS income_verification_ratio,

    CASE
        WHEN ei.verified_income > 0
        THEN (a.requested_amount / ei.verified_income)::double precision
        ELSE NULL
    END AS requested_amount_to_income,

    CASE
        WHEN ei.verified_income > 0
        THEN (l.principal_amount / ei.verified_income)::double precision
        ELSE NULL
    END AS principal_amount_to_income,

    CASE
        WHEN ei.verified_income > 0
        THEN (b.outstanding_debt / ei.verified_income)::double precision
        ELSE NULL
    END AS bureau_debt_to_income,

    CASE
        WHEN b.total_limit > 0
        THEN (b.outstanding_debt / b.total_limit)::double precision
        ELSE NULL
    END AS calculated_bureau_utilization,

    -- LGD/EAD helper columns for later, not for PD feature training
    lt.ead_proxy_12m::double precision AS ead_proxy_12m,
    COALESCE(col.collections_actions_count, 0) AS collections_actions_count,
    COALESCE(col.total_recovered_amount, 0)::double precision AS total_recovered_amount,
    COALESCE(col.ever_contact_success_flag, 0) AS ever_contact_success_flag,

    CASE
        WHEN lt.ead_proxy_12m > 0
        THEN GREATEST(
            0,
            LEAST(
                1,
                1 - COALESCE(col.total_recovered_amount, 0) / lt.ead_proxy_12m
            )
        )::double precision
        ELSE NULL
    END AS lgd_proxy

FROM loans l
JOIN applications a
    ON l.application_id = a.application_id
JOIN clients c
    ON l.client_id = c.client_id
LEFT JOIN client_contact_info ci
    ON l.client_id = ci.client_id
LEFT JOIN bureau_snapshot b
    ON l.application_id = b.application_id
LEFT JOIN employment_income_snapshot ei
    ON l.application_id = ei.application_id
LEFT JOIN fraud_flags f
    ON l.application_id = f.application_id
LEFT JOIN decision_engine_results der
    ON l.application_id = der.application_id
LEFT JOIN macro_monthly_factors m
    ON DATE_TRUNC('month', a.application_date)::date = m.month
LEFT JOIN loan_targets lt
    ON l.loan_id = lt.loan_id
LEFT JOIN collections col
    ON l.loan_id = col.loan_id

-- Need at least 12 observed months: MOB 0..11
WHERE lt.max_observed_mob >= 11;

CREATE INDEX idx_credit_risk_modeling_mart_app_date
    ON credit_risk_modeling_mart(application_date);

CREATE INDEX idx_credit_risk_modeling_mart_target
    ON credit_risk_modeling_mart(target_default_90dpd_12m);

CREATE INDEX idx_credit_risk_modeling_mart_loan
    ON credit_risk_modeling_mart(loan_id);

ANALYZE credit_risk_modeling_mart;