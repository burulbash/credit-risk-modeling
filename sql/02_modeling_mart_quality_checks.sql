--Quality checks for credit_risk_modeling_mart

--General mart summary
SELECT
    COUNT(*) AS rows_count,
    COUNT(DISTINCT loan_id) AS unique_loans,
    COUNT(DISTINCT application_id) AS unique_applications,
    COUNT(DISTINCT client_id) AS unique_clients,
    MIN(application_date) AS min_application_date,
    MAX(application_date) AS max_application_date,
    AVG(target_default_90dpd_12m::numeric) AS default_90dpd_12m_rate,
    AVG(target_ever30_6m::numeric) AS ever30_6m_rate
FROM credit_risk_modeling_mart;


-- Duplicate check
SELECT
    COUNT(*) AS rows_count,
    COUNT(DISTINCT loan_id) AS unique_loan_id,
    COUNT(*) - COUNT(DISTINCT loan_id) AS duplicate_loan_rows
FROM credit_risk_modeling_mart;


--Default rate by product
SELECT
    product_type,
    COUNT(*) AS loans,
    AVG(target_default_90dpd_12m::numeric) AS default_90dpd_12m_rate
FROM credit_risk_modeling_mart
GROUP BY product_type
ORDER BY default_90dpd_12m_rate DESC;


-- Default rate by application year
SELECT
    EXTRACT(YEAR FROM application_date)::integer AS application_year,
    COUNT(*) AS loans,
    AVG(target_default_90dpd_12m::numeric) AS default_90dpd_12m_rate
FROM credit_risk_modeling_mart
GROUP BY EXTRACT(YEAR FROM application_date)
ORDER BY application_year;