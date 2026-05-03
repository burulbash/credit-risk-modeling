# Data Dictionary

All names are in `snake_case`. All data is fully synthetic.

## 1) clients

One row per client entity.

| column | type | description |
|---|---|---|
| client_id | bigint | Surrogate client key |
| registration_date | date | Date client first registered in lender ecosystem |
| birth_date | date | Synthetic birth date |
| gender | text | `female`, `male`, `unknown` |
| marital_status | text | Marital status at current profile state |
| education_level | text | Education bucket |
| region_code | text | Kazakhstan-like region code |
| city_type | text | `metro`, `urban`, `semi_urban`, `rural` |
| residence_type | text | Residence bucket |
| dependents_count | int | Number of dependents |
| citizenship_flag | smallint | 1 = citizen/resident flag |
| segment | text | Synthetic business segment (`new_to_credit`, `thin_file`, `salaried`, `self_employed`, `risky_repeat`, `good_repeat`, `near_prime`, `subprime`) |
| is_existing_customer | smallint | 1 if client is treated as existing at later applications |
| created_at | timestamp | Technical profile creation timestamp |

## 2) client_contact_info

One row per client. Intentionally a little dirty.

| column | type | description |
|---|---|---|
| client_id | bigint | FK to clients |
| phone_raw | text | Raw phone string with occasional formatting noise |
| email_raw | text | Raw email string; some missing / invalid |
| phone_normalized_flag | smallint | 1 if phone appears normalized |
| email_valid_flag | smallint | 1 if email passes simple validity logic |
| address_text | text | Free-text address |
| employer_name_raw | text | Raw employer name with case / spacing / spelling variations |
| employer_industry | text | Employer industry from profile/contact source |
| contact_update_date | date | Last contact/profile update date |

## 3) applications

One row per application.

| column | type | description |
|---|---|---|
| application_id | bigint | Application key |
| client_id | bigint | FK to clients |
| application_date | timestamp | Application creation timestamp |
| product_type | text | `cash_loan`, `installment_loan`, `credit_line` |
| channel | text | Origination channel |
| branch_code | text | Branch / POS code if applicable |
| digital_channel | text | Mobile/web sub-channel |
| requested_amount | numeric | Requested principal |
| requested_term_months | int | Requested term in months |
| purpose_code | text | Loan purpose bucket |
| promo_flag | smallint | Promo participation flag |
| refinance_flag | smallint | Refinance intent flag |
| repeat_application_flag | smallint | 1 if client had earlier applications |
| application_status | text | Final application status (`approved`, `declined`, `expired`, `cancelled`) |
| application_region | text | Region recorded at application time |
| device_type | text | Digital device category |
| utm_source | text | Digital acquisition source |
| campaign_id | text | Optional acquisition campaign id |

## 4) bureau_snapshot

Bureau-at-application snapshot.

| column | type | description |
|---|---|---|
| application_id | bigint | FK to applications |
| bureau_pull_date | timestamp | Bureau retrieval timestamp |
| bureau_score | int | External-like bureau score |
| active_loans_count | int | Open trades count |
| closed_loans_count | int | Closed trades count |
| delinquency_30_12m | int | Number of 30+ delinquency events in prior 12m |
| delinquency_60_12m | int | Number of 60+ delinquency events in prior 12m |
| delinquency_90_24m | int | Number of 90+ delinquency events in prior 24m |
| max_dpd_24m | int | Maximum DPD observed over lookback |
| inquiries_30d | int | Recent inquiries |
| inquiries_90d | int | 90d inquiries |
| outstanding_debt | numeric | Outstanding debt estimate |
| total_limit | numeric | Total bureau-visible limits |
| utilization_rate | numeric | Utilization estimate |
| oldest_trade_months | int | Age of oldest trade |
| bureau_file_thin_flag | smallint | Thin-file indicator |
| external_collections_flag | smallint | External collections indicator |

## 5) employment_income_snapshot

Employment / income snapshot used for underwriting.

| column | type | description |
|---|---|---|
| application_id | bigint | FK to applications |
| employment_type | text | Employment type bucket |
| employer_industry | text | Industry bucket |
| job_tenure_months | int | Tenure at current job |
| total_work_experience_months | int | Total work experience |
| declared_income | numeric | Applicant-declared monthly income |
| verified_income | numeric | Verified monthly income (may differ from declared) |
| other_income | numeric | Other monthly income |
| income_stability_score | int | Internal stability score |
| salary_project_flag | smallint | 1 if salary project / payroll relationship inferred |
| pension_contrib_months_12m | int | Pension contribution continuity proxy |
| debt_to_income_est | numeric | Estimated DTI |
| expense_to_income_est | numeric | Estimated expense ratio |

## 6) fraud_flags

Optional but included. Pre-decision fraud / verification layer.

| column | type | description |
|---|---|---|
| application_id | bigint | FK to applications |
| device_risk_score | numeric | Device / session risk score |
| doc_mismatch_flag | smallint | Document mismatch flag |
| synthetic_identity_flag | smallint | Synthetic identity suspicion flag |
| fraud_suspect_flag | smallint | Aggregate suspicion flag |
| verification_status | text | `passed`, `manual_review`, `failed`, `not_required` |
| aml_pep_flag | smallint | Synthetic AML/PEP marker |
| blacklist_hit_flag | smallint | Blacklist hit flag |

## 7) decision_engine_results

Decision engine output at application level.

| column | type | description |
|---|---|---|
| application_id | bigint | FK to applications |
| scorecard_score | int | Internal scorecard-like score |
| pd_estimate | numeric | Internal PD estimate |
| risk_grade | text | Coarse risk band `A`..`E` |
| hard_decline_flag | smallint | 1 if rules caused hard decline |
| hard_decline_reason | text | Main hard-decline reason |
| manual_review_flag | smallint | 1 if sent to manual review |
| policy_version | text | Policy version active at decision date |
| cutoff_value | int | Score cutoff used |
| offered_amount | numeric | Approved / counter-offered amount |
| offered_term_months | int | Offered term |
| offered_rate | numeric | Offered annualized rate |
| decision_final | text | `approved` or `declined` |
| decision_timestamp | timestamp | Final decision timestamp |
| model_version | text | Scoring model version |
| rule_hits_count | int | Number of rules / alerts triggered |

## 8) loans

Booked loans only.

| column | type | description |
|---|---|---|
| loan_id | bigint | Loan key |
| application_id | bigint | FK to booked application |
| client_id | bigint | FK to client |
| disbursement_date | date | Disbursement date |
| product_type | text | Product type |
| principal_amount | numeric | Booked principal |
| term_months | int | Contractual term |
| interest_rate | numeric | Contract rate |
| monthly_payment_amount | numeric | Contractual monthly payment / minimum due approximation |
| origination_fee | numeric | Origination fee |
| insurance_fee | numeric | Insurance fee |
| status_current | text | Current as-of-last-observation status |
| close_date | date | Close/writeoff date if applicable |
| restructuring_flag | smallint | 1 if ever restructured |
| writeoff_flag | smallint | 1 if written off |

## 9) payment_schedule

Contractual schedule.

| column | type | description |
|---|---|---|
| loan_id | bigint | FK to loans |
| installment_no | int | Installment sequence number |
| due_date | date | Due date |
| principal_due | numeric | Principal due for installment |
| interest_due | numeric | Interest due |
| fee_due | numeric | Fee due |
| total_due | numeric | Total due |

## 10) payments

Actual payment facts.

| column | type | description |
|---|---|---|
| payment_id | bigint | Payment key |
| loan_id | bigint | FK to loans |
| payment_date | date | Payment date |
| amount_paid | numeric | Total amount paid |
| principal_paid | numeric | Principal component |
| interest_paid | numeric | Interest component |
| fee_paid | numeric | Fee component |
| payment_channel | text | Payment channel |
| is_partial_payment | smallint | 1 if partial / split payment logic applies |
| days_from_due | int | Signed/lagged payment timing relative to due date |
| payment_source | text | `borrower`, `salary_deduction`, `collection_agent`, etc. |

## 11) loan_monthly_snapshot

Month-end-like behavioral panel. Key table for MoB / vintage / roll-rate work.

| column | type | description |
|---|---|---|
| loan_id | bigint | FK to loans |
| snapshot_month | date | Month bucket |
| mob | int | Months on book |
| os_principal | numeric | Outstanding principal |
| accrued_interest | numeric | Accrued interest proxy |
| dpd | int | Days past due |
| delinquency_bucket | text | `current`, `1_29`, `30_59`, `60_89`, `90_plus`, `closed`, `writeoff` |
| status_month_end | text | End-of-month state |
| ever_30_flag | smallint | Ever 1+ late / light delinquency marker in this synthetic setup |
| ever_60_flag | smallint | Ever 30+/60+ progression marker |
| ever_90_flag | smallint | Ever 90+ |
| payment_ratio_month | numeric | Actual payment / expected payment proxy |
| utilization_like_metric | numeric | Revolving-like utilization proxy for credit-line products |
| cure_flag | smallint | 1 if bucket improved vs prior month |
| restructuring_flag | smallint | Restructuring indicator for that month |
| recovery_stage | text | `none`, `soft`, `hard`, `legal`, `restructured`, `writeoff`, etc. |

## 12) collections_actions

Collections events / interventions.

| column | type | description |
|---|---|---|
| action_id | bigint | Action key |
| loan_id | bigint | FK to loans |
| action_date | date | Action date |
| action_stage | text | `soft`, `hard`, `legal`, `pre_writeoff` |
| action_type | text | Call / SMS / notice / agency / restructure offer, etc. |
| agency_type | text | `internal`, `outsourced`, `legal` |
| promise_to_pay_flag | smallint | Promise-to-pay indicator |
| outcome_code | text | Outcome bucket |
| recovered_amount | numeric | Recovered amount linked to action if any |
| contact_success_flag | smallint | 1 if contact was successful |

## 13) macro_monthly_factors

Macro / environment layer by month.

| column | type | description |
|---|---|---|
| month | date | Month key |
| base_rate | numeric | Synthetic base rate |
| inflation_index | numeric | Synthetic inflation index |
| unemployment_proxy | numeric | Synthetic unemployment proxy |
| fx_stress_index | numeric | Synthetic FX / market stress proxy |
| consumer_confidence_proxy | numeric | Synthetic consumer confidence proxy |
| stress_regime_flag | smallint | 1 if month is in stress regime |

## 14) application_events

Optional event trail for funnel / session analysis.

| column | type | description |
|---|---|---|
| event_id | bigint | Event key |
| application_id | bigint | FK to applications |
| event_timestamp | timestamp | Event timestamp |
| event_type | text | `submitted`, `bureau_requested`, `score_calculated`, `manual_review`, `decision_sent`, `offer_shown`, `disbursed`, `abandoned` |
| channel | text | Channel at event time |
| device_type | text | Device type |
| session_minutes | numeric | Session duration proxy |
| event_value | numeric | Small synthetic metric for event analytics |

## Business keys / join tips

Typical joins:
- `clients.client_id = applications.client_id`
- `applications.application_id = bureau_snapshot.application_id`
- `applications.application_id = employment_income_snapshot.application_id`
- `applications.application_id = fraud_flags.application_id`
- `applications.application_id = decision_engine_results.application_id`
- `applications.application_id = loans.application_id`
- `loans.loan_id = payment_schedule.loan_id`
- `loans.loan_id = payments.loan_id`
- `loans.loan_id = loan_monthly_snapshot.loan_id`
- `loans.loan_id = collections_actions.loan_id`
- `applications.application_id = application_events.application_id`

## Leakage note

For application-level modeling, do **not** join post-disbursement tables unless you are explicitly building a post-origination model.
