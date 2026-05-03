# QA / Design Report

## 1) Embedded dependencies and business logic

### Approval / decision logic
Approval is driven by a noisy combination of:
- bureau score,
- recent delinquency history,
- DTI,
- inquiries,
- income stability,
- salary-project indicator,
- existing / repeat-customer effect,
- product type,
- channel,
- fraud / verification flags,
- policy version,
- macro stress month.

Important properties:
- policy versions change cutoffs over time,
- `manual_review_flag` can override both approve and decline,
- `hard_decline_flag` is not purely score-based,
- approved amount / term / rate are risk-sensitive,
- channel and product mix shifts over time.

### Default / delinquency logic
Behavioral deterioration is driven by:
- origination PD estimate,
- DTI,
- bureau delinquency,
- thin-file effect,
- product type,
- requested / booked amount vs income,
- salary stability,
- repeat quality,
- stress regime,
- post-book payment behavior.

Important properties:
- not all high-score clients behave well,
- not all low-score clients default,
- trees should find useful interactions,
- linear models still have useful signal,
- macro stress months worsen both approval and performance.

### Time effects / drift
Embedded temporal effects:
- stress periods in `macro_monthly_factors`,
- policy changes across several version blocks,
- changing cutoffs and appetite,
- changing application volume,
- changing digital / branch mix,
- performance sensitivity to stress regime.

This supports:
- time-based train / test split,
- vintage curves,
- before / after policy analysis,
- drift monitoring,
- PSI-style comparisons.

## 2) Intentional data dirt

The generator adds moderate, realistic imperfections.

### Missing values
Typical 3%–10% pockets of missingness in selected fields:
- invalid or missing emails,
- missing address / employer fields,
- thin-file bureau fields null,
- occasional missing bureau score / inquiries,
- occasional missing verified income / pension contribution months,
- optional acquisition fields missing for non-digital and some digital records.

### Raw-text quality issues
- `phone_raw` has inconsistent formatting and occasional malformed strings,
- `email_raw` includes occasional invalid formats,
- `employer_name_raw` includes case changes, spacing noise, spelling-like variations, trailing spaces.

### Duplicate-like identities
Rare client-contact duplication is introduced through copied phone/email/address patterns across different `client_id`s. This simulates household overlap / duplicate registration / entity-resolution challenges without real PII.

### Boundary outliers
- some income outliers,
- some requested amount outliers,
- mismatches between declared and verified income,
- small number of status / timestamp inconsistencies,
- small number of backfilled close dates / bureau pull dates.

### What is intentionally *not* present at scale
- massive impossible values,
- systemic FK corruption,
- fully broken dates,
- chaotic random nulling,
- unrealistic perfect separability.

## 3) Leakage traps (deliberate)

The database is built so that a careless analyst can leak future information.

### Common leakage mistakes
1. Joining `loan_monthly_snapshot` to application-level base without filtering to horizon / post-date logic.
2. Joining `payments` into application-date training base.
3. Using `collections_actions` or `recovery_stage` for origination PD model.
4. Using `application_events` after final decision or `disbursed` event as a feature for approval / early-PD modeling.
5. Using `loans.status_current`, `close_date`, `writeoff_flag`, `restructuring_flag` in origination model.
6. Using `decision_engine_results` when the intended model point-in-time is **before** internal score calculation.

### Safe origination modeling inputs
At application time, safest features are:
- client profile,
- contact quality indicators,
- application fields,
- bureau snapshot,
- income/employment snapshot,
- fraud/verification snapshot,
- macro factors as of application month.

### Good learning value
This setup is intentional so you can practice:
- leakage-safe feature cut-off,
- one-row-per-application marts,
- horizon-aware target building,
- train / valid splits by time.

## 4) Practical targets supported

### Directly derivable from booked-loan behavior
- `target_default_30dpd_6m`
- `target_default_90dpd_12m`
- `fpd_flag`
- `ever_30`
- `ever_60`
- `ever_90`
- roll-rates between delinquency buckets
- early closure / prepayment flag
- cure rate
- collections recovery effectiveness

### Business monitoring targets
- approval rate
- booking rate
- bad rate by score band / policy version / product / channel / region
- vintage bad rate by disbursement month and MoB
- collections promise-to-pay effectiveness
- manual review override quality

## 5) Expected model behavior

The generator is designed so that:
- logistic regression should get meaningful but not absurd performance,
- trees / RF / boosting should benefit from nonlinearities and interactions,
- naive leakage can make results look unrealistically strong,
- time-based validation should behave differently from random split,
- score, DTI, delinquency history, income verification gap, repeat quality, and macro stress will usually be among the stronger signals.

## 6) Recommended QA checks after generation

Run these sanity checks locally after exporting:
- row counts by table,
- approval rate,
- booking rate,
- share of loans with `default_90dpd_12m`,
- FPD rate,
- share of thin-file applications,
- null ratios by field,
- average term / amount by product,
- monthly volume trend,
- approval rate by policy version,
- bad rate by origination month,
- ratio of collections actions to delinquent snapshots.

## 7) Validation note

In this environment, the package was prepared and the generator was syntax-checked. Depending on local hardware and chosen size, full generation can take noticeable time because the dataset is intentionally non-trivial and export-heavy.
