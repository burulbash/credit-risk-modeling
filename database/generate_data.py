#!/usr/bin/env python3
"""
Synthetic retail credit risk / lending analytics dataset generator.

Creates a PostgreSQL-friendly relational dataset with realistic temporal drift,
moderate data quality issues, and post-origination behavior suitable for:
- SQL practice
- pandas / numpy EDA
- baseline PD / risk modelling in sklearn
- vintage / FPD / MoB / roll-rate / collections analytics

Usage example:
    python generate_data.py --size medium --seed 42 --output-dir ./exports_medium
"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from faker import Faker  # type: ignore
except Exception:  # pragma: no cover
    Faker = None


# -----------------------------
# Helpers and configuration
# -----------------------------


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def clip_series(s: pd.Series, low=None, high=None):
    return s.clip(lower=low, upper=high)


REGION_WEIGHTS = {
    "almaty": 0.19,
    "astana": 0.12,
    "shymkent": 0.10,
    "karaganda": 0.08,
    "aktobe": 0.07,
    "atyrau": 0.05,
    "pavlodar": 0.05,
    "turkistan": 0.11,
    "kostanay": 0.06,
    "east_kz": 0.07,
    "west_kz": 0.05,
    "zhambyl": 0.05,
}

REGION_RISK = {
    "almaty": -0.10,
    "astana": -0.08,
    "shymkent": 0.03,
    "karaganda": 0.02,
    "aktobe": 0.01,
    "atyrau": -0.02,
    "pavlodar": 0.00,
    "turkistan": 0.08,
    "kostanay": -0.01,
    "east_kz": 0.04,
    "west_kz": 0.05,
    "zhambyl": 0.07,
}

SEGMENT_PROBS = {
    "new_to_credit": 0.12,
    "thin_file": 0.10,
    "salaried": 0.27,
    "self_employed": 0.13,
    "risky_repeat": 0.08,
    "good_repeat": 0.12,
    "near_prime": 0.10,
    "subprime": 0.08,
}

SEGMENT_RISK = {
    "new_to_credit": 0.28,
    "thin_file": 0.40,
    "salaried": -0.18,
    "self_employed": 0.12,
    "risky_repeat": 0.72,
    "good_repeat": -0.55,
    "near_prime": 0.04,
    "subprime": 0.95,
}

EDU_LEVELS = ["secondary", "vocational", "bachelor", "master", "other"]
EMPLOYMENT_TYPES = [
    "formal_salaried",
    "informal_salaried",
    "self_employed",
    "micro_business_owner",
    "contractor",
    "unemployed",
    "pensioner",
]
INDUSTRIES = [
    "retail",
    "public_sector",
    "oil_and_gas",
    "logistics",
    "construction",
    "education",
    "healthcare",
    "manufacturing",
    "agriculture",
    "hospitality",
    "it_services",
    "finance",
    "telecom",
]
PRODUCTS = ["cash_loan", "installment_loan", "credit_line"]
CHANNELS = ["mobile_app", "web", "branch", "partner_pos"]
PURPOSES = [
    "working_capital",
    "consumer_purchase",
    "medical",
    "education",
    "home_repair",
    "travel",
    "debt_consolidation",
    "electronics",
    "wedding",
    "other",
]
EMAIL_DOMAINS = ["gmail.com", "mail.ru", "yandex.kz", "outlook.com", "bk.ru"]
EMPLOYER_PREFIX = [
    "Too",
    "LLP",
    "IP",
    "JSC",
    "AO",
    "Tech",
    "Kaz",
    "Global",
    "Prime",
    "Silk",
    "Steppe",
    "Alem",
]
EMPLOYER_CORE = [
    "Trade",
    "Logistics",
    "Retail",
    "Service",
    "Consult",
    "Market",
    "Energy",
    "Pharma",
    "Build",
    "Finance",
    "Food",
    "Agro",
    "Systems",
]
EMPLOYER_SUFFIX = ["Group", "Holding", "Solutions", "Partners", "Company", "KZ", "Center"]


@dataclass
class SizeConfig:
    name: str
    n_clients: int
    n_applications: int
    start_date: str = "2022-01-01"
    app_end_date: str = "2025-03-31"
    obs_end_month: str = "2026-03-01"


SIZE_CONFIGS = {
    "small": SizeConfig("small", 18000, 32000),
    "medium": SizeConfig("medium", 82000, 150000),
    "large": SizeConfig("large", 180000, 340000),
}


class CreditRiskDataGenerator:
    def __init__(self, size: str = "medium", seed: int = 42, output_dir: str = "exports_medium"):
        if size not in SIZE_CONFIGS:
            raise ValueError(f"Unknown size: {size}")
        self.cfg = SIZE_CONFIGS[size]
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        self.fake = Faker("en_US") if Faker is not None else None
        if self.fake is not None:
            self.fake.seed_instance(seed)
        self.tables: Dict[str, pd.DataFrame] = {}
        self.debug: Dict[str, pd.DataFrame] = {}

    # -----------------------------
    # Macro factors and clients
    # -----------------------------
    def generate_macro_monthly_factors(self) -> pd.DataFrame:
        months = pd.date_range(self.cfg.start_date, self.cfg.obs_end_month, freq="MS")
        n = len(months)
        t = np.arange(n)
        stress = np.zeros(n, dtype=int)
        stress[((months >= "2023-04-01") & (months <= "2023-08-01"))] = 1
        stress[((months >= "2024-11-01") & (months <= "2025-02-01"))] = 1
        base_rate = 14.0 + 0.6 * np.sin(t / 3.2) + 0.25 * np.cos(t / 1.9) + 1.1 * stress + 0.05 * t
        inflation = 100 + np.cumsum(0.48 + 0.08 * np.sin(t / 2.7) + 0.22 * stress + self.rng.normal(0, 0.08, n))
        unemployment = 4.7 + 0.12 * np.sin(t / 4.1) + 0.28 * stress + self.rng.normal(0, 0.05, n)
        fx_stress = 42 + 8 * stress + 4 * np.sin(t / 2.2) + self.rng.normal(0, 1.1, n)
        consumer_conf = 104 - 2.5 * stress + 1.2 * np.sin(t / 3.9) + self.rng.normal(0, 0.5, n)

        macro = pd.DataFrame(
            {
                "month": months.date,
                "base_rate": np.round(base_rate, 2),
                "inflation_index": np.round(inflation, 2),
                "unemployment_proxy": np.round(unemployment, 2),
                "fx_stress_index": np.round(fx_stress, 2),
                "consumer_confidence_proxy": np.round(consumer_conf, 2),
                "stress_regime_flag": stress.astype(int),
            }
        )
        self.tables["macro_monthly_factors"] = macro
        return macro

    def _random_dates(self, start: str, end: str, n: int) -> pd.Series:
        start_ts = pd.Timestamp(start)
        end_ts = pd.Timestamp(end)
        delta = (end_ts - start_ts).days
        offsets = self.rng.integers(0, delta + 1, size=n)
        return pd.Series(start_ts + pd.to_timedelta(offsets, unit="D"))

    def generate_clients(self) -> pd.DataFrame:
        n = self.cfg.n_clients
        segment = self.rng.choice(list(SEGMENT_PROBS.keys()), size=n, p=list(SEGMENT_PROBS.values()))
        registration_date = self._random_dates("2021-01-01", self.cfg.app_end_date, n)
        age = np.clip(self.rng.normal(36, 10.5, n), 19, 71).round().astype(int)
        birth_date = registration_date - pd.to_timedelta(age * 365 + self.rng.integers(0, 365, n), unit="D")
        gender = self.rng.choice(["female", "male", "unknown"], size=n, p=[0.51, 0.47, 0.02])
        marital = self.rng.choice(
            ["single", "married", "divorced", "widowed"],
            size=n,
            p=[0.39, 0.46, 0.11, 0.04],
        )
        education = self.rng.choice(EDU_LEVELS, size=n, p=[0.25, 0.23, 0.34, 0.10, 0.08])
        regions = self.rng.choice(list(REGION_WEIGHTS.keys()), size=n, p=list(REGION_WEIGHTS.values()))
        city_type = self.rng.choice(["metro", "urban", "semi_urban", "rural"], size=n, p=[0.19, 0.44, 0.22, 0.15])
        residence_type = self.rng.choice(
            ["owned", "family_owned", "rented", "dormitory", "other"],
            size=n,
            p=[0.34, 0.22, 0.31, 0.06, 0.07],
        )
        dep_base = np.where(marital == "married", 1.6, 0.5)
        dependents = np.clip(self.rng.poisson(dep_base), 0, 6)
        citizenship = self.rng.choice([1, 0], size=n, p=[0.964, 0.036])
        existing = (
            (registration_date < pd.Timestamp("2023-12-01")).astype(int)
            * (self.rng.random(n) < (0.23 + 0.30 * pd.Series(segment).isin(["good_repeat", "risky_repeat"]).astype(float)))
        ).astype(int)
        created_at = registration_date + pd.to_timedelta(self.rng.integers(0, 48, size=n), unit="h")

        clients = pd.DataFrame(
            {
                "client_id": np.arange(1, n + 1, dtype=np.int64),
                "registration_date": registration_date.dt.date,
                "birth_date": birth_date.dt.date,
                "gender": gender,
                "marital_status": marital,
                "education_level": education,
                "region_code": regions,
                "city_type": city_type,
                "residence_type": residence_type,
                "dependents_count": dependents.astype(int),
                "citizenship_flag": citizenship.astype(int),
                "segment": segment,
                "is_existing_customer": existing.astype(int),
                "created_at": created_at,
            }
        )

        # Internal latent fields for downstream generation (not exported)
        segment_risk = pd.Series(segment).map(SEGMENT_RISK).astype(float)
        edu_adj = pd.Series(education).map(
            {
                "secondary": -0.05,
                "vocational": 0.00,
                "bachelor": 0.08,
                "master": 0.12,
                "other": -0.08,
            }
        )
        city_adj = pd.Series(city_type).map({"metro": 0.08, "urban": 0.03, "semi_urban": -0.02, "rural": -0.07})
        industry_pref = self.rng.choice(INDUSTRIES, size=n)
        income_anchor = np.exp(
            12.15
            + 0.34 * edu_adj.to_numpy()
            + 0.22 * city_adj.to_numpy()
            - 0.18 * segment_risk.to_numpy()
            + self.rng.normal(0, 0.42, n)
        )
        income_anchor = np.clip(income_anchor, 70000, 2200000)
        if n > 20:
            outlier_ix = self.rng.choice(n, size=max(8, int(n * 0.003)), replace=False)
            income_anchor[outlier_ix] *= self.rng.uniform(1.9, 4.5, size=len(outlier_ix))
            income_anchor = np.clip(income_anchor, 70000, 6500000)
        clients_debug = clients.copy()
        clients_debug["_segment_risk"] = segment_risk
        clients_debug["_income_anchor"] = income_anchor
        clients_debug["_region_risk"] = pd.Series(regions).map(REGION_RISK).astype(float)
        clients_debug["_preferred_industry"] = industry_pref
        self.debug["clients"] = clients_debug
        self.tables["clients"] = clients
        return clients

    def generate_client_contact_info(self) -> pd.DataFrame:
        clients = self.debug["clients"]
        n = len(clients)
        client_ids = clients["client_id"].to_numpy()

        def make_phone() -> str:
            body = "".join(str(x) for x in self.rng.integers(0, 10, size=9))
            fmt = self.rng.choice([
                f"+7{body}",
                f"8 ({body[0:3]}) {body[3:6]}-{body[6:8]}-{body[8]}{self.rng.integers(0,10)}",
                f"+7 ({body[0:3]}) {body[3:6]} {body[6:8]} {body[8]}{self.rng.integers(0,10)}",
                f"7{body}",
            ])
            return fmt

        phones = [make_phone() for _ in range(n)]
        phone_normalized_flag = (self.rng.random(n) > 0.08).astype(int)
        malformed_ix = self.rng.choice(n, size=max(15, int(n * 0.035)), replace=False)
        for i in malformed_ix[: len(malformed_ix) // 2]:
            phones[i] = phones[i].replace("+", "")[: self.rng.integers(7, 11)]
            phone_normalized_flag[i] = 0
        for i in malformed_ix[len(malformed_ix) // 2 :]:
            phones[i] = phones[i] + self.rng.choice(["  ", " ext", "-00", " / "])
        email_present = self.rng.random(n) < 0.68
        email_valid = (self.rng.random(n) > 0.085).astype(int)
        email_raw = np.array([None] * n, dtype=object)
        local_parts = [f"client{cid}_{self.rng.integers(10,9999)}" for cid in client_ids]
        domains = self.rng.choice(EMAIL_DOMAINS, size=n)
        email_ix = np.where(email_present)[0]
        for i in email_ix:
            email_raw[i] = f"{local_parts[i]}@{domains[i]}"
        invalid_ix = self.rng.choice(email_ix, size=max(10, int(len(email_ix) * 0.07)), replace=False)
        for i in invalid_ix:
            email_raw[i] = self.rng.choice(
                [
                    f"{local_parts[i]}@@{domains[i]}",
                    f"{local_parts[i]}{domains[i]}",
                    f" {local_parts[i]}@{domains[i]} ",
                    f"{local_parts[i]}@mail",
                ]
            )
            email_valid[i] = 0

        preferred_industry = clients["_preferred_industry"].to_numpy()
        employer_industry = preferred_industry.copy()
        prefix = self.rng.choice(EMPLOYER_PREFIX, size=n)
        core = self.rng.choice(EMPLOYER_CORE, size=n)
        suffix = self.rng.choice(EMPLOYER_SUFFIX, size=n)
        employer_name = np.array([f"{p} {c} {s}" for p, c, s in zip(prefix, core, suffix)], dtype=object)
        noisy_ix = self.rng.choice(n, size=max(20, int(n * 0.05)), replace=False)
        for i in noisy_ix:
            employer_name[i] = self.rng.choice(
                [
                    employer_name[i].lower(),
                    employer_name[i].upper(),
                    employer_name[i] + " ",
                    employer_name[i].replace(" ", "  "),
                    employer_name[i].replace("LLP", "Llp").replace("Too", "TOO"),
                ]
            )

        address = np.array(
            [
                f"{row.region_code} {self.rng.choice(['district', 'microdistrict', 'avenue', 'street'])} {self.rng.integers(1,120)}"
                for row in clients.itertuples(index=False)
            ],
            dtype=object,
        )
        contact_update_date = self._random_dates("2022-01-01", self.cfg.obs_end_month, n)

        # Duplicate-like contact identities across different client_ids
        dup_n = max(12, int(n * 0.007))
        src_ix = self.rng.choice(n, size=dup_n, replace=False)
        dst_ix = self.rng.choice(n, size=dup_n, replace=False)
        for s, d in zip(src_ix, dst_ix):
            if s != d:
                phones[d] = phones[s]
                if email_raw[s] is not None and self.rng.random() < 0.65:
                    email_raw[d] = email_raw[s]
                if self.rng.random() < 0.55:
                    address[d] = address[s]

        contact = pd.DataFrame(
            {
                "client_id": client_ids,
                "phone_raw": phones,
                "email_raw": email_raw,
                "phone_normalized_flag": phone_normalized_flag.astype(int),
                "email_valid_flag": email_valid.astype(int),
                "address_text": address,
                "employer_name_raw": employer_name,
                "employer_industry": employer_industry,
                "contact_update_date": contact_update_date.dt.date,
            }
        )
        # Missingness / dirt
        miss_addr_ix = self.rng.choice(n, size=max(20, int(n * 0.035)), replace=False)
        contact.loc[miss_addr_ix, "address_text"] = None
        miss_emp_ix = self.rng.choice(n, size=max(20, int(n * 0.04)), replace=False)
        contact.loc[miss_emp_ix, ["employer_name_raw", "employer_industry"]] = None

        self.tables["client_contact_info"] = contact
        return contact

    # -----------------------------
    # Applications and pre-origination
    # -----------------------------
    def generate_applications(self) -> pd.DataFrame:
        clients = self.debug["clients"]
        macro = self.tables["macro_monthly_factors"].copy()
        macro["month"] = pd.to_datetime(macro["month"])
        app_months = pd.date_range(self.cfg.start_date, self.cfg.app_end_date, freq="MS")
        macro_sub = macro[macro["month"].isin(app_months)].copy()

        # Volume drift: more digital / growth waves, lower during stress.
        volume_index = 1.0 + 0.25 * np.sin(np.arange(len(macro_sub)) / 3.0) - 0.12 * macro_sub["stress_regime_flag"].to_numpy()
        volume_index = np.clip(volume_index, 0.65, 1.35)
        volume_probs = volume_index / volume_index.sum()
        month_choices = self.rng.choice(app_months, size=self.cfg.n_applications, p=volume_probs)
        month_choices = pd.to_datetime(month_choices)
        day_offsets = self.rng.integers(0, 28, size=self.cfg.n_applications)
        application_date = month_choices + pd.to_timedelta(day_offsets, unit="D")

        # Existing customers / repeat clients slightly overrepresented in applications.
        client_weights = 1.0 + 0.55 * clients["is_existing_customer"].to_numpy() + 0.18 * clients["segment"].isin(["good_repeat", "risky_repeat"]).astype(float).to_numpy()
        client_weights = client_weights / client_weights.sum()
        chosen_client_ids = self.rng.choice(clients["client_id"].to_numpy(), size=self.cfg.n_applications, replace=True, p=client_weights)

        apps = pd.DataFrame(
            {
                "application_id": np.arange(1, self.cfg.n_applications + 1, dtype=np.int64),
                "client_id": chosen_client_ids,
                "application_date": pd.to_datetime(application_date),
            }
        )
        apps = apps.merge(clients, on="client_id", how="left", suffixes=("", "_client"))
        apps["app_month"] = apps["application_date"].dt.to_period("M").dt.to_timestamp()
        apps = apps.merge(macro_sub.rename(columns={"month": "app_month"}), on="app_month", how="left")
        apps = apps.sort_values(["client_id", "application_date", "application_id"]).reset_index(drop=True)
        apps["repeat_application_flag"] = apps.groupby("client_id").cumcount().gt(0).astype(int)
        prev_apps = apps.groupby("client_id").cumcount()
        repeat_n = prev_apps.to_numpy()

        # Product and channel mix.
        segment = apps["segment"]
        product = np.where(
            segment.isin(["thin_file", "new_to_credit"]),
            self.rng.choice(PRODUCTS, size=len(apps), p=[0.44, 0.37, 0.19]),
            self.rng.choice(PRODUCTS, size=len(apps), p=[0.47, 0.25, 0.28]),
        )
        channel = np.where(
            product == "installment_loan",
            self.rng.choice(CHANNELS, size=len(apps), p=[0.15, 0.05, 0.18, 0.62]),
            self.rng.choice(CHANNELS, size=len(apps), p=[0.44, 0.18, 0.28, 0.10]),
        )
        branch_code = np.where(
            pd.Series(channel).isin(["branch", "partner_pos"]),
            [f"BR_{x:03d}" for x in self.rng.integers(1, 65, len(apps))],
            None,
        )
        digital_channel = np.where(
            pd.Series(channel).isin(["mobile_app", "web"]),
            np.where(pd.Series(channel) == "mobile_app", self.rng.choice(["android_app", "ios_app"], len(apps), p=[0.78, 0.22]), "web_portal"),
            None,
        )
        device_type = np.where(
            pd.Series(channel).isin(["mobile_app", "web", "partner_pos"]),
            self.rng.choice(["android", "ios", "desktop", "tablet"], len(apps), p=[0.54, 0.19, 0.22, 0.05]),
            None,
        )
        promo_flag = (self.rng.random(len(apps)) < (0.12 + 0.05 * (product == "installment_loan"))).astype(int)
        refinance_flag = (self.rng.random(len(apps)) < (0.06 + 0.10 * apps["repeat_application_flag"])) .astype(int)
        purpose = self.rng.choice(PURPOSES, size=len(apps), p=[0.10, 0.24, 0.07, 0.05, 0.12, 0.05, 0.13, 0.12, 0.03, 0.09])
        utm_source = np.where(
            pd.Series(channel).isin(["mobile_app", "web"]),
            self.rng.choice(["seo", "cpa_network", "social", "crm_push", "affiliate", "direct", None], size=len(apps), p=[0.15, 0.15, 0.19, 0.14, 0.08, 0.19, 0.10]),
            None,
        )
        campaign_id = np.where(
            pd.Series(channel).isin(["mobile_app", "web"]),
            np.where(self.rng.random(len(apps)) < 0.62, [f"CMP_{x:05d}" for x in self.rng.integers(100, 99999, len(apps))], None),
            None,
        )

        income_anchor = apps["_income_anchor"].to_numpy()
        seg_risk = apps["_segment_risk"].to_numpy()
        region_risk = apps["_region_risk"].to_numpy()
        stress = apps["stress_regime_flag"].fillna(0).to_numpy()
        repeat_good = apps["segment"].isin(["good_repeat", "salaried"]).astype(float).to_numpy()

        base_amount = (
            income_anchor
            * np.where(product == "cash_loan", self.rng.uniform(1.6, 4.8, len(apps)), np.where(product == "installment_loan", self.rng.uniform(0.55, 2.6, len(apps)), self.rng.uniform(0.9, 3.8, len(apps))))
            * (1.0 + 0.12 * apps["repeat_application_flag"].to_numpy() + 0.08 * repeat_good - 0.08 * stress + self.rng.normal(0, 0.18, len(apps)))
        )
        requested_amount = np.clip(base_amount, 25000, 4500000)
        outlier_ix = self.rng.choice(len(apps), size=max(25, int(len(apps) * 0.004)), replace=False)
        requested_amount[outlier_ix] *= self.rng.uniform(1.25, 2.1, len(outlier_ix))
        requested_amount = np.clip(requested_amount, 25000, 6800000)

        term = np.where(
            product == "cash_loan",
            self.rng.choice([6, 9, 12, 18, 24, 30, 36], size=len(apps), p=[0.05, 0.08, 0.22, 0.24, 0.22, 0.10, 0.09]),
            np.where(
                product == "installment_loan",
                self.rng.choice([3, 6, 9, 12, 18, 24], size=len(apps), p=[0.10, 0.17, 0.18, 0.28, 0.17, 0.10]),
                self.rng.choice([6, 12, 18, 24], size=len(apps), p=[0.12, 0.52, 0.18, 0.18]),
            ),
        )

        application_region = apps["region_code"].copy()
        mismatch_ix = self.rng.choice(len(apps), size=max(20, int(len(apps) * 0.025)), replace=False)
        application_region.iloc[mismatch_ix] = self.rng.choice(list(REGION_WEIGHTS.keys()), size=len(mismatch_ix))

        apps_export = pd.DataFrame(
            {
                "application_id": apps["application_id"].astype(np.int64),
                "client_id": apps["client_id"].astype(np.int64),
                "application_date": apps["application_date"],
                "product_type": product,
                "channel": channel,
                "branch_code": branch_code,
                "digital_channel": digital_channel,
                "requested_amount": np.round(requested_amount, 2),
                "requested_term_months": term.astype(int),
                "purpose_code": purpose,
                "promo_flag": promo_flag.astype(int),
                "refinance_flag": refinance_flag.astype(int),
                "repeat_application_flag": apps["repeat_application_flag"].astype(int),
                "application_status": "submitted",
                "application_region": application_region,
                "device_type": device_type,
                "utm_source": utm_source,
                "campaign_id": campaign_id,
            }
        )

        # Controlled missingness
        miss_device = self.rng.choice(len(apps_export), size=max(25, int(len(apps_export) * 0.04)), replace=False)
        apps_export.loc[miss_device, "device_type"] = None
        miss_campaign = self.rng.choice(len(apps_export), size=max(25, int(len(apps_export) * 0.09)), replace=False)
        apps_export.loc[miss_campaign, "campaign_id"] = None

        apps_debug = apps_export.copy()
        apps_debug["segment"] = apps["segment"].to_numpy()
        apps_debug["is_existing_customer"] = apps["is_existing_customer"].to_numpy()
        apps_debug["education_level"] = apps["education_level"].to_numpy()
        apps_debug["region_code"] = apps["region_code"].to_numpy()
        apps_debug["city_type"] = apps["city_type"].to_numpy()
        apps_debug["dependents_count"] = apps["dependents_count"].to_numpy()
        apps_debug["registration_date"] = apps["registration_date"].to_numpy()
        apps_debug["birth_date"] = apps["birth_date"].to_numpy()
        apps_debug["_income_anchor"] = income_anchor
        apps_debug["_segment_risk"] = seg_risk
        apps_debug["_region_risk"] = region_risk
        apps_debug["_stress"] = stress
        apps_debug["_base_rate"] = apps["base_rate"].to_numpy()
        apps_debug["_inflation_index"] = apps["inflation_index"].to_numpy()
        apps_debug["_repeat_n"] = repeat_n
        self.debug["applications"] = apps_debug
        self.tables["applications"] = apps_export
        return apps_export

    def generate_fraud_flags(self) -> pd.DataFrame:
        apps = self.debug["applications"]
        n = len(apps)
        risky_segment = apps["segment"].isin(["subprime", "thin_file", "risky_repeat"]).astype(float).to_numpy()
        digital = apps["channel"].isin(["mobile_app", "web"]).astype(float).to_numpy()
        repeat = apps["repeat_application_flag"].astype(float).to_numpy()
        device_risk = np.clip(35 + 18 * risky_segment + 7 * digital - 5 * repeat + self.rng.normal(0, 10, n), 1, 99)
        doc_mismatch = (self.rng.random(n) < sigmoid(-3.1 + 0.03 * device_risk + 0.25 * risky_segment)).astype(int)
        synthetic_identity = (self.rng.random(n) < sigmoid(-4.2 + 0.045 * device_risk + 0.48 * risky_segment + 0.12 * digital)).astype(int)
        blacklist_hit = (self.rng.random(n) < sigmoid(-5.1 + 0.06 * device_risk + 0.7 * synthetic_identity + 0.45 * doc_mismatch)).astype(int)
        aml_pep = (self.rng.random(n) < 0.004).astype(int)
        suspect = ((device_risk > 68).astype(int) + doc_mismatch + synthetic_identity + blacklist_hit >= 2).astype(int)
        verification_status = np.where(
            suspect == 1,
            self.rng.choice(["manual_review", "failed", "passed"], size=n, p=[0.58, 0.26, 0.16]),
            self.rng.choice(["passed", "not_required", "manual_review"], size=n, p=[0.64, 0.26, 0.10]),
        )
        fraud = pd.DataFrame(
            {
                "application_id": apps["application_id"].astype(np.int64),
                "device_risk_score": np.round(device_risk, 2),
                "doc_mismatch_flag": doc_mismatch.astype(int),
                "synthetic_identity_flag": synthetic_identity.astype(int),
                "fraud_suspect_flag": suspect.astype(int),
                "verification_status": verification_status,
                "aml_pep_flag": aml_pep.astype(int),
                "blacklist_hit_flag": blacklist_hit.astype(int),
            }
        )
        self.tables["fraud_flags"] = fraud
        return fraud

    def generate_bureau_snapshot(self) -> pd.DataFrame:
        apps = self.debug["applications"]
        n = len(apps)
        thin = apps["segment"].isin(["new_to_credit", "thin_file"]).astype(float).to_numpy()
        repeat_good = apps["segment"].isin(["good_repeat", "salaried"]).astype(float).to_numpy()
        repeat_bad = apps["segment"].isin(["risky_repeat", "subprime"]).astype(float).to_numpy()
        seg_risk = apps["_segment_risk"].to_numpy()
        age_years = (pd.Timestamp("2026-01-01") - pd.to_datetime(apps["birth_date"])) .dt.days.to_numpy() / 365.25
        base = 665 - 105 * seg_risk - 11 * thin + 24 * repeat_good - 14 * repeat_bad - 0.22 * np.abs(age_years - 36) + self.rng.normal(0, 38, n)
        bureau_score = np.clip(base, 280, 910)
        active_loans = np.clip(self.rng.poisson(1.25 + 0.85 * repeat_good + 0.40 * repeat_bad - 0.75 * thin), 0, 11)
        closed_loans = np.clip(self.rng.poisson(1.8 + 1.6 * repeat_good + 0.4 * repeat_bad - 1.15 * thin), 0, 28)
        del30 = np.clip(self.rng.poisson(sigmoid(-1.6 + 1.25 * seg_risk + 0.55 * repeat_bad - 0.50 * repeat_good) * 3.1), 0, 8)
        del60 = np.clip(self.rng.poisson(sigmoid(-2.3 + 1.45 * seg_risk + 0.78 * repeat_bad) * 1.8), 0, 5)
        del90 = np.clip(self.rng.poisson(sigmoid(-2.8 + 1.70 * seg_risk + 0.85 * repeat_bad) * 1.2), 0, 4)
        max_dpd = np.where(
            del90 > 0,
            self.rng.integers(90, 240, n),
            np.where(del60 > 0, self.rng.integers(60, 90, n), np.where(del30 > 0, self.rng.integers(15, 60, n), self.rng.integers(0, 15, n))),
        )
        inquiries30 = np.clip(self.rng.poisson(0.6 + 0.5 * seg_risk + 0.35 * thin), 0, 8)
        inquiries90 = inquiries30 + np.clip(self.rng.poisson(1.1 + 0.7 * seg_risk + 0.4 * thin), 0, 12)
        total_limit = np.where(
            thin == 1,
            np.maximum(0, self.rng.normal(120000, 90000, n)),
            np.maximum(0, self.rng.normal(650000 + 280000 * repeat_good - 90000 * repeat_bad, 260000, n)),
        )
        utilization = np.clip(self.rng.beta(1.7 + 1.1 * repeat_bad + 0.3 * thin, 2.7 + 1.3 * repeat_good, n), 0, 1)
        outstanding = np.round(total_limit * utilization + active_loans * self.rng.normal(35000, 24000, n), 2)
        oldest_trade = np.clip((active_loans + closed_loans) * self.rng.normal(9, 3, n) + 3 * repeat_good - 4 * thin, 0, 320)
        thin_flag = ((thin == 1) | ((active_loans + closed_loans) <= 1)).astype(int)
        ext_collections = (self.rng.random(n) < sigmoid(-4.4 + 1.2 * repeat_bad + 0.014 * max_dpd + 0.25 * thin_flag)).astype(int)
        bureau_date = pd.to_datetime(apps["application_date"]) - pd.to_timedelta(self.rng.integers(0, 4, n), unit="D")
        backfill_ix = self.rng.choice(n, size=max(30, int(n * 0.006)), replace=False)
        bureau_date.iloc[backfill_ix] = pd.to_datetime(apps.iloc[backfill_ix]["application_date"]) + pd.to_timedelta(self.rng.integers(0, 3, len(backfill_ix)), unit="D")

        bureau = pd.DataFrame(
            {
                "application_id": apps["application_id"].astype(np.int64),
                "bureau_pull_date": bureau_date,
                "bureau_score": np.round(bureau_score, 0),
                "active_loans_count": active_loans.astype(int),
                "closed_loans_count": closed_loans.astype(int),
                "delinquency_30_12m": del30.astype(int),
                "delinquency_60_12m": del60.astype(int),
                "delinquency_90_24m": del90.astype(int),
                "max_dpd_24m": max_dpd.astype(int),
                "inquiries_30d": inquiries30.astype(int),
                "inquiries_90d": inquiries90.astype(int),
                "outstanding_debt": np.round(np.clip(outstanding, 0, None), 2),
                "total_limit": np.round(np.clip(total_limit, 0, None), 2),
                "utilization_rate": np.round(utilization, 4),
                "oldest_trade_months": np.round(oldest_trade, 0).astype(int),
                "bureau_file_thin_flag": thin_flag.astype(int),
                "external_collections_flag": ext_collections.astype(int),
            }
        )

        # Missing bureau fields for thin-file profiles and operational misses.
        null_ix = bureau.index[(thin_flag == 1) & (self.rng.random(n) < 0.22)]
        bureau.loc[null_ix, ["utilization_rate", "total_limit", "outstanding_debt", "oldest_trade_months"]] = None
        rare_null_ix = self.rng.choice(n, size=max(20, int(n * 0.012)), replace=False)
        bureau.loc[rare_null_ix, ["bureau_score", "inquiries_90d"]] = None
        self.tables["bureau_snapshot"] = bureau
        return bureau

    def generate_employment_income_snapshot(self) -> pd.DataFrame:
        apps = self.debug["applications"]
        bureau = self.tables["bureau_snapshot"].set_index("application_id")
        n = len(apps)
        segment = apps["segment"]
        employment_type = np.where(
            segment.isin(["salaried", "good_repeat", "near_prime"]),
            self.rng.choice(EMPLOYMENT_TYPES, size=n, p=[0.58, 0.12, 0.08, 0.04, 0.08, 0.04, 0.06]),
            self.rng.choice(EMPLOYMENT_TYPES, size=n, p=[0.27, 0.14, 0.20, 0.10, 0.11, 0.11, 0.07]),
        )
        employer_industry = np.where(
            pd.Series(employment_type).isin(["unemployed", "pensioner"]),
            None,
            self.rng.choice(INDUSTRIES, size=n),
        )
        job_tenure = np.clip(
            self.rng.gamma(shape=2.8 + 0.8 * pd.Series(employment_type).isin(["formal_salaried", "pensioner"]).astype(float), scale=9.0, size=n),
            0,
            360,
        )
        work_exp = np.clip(job_tenure + self.rng.gamma(shape=2.3, scale=14.0, size=n), 0, 540)
        income_anchor = apps["_income_anchor"].to_numpy()
        stability_base = (
            54
            + 12 * pd.Series(employment_type).isin(["formal_salaried", "pensioner"]).astype(float).to_numpy()
            - 9 * pd.Series(employment_type).isin(["self_employed", "micro_business_owner"]).astype(float).to_numpy()
            + 0.06 * np.minimum(job_tenure, 96)
            - 6 * apps["segment"].isin(["risky_repeat", "subprime"]).astype(float).to_numpy()
            + self.rng.normal(0, 8, n)
        )
        stability = np.clip(stability_base, 5, 99)
        declared_income = income_anchor * np.exp(self.rng.normal(0, 0.18, n))
        under_verification_penalty = np.where(
            pd.Series(employment_type).isin(["self_employed", "micro_business_owner", "contractor"]),
            self.rng.uniform(0.68, 1.00, n),
            self.rng.uniform(0.82, 1.05, n),
        )
        verified_income = declared_income * under_verification_penalty
        other_income = np.where(self.rng.random(n) < 0.34, declared_income * self.rng.uniform(0.02, 0.30, n), 0.0)
        salary_project = (self.rng.random(n) < sigmoid(-0.6 + 0.06 * stability + 0.8 * pd.Series(employment_type).isin(["formal_salaried"]).astype(float).to_numpy() + 0.45 * apps["segment"].isin(["good_repeat"]).astype(float).to_numpy())).astype(int)
        pension_contrib = np.clip(
            np.round(
                12 * sigmoid(-1.4 + 0.05 * stability + 0.7 * pd.Series(employment_type).isin(["formal_salaried", "pensioner"]).astype(float).to_numpy() - 0.4 * pd.Series(employment_type).isin(["informal_salaried", "unemployed"]).astype(float).to_numpy())
                + self.rng.normal(0, 1.8, n)
            ),
            0,
            12,
        ).astype(int)
        outstanding = bureau.loc[apps["application_id"], "outstanding_debt"].fillna(0).to_numpy()
        monthly_requested = apps["requested_amount"].to_numpy() / np.maximum(apps["requested_term_months"].to_numpy(), 1)
        debt_to_income = np.clip((outstanding / 18 + monthly_requested) / np.maximum(verified_income + other_income * 0.6, 40000), 0.01, 1.75)
        expense_to_income = np.clip(0.28 + 0.06 * apps["dependents_count"].to_numpy() + 0.04 * apps["city_type"].isin(["metro", "urban"]).astype(float).to_numpy() + self.rng.normal(0, 0.06, n), 0.10, 0.92)

        emp = pd.DataFrame(
            {
                "application_id": apps["application_id"].astype(np.int64),
                "employment_type": employment_type,
                "employer_industry": employer_industry,
                "job_tenure_months": np.round(job_tenure, 0).astype(int),
                "total_work_experience_months": np.round(work_exp, 0).astype(int),
                "declared_income": np.round(np.clip(declared_income, 35000, 9000000), 2),
                "verified_income": np.round(np.clip(verified_income, 0, 9000000), 2),
                "other_income": np.round(np.clip(other_income, 0, 4000000), 2),
                "income_stability_score": np.round(stability, 0).astype(int),
                "salary_project_flag": salary_project.astype(int),
                "pension_contrib_months_12m": pension_contrib,
                "debt_to_income_est": np.round(debt_to_income, 4),
                "expense_to_income_est": np.round(expense_to_income, 4),
            }
        )
        miss_ix = self.rng.choice(n, size=max(20, int(n * 0.035)), replace=False)
        emp.loc[miss_ix, ["verified_income", "pension_contrib_months_12m"]] = None
        noise_ix = self.rng.choice(n, size=max(20, int(n * 0.018)), replace=False)
        emp.loc[noise_ix, "declared_income"] = np.round(emp.loc[noise_ix, "declared_income"] * self.rng.uniform(1.15, 1.6, len(noise_ix)), 2)

        self.tables["employment_income_snapshot"] = emp
        return emp

    def generate_decision_engine_results(self) -> pd.DataFrame:
        apps = self.debug["applications"].set_index("application_id")
        bureau = self.tables["bureau_snapshot"].set_index("application_id")
        emp = self.tables["employment_income_snapshot"].set_index("application_id")
        fraud = self.tables["fraud_flags"].set_index("application_id")
        n = len(apps)

        app_date = pd.to_datetime(apps["application_date"])
        policy_version = np.select(
            [
                app_date < pd.Timestamp("2023-07-01"),
                (app_date >= pd.Timestamp("2023-07-01")) & (app_date < pd.Timestamp("2024-03-01")),
                (app_date >= pd.Timestamp("2024-03-01")) & (app_date < pd.Timestamp("2024-11-01")),
                app_date >= pd.Timestamp("2024-11-01"),
            ],
            ["policy_v1_growth", "policy_v2_tight", "policy_v3_relax", "policy_v4_guard"],
            default="policy_v3_relax",
        )
        model_version = np.where(app_date < pd.Timestamp("2024-05-01"), "score_v1", "score_v2")
        policy_adj = pd.Series(policy_version).map(
            {
                "policy_v1_growth": -8,
                "policy_v2_tight": 12,
                "policy_v3_relax": -4,
                "policy_v4_guard": 16,
            }
        ).to_numpy()
        score = (
            575
            + 0.38 * bureau["bureau_score"].fillna(615).to_numpy()
            - 165 * emp["debt_to_income_est"].fillna(0.62).to_numpy()
            - 22 * bureau["delinquency_30_12m"].fillna(1).to_numpy()
            - 30 * bureau["delinquency_60_12m"].fillna(0).to_numpy()
            - 46 * bureau["delinquency_90_24m"].fillna(0).to_numpy()
            - 10 * bureau["inquiries_90d"].fillna(2).to_numpy()
            + 0.75 * emp["income_stability_score"].fillna(55).to_numpy()
            + 18 * emp["salary_project_flag"].fillna(0).to_numpy()
            + 20 * apps["segment"].isin(["good_repeat"]).astype(float).to_numpy()
            - 18 * apps["segment"].isin(["risky_repeat", "subprime"]).astype(float).to_numpy()
            - 12 * apps["product_type"].isin(["credit_line"]).astype(float).to_numpy()
            + 7 * apps["channel"].isin(["branch", "partner_pos"]).astype(float).to_numpy()
            - 0.000018 * apps["requested_amount"].to_numpy()
            - 0.8 * fraud["device_risk_score"].to_numpy()
            + 9 * apps["is_existing_customer"].to_numpy()
            - 9 * apps["_stress"].to_numpy()
            - policy_adj
            + self.rng.normal(0, 28, n)
        )
        score = np.clip(score, 290, 915)
        pd_estimate = sigmoid(2.45 - score / 110 + 1.25 * emp["debt_to_income_est"].fillna(0.60).to_numpy() + 0.08 * bureau["delinquency_30_12m"].fillna(1).to_numpy() + 0.17 * bureau["external_collections_flag"].fillna(0).to_numpy())
        pd_estimate = np.clip(pd_estimate, 0.012, 0.65)
        risk_grade = pd.cut(score, bins=[0, 540, 610, 680, 745, 1000], labels=["E", "D", "C", "B", "A"], right=False).astype(str)

        hard_decline_flag = (
            (bureau["external_collections_flag"].fillna(0).to_numpy() == 1)
            | (fraud["blacklist_hit_flag"].to_numpy() == 1)
            | (fraud["synthetic_identity_flag"].to_numpy() == 1)
            | (emp["debt_to_income_est"].fillna(0.8).to_numpy() > 0.95)
            | (bureau["delinquency_90_24m"].fillna(0).to_numpy() >= 2)
        ).astype(int)
        hard_reason = np.where(
            fraud["blacklist_hit_flag"].to_numpy() == 1,
            "blacklist_hit",
            np.where(
                fraud["synthetic_identity_flag"].to_numpy() == 1,
                "synthetic_identity",
                np.where(
                    bureau["external_collections_flag"].fillna(0).to_numpy() == 1,
                    "external_collections",
                    np.where(emp["debt_to_income_est"].fillna(0.8).to_numpy() > 0.95, "high_dti", np.where(bureau["delinquency_90_24m"].fillna(0).to_numpy() >= 2, "severe_delinquency_history", None)),
                ),
            ),
        )
        manual_review_flag = (
            (np.abs(score - (640 - policy_adj)) < 24)
            | (apps["requested_amount"].to_numpy() > np.nanpercentile(apps["requested_amount"].to_numpy(), 83))
            | (fraud["verification_status"].isin(["manual_review"]).astype(int).to_numpy() == 1)
            | (emp["verified_income"].fillna(0).to_numpy() < emp["declared_income"].fillna(0).to_numpy() * 0.72)
        ).astype(int)

        cutoff = np.where(
            apps["product_type"].to_numpy() == "credit_line",
            655 + policy_adj,
            np.where(apps["product_type"].to_numpy() == "cash_loan", 632 + policy_adj, 620 + policy_adj),
        )
        base_approve = (score >= cutoff) & (hard_decline_flag == 0)

        # Manual review overrides both ways.
        borderline = np.abs(score - cutoff) < 28
        good_repeat = apps["segment"].isin(["good_repeat"]).astype(float).to_numpy()
        suspicious = fraud["fraud_suspect_flag"].to_numpy() + (emp["debt_to_income_est"].fillna(0.60).to_numpy() > 0.72).astype(int) + (bureau["bureau_file_thin_flag"].fillna(0).to_numpy() == 1).astype(int)
        override_to_approve = (manual_review_flag == 1) & borderline & (self.rng.random(n) < sigmoid(-1.15 + 1.1 * good_repeat + 0.18 * (score - cutoff) / 10 - 0.45 * suspicious))
        override_to_decline = (manual_review_flag == 1) & base_approve & (self.rng.random(n) < sigmoid(-2.5 + 0.65 * suspicious + 0.18 * (cutoff - score) / 10 + 0.45 * (fraud["verification_status"].isin(["failed"]).astype(int).to_numpy())))
        final_approve = (base_approve | override_to_approve) & (~override_to_decline)
        final_approve = final_approve.astype(int)

        decision_final = np.where(final_approve == 1, "approved", "declined")
        # A small share of approved applications expire or get cancelled before disbursement.
        application_status = np.where(
            decision_final == "declined",
            self.rng.choice(["declined", "declined", "cancelled", "expired"], size=n, p=[0.82, 0.08, 0.06, 0.04]),
            self.rng.choice(["approved", "approved", "cancelled", "expired"], size=n, p=[0.82, 0.08, 0.05, 0.05]),
        )

        offered_amount = apps["requested_amount"].to_numpy() * np.clip(0.72 + (score - cutoff) / 260 + self.rng.normal(0, 0.08, n), 0.35, 1.12)
        offered_amount = np.where(final_approve == 1, np.clip(offered_amount, 20000, apps["requested_amount"].to_numpy() * 1.05), 0)
        offered_term = np.where(
            final_approve == 1,
            np.maximum(3, apps["requested_term_months"].to_numpy() - self.rng.choice([0, 0, 0, 3, 6], size=n, p=[0.44, 0.18, 0.12, 0.18, 0.08])),
            0,
        )
        base_rate = np.where(
            apps["product_type"].to_numpy() == "installment_loan",
            0.19,
            np.where(apps["product_type"].to_numpy() == "cash_loan", 0.27, 0.31),
        )
        offered_rate = np.where(final_approve == 1, base_rate + 0.09 * pd_estimate + 0.012 * apps["_stress"].to_numpy() + self.rng.normal(0, 0.015, n), 0.0)
        rule_hits = (bureau["delinquency_30_12m"].fillna(0).to_numpy() > 0).astype(int) + (emp["debt_to_income_est"].fillna(0.60).to_numpy() > 0.55).astype(int) + fraud["fraud_suspect_flag"].to_numpy() + bureau["inquiries_90d"].fillna(0).to_numpy() // 4 + bureau["bureau_file_thin_flag"].fillna(0).to_numpy()
        decision_ts = pd.to_datetime(apps["application_date"]) + pd.to_timedelta(self.rng.integers(5, 72, n), unit="h")

        der = pd.DataFrame(
            {
                "application_id": apps.index.astype(np.int64),
                "scorecard_score": np.round(score, 0).astype(int),
                "pd_estimate": np.round(pd_estimate, 5),
                "risk_grade": risk_grade,
                "hard_decline_flag": hard_decline_flag.astype(int),
                "hard_decline_reason": hard_reason,
                "manual_review_flag": manual_review_flag.astype(int),
                "policy_version": policy_version,
                "cutoff_value": np.round(cutoff, 0).astype(int),
                "offered_amount": np.round(offered_amount, 2),
                "offered_term_months": offered_term.astype(int),
                "offered_rate": np.round(np.clip(offered_rate, 0, 0.72), 4),
                "decision_final": decision_final,
                "decision_timestamp": decision_ts,
                "model_version": model_version,
                "rule_hits_count": np.clip(rule_hits, 0, 12).astype(int),
            }
        )

        self.tables["decision_engine_results"] = der
        # Update application status in exported applications
        self.tables["applications"]["application_status"] = application_status
        self.debug["applications"]["application_status"] = application_status
        return der

    def generate_loans(self) -> pd.DataFrame:
        apps = self.debug["applications"].set_index("application_id")
        der = self.tables["decision_engine_results"].set_index("application_id")
        bureau = self.tables["bureau_snapshot"].set_index("application_id")
        emp = self.tables["employment_income_snapshot"].set_index("application_id")

        approved_ids = der.index[der["decision_final"] == "approved"]
        approved = apps.loc[approved_ids].copy()
        der_a = der.loc[approved_ids]
        take_up_prob = sigmoid(
            1.95
            + 0.22 * approved["channel"].isin(["branch", "partner_pos"]).astype(float).to_numpy()
            + 0.15 * approved["segment"].isin(["good_repeat", "salaried"]).astype(float).to_numpy()
            - 0.55 * approved["product_type"].isin(["credit_line"]).astype(float).to_numpy()
            - 0.35 * (der_a["offered_amount"].to_numpy() < approved["requested_amount"].to_numpy() * 0.72)
            - 0.18 * approved["_stress"].to_numpy()
        )
        take_up = self.rng.random(len(approved)) < take_up_prob
        booked = approved.loc[take_up].copy()
        der_b = der_a.loc[booked.index]
        emp_b = emp.loc[booked.index]
        bureau_b = bureau.loc[booked.index]

        n = len(booked)
        disb_date = pd.to_datetime(der_b["decision_timestamp"]).dt.floor("D") + pd.to_timedelta(self.rng.integers(0, 6, n), unit="D")
        principal = np.round(np.clip(der_b["offered_amount"].to_numpy() * self.rng.uniform(0.95, 1.0, n), 15000, None), 2)
        term = np.where(der_b["offered_term_months"].to_numpy() > 0, der_b["offered_term_months"].to_numpy(), booked["requested_term_months"].to_numpy()).astype(int)
        rate = np.round(np.clip(der_b["offered_rate"].to_numpy(), 0.12, 0.72), 4)
        monthly_r = rate / 12
        with np.errstate(divide="ignore", invalid="ignore"):
            annuity = np.where(
                monthly_r > 0,
                principal * monthly_r / (1 - np.power(1 + monthly_r, -term)),
                principal / term,
            )
        annuity = np.clip(annuity, 5000, None)
        orig_fee = np.round(principal * np.where(booked["product_type"].to_numpy() == "installment_loan", self.rng.uniform(0.005, 0.025, n), self.rng.uniform(0.008, 0.04, n)), 2)
        insurance = np.round(np.where(self.rng.random(n) < 0.42, principal * self.rng.uniform(0.0, 0.025, n), 0), 2)

        orig_risk = sigmoid(
            -2.15
            + 3.2 * der_b["pd_estimate"].to_numpy()
            + 0.55 * emp_b["debt_to_income_est"].fillna(0.6).to_numpy()
            + 0.17 * bureau_b["delinquency_30_12m"].fillna(0).to_numpy()
            + 0.33 * bureau_b["delinquency_60_12m"].fillna(0).to_numpy()
            + 0.48 * bureau_b["bureau_file_thin_flag"].fillna(0).to_numpy()
            + 0.18 * booked["product_type"].isin(["credit_line"]).astype(float).to_numpy()
            + 0.10 * (principal / np.maximum(emp_b["verified_income"].fillna(emp_b["declared_income"]).fillna(150000).to_numpy(), 50000))
            + 0.10 * booked["_stress"].to_numpy()
            - 0.35 * booked["segment"].isin(["good_repeat", "salaried"]).astype(float).to_numpy()
            + self.rng.normal(0, 0.18, n)
        )
        prepay_propensity = sigmoid(
            -1.9
            + 0.7 * booked["segment"].isin(["good_repeat", "salaried", "near_prime"]).astype(float).to_numpy()
            + 0.35 * emp_b["salary_project_flag"].fillna(0).to_numpy()
            - 1.2 * orig_risk
            + 0.08 * (rate < np.median(rate))
        )

        loans = pd.DataFrame(
            {
                "loan_id": np.arange(1, n + 1, dtype=np.int64),
                "application_id": booked.index.astype(np.int64),
                "client_id": booked["client_id"].astype(np.int64).to_numpy(),
                "disbursement_date": disb_date,
                "product_type": booked["product_type"].to_numpy(),
                "principal_amount": principal,
                "term_months": term,
                "interest_rate": rate,
                "monthly_payment_amount": np.round(annuity, 2),
                "origination_fee": orig_fee,
                "insurance_fee": insurance,
                "status_current": "active",
                "close_date": pd.NaT,
                "restructuring_flag": 0,
                "writeoff_flag": 0,
            }
        )
        self.tables["loans"] = loans

        loans_debug = loans.copy()
        loans_debug["orig_risk_dbg"] = np.round(orig_risk, 6)
        loans_debug["prepay_propensity_dbg"] = np.round(prepay_propensity, 6)
        loans_debug["verified_income_dbg"] = emp_b["verified_income"].fillna(emp_b["declared_income"]).fillna(150000).to_numpy()
        loans_debug["salary_project_flag_dbg"] = emp_b["salary_project_flag"].fillna(0).to_numpy()
        loans_debug["segment_dbg"] = booked["segment"].to_numpy()
        loans_debug["bureau_score_dbg"] = bureau_b["bureau_score"].fillna(620).to_numpy()
        loans_debug["stress_at_book_dbg"] = booked["_stress"].to_numpy()
        self.debug["loans"] = loans_debug
        return loans

    # -----------------------------
    # Post-origination behaviour
    # -----------------------------
    def generate_payment_schedule(self) -> pd.DataFrame:
        loans = self.tables["loans"].copy()
        rows: List[Tuple] = []
        for row in loans.itertuples(index=False):
            os_bal = float(row.principal_amount)
            monthly_r = float(row.interest_rate) / 12
            due_day = min(pd.Timestamp(row.disbursement_date).day, 28)
            first_due = (pd.Timestamp(row.disbursement_date) + pd.offsets.MonthBegin(1)).replace(day=due_day)
            for inst in range(1, int(row.term_months) + 1):
                due_date = first_due + pd.DateOffset(months=inst - 1)
                interest_due = os_bal * monthly_r
                total_due = float(row.monthly_payment_amount)
                principal_due = max(total_due - interest_due, 0.0)
                if inst == int(row.term_months):
                    principal_due = os_bal
                    total_due = principal_due + interest_due
                fee_due = 0.0
                rows.append(
                    (
                        int(row.loan_id),
                        inst,
                        due_date.date(),
                        round(principal_due, 2),
                        round(interest_due, 2),
                        round(fee_due, 2),
                        round(total_due, 2),
                    )
                )
                os_bal = max(os_bal - principal_due, 0.0)
        schedule = pd.DataFrame(
            rows,
            columns=[
                "loan_id",
                "installment_no",
                "due_date",
                "principal_due",
                "interest_due",
                "fee_due",
                "total_due",
            ],
        )
        self.tables["payment_schedule"] = schedule
        return schedule

    def generate_loan_monthly_snapshot_and_payments(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        loans = self.debug["loans"].copy().set_index("loan_id")
        schedule = self.tables["payment_schedule"].copy()
        sched_map = {}
        for lid, df in schedule.groupby("loan_id", sort=False):
            sched_map[lid] = (
                pd.to_datetime(df["due_date"]).tolist(),
                df["principal_due"].to_numpy(dtype=float),
                df["interest_due"].to_numpy(dtype=float),
                df["total_due"].to_numpy(dtype=float),
            )
        macro = self.tables["macro_monthly_factors"].copy()
        macro["month"] = pd.to_datetime(macro["month"])
        macro_map = macro.set_index("month").to_dict("index")
        obs_end = pd.Timestamp(self.cfg.obs_end_month)

        snap_rows: List[Tuple] = []
        payment_rows: List[Tuple] = []
        payment_id = 1
        restructuring_loans = set()
        writeoff_loans = set()
        close_dates: Dict[int, pd.Timestamp] = {}
        status_current: Dict[int, str] = {}

        for loan in loans.itertuples():
            loan_id = int(loan.Index)
            principal = float(loan.principal_amount)
            term = int(loan.term_months)
            rate = float(loan.interest_rate)
            orig_risk = float(loan.orig_risk_dbg)
            prepay_prop = float(loan.prepay_propensity_dbg)
            salary_project = float(loan.salary_project_flag_dbg)
            seg = str(loan.segment_dbg)
            due_dates, sched_principal_arr, sched_interest_arr, sched_total_arr = sched_map[loan_id]
            sched_len = len(due_dates)
            start_month = pd.Timestamp(loan.disbursement_date).to_period("M").to_timestamp()
            months_avail = int((obs_end.to_period("M") - start_month.to_period("M")).n) + 1
            months_to_generate = max(1, min(months_avail, term + 8))
            os_bal = principal
            state = 0  # 0 current,1 1-29,2 30-59,3 60-89,4 90+,5 closed,6 writeoff
            ever30 = ever60 = ever90 = 0
            severe_streak = 0
            prev_state = 0
            last_effective_mob = 0

            for mob in range(months_to_generate):
                snap_month = start_month + pd.DateOffset(months=mob)
                macro_row = macro_map.get(pd.Timestamp(snap_month), None)
                stress = macro_row["stress_regime_flag"] if macro_row is not None else 0
                base_rate = macro_row["base_rate"] if macro_row is not None else 15.0
                if mob < sched_len:
                    due_date = due_dates[mob]
                    scheduled_total = float(sched_total_arr[mob])
                    scheduled_principal = float(sched_principal_arr[mob])
                    scheduled_interest = float(sched_interest_arr[mob])
                else:
                    due_date = None
                    scheduled_total = 0.0
                    scheduled_principal = 0.0
                    scheduled_interest = 0.0

                if state in (5, 6):
                    break

                # Delinquency / cure / prepay transitions.
                p_prepay = float(sigmoid(-4.2 + 2.4 * prepay_prop - 1.3 * orig_risk + 0.12 * mob + 0.18 * salary_project - 0.18 * stress)) if state == 0 else 0.0
                p_miss = float(sigmoid(-3.8 + 3.3 * orig_risk + 0.38 * stress + 0.32 * (1 <= mob <= 4) + 0.20 * (seg in ["subprime", "risky_repeat", "thin_file"]) - 0.18 * salary_project))
                p_cure = float(sigmoid(0.9 - 2.15 * orig_risk - 0.36 * stress + 0.26 * salary_project - 0.10 * state))
                p_worse = float(sigmoid(-0.95 + 1.95 * orig_risk + 0.30 * stress + 0.15 * state))

                transition_rand = self.rng.random()
                closed_this_month = False
                writeoff_this_month = False
                cure_flag = 0
                restructured = 0
                utilization_like = None
                recovery_stage = "none"

                if state == 0:
                    if mob > 2 and transition_rand < p_prepay and os_bal > 10000:
                        state = 5
                        closed_this_month = True
                        payment_ratio = float(self.rng.uniform(1.8, 4.8))
                    elif transition_rand < p_miss:
                        state = 2 if self.rng.random() < sigmoid(-3.1 + 3.2 * orig_risk) else 1
                        payment_ratio = float(np.clip(self.rng.normal(0.45, 0.28), 0.0, 1.1))
                    else:
                        state = 0
                        payment_ratio = float(np.clip(self.rng.normal(1.0, 0.10), 0.65, 1.35))
                elif state == 1:
                    if transition_rand < p_cure:
                        state = 0
                        cure_flag = 1
                        payment_ratio = float(np.clip(self.rng.normal(1.35, 0.28), 0.8, 2.4))
                    elif transition_rand < p_cure + p_worse * 0.55:
                        state = 2
                        payment_ratio = float(np.clip(self.rng.normal(0.25, 0.18), 0.0, 0.8))
                    else:
                        state = 1
                        payment_ratio = float(np.clip(self.rng.normal(0.55, 0.20), 0.0, 1.0))
                elif state == 2:
                    if transition_rand < p_cure * 0.72:
                        state = 1 if self.rng.random() < 0.55 else 0
                        cure_flag = 1
                        payment_ratio = float(np.clip(self.rng.normal(1.15, 0.35), 0.5, 2.1))
                    elif transition_rand < p_cure * 0.72 + p_worse * 0.60:
                        state = 3
                        payment_ratio = float(np.clip(self.rng.normal(0.12, 0.10), 0.0, 0.45))
                    else:
                        state = 2
                        payment_ratio = float(np.clip(self.rng.normal(0.22, 0.14), 0.0, 0.65))
                elif state == 3:
                    if transition_rand < p_cure * 0.48:
                        state = 2
                        cure_flag = 1
                        payment_ratio = float(np.clip(self.rng.normal(0.85, 0.30), 0.2, 1.8))
                    elif transition_rand < p_cure * 0.48 + p_worse * 0.62:
                        state = 4
                        payment_ratio = float(np.clip(self.rng.normal(0.08, 0.08), 0.0, 0.30))
                    else:
                        state = 3
                        payment_ratio = float(np.clip(self.rng.normal(0.10, 0.08), 0.0, 0.35))
                elif state == 4:
                    severe_streak += 1
                    recovery_stage = "soft" if severe_streak <= 2 else ("hard" if severe_streak <= 5 else "legal")
                    if severe_streak >= 6 and self.rng.random() < sigmoid(-0.8 + 2.0 * orig_risk + 0.22 * stress):
                        state = 6
                        writeoff_this_month = True
                        payment_ratio = float(np.clip(self.rng.normal(0.03, 0.04), 0.0, 0.12))
                    elif transition_rand < p_cure * 0.22:
                        state = 3
                        cure_flag = 1
                        payment_ratio = float(np.clip(self.rng.normal(0.42, 0.22), 0.0, 1.1))
                    else:
                        state = 4
                        payment_ratio = float(np.clip(self.rng.normal(0.05, 0.05), 0.0, 0.22))
                else:
                    payment_ratio = 0.0

                if state < 4:
                    severe_streak = 0
                if state >= 2 and mob >= 4 and self.rng.random() < sigmoid(-5.4 + 3.8 * orig_risk + 0.2 * stress):
                    restructured = 1
                    restructuring_loans.add(loan_id)
                    payment_ratio = max(payment_ratio, float(self.rng.uniform(0.35, 0.95)))
                    recovery_stage = "restructured"

                if state >= 1:
                    ever30 = 1
                if state >= 2:
                    ever60 = 1
                if state >= 4:
                    ever90 = 1

                dpd = (
                    int(self.rng.integers(1, 30)) if state == 1 else
                    int(self.rng.integers(30, 60)) if state == 2 else
                    int(self.rng.integers(60, 90)) if state == 3 else
                    int(self.rng.integers(90, 180)) if state == 4 else
                    0
                )
                bucket = {0: "current", 1: "1_29", 2: "30_59", 3: "60_89", 4: "90_plus", 5: "closed", 6: "writeoff"}[state]

                if loan.product_type == "credit_line":
                    utilization_like = round(float(np.clip(sigmoid(-0.6 + 1.6 * orig_risk + 0.22 * state + self.rng.normal(0, 0.5)), 0.05, 0.99)), 4)

                accrued_interest = round(os_bal * rate / 12 * (1 + 0.15 * (state >= 2)), 2)
                actual_total_paid = round(max(0.0, scheduled_total * payment_ratio), 2)
                if closed_this_month:
                    actual_total_paid = round(max(actual_total_paid, os_bal + accrued_interest), 2)
                principal_paid = min(os_bal, max(0.0, scheduled_principal * min(payment_ratio, 1.0) + max(actual_total_paid - scheduled_total, 0.0)))
                if state >= 4:
                    principal_paid = min(principal_paid, scheduled_principal * 0.2)
                interest_paid = min(accrued_interest, max(0.0, actual_total_paid - principal_paid))
                fee_paid = round(max(0.0, actual_total_paid - principal_paid - interest_paid), 2)
                os_bal = round(max(0.0, os_bal - principal_paid), 2)
                if os_bal < 250 and mob >= 1:
                    closed_this_month = True
                    state = 5
                    bucket = "closed"
                    dpd = 0
                    recovery_stage = recovery_stage if recovery_stage != "none" else "closed"
                if writeoff_this_month:
                    writeoff_loans.add(loan_id)
                    recovery_stage = "writeoff"
                if state == 5 and loan_id not in close_dates:
                    close_dates[loan_id] = snap_month + pd.offsets.MonthEnd(0)
                if state == 6 and loan_id not in close_dates:
                    close_dates[loan_id] = snap_month + pd.offsets.MonthEnd(0)

                status_month_end = (
                    "closed" if state == 5 else
                    "written_off" if state == 6 else
                    "restructured" if restructured == 1 else
                    "delinquent" if state >= 1 else
                    "active"
                )
                last_effective_mob = mob
                snap_rows.append(
                    (
                        loan_id,
                        snap_month.date(),
                        mob,
                        round(os_bal, 2),
                        round(accrued_interest, 2),
                        dpd,
                        bucket,
                        status_month_end,
                        ever30,
                        ever60,
                        ever90,
                        round(payment_ratio, 4),
                        utilization_like,
                        cure_flag,
                        restructured,
                        recovery_stage,
                    )
                )

                # Payments generated from monthly ratio and state. Some months have split payments.
                if actual_total_paid > 0 and due_date is not None:
                    split = 2 if actual_total_paid > scheduled_total * 1.1 and self.rng.random() < 0.18 else (2 if self.rng.random() < 0.10 and state <= 1 else 1)
                    residual = actual_total_paid
                    residual_pr = principal_paid
                    residual_int = interest_paid
                    residual_fee = fee_paid
                    for k in range(split):
                        if k == split - 1:
                            amt = round(residual, 2)
                            pr = round(residual_pr, 2)
                            intr = round(residual_int, 2)
                            fee = round(residual_fee, 2)
                        else:
                            share = float(self.rng.uniform(0.25, 0.65))
                            amt = round(actual_total_paid * share, 2)
                            pr = round(principal_paid * share, 2)
                            intr = round(interest_paid * share, 2)
                            fee = round(max(0.0, amt - pr - intr), 2)
                            residual -= amt
                            residual_pr -= pr
                            residual_int -= intr
                            residual_fee -= fee
                        if state == 0:
                            days_from_due = int(self.rng.integers(-3, 8))
                        elif state == 1:
                            days_from_due = int(self.rng.integers(1, 28))
                        elif state == 2:
                            days_from_due = int(self.rng.integers(18, 52))
                        else:
                            days_from_due = int(self.rng.integers(30, 85))
                        pay_date = pd.Timestamp(due_date) + pd.to_timedelta(days_from_due, unit="D")
                        pay_channel = self.rng.choice(["mobile_app", "bank_transfer", "cash_terminal", "branch_cash", "autopay"], p=[0.34, 0.26, 0.18, 0.08, 0.14])
                        pay_source = self.rng.choice(["borrower", "salary_deduction", "collection_agent", "family_member"], p=[0.78, 0.10, 0.08, 0.04])
                        payment_rows.append(
                            (
                                payment_id,
                                loan_id,
                                pay_date.date(),
                                round(amt, 2),
                                round(pr, 2),
                                round(intr, 2),
                                round(fee, 2),
                                pay_channel,
                                int(split > 1 or payment_ratio < 0.99),
                                days_from_due,
                                pay_source,
                            )
                        )
                        payment_id += 1

                prev_state = state
                if state in (5, 6):
                    break

            status_current[loan_id] = (
                "written_off" if loan_id in writeoff_loans else "closed" if loan_id in close_dates and close_dates[loan_id] < (obs_end + pd.offsets.MonthEnd(0)) else ("delinquent" if prev_state >= 1 else "active")
            )

        snapshots = pd.DataFrame(
            snap_rows,
            columns=[
                "loan_id",
                "snapshot_month",
                "mob",
                "os_principal",
                "accrued_interest",
                "dpd",
                "delinquency_bucket",
                "status_month_end",
                "ever_30_flag",
                "ever_60_flag",
                "ever_90_flag",
                "payment_ratio_month",
                "utilization_like_metric",
                "cure_flag",
                "restructuring_flag",
                "recovery_stage",
            ],
        )
        payments = pd.DataFrame(
            payment_rows,
            columns=[
                "payment_id",
                "loan_id",
                "payment_date",
                "amount_paid",
                "principal_paid",
                "interest_paid",
                "fee_paid",
                "payment_channel",
                "is_partial_payment",
                "days_from_due",
                "payment_source",
            ],
        )

        # Update loans with final statuses.
        loans_export = self.tables["loans"].copy()
        loans_export["restructuring_flag"] = loans_export["loan_id"].isin(restructuring_loans).astype(int)
        loans_export["writeoff_flag"] = loans_export["loan_id"].isin(writeoff_loans).astype(int)
        loans_export["close_date"] = loans_export["loan_id"].map(close_dates)
        loans_export["status_current"] = loans_export["loan_id"].map(status_current).fillna("active")
        self.tables["loans"] = loans_export
        self.tables["loan_monthly_snapshot"] = snapshots
        self.tables["payments"] = payments
        return snapshots, payments

    def generate_collections_actions(self) -> pd.DataFrame:
        snaps = self.tables["loan_monthly_snapshot"].copy()
        loans = self.tables["loans"].set_index("loan_id")
        delinquent = snaps[snaps["dpd"] > 0].copy()
        rows: List[Tuple] = []
        action_id = 1
        for row in delinquent.itertuples(index=False):
            bucket = row.delinquency_bucket
            n_actions = 1 + int(self.rng.random() < (0.35 if bucket in ["30_59", "60_89", "90_plus"] else 0.12))
            for _ in range(n_actions):
                if bucket == "1_29":
                    stage = "soft"
                    action_type = self.rng.choice(["sms", "reminder_call", "robo_call", "push_notification"], p=[0.40, 0.30, 0.18, 0.12])
                    agency = "internal"
                elif bucket in ["30_59", "60_89"]:
                    stage = "hard"
                    action_type = self.rng.choice(["collection_call", "field_visit", "promise_to_pay", "restructure_offer"], p=[0.45, 0.13, 0.22, 0.20])
                    agency = self.rng.choice(["internal", "outsourced"], p=[0.72, 0.28])
                else:
                    stage = self.rng.choice(["hard", "legal", "pre_writeoff"], p=[0.32, 0.44, 0.24])
                    action_type = self.rng.choice(["legal_notice", "external_agency_assignment", "field_visit", "settlement_offer"], p=[0.28, 0.30, 0.18, 0.24])
                    agency = self.rng.choice(["outsourced", "legal", "internal"], p=[0.48, 0.32, 0.20])
                promise = int(self.rng.random() < (0.09 if stage == "soft" else 0.18))
                success = int(self.rng.random() < (0.42 if stage == "soft" else 0.28 if stage == "hard" else 0.18))
                outcome = self.rng.choice(
                    ["no_contact", "contacted", "promise_kept", "promise_broken", "paid_partial", "paid_full", "restructured", "legal_transfer"],
                    p=[0.22, 0.18, 0.08, 0.11, 0.18, 0.07, 0.08, 0.08],
                )
                recovered = 0.0
                if outcome in ["paid_partial", "paid_full", "promise_kept"]:
                    recovered = round(float(self.rng.uniform(3000, max(7000, loans.loc[row.loan_id, "monthly_payment_amount"] * (2.8 if outcome == "paid_full" else 1.1)))), 2)
                action_date = pd.Timestamp(row.snapshot_month) + pd.to_timedelta(int(self.rng.integers(1, 27)), unit="D")
                rows.append(
                    (
                        action_id,
                        int(row.loan_id),
                        action_date.date(),
                        stage,
                        action_type,
                        agency,
                        promise,
                        outcome,
                        recovered,
                        success,
                    )
                )
                action_id += 1
        actions = pd.DataFrame(
            rows,
            columns=[
                "action_id",
                "loan_id",
                "action_date",
                "action_stage",
                "action_type",
                "agency_type",
                "promise_to_pay_flag",
                "outcome_code",
                "recovered_amount",
                "contact_success_flag",
            ],
        )
        self.tables["collections_actions"] = actions
        return actions

    def generate_application_events(self) -> pd.DataFrame:
        apps = self.tables["applications"].copy().set_index("application_id")
        der = self.tables["decision_engine_results"].set_index("application_id")
        loans = self.tables["loans"].set_index("application_id")
        rows: List[Tuple] = []
        event_id = 1
        for app in apps.itertuples():
            base_ts = pd.Timestamp(app.application_date)
            device = app.device_type
            steps = [
                ("submitted", 0),
                ("bureau_requested", int(self.rng.integers(0, 3))),
                ("score_calculated", int(self.rng.integers(1, 6))),
            ]
            if int(der.loc[app.Index, "manual_review_flag"]) == 1:
                steps.append(("manual_review", int(self.rng.integers(6, 48))))
            steps.append(("decision_sent", int(self.rng.integers(6, 72))))
            if app.application_status in ["approved", "expired", "cancelled"]:
                steps.append(("offer_shown", int(self.rng.integers(8, 80))))
            if app.Index in loans.index:
                steps.append(("disbursed", int((pd.Timestamp(loans.loc[app.Index, "disbursement_date"]) - base_ts).days * 24 + self.rng.integers(1, 24))))
            elif app.application_status in ["expired", "cancelled"]:
                steps.append(("abandoned", int(self.rng.integers(8, 120))))
            for event_type, hours_after in steps:
                rows.append(
                    (
                        event_id,
                        int(app.Index),
                        (base_ts + pd.to_timedelta(hours_after, unit="h")),
                        event_type,
                        app.channel,
                        device,
                        round(float(np.clip(self.rng.normal(8 if app.channel in ["mobile_app", "web"] else 15, 4), 1, 90)), 2),
                        round(float(self.rng.uniform(0, 1)), 4),
                    )
                )
                event_id += 1
        events = pd.DataFrame(
            rows,
            columns=[
                "event_id",
                "application_id",
                "event_timestamp",
                "event_type",
                "channel",
                "device_type",
                "session_minutes",
                "event_value",
            ],
        )
        self.tables["application_events"] = events
        return events

    # -----------------------------
    # Export and QA summaries
    # -----------------------------
    def run(self) -> Dict[str, pd.DataFrame]:
        self.generate_macro_monthly_factors()
        self.generate_clients()
        self.generate_client_contact_info()
        self.generate_applications()
        self.generate_fraud_flags()
        self.generate_bureau_snapshot()
        self.generate_employment_income_snapshot()
        self.generate_decision_engine_results()
        self.generate_loans()
        self.generate_payment_schedule()
        self.generate_loan_monthly_snapshot_and_payments()
        self.generate_collections_actions()
        self.generate_application_events()
        self._final_cleanups()
        self.export_csvs()
        return self.tables

    def _final_cleanups(self):
        # Mild status/timestamp inconsistencies by design.
        apps = self.tables["applications"]
        der = self.tables["decision_engine_results"]
        loans = self.tables["loans"]
        loan_app_ids = set(loans["application_id"].tolist())
        apps.loc[apps["application_id"].isin(loan_app_ids), "application_status"] = "approved"
        odd_ix = self.rng.choice(apps.index, size=max(10, int(len(apps) * 0.0015)), replace=False)
        apps.loc[odd_ix, "application_status"] = self.rng.choice(["expired", "cancelled", "approved"], size=len(odd_ix), p=[0.34, 0.18, 0.48])

        # Small number of backfilled close dates.
        close_ix = loans.index[loans["close_date"].notna()]
        if len(close_ix) > 0:
            bf_ix = self.rng.choice(close_ix, size=max(8, int(len(close_ix) * 0.012)), replace=False)
            loans.loc[bf_ix, "close_date"] = pd.to_datetime(loans.loc[bf_ix, "close_date"]) + pd.to_timedelta(self.rng.integers(-3, 4, len(bf_ix)), unit="D")
        self.tables["applications"] = apps
        self.tables["decision_engine_results"] = der
        self.tables["loans"] = loans

    def export_csvs(self):
        ordered_tables = [
            "clients",
            "client_contact_info",
            "applications",
            "bureau_snapshot",
            "employment_income_snapshot",
            "fraud_flags",
            "decision_engine_results",
            "loans",
            "payment_schedule",
            "payments",
            "loan_monthly_snapshot",
            "collections_actions",
            "macro_monthly_factors",
            "application_events",
        ]
        for name in ordered_tables:
            df = self.tables[name].copy()
            out = self.output_dir / f"{name}.csv"
            df.to_csv(out, index=False)

        summary = self.build_summary()
        summary.to_csv(self.output_dir / "_generation_summary.csv", index=False)

    def build_summary(self) -> pd.DataFrame:
        loans = self.tables["loans"]
        apps = self.tables["applications"]
        der = self.tables["decision_engine_results"]
        snaps = self.tables["loan_monthly_snapshot"]
        loan_targets = snaps.groupby("loan_id").agg(
            fpd_flag=("dpd", lambda s: int((s.iloc[1:2] >= 1).any()) if len(s) > 1 else 0),
            ever_30=("ever_30_flag", "max"),
            ever_60=("ever_60_flag", "max"),
            ever_90=("ever_90_flag", "max"),
            default_30dpd_6m=("dpd", lambda s: int((s.iloc[:6] >= 30).any())),
            default_90dpd_12m=("dpd", lambda s: int((s.iloc[:12] >= 90).any())),
        ).reset_index()
        loan_with_targets = loans.merge(loan_targets, on="loan_id", how="left")
        approval_rate = (der["decision_final"] == "approved").mean()
        booked_rate = len(loans) / len(apps)
        return pd.DataFrame(
            {
                "metric": [
                    "clients",
                    "applications",
                    "approved_applications",
                    "booked_loans",
                    "payment_schedule_rows",
                    "payments_rows",
                    "loan_monthly_snapshot_rows",
                    "collections_actions_rows",
                    "approval_rate",
                    "booking_rate",
                    "fpd_rate",
                    "default_30dpd_6m_rate",
                    "default_90dpd_12m_rate",
                    "ever_30_rate",
                    "ever_60_rate",
                    "ever_90_rate",
                    "writeoff_rate",
                    "restructuring_rate",
                ],
                "value": [
                    len(self.tables["clients"]),
                    len(apps),
                    int((der["decision_final"] == "approved").sum()),
                    len(loans),
                    len(self.tables["payment_schedule"]),
                    len(self.tables["payments"]),
                    len(self.tables["loan_monthly_snapshot"]),
                    len(self.tables["collections_actions"]),
                    round(float(approval_rate), 6),
                    round(float(booked_rate), 6),
                    round(float(loan_with_targets["fpd_flag"].mean()), 6),
                    round(float(loan_with_targets["default_30dpd_6m"].mean()), 6),
                    round(float(loan_with_targets["default_90dpd_12m"].mean()), 6),
                    round(float(loan_with_targets["ever_30"].mean()), 6),
                    round(float(loan_with_targets["ever_60"].mean()), 6),
                    round(float(loan_with_targets["ever_90"].mean()), 6),
                    round(float(loans["writeoff_flag"].mean()), 6),
                    round(float(loans["restructuring_flag"].mean()), 6),
                ],
            }
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic credit risk relational dataset")
    p.add_argument("--size", choices=list(SIZE_CONFIGS.keys()), default="medium")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir or f"exports_{args.size}"
    gen = CreditRiskDataGenerator(size=args.size, seed=args.seed, output_dir=output_dir)
    gen.run()
    summary = gen.build_summary()
    print(summary.to_string(index=False))
    print(f"\nCSV export completed: {Path(output_dir).resolve()}")


if __name__ == "__main__":
    main()
