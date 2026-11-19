"""Cleaning + churn feature engineering (pandas path).

The same transformations are expressed as a PySpark job in
`glue_job.py` for AWS Glue; this pandas version is the local/test path
so the pipeline runs anywhere without a Spark cluster.
"""
from __future__ import annotations

import pandas as pd

CONTRACT_MAP = {"month-to-month": 0, "one-year": 1, "two-year": 2}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["contract_type"] = (
        df["contract_type"].astype(str).str.strip().str.lower()
    )
    df["monthly_charges"] = df["monthly_charges"].fillna(df["monthly_charges"].median())
    df = df.drop_duplicates(subset=["customer_id"])
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = clean(df)
    df["contract_code"] = df["contract_type"].map(CONTRACT_MAP).fillna(0)
    df["charges_per_month_tenure"] = df["monthly_charges"] / df["tenure_months"].clip(lower=1)
    df["total_charges_est"] = df["monthly_charges"] * df["tenure_months"]
    df["tickets_per_login"] = df["support_tickets"] / (df["logins_last_30d"] + 1)
    df["usage_per_login"] = df["usage_gb"] / (df["logins_last_30d"] + 1)
    df["is_new_customer"] = (df["tenure_months"] <= 6).astype(int)
    df["has_late_payment"] = (df["late_payments"] > 0).astype(int)
    df["high_value"] = (df["monthly_charges"] >= 100).astype(int)
    return df


FEATURE_COLUMNS = [
    "tenure_months", "monthly_charges", "support_tickets", "logins_last_30d",
    "usage_gb", "late_payments", "num_products", "contract_code",
    "charges_per_month_tenure", "total_charges_est", "tickets_per_login",
    "usage_per_login", "is_new_customer", "has_late_payment", "high_value",
]
TARGET = "churn"
