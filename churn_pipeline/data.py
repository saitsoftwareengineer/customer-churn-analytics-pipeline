"""Synthetic customer data generation (no real customer data in this repo)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_customers(n: int = 3000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "customer_id": [f"C{1000 + i}" for i in range(n)],
        "tenure_months": rng.integers(1, 72, n),
        "monthly_charges": rng.normal(72, 26, n).clip(15, 200).round(2),
        "contract_type": rng.choice(["month-to-month", "one-year", "two-year"], n, p=[0.55, 0.27, 0.18]),
        "payment_method": rng.choice(["card", "bank", "wallet"], n),
        "support_tickets": rng.poisson(2.2, n),
        "logins_last_30d": rng.poisson(16, n),
        "usage_gb": rng.normal(130, 65, n).clip(0, 450).round(1),
        "late_payments": rng.poisson(0.8, n),
        "num_products": rng.integers(1, 5, n),
    })
    # Introduce a few realistic data-quality issues for the cleaning step
    idx = rng.choice(n, size=max(5, n // 200), replace=False)
    df.loc[idx, "monthly_charges"] = np.nan
    df.loc[df.index[:3], "contract_type"] = " Month-to-Month "

    contract_risk = df["contract_type"].str.strip().str.lower().map(
        {"month-to-month": 1.1, "one-year": 0.0, "two-year": -0.9}
    ).fillna(0.9)
    logit = (
        -0.07 * df["tenure_months"] + 0.025 * df["monthly_charges"].fillna(72)
        + 0.6 * df["support_tickets"] - 0.07 * df["logins_last_30d"]
        + 0.8 * df["late_payments"] - 0.4 * df["num_products"]
        + contract_risk - 0.4
    )
    p = 1 / (1 + np.exp(-logit))
    churn = (rng.random(n) < p).astype(int)
    flip = rng.random(n) < 0.04
    df["churn"] = np.where(flip, 1 - churn, churn)
    return df
