"""End-to-end pipeline: generate -> clean/feature -> train -> write outputs."""
from __future__ import annotations

import json
import os

from churn_pipeline.data import generate_customers
from churn_pipeline.features import FEATURE_COLUMNS
from churn_pipeline.train import train_and_evaluate


def run(n: int = 3000, out_dir: str = "outputs") -> dict:
    df = generate_customers(n=n)
    model, featured, metrics = train_and_evaluate(df)
    os.makedirs(out_dir, exist_ok=True)
    featured["churn_probability"] = model.predict_proba(featured[FEATURE_COLUMNS])[:, 1].round(4)
    featured["churn_prediction"] = (featured["churn_probability"] >= 0.5).astype(int)
    cols = ["customer_id", "churn_probability", "churn_prediction", "churn"]
    featured[cols].to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    print(run())
