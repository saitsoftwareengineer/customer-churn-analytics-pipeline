# Customer Churn Analytics Pipeline

End-to-end churn analytics: synthetic customer data generation, data
cleaning, churn feature engineering, a scikit-learn classifier with
reported accuracy / F1 / ROC-AUC, and prediction outputs written for
downstream analytics.

This is the project behind the resume line: *"Produced churn predictions
on a held-out test set by cleaning customer activity data, engineering
churn features, training a classifier, and preparing model outputs for
downstream analytics."*

## Tech stack
Python · Pandas (local path) · PySpark / AWS Glue (production path, see
`glue_job.py`) · Amazon S3 · Scikit-learn

> **PySpark note:** the feature logic is written twice on purpose -- a
> PySpark version for Glue (`glue_job.py`, Glue supplies the Spark
> runtime and reads/writes S3) and an identical pandas version
> (`churn_pipeline/`) so the pipeline and its tests run anywhere,
> with no Spark cluster required.

## How to run
```bash
pip install -r requirements.txt
python run_pipeline.py     # writes outputs/predictions.csv + outputs/metrics.json
pytest                     # run the tests
```

## Project structure
```
churn_pipeline/data.py      synthetic customer data (with realistic quality issues)
churn_pipeline/features.py  cleaning + churn feature engineering
churn_pipeline/train.py     classifier training + evaluation metrics
churn_pipeline/pipeline.py  end-to-end run + output writing
glue_job.py                 AWS Glue / PySpark version of the feature job
run_pipeline.py             CLI entry point
tests/                      data, feature, training and pipeline tests
```

## Features engineered
Tenure, monthly charges, support tickets, logins, usage, late payments,
products, contract type, charges-per-tenure, estimated total charges,
tickets-per-login, usage-per-login, new-customer and high-value flags.
