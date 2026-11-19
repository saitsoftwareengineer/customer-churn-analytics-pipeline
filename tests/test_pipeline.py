from churn_pipeline.data import generate_customers
from churn_pipeline.features import FEATURE_COLUMNS, build_features
from churn_pipeline.pipeline import run
from churn_pipeline.train import train_and_evaluate


def test_synthetic_data_shape_and_churn_rate():
    df = generate_customers(n=500)
    assert len(df) == 500
    assert 0.05 < df["churn"].mean() < 0.6


def test_features_created_and_cleaned():
    featured = build_features(generate_customers(n=300))
    assert featured["monthly_charges"].isna().sum() == 0
    for col in FEATURE_COLUMNS:
        assert col in featured.columns
    # contract values normalised (no stray whitespace / casing)
    assert featured["contract_type"].isin(
        ["month-to-month", "one-year", "two-year"]
    ).all()


def test_training_accuracy_is_reasonable():
    _, _, metrics = train_and_evaluate(generate_customers(n=1500))
    assert metrics["accuracy"] >= 0.70


def test_pipeline_writes_outputs(tmp_path):
    metrics = run(n=800, out_dir=str(tmp_path))
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert metrics["accuracy"] >= 0.65
