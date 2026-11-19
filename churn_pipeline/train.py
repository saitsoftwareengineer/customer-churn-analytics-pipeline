"""Train + evaluate the churn classifier."""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

from churn_pipeline.features import FEATURE_COLUMNS, TARGET, build_features


def train_and_evaluate(df):
    featured = build_features(df)
    X, y = featured[FEATURE_COLUMNS], featured[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    proba = model.predict_proba(Xte)[:, 1]
    metrics = {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "f1": round(float(f1_score(yte, pred)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "test_rows": int(len(yte)),
        "churn_rate_test": round(float(yte.mean()), 4),
    }
    return model, featured, metrics
