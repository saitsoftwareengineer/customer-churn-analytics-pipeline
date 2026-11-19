from churn_pipeline.pipeline import run

if __name__ == "__main__":
    metrics = run()
    print("Pipeline finished. Metrics:", metrics)
    print("Outputs: outputs/predictions.csv, outputs/metrics.json")
