"""AWS Glue / PySpark version of the churn feature job.

On Glue: pass --JOB_NAME and the S3 input/output paths as job arguments;
the job reads raw customer CSV from S3, applies the same cleaning and
feature logic as churn_pipeline/features.py (expressed in Spark), and
writes Parquet back to S3 for downstream analytics / model training.

Locally (no Spark cluster), churn_pipeline.pipeline falls back to the
pandas path -- the feature definitions are kept in sync deliberately.
"""
import sys

try:
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
except ImportError:  # pragma: no cover - local fallback
    SparkSession = None


def main():  # pragma: no cover - requires a Spark/Glue runtime
    if SparkSession is None:
        raise SystemExit(
            "PySpark is not installed. Run the pandas path instead: "
            "python run_pipeline.py"
        )
    spark = SparkSession.builder.appName("customer-churn-features").getOrCreate()
    src, dst = sys.argv[1], sys.argv[2]  # s3://bucket/raw/customers.csv, s3://bucket/features/
    df = spark.read.csv(src, header=True, inferSchema=True)
    df = (
        df.withColumn("contract_type", F.lower(F.trim(F.col("contract_type"))))
        .fillna({"monthly_charges": 72.0})
        .withColumn("total_charges_est", F.col("monthly_charges") * F.col("tenure_months"))
        .withColumn("tickets_per_login", F.col("support_tickets") / (F.col("logins_last_30d") + 1))
        .withColumn("is_new_customer", (F.col("tenure_months") <= 6).cast("int"))
    )
    df.write.mode("overwrite").parquet(dst)
    spark.stop()


if __name__ == "__main__":
    main()
