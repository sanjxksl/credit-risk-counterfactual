# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # Credit Risk — Spark EDA
# MAGIC
# MAGIC Exploratory analysis on the full 148,670-loan dataset using Spark.
# MAGIC Covers class distribution, feature statistics, and the categorical feature
# MAGIC audit that identified five artifact features removed before modelling.
# MAGIC
# MAGIC **Before running:** upload `cleaned_loan_data.csv` to DBFS:
# MAGIC ```
# MAGIC databricks fs cp data/cleaned_loan_data.csv dbfs:/FileStore/credit-risk/cleaned_loan_data.csv
# MAGIC ```
# MAGIC Or drag-and-drop via Data → Add Data → DBFS in the UI.

# COMMAND ----------

DATA_PATH = "dbfs:/FileStore/credit-risk/cleaned_loan_data.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(DATA_PATH)
)

print(f"Rows: {df.count():,}   Columns: {len(df.columns)}")
df.printSchema()

# COMMAND ----------

# MAGIC %md ## Class Distribution

# COMMAND ----------

from pyspark.sql.functions import col, count, round as spark_round

total = df.count()

df.groupBy("status").agg(
    count("*").alias("n")
).withColumn(
    "pct", spark_round(col("n") / total * 100, 2)
).orderBy("status").show()

# COMMAND ----------

# MAGIC %md ## Numeric Feature Summary

# COMMAND ----------

NUMERIC_COLS = ["loan_amount", "income", "property_value", "credit_score", "ltv", "dtir1", "term"]

df.select(NUMERIC_COLS).describe().show()

# COMMAND ----------

# MAGIC %md ## Default Rate by Numeric Bin
# MAGIC
# MAGIC Buckets credit score and LTV to show how default rate varies across ranges.

# COMMAND ----------

from pyspark.sql.functions import when, mean

df_binned = df.withColumn(
    "credit_score_bin",
    when(col("credit_score") < 580, "<580")
    .when(col("credit_score") < 620, "580-620")
    .when(col("credit_score") < 660, "620-660")
    .when(col("credit_score") < 720, "660-720")
    .otherwise("720+")
).withColumn(
    "ltv_bin",
    when(col("ltv") < 60, "<60")
    .when(col("ltv") < 80, "60-80")
    .when(col("ltv") < 90, "80-90")
    .when(col("ltv") < 100, "90-100")
    .otherwise("100+")
)

print("Default rate by credit score bin:")
df_binned.groupBy("credit_score_bin").agg(
    mean("status").alias("default_rate"),
    count("*").alias("n")
).orderBy("credit_score_bin").show()

print("Default rate by LTV bin:")
df_binned.groupBy("ltv_bin").agg(
    mean("status").alias("default_rate"),
    count("*").alias("n")
).orderBy("ltv_bin").show()

# COMMAND ----------

# MAGIC %md ## Categorical Feature Audit
# MAGIC
# MAGIC Checks every categorical feature for near-deterministic association with the
# MAGIC default label. Any category with 100% default rate and n > 10 is flagged as
# MAGIC a data artifact — it encodes a lookup rule rather than borrower risk.
# MAGIC
# MAGIC Five features were removed from the final model as a result of this audit.

# COMMAND ----------

from pyspark.sql.functions import avg

CATEGORICAL_COLS = [
    "loan_limit", "gender", "approv_in_adv", "loan_type", "loan_purpose",
    "credit_worthiness", "open_credit", "business_or_commercial",
    "neg_ammortization", "interest_only", "lump_sum_payment", "occupancy_type",
    "total_units", "age", "submission_of_application", "region",
    # artifact candidates — included here so the audit catches them
    "credit_type", "co-applicant_credit_type",
    "security_type", "construction_type", "secured_by",
]

existing_cols = [c for c in CATEGORICAL_COLS if c in df.columns]

for col_name in existing_cols:
    stats = (
        df.groupBy(col_name)
        .agg(
            avg("status").alias("default_rate"),
            count("*").alias("n")
        )
        .orderBy(col("default_rate").desc())
        .collect()
    )
    print(f"\n{col_name}:")
    for row in stats:
        flag = "  <-- ARTIFACT (100% default)" if row["default_rate"] == 1.0 and row["n"] > 10 else ""
        print(f"  {row[col_name]}: {row['default_rate']:.0%} (n={row['n']:,}){flag}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Audit Results
# MAGIC
# MAGIC | Feature | Artifact Category | n | Reason |
# MAGIC |---------|------------------|---|--------|
# MAGIC | `credit_type` | EQUI | 15,298 | Bureau lookup — encodes loan program, not risk |
# MAGIC | `co-applicant_credit_type` | — | — | Administrative metadata |
# MAGIC | `security_type` | Indriect (typo) | 33 | Same 33-loan cluster |
# MAGIC | `construction_type` | mh | 33 | Same 33-loan cluster |
# MAGIC | `secured_by` | land | 33 | Same 33-loan cluster |
# MAGIC
# MAGIC These five features are dropped in `feature_engineering.ipynb` before any modelling.

# COMMAND ----------

# MAGIC %md ## Missing Values

# COMMAND ----------

from pyspark.sql.functions import isnan, isnull, sum as spark_sum

missing = df.select([
    spark_sum(isnull(c).cast("int")).alias(c)
    for c in df.columns
]).collect()[0].asDict()

print("Columns with missing values:")
for col_name, n_missing in sorted(missing.items(), key=lambda x: -x[1]):
    if n_missing > 0:
        pct = n_missing / total * 100
        print(f"  {col_name}: {n_missing:,} ({pct:.1f}%)")

# COMMAND ----------

# MAGIC %md ## Class Imbalance and pos_weight
# MAGIC
# MAGIC The training split has a 24.6% default rate. Rather than resampling with SMOTE
# MAGIC (which changes the data distribution and requires corrective Platt scaling),
# MAGIC the imbalance is handled via `pos_weight` in `BCEWithLogitsLoss`.
# MAGIC
# MAGIC `pos_weight = n_negative / n_positive` — the gradient for each positive example
# MAGIC is scaled up by this factor, so the model sees them as 3× more important.

# COMMAND ----------

from pyspark.sql.functions import lit

train_path = "dbfs:/FileStore/credit-risk/train.csv"
train_df = spark.read.option("header", True).option("inferSchema", True).csv(train_path)

n_pos = train_df.filter(col("status") == 1).count()
n_neg = train_df.filter(col("status") == 0).count()
pos_weight = n_neg / n_pos

print(f"Training set: {train_df.count():,} rows")
print(f"  Positives (default):     {n_pos:,} ({n_pos / train_df.count():.1%})")
print(f"  Negatives (no default):  {n_neg:,} ({n_neg / train_df.count():.1%})")
print(f"  pos_weight:              {pos_weight:.4f}")
