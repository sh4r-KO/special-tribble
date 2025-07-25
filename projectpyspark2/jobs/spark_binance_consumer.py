from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json, split
from pyspark.sql.types import StructType, DoubleType, LongType

spark = (
    SparkSession.builder
    .appName("binance-hourly-bars-stream")
    .config(
        "spark.jars.packages",
        "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
        "io.delta:delta-spark_2.12:3.2.0"
    )
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
    # --- MinIO (S3A) credentials ---
    .config("spark.hadoop.fs.s3a.endpoint", "http://minio:9000")
    .config("spark.hadoop.fs.s3a.access.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.secret.key", "minioadmin")
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .getOrCreate()
)

schema = (
    StructType()
    .add("t", LongType())      # openTime ms
    .add("o", DoubleType())
    .add("h", DoubleType())
    .add("l", DoubleType())
    .add("c", DoubleType())
    .add("v", DoubleType())
)

raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", "kafka:9092")
    .option("subscribe", "binance.bars")
    .option("startingOffsets", "latest")
    .load()
)

bars = (
    raw.selectExpr("CAST(key AS STRING) AS k", "CAST(value AS STRING) AS j")
        .select(
            split(col("k"), "-")[0].alias("symbol"),
            split(col("k"), "-")[1].cast(LongType()).alias("t"),
            from_json("j", schema).alias("d")
        )
        .select("symbol", col("d.*"))
        .withWatermark("t", "3 hours")        # lateness window
        .dropDuplicates(["symbol", "t"])
)

(
    bars.writeStream
        .format("delta")
        .outputMode("append")
        .option("checkpointLocation", "/tmp/ckpt/binance_bars")
        .option("path", "s3a://datalake/binance_bars")
        .trigger(processingTime="1 minute")
        .start()
        .awaitTermination()
)
