from pyspark.sql import SparkSession

def create_spark():
    return (
        SparkSession.builder
        .appName("OpenFinancePrediction")
        .getOrCreate()
    )

spark = create_spark()
