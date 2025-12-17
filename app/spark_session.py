from typing import Optional


def create_spark():
    # local import to avoid importing pyspark at module import time
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder
        .appName("OpenFinancePrediction")
        .getOrCreate()
    )


_spark_instance: Optional[object] = None


def get_spark():
    """Retorna uma instância de SparkSession (lazy)."""
    global _spark_instance
    if _spark_instance is None:
        _spark_instance = create_spark()
    return _spark_instance
