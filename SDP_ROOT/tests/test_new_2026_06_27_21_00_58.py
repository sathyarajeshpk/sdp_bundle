"""
Unit tests for first_sdp_pipeline – fact_trips dataset.

Run with: pytest tests/test_new_2026_06_27_21_00_58.py -v

NOTE: These tests exercise the transformation logic directly using
      synthetic data, without needing to connect to the pipeline runtime
      or the samples.nyctaxi.trips source.
"""
import pytest
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructType, StructField,
    TimestampType, DoubleType, IntegerType,
)
from pyspark.testing import assertDataFrameEqual
import pyspark.sql.functions as F


# ── Transformation logic (mirrors transformations/my_transformation.py) ────

def transform_fact_trips(df):
    """
    Mirrors the fact_trips dataset definition.
    Currently a passthrough – ingests samples.nyctaxi.trips as-is.
    Update this function whenever new transformations are added to the pipeline.
    """
    return df


# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def spark():
    return (
        SparkSession.builder
        .master("local[*]")
        .appName("test_first_sdp_pipeline")
        .config("spark.sql.shuffle.partitions", "1")
        .getOrCreate()
    )


@pytest.fixture(scope="session")
def trips_schema():
    """Schema matching samples.nyctaxi.trips exactly."""
    return StructType([
        StructField("tpep_pickup_datetime",  TimestampType(), True),
        StructField("tpep_dropoff_datetime", TimestampType(), True),
        StructField("trip_distance",         DoubleType(),    True),
        StructField("fare_amount",           DoubleType(),    True),
        StructField("pickup_zip",            IntegerType(),   True),
        StructField("dropoff_zip",           IntegerType(),   True),
    ])


@pytest.fixture
def sample_trips(spark, trips_schema):
    data = [
        (datetime(2024, 1, 1,  8,  0), datetime(2024, 1, 1,  8, 15), 2.5, 10.0, 10001, 10110),
        (datetime(2024, 1, 1,  9,  0), datetime(2024, 1, 1,  9, 30), 5.0, 18.5, 10023, 10023),
        (datetime(2024, 1, 1, 10,  0), datetime(2024, 1, 1, 10,  5), 0.8,  5.0, 10018, 10001),
    ]
    return spark.createDataFrame(data, schema=trips_schema)


# ── Schema tests ─────────────────────────────────────────────────────────────

class TestFactTripsSchema:

    def test_expected_columns_present(self, sample_trips):
        result = transform_fact_trips(sample_trips)
        expected = {
            "tpep_pickup_datetime", "tpep_dropoff_datetime",
            "trip_distance", "fare_amount",
            "pickup_zip", "dropoff_zip",
        }
        missing = expected - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_row_count_preserved(self, sample_trips):
        assert transform_fact_trips(sample_trips).count() == 3

    def test_column_types(self, sample_trips):
        result = transform_fact_trips(sample_trips)
        type_map = {f.name: f.dataType for f in result.schema.fields}
        assert str(type_map["trip_distance"]) == "DoubleType()"
        assert str(type_map["fare_amount"])   == "DoubleType()"
        assert str(type_map["pickup_zip"])    == "IntegerType()"


# ── Data-quality tests ───────────────────────────────────────────────────────

class TestFactTripsDataQuality:

    def test_no_negative_trip_distance(self, sample_trips):
        result = transform_fact_trips(sample_trips)
        assert result.filter(F.col("trip_distance") < 0).count() == 0

    def test_no_negative_fare_amount(self, sample_trips):
        result = transform_fact_trips(sample_trips)
        assert result.filter(F.col("fare_amount") < 0).count() == 0

    def test_pickup_before_dropoff(self, sample_trips):
        result = transform_fact_trips(sample_trips)
        invalid = result.filter(
            F.col("tpep_pickup_datetime") >= F.col("tpep_dropoff_datetime")
        ).count()
        assert invalid == 0, f"{invalid} rows where pickup >= dropoff"

    def test_invalid_rows_surface_for_expectations(self, spark, trips_schema):
        """
        Rows violating data-quality rules must reach the output so that
        pipeline expectations (expect_or_drop / expect_or_fail) can act on them.
        """
        bad_data = [
            (datetime(2024, 1, 1, 8, 0), datetime(2024, 1, 1, 8, 15), -1.0, 10.0, 10001, 10110),
        ]
        bad_df = spark.createDataFrame(bad_data, schema=trips_schema)
        result = transform_fact_trips(bad_df)
        # Negative distance must not be silently swallowed by the transform
        assert result.filter(F.col("trip_distance") < 0).count() == 1


# ── Transformation correctness tests ─────────────────────────────────────────

class TestFactTripsTransformation:

    def test_output_equals_input(self, sample_trips):
        """Passthrough: output must be identical to input."""
        assertDataFrameEqual(transform_fact_trips(sample_trips), sample_trips)

    def test_idempotent(self, sample_trips):
        """Applying the transform twice must yield the same result."""
        once  = transform_fact_trips(sample_trips)
        twice = transform_fact_trips(once)
        assertDataFrameEqual(once, twice)
