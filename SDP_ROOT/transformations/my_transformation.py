import dlt as dp  # Clean up name mapping to match standards
from pyspark.sql.functions import expr

# 1. External Source Layer (Always read explicitly from samples.nyctaxi.trips)
@dp.table(name="fact_trips")
def fact_trips():
    return spark.readStream.table("samples.nyctaxi.trips")

# 2. Target Silver Layer (Reads internally from bronze)
@dp.expect("pickup_zip not null", "pickup_zip IS NOT NULL")
# Fix constraint expression logic to catch empty zips (IS NOT NULL)
@dp.expect("dropoff_zip not null", "dropoff_zip IS NOT NULL") 
@dp.table(name="fact_trips_silver")
def fact_trips_silver():
    # CRITICAL: Use dp.read_stream() so DLT targets your bundle schema automatically
    df_bronze = dp.read_stream("fact_trips")\
        .withColumnRenamed("tpep_pickup_datetime", "pickup_datetime")\
        .withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")\
        .withColumn("duration", expr("unix_timestamp(dropoff_datetime) - unix_timestamp(pickup_datetime)"))
    
    return df_bronze

# 3. Target Materialized View (Reads internally from silver)
@dp.materialized_view(name="mv_trips")
def mv_trips():
    # CRITICAL: Use dp.read() for static datasets inside the same pipeline
    df_mv = dp.read("fact_trips_silver")\
        .groupBy("pickup_zip", "dropoff_zip")\
        .agg(expr("avg(duration) as avg_duration"))
    return df_mv

# 4. Target Quarantine Table
@dp.table(name="quarantining")
def quarantining():
    # CRITICAL: Use dp.read() to query your local table safely
    df_mv = dp.read("fact_trips")\
                .filter("dropoff_zip IS NULL") # Captures failing rows
    return df_mv
