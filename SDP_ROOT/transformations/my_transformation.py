from pyspark import pipelines as dp
from pyspark.sql.functions import expr


#Create a new fact streaming table

@dp.table(name="fact_trips")
def fact_trips():
    return spark.readStream.table("samples.nyctaxi.trips")

@dp.expect("pickup_zip not null", "pickup_zip IS NOT NULL")
@dp.expect("dropoff_zip not null", "dropoff_zip IS NULL")
@dp.table(name="fact_trips_silver")
def fact_trips_silver():
    
    df_bronze = spark.readStream.table("fact_trips")\
        .withColumnRenamed("tpep_pickup_datetime", "pickup_datetime")\
        .withColumnRenamed("tpep_dropoff_datetime", "dropoff_datetime")\
        .withColumn("duration", expr("unix_timestamp(dropoff_datetime) - unix_timestamp(pickup_datetime)"))
    
    return df_bronze

@dp.materialized_view(name="mv_trips")
def mv_trips():
    df_mv = spark.read.table("fact_trips_silver")\
        .groupBy("pickup_zip", "dropoff_zip")\
        .agg(expr("avg(duration) as avg_duration"))
    return df_mv

@dp.table(name="quarantining")
def quarantining():
    df_mv = spark.read.table("fact_trips")\
                .filter("dropoff_zip IS NOT NULL")
    return df_mv