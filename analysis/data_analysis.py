"""
Author: Lin Sze Khoo
Created on: 24/01/2024
Last modified on: 25/06/2024
"""
import collections
import json
import math
import os
import haversine as hs
from datetime import datetime, timezone, timedelta
from itertools import product

import findspark
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import mysql.connector
import numpy as np
import pandas as pd
import pytz
from scipy.stats import pearsonr
from kneed import KneeLocator
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType, StructField, 
                               StructType, TimestampType, BooleanType, ArrayType, DoubleType)
from pyspark.sql.window import Window
from scipy.interpolate import interp1d
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


def get_user_table(cursor, table_name, device_id):
    """
    Retrieves table entries belonging to a specific device (participant).
    @param cursor: Database cursor for executing SQL queries
    @param device_id: Unique device ID assigned to each device (participant)
    @param table_name: Name of database table to be retrieved
    """
    cursor.execute(f"SELECT * FROM {table_name} WHERE device_id='{device_id}'")
    result = cursor.fetchall()
    headers = [i[0] for i in cursor.description]
    return headers, result

def get_user_device(cursor, user_id):
    cursor.execute(f"SELECT device_id FROM aware_device WHERE label='{user_id}'")
    result = cursor.fetchall()
    return result

def delete_single_entry(cursor, table_name, device_id):
    cursor.execute(f"DELETE FROM {table_name} WHERE device_id='{device_id}'")

def delete_invalid_device_entries(cursor, table_name, valid_device_ids):
    cursor.execute(f"DELETE FROM {table_name} WHERE device_id NOT IN {valid_device_ids}")

def get_valid_device_id(cursor):
    cursor.execute("SELECT DISTINCT device_id FROM aware_device")
    result = [i[0] for i in cursor.fetchall()]
    return result

def export_user_data(cursor, user_id):
    """
    Reads data belonging to input user_id from all tables (sensor data + ESM) and exports to CSV files.
    """
    cursor.execute(f"SELECT device_id FROM aware_device WHERE label='{user_id}'")
    device_id = cursor.fetchone()[0]
    for table in ALL_TABLES:
        table_headers, user_entries = get_user_table(cursor, table, device_id)
        if len(user_entries) > 0:
            table_df = pd.DataFrame.from_records(user_entries, columns=table_headers)
            table_df["datetime"] = [datetime.fromtimestamp(d/1000, TIMEZONE) for d in table_df["timestamp"]]
            table_df["date"] = [d.date().strftime("%Y-%m-%d") for d in table_df["datetime"]]
            table_df["hour"] = [d.time().hour for d in table_df["datetime"]]
            table_df["minute"] = [d.time().minute for d in table_df["datetime"]]
            # Day 1 to 7, 1 is equivalent to Monday
            table_df["day_of_the_week"] = [d.weekday() + 1 for d in table_df["datetime"]]
            for col in table_df.columns:
                if col == "activities" or col == "esm_json":
                    table_df[col] = [d.replace('"', "'") for d in table_df[col]]
            table_df = table_df.drop("datetime", axis=1)
            table_df.to_csv(f"{DATA_FOLDER}/{user_id}_{table}.csv", index=False)


def outliers_from_rolling_average(df, value_col):
    # Moving average to smooth the data and identify outliers
    smooth_window = (Window()\
        .partitionBy(F.col("date"))\
        .orderBy(F.col("hour").asc(), F.col("minute").asc())\
        .rangeBetween(-5, 0))
    df = df.withColumn("rolling_average", F.mean(value_col).over(smooth_window))
    df = df.withColumn("residuals", F.col(value_col) - F.col("rolling_average"))

    outlier_threshold = 2 * df.select(F.stddev("residuals")).first()[0]
    print(f"Outlier threshold: {outlier_threshold}")
    
    df = df.withColumn("potential_outlier", F.when((F.abs("residuals") > outlier_threshold), True).otherwise(False))
    outliers = df.filter(F.col('potential_outlier') == True)
    print(f"Number of rows flagged as outliers: {outliers.count()}")
    outliers.show()

def round_time_to_nearest_n_minutes(dt, n):
    """
    Rounds input timestamp to the nearest n minutes for synchronization.
    """
    seconds = (dt - dt.min).seconds
    rounding = (seconds + n * 30) // (n * 60) * (n * 60)
    return dt + timedelta(0, rounding - seconds, -dt.microsecond)


def fill_df_hour_minute(cur_df, minute_interval=1):
    """
    Fills dataframe with missing entries of hour and minute of available dates.
    """
    time_cols = ["date", "hour", "minute"]
    unique_dates = np.array(cur_df.select("date").distinct().collect()).flatten().tolist()
    minute_range = list(range(0, 60, minute_interval))
    hour_range = list(range(0, 24))
    date_hour_minute = list(product(unique_dates, hour_range, minute_range))
    time_df = spark.createDataFrame(date_hour_minute, schema=StructType(
        [StructField("date", StringType(), False),\
         StructField("hour", IntegerType(), False),
         StructField("minute", IntegerType(), False)]))\
         .sort(time_cols)
    
    # Round existing date time to nearest minute if the specified minute interval is greater than 1
    if minute_interval > 1:
        cur_df = cur_df.withColumn("date_time", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("rounded_date_time", udf_round_datetime_to_nearest_minute(F.col("date_time"), F.lit(minute_interval)))\
            .withColumn("date", udf_get_date_from_datetime("rounded_date_time"))\
            .withColumn("hour", udf_get_hour_from_datetime("rounded_date_time"))\
            .withColumn("minute", udf_get_minute_from_datetime("rounded_date_time"))\
            .drop("date_time", "rounded_date_time")
        cur_df = cur_df.groupBy(*time_cols).agg(F.mean("average_decibels_minute").alias("average_decibels_minute"))
    
    # First row of dataframe
    first_entry = cur_df.first()
    time_df = time_df.filter((F.col("date") > first_entry["date"]) |\
                             ((F.col("date") == first_entry["date"]) &\
                              (F.col("hour") > first_entry["hour"])) |
                              ((F.col("date") == first_entry["date"]) &\
                              (F.col("hour") == first_entry["hour"]) &\
                                (F.col("minute") >= first_entry["minute"])))
    
    # Last row of dataframe
    last_entry = cur_df.sort(time_cols, ascending=False).first()
    time_df = time_df.filter((F.col("date") < last_entry["date"]) |\
                             ((F.col("date") == last_entry["date"]) &\
                              (F.col("hour") < last_entry["hour"])) |\
                               ((F.col("date") == last_entry["date"]) &\
                              (F.col("hour") == last_entry["hour"]) &\
                                (F.col("minute") <= last_entry["minute"])))\
                                .sort(time_cols)

    cur_df = cur_df.join(time_df, time_cols, "right")\
        .sort(time_cols)
    return cur_df

def nearest_neighbour_interpolation(feature_list):
    """
    Nearest neighbour interpolation of missing data based on https://dl.acm.org/doi/pdf/10.1145/3510029
    """
    time = list(range(len(feature_list)))
    initial_time = [index for index in range(
        len(feature_list)) if not math.isnan(feature_list[index])]
    initial_feature_list = [
        feat for feat in feature_list if not math.isnan(feat)]
    interp_func = interp1d(
        initial_time, initial_feature_list, kind="nearest", fill_value="extrapolate")
    feature_list = interp_func(time)
    return feature_list

def process_light_data(user_id, data_minute_interval=1):
    """
    Ambient luminance data

    NOTE Assumptions:
    1. Latest sensing configuration sets to collect data every 10 mins but may have more frequent entries when an absolute change of 10% is detected.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_light.parquet"
    if not os.path.exists(parquet_filename):
        light_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_light.csv")
        time_cols = ["date", "hour", "minute"]

        # Moving average to smooth the data and identify outliers
        light_df = light_df.withColumn("double_light_lux", F.col("double_light_lux").cast(FloatType()))\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("second_timestamp", F.round(F.col("timestamp")/1000).cast(TimestampType()))\
            .withColumn("second_timestamp", udf_round_datetime_to_nearest_minute(F.col("second_timestamp"), F.lit(data_minute_interval)))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        time_window = Window.partitionBy("date").orderBy("timestamp")
        light_df = light_df.withColumn("prev_lux", F.lag(F.col("double_light_lux")).over(time_window))\
            .withColumn("percentage_change", F.abs(F.col("double_light_lux") - F.col("prev_lux")) / F.col("prev_lux"))\
            .filter(F.col("percentage_change") > 0.1)
        
        # Obtain statistical descriptors within each 1-minute time window (30 seconds buffer for the 30-second sampling duration)      
        stat_functions = [F.min, F.max, F.mean, F.stddev]
        stat_names = ["min", "max", "mean", "std"]
        agg_expressions = [stat_functions[index]("double_light_lux").alias(f"{stat_names[index]}_light_lux") for index in range(len(stat_functions))]

        light_df = light_df.withWatermark("second_timestamp", "1 minute")\
            .groupBy(F.window("second_timestamp", "1 minute"))\
            .agg(*agg_expressions)\
            .withColumn("start_timestamp", F.col("window.start"))\
            .withColumn("end_timestamp", F.col("window.end"))\
            .withColumn("date", udf_get_date_from_datetime("start_timestamp"))\
            .withColumn("hour", udf_get_hour_from_datetime("start_timestamp"))\
            .withColumn("minute", udf_get_minute_from_datetime("start_timestamp"))\
            .drop("window", "start_timestamp", "end_timestamp").sort(*time_cols)

        # minutes = lambda i: i * 60000 # Timestamp is in milliseconds
        # smooth_window = (Window()\
        #     .partitionBy(F.col("date"))\
        #     .orderBy(F.col("timestamp"))\
        #     .rangeBetween(-minutes(1), 0))
        
    #     # NOTE: Ignores extremely high lumination occuring at 01/02/2024 23:06 and outliers based on assumed threshold
    #     # light_df = light_df.filter(F.col("double_light_lux") <= 2000)
    #     light_df = light_df.withColumn("rolling_light_lux", F.mean("double_light_lux").over(smooth_window))
    #     light_df = light_df.withColumn("residuals", F.col("double_light_lux") - F.col("rolling_light_lux"))

    #     # NOTE: Exclude outliers based on standard deviation of difference between initial value and rolling average
    #     outlier_threshold = 2 * light_df.select(F.stddev("residuals")).first()[0]
    #     light_df = light_df.withColumn("potential_outlier", F.when((F.abs("residuals") > outlier_threshold), True).otherwise(False))
    #     light_df = light_df.filter(F.col("potential_outlier") == False)
        
    #     # Fill in missing time points
    #     light_df = fill_df_hour_minute(light_df, data_minute_interval)
    #     light_df = light_df.toPandas()
    #     feature_ts = light_df["average_lux_minute"]

    #     # Perform interpolation to fill in nan values
    #     feature_ts = [float(t) for t in feature_ts]
    #     if np.isnan(np.min(feature_ts)):
    #         feature_ts = nearest_neighbour_interpolation(feature_ts)
    #     light_df["average_lux_minute"] = feature_ts
    #     light_df = spark.createDataFrame(light_df.values.tolist(), schema=StructType([StructField("date", StringType()),\
    #                                                                        StructField("hour", IntegerType()),\
    #                                                                        StructField("minute", IntegerType()),\
    #                                                                        StructField("average_lux_minute", FloatType())]))
        light_df.write.parquet(parquet_filename)
    
    light_df = spark.read.parquet(parquet_filename)
    return light_df


def process_raw_noise_data(user_id):
    """
    Ambient audio data without aggregation.
    Used to get estimation of quartiles from the entire distribution for thresholding.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_raw_noise.parquet"
    if not os.path.exists(parquet_filename):
        noise_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_ambient_noise.csv")
        time_cols = ["date", "hour", "minute"]
        noise_df = noise_df.withColumn("double_frequency", F.col("double_frequency").cast(FloatType()))\
            .withColumn("double_decibels", F.col("double_decibels").cast(FloatType()))\
            .withColumn("double_rms", F.col("double_rms").cast(FloatType()))\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        noise_df.write.parquet(parquet_filename)
    
    noise_df = spark.read.parquet(parquet_filename)
    return noise_df


def process_noise_data_with_conv_estimate(user_id):
    """
    Ambient audio data with rough estimation of nearby conversation based on frequency, decibels, and rms thresholds.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_noise.parquet"
    if not os.path.exists(parquet_filename):
        raw_noise_df = process_raw_noise_data(user_id)\
            .withColumn("double_frequency", F.col("double_frequency").cast(FloatType()))\
            .withColumn("double_decibels", F.col("double_decibels").cast(FloatType()))\
            .withColumn("double_rms", F.col("double_rms").cast(FloatType()))\
            .withColumn("second_timestamp", F.round(F.col("timestamp")/1000).cast(TimestampType()))\
            .withColumn("second_timestamp", udf_round_datetime_to_nearest_minute(F.col("second_timestamp"), F.lit(1)))
        
        with open(f"{user_id}_contexts.json", "r") as f:
            contexts = json.load(f)
        raw_noise_df = raw_noise_df.withColumn("conversation_estimate", F.when((F.col("double_decibels") > F.lit(contexts["silent_threshold"])) &\
                                                                        (F.col("double_frequency") > F.lit(contexts["frequency_threshold"])) &\
                                                                        (F.col("double_rms") > F.lit(contexts["rms_threshold"])), True).otherwise(False))

        # Obtain statistical descriptors within each 1-minute time window
        time_cols = ["date", "hour", "minute"]   
        stat_functions = [F.min, F.max, F.mean, F.stddev]
        stat_names = ["min", "max", "mean", "std"]
        agg_expressions = []
        agg_cols = [col for col in raw_noise_df.columns if col.startswith("double_") and col != "double_silence_threshold"]
        for col in agg_cols:
            for index, func in enumerate(stat_functions):
                agg_expressions.append(func(col).alias(f"{stat_names[index]}_{col[7:]}"))
        noise_df = raw_noise_df.withWatermark("second_timestamp", "1 minute")\
            .groupBy(F.window("second_timestamp", "1 minute"))\
            .agg(*agg_expressions)\
            .withColumn("start_timestamp", F.col("window.start"))\
            .withColumn("end_timestamp", F.col("window.end"))\
            .withColumn("date", udf_get_date_from_datetime("start_timestamp"))\
            .withColumn("hour", udf_get_hour_from_datetime("start_timestamp"))\
            .withColumn("minute", udf_get_minute_from_datetime("start_timestamp"))\
            .drop("window", "start_timestamp", "end_timestamp").sort(*time_cols)

        conversation_noise_df = raw_noise_df.withWatermark("second_timestamp", "1 minute")\
            .groupBy(F.window("second_timestamp", "1 minute"), F.col("conversation_estimate"))\
            .agg(F.count("timestamp").alias("estimate_count"))\
            .withColumn("row_number", F.row_number().over(Window.partitionBy("window").orderBy(F.desc("estimate_count"))))\
            .filter(F.col("row_number") == 1)\
            .withColumn("start_timestamp", F.col("window.start"))\
            .withColumn("date", udf_get_date_from_datetime("start_timestamp"))\
            .withColumn("hour", udf_get_hour_from_datetime("start_timestamp"))\
            .withColumn("minute", udf_get_minute_from_datetime("start_timestamp"))\
            .drop("window", "start_timestamp").sort(*time_cols)

        noise_df = noise_df.join(conversation_noise_df.select(*time_cols + ["conversation_estimate"]), time_cols)\
            .dropDuplicates().sort(*time_cols)

        noise_df.write.parquet(parquet_filename)

    noise_df = spark.read.parquet(parquet_filename)
    return noise_df


def extract_max_confidence_activity(activity_confidence):
    """
    Extracts one or more activities with the highest confidence
    
    NOTE: Running and walking are subtypes of on_foot, so retain the prior to be more precise if both co-exist
    https://developers.google.com/android/reference/com/google/android/gms/location/DetectedActivity
    """
    max_confidence = max(activity_confidence, key=lambda x: x["confidence"])["confidence"]  
    max_activities = [conf for conf in activity_confidence if conf["confidence"] == max_confidence]
    activity_list = [conf["activity"] for conf in max_activities]

    # Check if on_foot coexists with more precise walking or running
    if len(activity_list) > 1 and "on_foot" in activity_list:
        remove_index = activity_list.index("on_foot")
        max_activities.pop(remove_index)

    return max_activities

def process_activity_data(user_id):
    """
    (Event-based)
    Activity recognition data

    NOTE Assumptions:
    1. Remove activity inference with confidence lower than 50 (majorly "unknown" activity)
    2. Retain all activity inferences with maximum confidence (may have one or more for each entry)
    3. Retain only "walking" or "running" activity type if they co-exist with "on_foot" since they are more precise subtypes.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_activity.parquet"
    if not os.path.exists(parquet_filename):
        activity_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_google_activity_recognition.csv")
        time_cols = ["date", "hour", "minute"]

        activity_df = activity_df.withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .withColumn("day_of_the_week", F.col("day_of_the_week").cast(IntegerType()))\
            .withColumn("activities", F.from_json(F.col("activities"), activity_confidence_schema))

        # Find activities with max confidence of above 50 and explode into one or more rows
        activity_df = activity_df.withColumn("max_activities", udf_extract_max_confidence_activity("activities"))\
            .withColumn("activities", F.explode(F.col("max_activities")))\
            .withColumn("activity_name", F.col("activities.activity"))\
            .withColumn("confidence", F.col("activities.confidence"))\
            .withColumn("activity_type", udf_map_activity_name_to_type("activity_name"))\
            .filter(F.col("confidence") > 50).sort(*time_cols)
        activity_df = activity_df.select(*time_cols + ["timestamp", "activity_name", "activity_type", "confidence"]).sort("timestamp")
        activity_df.write.parquet(parquet_filename)
    
    activity_df = spark.read.parquet(parquet_filename)

    return activity_df

def process_screen_data(user_id):
    """
    (Event-based)
    Screen status data: currently does not attempt to identify missing data
    NOTE: Device usage plugin readily provides usage (screen unlocked) and non-usage duration (screen off)
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_screen.parquet"
    if not os.path.exists(parquet_filename):
        screen_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_screen.csv")
        time_cols = ["date", "hour", "minute"]
        time_window = Window.partitionBy("date").orderBy("timestamp")
        screen_df = screen_df.withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .withColumn("day_of_the_week", F.col("day_of_the_week").cast(IntegerType()))\
            .sort(time_cols)
        screen_df = screen_df.withColumn("prev_timestamp", F.lag(F.col("timestamp")).over(time_window))\
            .withColumn("duration (ms)", (F.col("timestamp") - F.col("prev_timestamp")).cast(IntegerType()))\
            .withColumn("prev_status", F.lag(F.col("screen_status")).over(time_window))\
            .sort("timestamp")
        screen_df.write.parquet(parquet_filename)
    
    screen_df = spark.read.parquet(parquet_filename)
    return screen_df

def process_application_usage_data(user_id):
    """
    (Event-based)
    Application usage: does not attempt to identify missing data
    1. Combine information from foreground applications and device usage plugin.
    2. Include: Non-system applications running in the foreground when device screen is on.

    NOTE Assumptions:
    1. Assume that the usage duration of an application is the duration between the current usage timestamp and
    that of the subsequent foreground application or the end of device usage duration.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_app_usage.parquet"
    if not os.path.exists(parquet_filename):
        # V1
        phone_use_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")
        phone_in_use_df = phone_use_df.filter(F.col("double_elapsed_device_on") > 0)\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("start_timestamp", F.col("timestamp") - F.col("double_elapsed_device_on"))\
            .withColumnRenamed("timestamp", "end_timestamp")

        # V2: Manual computation through screen states that consider duration between state 3 and state 2
        # NOTE: This method does not consider that certain apps might still be used in locked state (value 2) 

        # # Compute start and end usage event where state 3 indicates the start timepoint and the first subsequent state 2 indicates the end timepoint.
        # prev_consecutive_time_window = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, -1)
        # phone_usage_df = screen_df.withColumn("start_usage", F.when(F.col("screen_status") == 3, F.col("timestamp")))\
        #     .withColumn("end_usage", F.when(F.col("screen_status") == 2, F.col("timestamp")))\
        #     .withColumn("last_start_usage", F.last("start_usage", True).over(prev_consecutive_time_window))\
        #     .withColumn("last_end_usage", F.last("end_usage", True).over(prev_consecutive_time_window))\
        #     .withColumn("start_usage", F.when(F.col("start_usage").isNotNull(), F.col("start_usage"))\
        #         .otherwise(F.when((F.col("end_usage").isNotNull()) | (F.col("last_start_usage") > F.col("last_end_usage")), F.col("last_start_usage"))))\
        #     .filter(F.col("last_end_usage") <= F.col("start_usage")).sort("timestamp")
        
        # # Aggregate consecutive start events since the last end event without any new end event in between
        # phone_in_use_df = phone_usage_df\
        #     .groupBy("last_end_usage").agg(F.min("start_usage").alias("start_timestamp"),\
        #                 F.min("end_usage").alias("end_timestamp"))\
        #     .sort("start_timestamp")
        # -- End of v2 --

        # Might interleave with system process for rendering UI
        substring_regex_pattern = "|".join(["systemui", "launcher", "biometrics"])
        app_usage_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_applications_foreground.csv")\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .filter((F.col("is_system_app") == 0) | ~(F.expr(f"package_name rlike '{substring_regex_pattern}'")) |\
                    ~(F.lower(F.col("application_name")).contains("launcher")))

        # Obtain intersect of phone screen in use and having applications running in foreground
        time_window = Window.orderBy("timestamp")
        in_use_app_df = app_usage_df.join(phone_in_use_df, (phone_in_use_df["start_timestamp"] <= app_usage_df["timestamp"]) &\
                                        (phone_in_use_df["end_timestamp"] >= app_usage_df["timestamp"]))\
            .select("application_name", "timestamp", "start_timestamp", "end_timestamp", "is_system_app", "double_elapsed_device_on")\
            .dropDuplicates()
        in_use_app_df = in_use_app_df.withColumn("next_timestamp", F.lead(F.col("timestamp")).over(time_window))\
            .withColumn("next_timestamp", F.when((F.col("next_timestamp") <= F.col("end_timestamp")), F.col("next_timestamp")).otherwise(F.col("end_timestamp")))\
            .withColumn("usage_duration", (F.col("next_timestamp") - F.col("timestamp"))/1000)\
            .filter(F.col("usage_duration") > 0).drop("start_timestamp", "end_timestamp").dropDuplicates()\
            .withColumnRenamed("timestamp", "start_timestamp")\
            .withColumnRenamed("next_timestamp", "end_timestamp").sort("start_timestamp")
    
    #     in_use_app_df.write.parquet(parquet_filename)
    
    # in_use_app_df = spark.read.parquet(parquet_filename)
    return in_use_app_df

def process_bluetooth_data(user_id):
    """
    (Event-based)
    Bluetooth data: does not attempt to identify missing data
    1. RSSI values indicate signal strength: -100 dBm (weaker) to 0 dBm (strongest)
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_bluetooth.parquet"
    if not os.path.exists(parquet_filename):
        bluetooth_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_bluetooth.csv")
        bluetooth_df.write.parquet(parquet_filename)
    
    bluetooth_df = spark.read.parquet(parquet_filename)

    return bluetooth_df

def optimize_cluster(loc_df):
    """
    Optimizes epsilon and mininum number of points based on silhouette score of clusters from DBSCAN clustering.
    Reference: https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/
    """
    min_points_decrement = 10
    num_neighbors = math.floor(
        len(loc_df) / min_points_decrement) * min_points_decrement
    best_epsilon = 0
    best_num_neighbors = 0
    best_silhouette_score = 0
    temp_epsilon = 0
    temp_num_neighbors = 0

    # Optimize epsilon value
    while num_neighbors > 0:
        # Distance from each point to its closest neighbour
        nb_learner = NearestNeighbors(n_neighbors=num_neighbors)
        nearest_neighbours = nb_learner.fit(loc_df)
        distances, _ = nearest_neighbours.kneighbors(loc_df)
        distances = np.sort(distances[:, num_neighbors-1], axis=0)

        # Optimal epsilon value: point with max curvature
        knee = KneeLocator(np.arange(len(distances)), distances, curve="convex",
                           direction="increasing", interp_method="polynomial")
        # knee.plot_knee()
        # plt.show()

        # Knee not found
        if knee.knee is None:
            opt_eps = 0
        else:
            opt_eps = round(distances[knee.knee], 3)

        if opt_eps > 0:
            dbscan_cluster = DBSCAN(eps=opt_eps, min_samples=num_neighbors)
            dbscan_cluster.fit(loc_df)
            # Number of clusters
            labels = dbscan_cluster.labels_
            n_clusters = len(set(labels))-(1 if -1 in labels else 0)
            if n_clusters > 1:
                # Can only calculate silhouette score when there are more than 1 clusters
                score = silhouette_score(loc_df, labels)
                if score > best_silhouette_score:
                    best_epsilon = opt_eps
                    best_num_neighbors = num_neighbors
                    best_silhouette_score = score
            elif n_clusters == 1:
                # Store temporarily in case only 1 cluster can be found after optimization
                temp_epsilon = opt_eps
                temp_num_neighbors = num_neighbors

        num_neighbors -= min_points_decrement

    if best_epsilon == 0:
        best_epsilon = temp_epsilon
        best_num_neighbors = temp_num_neighbors

    print(f"Best silhouette score: {best_silhouette_score}")
    print(f"Best epsilon value: {best_epsilon}")
    print(f"Best min points: {best_num_neighbors}")
    return best_epsilon, best_num_neighbors, best_silhouette_score


def cluster_locations(user_id, loc_df, latitude_col, longitude_col):
    """
    Performs DBSCAN clustering on unique combination of latitude and longitude using optimized epsilon values.
    Reference: https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/

    NOTE: Pre-saved location clusters are based on location coordinates rounded to 5 decimal places.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_location_clusters.parquet"
    if not os.path.exists(parquet_filename):
        loc_df = loc_df.withColumn(latitude_col, F.round(F.col(latitude_col), 5))\
            .withColumn(longitude_col, F.round(F.col(longitude_col), 5))
        loc_df = loc_df.select(latitude_col, longitude_col).distinct()
        latitude = np.array(loc_df.select(latitude_col).collect()).flatten()
        longitude = np.array(loc_df.select(longitude_col).collect()).flatten()
        loc_df = np.transpose(np.vstack([latitude, longitude]))
        epsilon, min_points, silhouette_score = optimize_cluster(loc_df)
        if epsilon == 0:
            return [-1 for i in range(len(loc_df))], min_points, silhouette_score
        dbscan_cluster = DBSCAN(eps=epsilon, min_samples=min_points)
        dbscan_cluster.fit(loc_df)

        cluster_df_data = []
        for index, (lat, long) in enumerate(loc_df):
            cluster_df_data.append((float(lat), float(long), int(dbscan_cluster.labels_[index])))
        cluster_df = spark.createDataFrame(cluster_df_data, schema=StructType([StructField(latitude_col, FloatType()),\
                                                                            StructField(longitude_col, FloatType()),\
                                                                            StructField("cluster_id", IntegerType())]))
        cluster_df.write.parquet(parquet_filename)
    
    cluster_df = spark.read.parquet(parquet_filename)
    return cluster_df


def process_location_data(user_id):
    """
    (Event-based)
    Location data: does not attempt to identify missing value

    NOTE Assumptions:
    1. When there are multiple entries in a given minute, retain the entry with the highest accuracy
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_locations.parquet"
    if not os.path.exists(parquet_filename):
        location_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_locations.csv")
        float_cols = ["timestamp", "double_latitude", "double_longitude", "double_bearing", "double_speed", "double_altitude", "accuracy"]
        for col in float_cols:
            location_df = location_df.withColumn(col, F.col(col).cast(FloatType()))
        time_cols = ["date", "hour", "minute"]
        max_accuracy_location = location_df.groupBy(time_cols).agg(F.max(F.col("accuracy")).alias("accuracy"))
        location_df = location_df.join(max_accuracy_location, time_cols + ["accuracy"])\
            .dropDuplicates()
        location_df.write.parquet(parquet_filename)
    
    location_df = spark.read.parquet(parquet_filename)
    return location_df


def process_wifi_data(user_id):
    """
    (Event-based)
    WiFi data: does not attempt to identify missing value

    NOTE Assumptions:
    1. Only retain unique ssid for nearby WiFi devices at a given time
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_wifi.parquet"
    if not os.path.exists(parquet_filename):
        wifi_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_wifi.csv")
        time_cols = ["date", "hour", "minute"]
        wifi_df = wifi_df.filter(F.col("ssid") != "null").select(*time_cols + ["ssid"]).distinct()\
            .sort(time_cols)
        wifi_df.write.parquet(parquet_filename)
    
    wifi_df = spark.read.parquet(parquet_filename)
    return wifi_df

@F.udf(IntegerType())
def resolve_cluster_id(cluster_ids, current_ts, prev_ts, next_ts, prev_cluster_ids, next_cluster_ids):
    """
    Resolves discrepancies in cluster IDs at the same time point by obtaining the mode cluster ID.
    If more than one mode cluster IDs are involved, use the cluster ID that co-occur at an adjacent time point.
    """
    counts = collections.Counter(cluster_ids)
    max_count = np.max(list(counts.values()))
    most_frequent_clusters = []
    for cluster_id, count in counts.items():
        if count == max_count:
            most_frequent_clusters.append(cluster_id)
    most_frequent_clusters = list(set(most_frequent_clusters))

    if len(most_frequent_clusters) > 1:
        prev_distance = abs(current_ts - prev_ts) if prev_ts is not None else np.inf
        next_distance = abs(current_ts - next_ts) if next_ts is not None else np.inf
        adj_clusters = None

        if prev_distance < next_distance:
            adj_clusters = prev_cluster_ids
        else:
            adj_clusters = next_cluster_ids
        
        adj_cluster_counts = collections.Counter(adj_clusters)
        max_adj_count = np.max(list(adj_cluster_counts.values()))
        most_frequent_adj_clusters = []
        for cluster_id, count in adj_cluster_counts.items():
            if count == max_adj_count:
                most_frequent_adj_clusters.append(cluster_id)
        most_frequent_adj_clusters = list(set(most_frequent_adj_clusters))
        # Make sure that the options for adjacent cluster IDs also exist in the current one
        most_frequent_adj_clusters = [c for c in most_frequent_adj_clusters if c in most_frequent_clusters]
        if len(most_frequent_adj_clusters) > 0:
            most_frequent_clusters = most_frequent_adj_clusters

    return most_frequent_clusters[0]


def complement_location_data(user_id):
    """
    Combines location and WiFi data to provide location semantics.
    Clusters locations and identify WiFi devices available in each cluster.

    NOTE:
    1. Location coordinates are rounded to 5 decimal places (up to 1 metres) - can also be rounded to 4 dp for 10 metres allowance
    2. Fill in cluster IDs based on groups of WiFi devices that have co-occur at clusters with known location coordinates.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_combined_location.parquet"
    if not os.path.exists(parquet_filename):
        time_cols = ["date", "hour", "minute"]
        coordinate_cols = ["double_latitude", "double_longitude"]
        location_df = process_location_data(user_id)\
            .withColumn("double_latitude", F.round(F.col("double_latitude"), 5))\
            .withColumn("double_longitude", F.round(F.col("double_longitude"), 5))\
            .withColumn("double_altitude", F.round(F.col("double_altitude"), 5))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))
        cluster_df = cluster_locations(user_id, location_df, "double_latitude", "double_longitude")
        # Location clusters are derived from all unique coordinates in location_df so there will be no null cluster_id
        location_df = location_df.join(cluster_df, coordinate_cols).dropDuplicates()
        wifi_df = process_wifi_data(user_id)\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))

        # Combine coordinates and WiFi devices information by date time
        all_locations = location_df.select(*time_cols + coordinate_cols + ["double_altitude", "cluster_id"])\
            .join(wifi_df, time_cols, "outer").dropDuplicates().sort(*time_cols)\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
        
        # Unique WiFi devices that can be directly mapped to location clusters
        unique_ssid_clusters = all_locations.select("cluster_id", "ssid").distinct().dropna()

        # Fill in cluster_ids for all time points of WiFi devices with known location clusters
        all_locations = all_locations.join(unique_ssid_clusters.withColumnRenamed("cluster_id", "temp_cluster_id"), "ssid", "left")\
            .withColumn("cluster_id", F.coalesce(F.col("cluster_id"), F.col("temp_cluster_id")))\
            .drop("temp_cluster_id").sort(*time_cols)

        # Compile WiFi devices that exist together at the same time point
        timewise_wifi_devices = all_locations.groupBy(time_cols).agg(F.collect_set("ssid").alias("WiFi_devices"))\
            .sort(*time_cols)
        
        prev_null_count = 0
        cur_null_count = all_locations.filter(F.col("cluster_id").isNull()).count()

        # Filling in cluster_ids for existing WiFi devices based on direct and indirect coexistence
        while cur_null_count != prev_null_count:
            prev_null_count = cur_null_count

            # Compile list of WiFi devices coexisting at various time points with those currently having known location clusters
            unique_ssid_clusters = all_locations.select("cluster_id", "ssid").distinct().dropna()
            coexist_ssids = unique_ssid_clusters\
                .select("ssid").distinct().join(timewise_wifi_devices, F.array_contains(F.col("WiFi_devices"), F.col("ssid")))\
                .groupBy("ssid").agg(F.collect_list("WiFi_devices").alias("WiFi_devices"))\
                .withColumn("WiFi_devices", F.sort_array(F.flatten(F.col("WiFi_devices"))))\
                .dropDuplicates().sort("ssid")
            coexist_ssids = coexist_ssids.join(unique_ssid_clusters, "ssid")\
                .dropDuplicates().sort("ssid")\
                .withColumnRenamed("cluster_id", "temp_cluster_id")
            
            # Map all available WiFi devices to known clusters based on coexistence
            all_locations = all_locations.join(coexist_ssids.select("WiFi_devices", "temp_cluster_id").distinct(),\
                                            F.array_contains(F.col("WiFi_devices"), F.col("ssid")), "left")\
                .withColumn("cluster_id", F.coalesce(F.col("cluster_id"), F.col("temp_cluster_id")))\
                .select(*all_locations.columns).dropDuplicates().sort(*time_cols)

            # Make sure that the filled clusters are consistent if the WiFi devices co-occur at the same time point
            time_window = Window().partitionBy("date").orderBy("datetime")
            resolved_cluster = all_locations.groupBy(*time_cols + ["datetime"]).agg(F.collect_list("cluster_id").alias("cluster_ids"))\
                .withColumn("prev_cluster_ids", F.lag(F.col("cluster_ids")).over(time_window))\
                .withColumn("next_cluster_ids", F.lead(F.col("cluster_ids")).over(time_window))\
                .withColumn("prev_datetime", F.lag(F.col("datetime")).over(time_window))\
                .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
                .withColumn("resolved_cluster_id", resolve_cluster_id(F.col("cluster_ids"), F.col("datetime"), 
                    F.col("prev_datetime"), F.col("next_datetime"), 
                    F.col("prev_cluster_ids"), F.col("next_cluster_ids")))\
                .withColumnRenamed("resolved_cluster_id", "cluster_id")

            all_locations = all_locations.drop("cluster_id").join(resolved_cluster.select(*time_cols + ["datetime", "cluster_id"]),\
                time_cols + ["datetime"], "left").dropDuplicates().sort(*time_cols)

            # Fill in those that coexist at the same time points with the newly filled WiFi devices
            # remaining_coexist_ssids = all_locations.groupBy(*time_cols)\
            #     .agg(F.first("cluster_id", ignorenulls=True).alias("temp_cluster_id"))
            # all_locations = all_locations.join(remaining_coexist_ssids, time_cols)\
            #     .withColumn("cluster_id", F.coalesce(F.col("cluster_id"), F.col("temp_cluster_id")))\
            #     .drop("temp_cluster_id").sort(*time_cols)
            cur_null_count = all_locations.filter(F.col("cluster_id").isNull()).count()
    
        # # Fill in null coordinates and list of nearby WiFi devices based on available data
        # filled_locations = all_locations.join(unique_coordinates_ssid.withColumnRenamed("WiFi_devices", "available_WiFi_devices"), coordinate_cols, "left")\
        #     .withColumn("WiFi_devices", F.coalesce("WiFi_devices", "available_WiFi_devices"))\
        #     .drop("available_WiFi_devices").dropDuplicates().sort(*time_cols)
        # filled_locations = filled_locations.join(unique_coordinates_ssid.withColumnRenamed("double_latitude", "available_latitude")\
        #                                     .withColumnRenamed("double_longitude", "available_longitude"), "WiFi_devices", "left")\
        #     .withColumn("double_latitude", F.coalesce("double_latitude", "available_latitude"))\
        #     .withColumn("double_longitude", F.coalesce("double_longitude", "available_longitude"))\
        #     .drop("available_latitude", "available_longitude").dropDuplicates().sort(*time_cols)
        all_locations.write.parquet(parquet_filename)

    all_locations = spark.read.parquet(parquet_filename)
    return all_locations


def interval_join(df1, df2):
    """
    Joins and finds overlapping duration of start and end times between input dataframes.
    """
    return df1.crossJoin(df2)\
        .filter((df1["start_datetime"] <= df2["end_datetime"]) & (df1["end_datetime"] >= df2["start_datetime"]))\
        .select(F.greatest(df1["start_datetime"], df2["start_datetime"]).alias("start_datetime"),\
                F.least(df1["end_datetime"], df2["end_datetime"]).alias("end_datetime"),\
                *[col for col in df1.columns + df2.columns if "_datetime" not in col])\
        .sort("start_datetime")

def outer_join_by_intervals(df1, df2, condition):
    """
    Performs outer join to accept situations where input df2 does not fulfill all conditions.
    """
    return df1.join(df2, (df2["start_datetime"] <= df1["overall_end_datetime"]) &\
                (df2["end_datetime"] >= df1["overall_start_datetime"]), "right")\
        .withColumn("condition", F.when(F.col("overall_start_datetime").isNull(), condition).otherwise("all"))\
        .withColumn("overall_start_datetime", F.greatest(F.col("overall_start_datetime"), F.col("start_datetime")))\
        .withColumn("overall_end_datetime", F.least(F.col("overall_end_datetime"), F.col("end_datetime")))\
        .drop("start_datetime", "end_datetime").dropDuplicates().sort("overall_start_datetime")

def estimate_sleep(user_id, with_allowance=True):
    """
    Estimate sleep duration based on information from light, noise, activity and phone screen.
    V1 method finds overlapping duration when all conditions are fulfilled (dark env, quiet env, stationary, phone not in use).
    V2 method allows if only 3 out of the 4 conditions are fulfilled.
    2 methods of obtaining phone usage: (1) directly via device usage plugin, (2) manual computation using screen states.

    NOTE Assumptions:
    1. When conditions are fulfilled for at least 5 minutes consecutively
    2. Environment may not be completely quiet and/or dark - thresholds are computed based on relative distribution
    3. Remove those with duration shorter than 1 hour
    4. Provide allowance by default with less strict constraints to offer more possible entries.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_sleep" + (
        "_optional" if with_allowance else "") + ".parquet"
    if not os.path.exists(parquet_filename):
        time_cols = ["date", "hour", "minute"]
        time_window = Window.partitionBy("date").orderBy(*time_cols)
        consecutive_min = 5

        light_df = process_light_data(user_id)
        brightness_quartiles = light_df.approxQuantile("mean_light_lux", [0.25, 0.50], 0.01)
        dark_threshold = round(brightness_quartiles[1])

        light_df = light_df.withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("is_dark", F.when(F.col("mean_light_lux") <= dark_threshold, 1).otherwise(0))\
            .withColumn("prev_is_dark", F.lag(F.col("is_dark")).over(time_window))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))
        # Consolidate consecutive rows with the same condition by assigning them to the same group until a transition occurs
        # NOTE: There are only 2 states so a new entry indicates a transition from the other state
        light_df = light_df.withColumn("new_group", (F.col("is_dark") != F.col("prev_is_dark")).cast("int"))\
            .withColumn("group_id", F.sum("new_group").over(Window.orderBy("datetime").rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
            .groupBy("group_id", "is_dark")\
            .agg(F.min("datetime").alias("start_datetime"),\
                    F.max("next_datetime").alias("end_datetime"), \
                    F.mean("mean_light_lux").alias("mean_light_lux"))\
            .withColumn("consecutive_duration", F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime")))\
            .drop("group_id").sort("start_datetime")
        dark_df = light_df.filter((F.col("is_dark") == 1) & (F.col("consecutive_duration") > consecutive_min*60))\
            .drop("is_dark", "consecutive_duration").sort("start_datetime")
        

        noise_df = process_noise_data(user_id)
        audio_quartiles = noise_df.approxQuantile("mean_decibels", [0.25, 0.50], 0.01)
        # silence_threshold = round((audio_quartiles[1]-audio_quartiles[0]) / 2)
        silence_threshold = round(audio_quartiles[1])

        noise_df = noise_df.withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("is_quiet", F.when(F.col("mean_decibels") <= silence_threshold, 1).otherwise(0))\
            .withColumn("prev_is_quiet", F.lag(F.col("is_quiet")).over(time_window))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))
        noise_df = noise_df.withColumn("new_group", (F.col("is_quiet") != F.col("prev_is_quiet")).cast("int"))\
            .withColumn("group_id", F.sum("new_group").over(Window.orderBy("datetime").rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
            .groupBy("group_id", "is_quiet")\
            .agg(F.min("datetime").alias("start_datetime"),\
                    F.max("next_datetime").alias("end_datetime"), \
                    F.mean("mean_decibels").alias("mean_decibels"))\
            .withColumn("consecutive_duration", F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime")))\
            .drop("group_id").sort("start_datetime")
        quiet_df = noise_df.filter((F.col("is_quiet") == 1) & (F.col("consecutive_duration") > consecutive_min*60))\
            .drop("is_quiet", "consecutive_duration").sort("start_datetime")
        

        activity_df = process_activity_data(user_id)\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
            .withColumn("prev_activity", F.lag(F.col("activity_type")).over(time_window))\
            .sort("datetime")
        activity_df = activity_df.withColumn("new_group", (F.col("activity_type") != F.col("prev_activity")).cast("int"))\
            .withColumn("group_id", F.sum("new_group").over(Window.orderBy("datetime").rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
            .groupBy("group_id", "activity_type")\
            .agg(F.min("datetime").alias("start_datetime"),\
                    F.max("next_datetime").alias("end_datetime")) \
            .withColumn("consecutive_duration", F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime")))\
            .drop("group_id").sort("start_datetime")
        stationary_df = activity_df.filter((F.col("activity_type") == 3) & (F.col("consecutive_duration") >= consecutive_min*60))\
            .select("start_datetime", "end_datetime").sort("start_datetime")

        # V1: Manual computation through screen states
        # screen_df = process_screen_data(user_id)
        # # Only exclude active usage time because phone screen might still be activated by notifications
        # phone_usage_df = screen_df.withColumn("is_in_use", F.when(F.col("prev_status") == 3, 1).otherwise(0))\
        #     .withColumn("prev_in_use", F.lag(F.col("is_in_use")).over(time_window))\
        #     .withColumn("new_group", (F.col("prev_in_use") != F.col("is_in_use")).cast("int"))\
        #     .withColumn("group_id", F.sum("new_group").over(Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
        #     .filter(F.col("prev_timestamp").isNotNull())\
        #     .groupBy("group_id", "is_in_use")\
        #     .agg(F.min("prev_timestamp").alias("start_datetime"),\
        #             F.max("timestamp").alias("end_datetime"))\
        #     .withColumn("consecutive_duration", F.round((F.col("end_datetime") - F.col("start_datetime"))/1000).cast(IntegerType()))\
        #     .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_datetime")))\
        #     .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("end_datetime")))\
        #     .drop("group_id").sort("start_datetime")

        # # With considerations of screen activation caused by notifications or active phone checking
        # not_in_use_df = phone_usage_df.filter((F.col("is_in_use") == 0) & (F.col("consecutive_duration") > consecutive_min*60))\
        #     .drop("is_in_use", "consecutive_duration").sort("start_datetime")
        
        # V2: Used plugin data directly
        phone_use_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")
        not_in_use_df = phone_use_df.filter(F.col("double_elapsed_device_off") > 0)\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("start_timestamp", F.col("timestamp") - F.col("double_elapsed_device_off"))\
            .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_timestamp")))\
            .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("timestamp")))\
            .withColumn("consecutive_duration", F.round(F.col("double_elapsed_device_off")/1000).cast(IntegerType()))\
            .select("start_datetime", "end_datetime").sort("start_datetime")

        # V1: Find overlapping intervals when all 4 conditions are fulfilled
        if not with_allowance:
            sleep_df = interval_join(interval_join(interval_join(dark_df, quiet_df), stationary_df), not_in_use_df)\
                .withColumn("condition", F.lit("all"))
        else:
            # V2: Optimize to include situations where 3 of the 4 conditions are fulfilled
            stationary_off_screen_df = interval_join(stationary_df, not_in_use_df)
            stationary_off_screen_quiet_df = interval_join(stationary_off_screen_df, quiet_df)
            stationary_off_screen_dark_df = interval_join(stationary_off_screen_df, dark_df)

            dark_quiet_df = interval_join(dark_df, quiet_df)
            off_screen_dark_quiet_df = interval_join(dark_quiet_df, not_in_use_df)
            stationary_dark_quiet_df = interval_join(dark_quiet_df, stationary_df)

            all_conditions_df = interval_join(stationary_off_screen_df, dark_quiet_df)\
                .withColumnRenamed("start_datetime", "overall_start_datetime")\
                .withColumnRenamed("end_datetime", "overall_end_datetime")
            
            all_cols = ["overall_start_datetime", "overall_end_datetime", "mean_light_lux", "mean_decibels", "condition"]
            sleep_df = outer_join_by_intervals(all_conditions_df.drop("mean_decibels"), stationary_off_screen_quiet_df, "stationary, off screen, quiet")\
                .select(*all_cols)
            sleep_df = sleep_df.union(\
                outer_join_by_intervals(all_conditions_df.drop("mean_light_lux"), stationary_off_screen_dark_df, "stationary, off screen, dark")\
                    .select(*all_cols))\
                .dropDuplicates().sort("overall_start_datetime")
            sleep_df = sleep_df.union(\
                outer_join_by_intervals(all_conditions_df.drop("mean_light_lux", "mean_decibels"), off_screen_dark_quiet_df, "off screen, quiet, dark"))\
                .select(*all_cols)\
                .dropDuplicates().sort("overall_start_datetime")
            sleep_df = sleep_df.union(\
                outer_join_by_intervals(all_conditions_df.drop("mean_light_lux", "mean_decibels"), stationary_dark_quiet_df, "stationary, quiet, dark"))\
                .select(*all_cols)\
                .dropDuplicates().sort("overall_start_datetime")
            sleep_df = sleep_df.withColumnRenamed("overall_start_datetime", "start_datetime")\
                .withColumnRenamed("overall_end_datetime", "end_datetime")

        # NOTE: This block is shared by both V1 and V2 methods
        sleep_df = sleep_df.withColumn("duration", F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime")))\
            .filter(F.col("duration") >= 3600)\
            .withColumn("duration (hr)", F.col("duration") / 3600)\
            .select("start_datetime", "end_datetime", "duration (hr)", "condition", "mean_light_lux", "mean_decibels")\
            .sort("start_datetime")

        # There may have subsequent entries that are within the same time window
        sleep_df = sleep_df.withColumn("prev_end_datetime", F.lag(F.col("end_datetime")).over(Window.orderBy("start_datetime")))\
            .withColumn("prev_condition", F.lag(F.col("condition")).over(Window.orderBy("start_datetime")))\
            .withColumn("new_group", ((F.col("prev_end_datetime") != F.col("start_datetime")) | (F.col("condition") != F.col("prev_condition"))).cast("int"))\
            .withColumn("group_id", F.sum("new_group").over(Window.orderBy("start_datetime").rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
            .groupBy("group_id", "condition")\
            .agg(F.min("start_datetime").alias("start_datetime"),\
                F.max("end_datetime").alias("end_datetime"),\
                F.sum("duration (hr)").alias("duration (hr)"),\
                F.mean("mean_light_lux").alias("mean_light_lux"),\
                F.mean("mean_decibels").alias("mean_decibels"))\
            .drop("group_id").sort("start_datetime")

        sleep_df.write.parquet(parquet_filename)
    
    sleep_df = spark.read.parquet(parquet_filename)
    return sleep_df


"""
Referenced from StudentLife paper: https://dl.acm.org/doi/abs/10.1145/2632048.2632054
1. High-level overview of day of the week vs time of the day using heatmap
2. Accessing stability of data collection
3. Accessing high-level user compliance of EMA
"""

@F.udf(FloatType())
def distance(lat1, long1, lat2, long2):
    """
    Distance between two pairs of location latitude and longitude.
    Haversine distance is more accurate compared to Euclidean distance from https://peerj.com/articles/2537/
    """
    return hs.haversine((lat1, long1), (lat2, long2), unit=hs.Unit.METERS)


@F.udf(StringType())
def resolve_activity_priority(activity_list):
    """
    Resolves instances where multiple activities share the same confidence to prioritize those of a higher priority (more fine-grained).
    """
    highest_priority_index = None
    priority_activity = None
    for act in activity_list:
        index = ACTIVITY_PRIORITIES.index(act)
        if not highest_priority_index or index < highest_priority_index:
            highest_priority_index = index
            priority_activity = act
    return priority_activity


def extract_daily_features(user_id, date=None):
    """
    Extracts day-level features and returns as a dictionary with 31 keys (date as the first key):
    1. Total duration of each activity state (feature 1-7)
    2. Total duration in dark environment (feature 8)
    3. Total duration in silent environment (feature 9)
    4. Total duration of phone use (feature 10)
    5. Total duration of using each application category, normalized by the duration of phone use (feature 11-20)
    6. Number of unique WiFi devices (feature 21)
    7. Number of unique Bluetooth devices (feature 22)
    8. Total distance traveled (feature 23)
    9. Location variance (feature 24)
    11. Cluster epoch features (# features = # unique clusters * 4 epochs of the day)
    10. Total number of clusters (feature 25)
    11. Total number of unique clusters (feature 26)
    12. Number of entries with -1 cluster ID, normalized by number of unique location entries (feature 27, 28)
    13. Location entropy and normalized entropy (feature 29, 30)
    14. Time spent at primary and secondary location clusters (feature 31, 32)


    NOTE Assumptions:
    1. Wake time as the minimum "end_datetime" of estimated sleep duration for each day
    2. The most frequently seen cluster at wake time as primary location
    
    # TODO:
    1. Map light with noise information - sleep, work vs outdoor environment
    2. People/device around - Bluetooth

    References:
    1. https://peerj.com/articles/2537/ (computation of):
        a. Location variance
        b. Total distance traveled
        c. Entropy and normalized entropy
    """
    time_cols = ["date", "hour", "minute"]
    time_window = Window().orderBy("datetime")

    # Retrieves pre-saved contexts
    with open(f"{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)
    
    if date is None:
        sleep_df = estimate_sleep(user_id)\
            .withColumn("date", udf_get_date_from_datetime("end_datetime"))
        cur_day = np.array(sleep_df.select("date").distinct().sort("date").collect()).flatten()[0]
    else:
        cur_day = date
    
    features = {"date": cur_day}

    # Filter data for the particular day
    physical_mobility = process_activity_data(user_id)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
        .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))
    # Get the last entry of the previous day to get data at 00:00
    physical_mobility = physical_mobility.filter(F.col("date") < cur_day).orderBy(F.col("datetime").desc()).limit(1)\
        .withColumn("datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0))+timedelta(hours=2))\
        .union(physical_mobility.filter(F.col("date") == cur_day)).sort("datetime")
    # Resolve multiple activity entries at the same time point with custom granularity priorities
    physical_mobility = physical_mobility.groupBy("datetime").agg(F.collect_list("activity_name").alias("activity_list"))\
        .withColumn("activity_name", resolve_activity_priority("activity_list"))
    physical_mobility = physical_mobility.withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
        .filter(F.col("next_datetime").isNotNull())\
        .withColumn("duration", F.unix_timestamp("next_datetime") - F.unix_timestamp("datetime"))
    activity_duration = physical_mobility.groupBy("activity_name")\
        .agg(F.sum("duration").alias("total_activity_duration")).toPandas()

    activity_names = activity_duration["activity_name"].to_list()
    durations = activity_duration["total_activity_duration"].to_list()
    for act in ["still", "in_vehicle", "on_bicycle", "on_foot", "tilting", "walking", "running"]:
        if act in activity_names:
            features[f"{act}_duration"] = durations[activity_names.index(act)]
        else:
            features[f"{act}_duration"] = 0


    ambient_light = process_light_data(user_id)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
        .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))
    ambient_light = ambient_light.filter(F.col("date") < cur_day).orderBy(F.col("datetime").desc()).limit(1)\
        .withColumn("datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0))+timedelta(hours=2))\
        .union(ambient_light.filter(F.col("date") == cur_day)).sort("datetime")
    ambient_light = ambient_light.withColumn("is_dark", F.when(F.col("mean_light_lux") <= contexts["dark_threshold"], 1).otherwise(0))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
            .filter(F.col("next_datetime").isNotNull())\
            .withColumn("duration", F.unix_timestamp("next_datetime") - F.unix_timestamp("datetime"))
    total_dark_duration = ambient_light.filter(F.col("is_dark")==1).agg(F.sum("duration")).collect()[0][0]
    if not total_dark_duration:
        total_dark_duration = 0
    features["dark_duration"] = total_dark_duration

    ambient_noise = process_noise_data_with_conv_estimate(user_id)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
        .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))
    ambient_noise = ambient_noise.filter(F.col("date") < cur_day).orderBy(F.col("datetime").desc()).limit(1)\
        .withColumn("datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0))+timedelta(hours=2))\
        .union(ambient_noise.filter(F.col("date") == cur_day)).sort("datetime")
    ambient_noise = ambient_noise.withColumn("is_quiet", F.when(F.col("mean_decibels") <= contexts["silent_threshold"], 1).otherwise(0))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
            .filter(F.col("next_datetime").isNotNull())\
            .withColumn("duration", F.unix_timestamp("next_datetime") - F.unix_timestamp("datetime"))
    total_quiet_duration = ambient_noise.filter(F.col("is_quiet")==1).agg(F.sum("duration")).collect()[0][0]
    if not total_quiet_duration:
        total_quiet_duration = 0
    features["quiet_duration"] = total_quiet_duration

    app_usage = process_application_usage_data(user_id)\
        .withColumn("start_timestamp", F.col("start_timestamp").cast(FloatType()))\
        .withColumn("start_datetime", udf_datetime_from_timestamp("start_timestamp")-timedelta(hours=2))\
        .withColumn("end_timestamp", F.col("end_timestamp").cast(FloatType()))\
        .withColumn("end_datetime", udf_datetime_from_timestamp("end_timestamp")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime("start_datetime"))\
        .filter(F.col("date") == cur_day)\
        .withColumn("is_system_app", F.col("is_system_app").cast(IntegerType()))\
        .withColumn("category", label_app_category(F.lit(user_id), F.col("application_name"), F.col("is_system_app")))

    total_phone_use_duration = app_usage.agg(F.sum("double_elapsed_device_on")).collect()[0][0]/1000
    features["phone_use_duration"] = total_phone_use_duration

    app_usage_duration = app_usage.groupBy("category")\
        .agg((F.sum("usage_duration")/60).alias("total_usage_duration"))\
        .withColumn("phone_use_normalized_usage_duration", F.col("total_usage_duration")/total_phone_use_duration)\
        .sort("total_usage_duration", ascending=False).toPandas()
    app_categories = list(contexts["app_categories"].keys())
    cur_apps = app_usage_duration["category"].to_list()
    app_durations = app_usage_duration["total_usage_duration"].to_list()
    normalized_app_durations = app_usage_duration["phone_use_normalized_usage_duration"].to_list()
    for category in app_categories + ["utilities", "others"]:
        if category in cur_apps:
            features[f"{category}_app_duration"] = app_durations[cur_apps.index(category)]
            features[f"{category}_app_normalized_duration"] = normalized_app_durations[cur_apps.index(category)]
        else:
            features[f"{category}_app_duration"] = 0
            features[f"{category}_app_normalized_duration"] = 0

    
    bluetooth_df = process_bluetooth_data(user_id)\
        .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
        .withColumn("bt_datetime", udf_datetime_from_timestamp("timestamp"))\
        .filter(F.col("date") == cur_day).sort("bt_datetime")
    features["unique_bluetooth_device"] = bluetooth_df.select("bt_address").distinct().count()


    locations = complement_location_data(user_id).withColumnRenamed("datetime", "location_datetime")\
        .filter(F.col("date") == cur_day).sort("location_datetime")
    # locations = locations.filter(F.col("date") < cur_day).orderBy(F.col("location_datetime").desc()).limit(1)\
    #     .withColumn("location_datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0)))\
    #     .union(locations.filter(F.col("date") == cur_day)).sort("location_datetime")

    wifi_df = locations.filter(F.col("ssid").isNotNull())
    features["unique_wifi_device"] = wifi_df.select("ssid").distinct().count()

    # Total distance travelled
    time_window = Window().orderBy("location_datetime")
    location_coordinates = locations.select("location_datetime", "double_latitude", "double_longitude")\
        .dropDuplicates().dropna().sort("location_datetime")
    distance_traveled = location_coordinates\
        .withColumn("next_latitude", F.lead(F.col("double_latitude")).over(time_window))\
        .withColumn("next_longitude", F.lead(F.col("double_longitude")).over(time_window))\
        .filter((F.col("next_latitude").isNotNull()) & (F.col("next_longitude").isNotNull()))\
        .withColumn("distance", distance(F.col("double_latitude"), F.col("double_longitude"),\
                                         F.col("next_latitude"), F.col("next_longitude")))
    total_distance = distance_traveled.agg(F.sum("distance")).collect()[0][0]
    features["total_distance_traveled"] = total_distance

    # Location variance
    latitude_variance = location_coordinates.agg(F.variance("double_latitude")).collect()[0][0]
    longitude_variance = location_coordinates.agg(F.variance("double_longitude")).collect()[0][0]
    location_variance = math.log(latitude_variance + longitude_variance)
    features["location_variance"] = location_variance

    # Group WiFi devices at each time point to compute cluster transitions between consecutive time points
    cluster_transitions = locations.groupBy(*[col for col in locations.columns if col != "ssid"])\
        .agg(F.concat_ws(", ", F.collect_set("ssid")).alias("WiFi_devices"))\
        .dropDuplicates().sort("location_datetime")
    cluster_transitions = cluster_transitions.withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(time_window))\
        .withColumn("prev_location_datetime", F.lag(F.col("location_datetime")).over(time_window))\
        .drop(*time_cols).sort("location_datetime")\
        .filter(F.col("prev_cluster") != F.col("cluster_id"))
    # visualize_day_contexts(user_id, cur_day, physical_mobility, ambient_light, ambient_noise, app_usage, cluster_transitions)

    # NOTE: (N+1) cluster analysis will be involved for N cluster transition points
    context_dfs = [physical_mobility, ambient_light, ambient_noise, app_usage, bluetooth_df, locations]
    context_df_datetime_cols = ["datetime", "datetime", "datetime", "start_datetime", "bt_datetime", "location_datetime"]

    # First row will always be the first filtered row for the day of interest
    location_transition_datetimes = np.array(cluster_transitions.select("location_datetime").collect()).flatten()
    if len(location_transition_datetimes) == 0:
        location_clusters = np.array(locations.select("cluster_id").distinct().collect()).flatten()
        features["cluster_count"] = 1
        features["unique_cluster_count"] = 1
    else:
        first_cluster = np.array(cluster_transitions.select("prev_cluster").collect()).flatten()[0]
        location_clusters = np.append(first_cluster, np.array(cluster_transitions.select("cluster_id").collect()).flatten())
        features["cluster_count"] = len(location_clusters)
        features["unique_cluster_count"] = len(list(set(location_clusters)))

    epoch_list = list(TIME_EPOCHS.keys())
    cluster_time_range = []
    cluster_context_dfs = [[] for _ in range(len(context_dfs))]
    for cluster_index, cluster in enumerate(location_clusters):
        if cluster_index == 0:
            if len(location_transition_datetimes) == 0:
                cluster_end_datetime = datetime.strptime(f"{cur_day} 23:59", "%Y-%m-%d %H:%M")
            else:
                cluster_end_datetime = location_transition_datetimes[cluster_index]
            # Compute start datetime as the minimum of all context dataframes
            min_context_datetime = []
            for context_index, context_df in enumerate(context_dfs):
                cur_df = context_df.filter(F.col(context_df_datetime_cols[context_index]) < cluster_end_datetime)
                min_context_datetime.append(cur_df.agg(F.min(context_df_datetime_cols[context_index])).collect()[0][0])
                cluster_context_dfs[context_index].append(cur_df)
            cluster_start_datetime = np.min(min_context_datetime)
            cluster_time_range.append((cluster_start_datetime, cluster_end_datetime))
        elif cluster_index < len(location_clusters)-1:
            for context_index, context_df in enumerate(context_dfs):
                location_datetime = location_transition_datetimes[cluster_index]
                cluster_context_dfs[context_index].append(context_df\
                    .filter((F.col(context_df_datetime_cols[context_index]) >= location_transition_datetimes[cluster_index-1]) &\
                        (F.col(context_df_datetime_cols[context_index]) < location_datetime)))
            cluster_time_range.append((location_transition_datetimes[cluster_index-1], location_datetime))
        else:
            for context_index, context_df in enumerate(context_dfs):
                location_datetime = location_transition_datetimes[-1]
                cluster_context_dfs[context_index].append(context_df.filter(F.col(context_df_datetime_cols[context_index]) >= location_datetime)) 
                cluster_time_range.append((location_transition_datetimes[-1], datetime.strptime(f"{cur_day} 23:59", "%Y-%m-%d %H:%M")))
        
        cluster_start_datetime = cluster_time_range[cluster_index][0]
        cluster_end_datetime = cluster_time_range[cluster_index][1]
        start_hour = cluster_start_datetime.time().hour
        end_hour = cluster_end_datetime.time().hour
        start_epoch = get_epoch_from_hour(start_hour)
        end_epoch =  get_epoch_from_hour(end_hour)
        # Check if the cluster spans across multiple epochs of the day
        if start_epoch != end_epoch:
            epoch_count = (end_epoch - start_epoch) % len(epoch_list)
            epoch_start_time = cluster_start_datetime
            for _ in range(epoch_count):
                # Compute individual cluster features for each epoch
                epoch_end_time = datetime.strptime(f"{cur_day} {TIME_EPOCHS[epoch_list[start_epoch]]['max']}:00", "%Y-%m-%d %H:%M")
                cluster_epoch_features = extract_cluster_contexts(*[user_id, (epoch_start_time, epoch_end_time)] +\
                    [cluster_context_dfs[i][cluster_index].filter((F.col(context_df_datetime_cols[i]) >= epoch_start_time) &\
                                              (F.col(context_df_datetime_cols[i]) < epoch_end_time))\
                                                for i in range(len(cluster_context_dfs))])
                for key, value in cluster_epoch_features.items():
                    features[f"cluster{cluster}_{epoch_list[get_epoch_from_hour(epoch_start_time.time().hour)]}_{key}"] = value
                epoch_start_time = epoch_end_time
                start_epoch = get_epoch_from_hour(epoch_start_time.time().hour)
            
            cluster_epoch_features = extract_cluster_contexts(*[user_id, (epoch_start_time, cluster_end_datetime)] +\
                [cluster_context_dfs[i][cluster_index].filter(F.col(context_df_datetime_cols[i]) >= epoch_start_time)\
                                            for i in range(len(cluster_context_dfs))])
            cluster_feature_keys = list(cluster_epoch_features.keys())
            for key, value in cluster_epoch_features.items():
                features[f"cluster{cluster}_{epoch_list[get_epoch_from_hour(epoch_start_time.time().hour)]}_{key}"] = value
        else:
            cluster_features = extract_cluster_contexts(*[user_id, cluster_time_range[cluster_index]] + [df[cluster_index] for df in cluster_context_dfs])
            cluster_feature_keys = list(cluster_features.keys())
            for key, value in cluster_features.items():
                features[f"cluster{cluster}_{epoch_list[start_epoch]}_{key}"] = value

    # Retrieves pre-computed clusters
    overall_location_info = contexts["location_clusters"]
    overall_clusters = [int(id) for id in overall_location_info.keys()]
    # Make sure that cluster features exist for each combination of location cluster and epoch
    for cluster_id in overall_clusters:
        for epoch in epoch_list:
            if f"cluster{cluster_id}_{epoch}_{cluster_feature_keys[0]}" not in features:
                for col in cluster_feature_keys:
                    features[f"cluster{cluster_id}_{epoch}_{col}"] = 0

    # Count of unknown locations, normalized by total location entries
    # Locations could have multiple entries at the same timestamp due to multiple WiFi devices
    unique_location_entries = locations.select(*["location_datetime", "cluster_id"] +\
                                               [col for col in locations.columns if "double_" in col]).dropDuplicates()
    unknown_location_count = unique_location_entries.filter(F.col("cluster_id") == -1).count()
    normalized_unknown_location_count = unknown_location_count / unique_location_entries.count()
    features["unknown_location_count"] = unknown_location_count
    features["normalized_unknown_location_count"] = normalized_unknown_location_count

    cluster_time_spent = [0 for _ in range(len(overall_clusters))]
    # Time spent in each cluster
    for cluster_index, cluster_id in enumerate(location_clusters):
        overall_index = overall_clusters.index(cluster_id)
        cluster_time_spent[overall_index] += (cluster_time_range[cluster_index][1] - cluster_time_range[cluster_index][0]).total_seconds()
    # Entropy and normalized entropy
    # Add a small negligible value to avoid log(0) when a specific cluster is not visited
    probability = np.array(cluster_time_spent)/(24*3600) + 1e-10
    entropy = - np.sum(probability * np.log(probability))
    normalized_entropy = entropy/math.log(len(overall_clusters))
    features["location_entropy"] = entropy
    features["normalized_location_entropy"] = normalized_entropy

    # Time spent at primary clusters
    time_spent_primary_cluster = 0
    time_spent_secondary_cluster = 0
    for index, cluster_id in enumerate(overall_clusters):
        if overall_location_info[str(cluster_id)]["is_primary"]:
            time_spent_primary_cluster = cluster_time_spent[index]
        elif overall_location_info[str(cluster_id)]["is_secondary"]:
            time_spent_secondary_cluster = cluster_time_spent[index]
    features["time_spent_primary_cluster"] = time_spent_primary_cluster
    features["time_spent_secondary_cluster"] = time_spent_secondary_cluster

    return features

def time_to_midnight_hours(t):
    """
    Maps current time to hours relative to midnight.
    """
    if t is None:
        return None
    hour = t.astimezone(TIMEZONE).hour
    minute = t.astimezone(TIMEZONE).minute
    if hour > 12:
        return (hour + minute / 60.0) - 24 
    return hour + minute / 60.0

@F.udf(FloatType())
def time_to_hours(t):
    if t is None:
        return None
    hour = t.astimezone(TIMEZONE).hour
    minute = t.astimezone(TIMEZONE).minute
    return hour + minute / 60.0

def map_overview_estimated_sleep_duration_to_sleep_ema(user_id, esm_ids):
    """
    (Overview - multiple days)
    Maps estimated sleep duration to self-reported sleep duration and sleep quality rating.
    Plots vertical bars showing estimated sleep duration colored based on quality rating and overlayed with reported sleep duration.
    """
    esm_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_esms.csv")\
        .withColumn("esm_id", udf_extract_esm_id("esm_json"))\
        .filter(F.col("esm_id").isin(esm_ids))\
        .groupBy("date").pivot("esm_id").agg(F.first("esm_user_answer"))\
        .withColumnRenamed(str(esm_ids[0]), "reported_sleep_time")\
        .withColumnRenamed(str(esm_ids[1]), "reported_wake_time")\
        .withColumnRenamed(str(esm_ids[2]), "sleep_quality_rating").sort("date")\
        .withColumn("reported_sleep_time", F.col("reported_sleep_time").cast(TimestampType()))\
        .withColumn("reported_wake_time", F.col("reported_wake_time").cast(TimestampType()))\
        .withColumn("sleep_quality_rating", F.col("sleep_quality_rating").cast(IntegerType()))

    sleep_df = estimate_sleep(user_id).withColumn("date", udf_get_date_from_datetime("end_datetime"))
    
    # Identify daily sleep duration from the earliest sleep wake datetime for each day
    wake_df = sleep_df.groupBy("date").agg(F.min("end_datetime").alias("end_datetime"))\
        .join(sleep_df, ["date", "end_datetime"])
    
    # Map estimated sleep duration to self-reported sleep duration
    combined_sleep_df = wake_df.join(esm_df, "date", "left")
    datetime_cols = ["start_datetime", "end_datetime", "reported_sleep_time", "reported_wake_time"]
    combined_sleep_df = combined_sleep_df.withColumn("start_datetime", F.col("start_datetime")-timedelta(hours=2))\
        .withColumn("end_datetime", F.col("end_datetime")-timedelta(hours=2))
    for col in datetime_cols:
        combined_sleep_df = combined_sleep_df.withColumn(col, udf_hours_relative_to_midnight(F.col(col)))
    combined_sleep_df = combined_sleep_df.select(*["date", "sleep_quality_rating"] + datetime_cols).sort("date").toPandas()

    _, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=1, vmax=5)

    for i, row in combined_sleep_df.iterrows():
        day_index = i  # x-axis position
        start = row["start_datetime"]
        end = row["end_datetime"]
        self_start = row["reported_sleep_time"]
        self_end = row["reported_wake_time"]
        
        # Plot estimated sleep duration and overlay with self-reported start and end datetime
        if np.isnan(row["sleep_quality_rating"]):
            bar_color = "grey"
        else:
            bar_color = cmap(norm(int(row["sleep_quality_rating"])))
        ax.bar(day_index, start-end, bottom=end, color=bar_color, edgecolor="black", width=0.5, alpha=0.9)
        ax.plot([day_index - 0.2, day_index + 0.2], [self_start, self_start], color="red", markersize=10, linestyle='-', linewidth=2)
        ax.plot([day_index - 0.2, day_index + 0.2], [self_end, self_end], color="red", markersize=10, linestyle='-', linewidth=2)
        ax.vlines(day_index, ymin=self_start, ymax=self_end, color="red", linestyle='-')
        # ax.barh(day_index, end - start, left=start, edgecolor='black', alpha=0.6)
        # ax.plot([self_start, self_end], [day_index, day_index], color='black', marker='|', markersize=15, linestyle='None')
        
    # X-axis as date
    ax.set_xticks(np.arange(len(combined_sleep_df)))
    ax.set_xticklabels(combined_sleep_df["date"])

    # Y-axis
    min_y = math.floor(combined_sleep_df[datetime_cols].min().min())
    max_y = math.ceil(combined_sleep_df[datetime_cols].max().max())
    ax.set_ylim(min_y, max_y)
    ax.set_yticks(range(min_y, max_y, 1))
    y_labels = []
    for time_diff in range(min_y, max_y, 1):
        if time_diff < 0:
            y_labels.append(f"{12 + time_diff} PM")
        elif time_diff == 0:
            y_labels.append("12 AM")
        else:
            y_labels.append(f"{time_diff} AM")
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Estimated sleep duration")
    ax.yaxis.grid()
    plt.gca().invert_yaxis()

    # Title and legend
    ax.set_title('Sleep Duration and Quality Rating Across A Week')
    legend_elements = [plt.Line2D([0], [0], color="red", lw=2, label="Self-reported sleep duration")]
    legend_elements.append(plt.Line2D([], [], color="none", label=""))
    legend_elements.append(plt.Line2D([], [], color="none", label="Sleep quality rating"))
    legend_elements.append(mpatches.Patch(facecolor="grey", edgecolor="black", label="No rating"))
    for index, rating in enumerate(SLEEP_QUALITY_RATINGS):
        legend_elements.append(mpatches.Patch(facecolor=cmap(norm(index+1)), edgecolor="black", label=rating))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def visualize_day_contexts(user_id, date, activity_df, light_df, noise_df, app_usage_df, location_df):
    """
    Plots day-level distribution of contexts, including physical activity, ambient light and noise, and application usage in location clusters.
    """
    time_window = Window().orderBy("datetime")

    activity_df = activity_df.withColumn("prev_activity", F.lag(F.col("activity_type")).over(time_window))\
        .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
        .withColumn("new_group", (F.col("activity_type") != F.col("prev_activity")).cast("int"))\
        .withColumn("group_id", F.sum("new_group").over(time_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
        .groupBy("group_id", "activity_type")\
        .agg(F.min("datetime").alias("start_datetime"),\
                F.max("next_datetime").alias("end_datetime"))\
        .drop("group_id").sort("start_datetime")\
        .withColumn("duration", (F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime"))) / 60)\
        .withColumn("start_datetime", time_to_hours("start_datetime"))\
        .withColumn("end_datetime", time_to_hours("end_datetime"))\
        .filter(F.col("activity_type") != 5)


    # Based on the original paper of approxQuantile computation https://dl.acm.org/doi/10.1145/375663.375670
    # The optimal error is 1/(2*S), where S is the number of quantiles (i.e. 100 in the current context)
    min_light_lux = light_df.agg(F.min("mean_light_lux")).collect()[0][0]
    max_light_lux = light_df.agg(F.max("mean_light_lux")).collect()[0][0]
    light_discretizer = QuantileDiscretizer(numBuckets=4, inputCol="mean_light_lux", outputCol="light_bin")
    light_discretizer_model = light_discretizer.fit(light_df)
    light_df = light_discretizer_model.transform(light_df)
    light_df = light_df.withColumn("prev_light_bin", F.lag(F.col("light_bin")).over(time_window))\
        .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
        .withColumn("new_group", (F.col("light_bin") != F.col("prev_light_bin")).cast("int"))\
        .withColumn("group_id", F.sum("new_group").over(time_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
        .groupBy("group_id", "light_bin")\
        .agg(F.min("datetime").alias("start_datetime"),\
                F.max("next_datetime").alias("end_datetime"))\
        .drop("group_id").sort("start_datetime")\
        .withColumn("duration", (F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime"))) / 60)\
        .withColumn("start_datetime", time_to_hours("start_datetime"))\
        .withColumn("end_datetime", time_to_hours("end_datetime"))
    quantile_splits = light_discretizer_model.getSplits()
    light_ranges = [min_light_lux] + quantile_splits[1:len(quantile_splits)-1] + [max_light_lux]
    light_ranges = [round(light_range, 1) for light_range in light_ranges]


    min_decibels = noise_df.agg(F.min("mean_decibels")).collect()[0][0]
    max_decibels = noise_df.agg(F.max("mean_decibels")).collect()[0][0]
    noise_discretizer = QuantileDiscretizer(numBuckets=4, inputCol="mean_decibels", outputCol="noise_bin")
    noise_discretizer_model = noise_discretizer.fit(noise_df)
    noise_df = noise_discretizer_model.transform(noise_df)
    noise_df = noise_df.withColumn("prev_noise_bin", F.lag(F.col("noise_bin")).over(time_window))\
        .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
        .withColumn("new_group", (F.col("noise_bin") != F.col("prev_noise_bin")).cast("int"))\
        .withColumn("group_id", F.sum("new_group").over(time_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
        .groupBy("group_id", "noise_bin")\
        .agg(F.min("datetime").alias("start_datetime"),\
                F.max("next_datetime").alias("end_datetime"))\
        .drop("group_id").sort("start_datetime")\
        .withColumn("duration", (F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime"))) / 60)\
        .withColumn("start_datetime", time_to_hours("start_datetime"))\
        .withColumn("end_datetime", time_to_hours("end_datetime"))
    quantile_splits = noise_discretizer_model.getSplits()
    noise_ranges = [min_decibels] + quantile_splits[1:len(quantile_splits)-1] + [max_decibels]
    noise_ranges = [round(noise_range, 1) for noise_range in noise_ranges]


    # Only show the usage of 5 most frequently used apps
    app_usage_df = app_usage_df.withColumn("start_datetime", time_to_hours("start_datetime"))\
        .withColumn("end_datetime", time_to_hours("end_datetime"))\
        .filter((F.col("is_system_app") == 0) & (F.col("application_name") != "AWARE-Light"))\
        .sort("start_datetime")
    top_apps = app_usage_df.groupBy("application_name").agg(F.sum("usage_duration").alias("total_usage_duration"))\
        .withColumn("app_rank", F.row_number().over(Window().orderBy(F.col("total_usage_duration").desc())))\
        .withColumn("app_rank", F.col("app_rank")-1)
    max_rank = top_apps.agg(F.max("app_rank")).collect()[0][0]

    # Top 5 through 0-indexing
    if max_rank > 4:
        max_rank = 4
    top_app_usage_df = app_usage_df.join(top_apps.filter(F.col("app_rank") <= max_rank), "application_name")\
        .withColumn("duration", F.col("usage_duration")/60).sort("start_datetime")
    top_apps = list(np.array(top_app_usage_df.select("application_name", "app_rank").distinct()\
                .sort("app_rank").select("application_name").collect()).flatten())
  

    # Plotting
    _, ax = plt.subplots(figsize=(12, 8))
    plot_cols = ["start_datetime", "duration"]
    df_color_cols = ["activity_type", "light_bin", "noise_bin", "app_rank"]
    dfs = [activity_df, light_df, noise_df, top_app_usage_df]
    plot_dfs = [dfs[index].select(*plot_cols + [df_color_cols[index]]).toPandas() for index in range(len(dfs))]
    df_color_maps = [plt.get_cmap(cm) for cm in ["Blues", "YlOrBr", "BuPu", "Set2"]]
    df_max_bins = [len(ACTIVITY_NAMES)-1, len(light_ranges)-2, len(noise_ranges)-2, max_rank]
    df_color_scales = [mcolors.Normalize(vmin=0, vmax=v_max) for v_max in df_max_bins]

    legend_positions = [[0.05, 0.775], [0.05, 0.555], [0.05, 0.355], [0.3, 0.155]]
    for index, context_df in enumerate(plot_dfs):
        legend_elements = []
        for _, row in context_df.iterrows():
            ax.barh(index, row["duration"], left=row["start_datetime"], height=0.4,\
                    color=df_color_maps[index](df_color_scales[index](int(row[df_color_cols[index]]))))
        if index == 0:
            legend_title = "Activity state"
            legend_labels = ACTIVITY_NAMES
        elif index == 1:
            legend_title = "Ambient light (luminance)"
            legend_labels = [f"{light_ranges[i]} - {light_ranges[i+1]}" for i in range(len(light_ranges)-1)]
        elif index == 2:
            legend_title = "Ambient noise (decibels)"
            legend_labels = [f"{noise_ranges[i]} - {noise_ranges[i+1]}" for i in range(len(noise_ranges)-1)]
        elif index == 3:
            legend_title = "Most frequently used apps"
            legend_labels = top_apps
        
        # df_max_ranges are inclusive
        for color_index in range(df_max_bins[index]+1):
            # To avoid adding label for non-existing activity type
            if legend_labels[color_index] != "":
                legend_elements.append(mpatches.Patch(facecolor=df_color_maps[index](df_color_scales[index](color_index)),\
                                                    label=legend_labels[color_index]))
            
        if index < 3:
            legend = ax.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=legend_positions[index], ncol=5, title=legend_title)
            ax.add_artist(legend)
        else:
            plt.legend(handles=legend_elements, loc="upper left", bbox_to_anchor=legend_positions[index], ncol=5, title=legend_title)

    # location_transitions = location_df.withColumn("datetime", time_to_hours("location_datetime"))\
    #     .withColumn("prev_datetime", time_to_hours("prev_location_datetime")).toPandas()
    # for trans_index, location_transition in location_transitions.iterrows():
    #     if trans_index == 0:
    #         boundary_start = 0
    #         boundary_end = location_transition["datetime"]
    #     else:
    #         boundary_start = location_transition["prev_datetime"]
    #         boundary_end = location_transition["datetime"]
    #     ax.axvline(x=boundary_end, color="r", linestyle='--')
    #     midpoint = (boundary_start + boundary_end) / 2
    #     ax.text(midpoint, -0.5, f"Cluster #{location_transition['prev_cluster']}", ha="center", va="bottom")

    #     # Last cluster
    #     if trans_index == location_transitions.shape[0]-1:
    #         ax.text((24 - location_transition["datetime"])/2, -0.5, f"Cluster #{location_transition['cluster_id']}", ha="center", va="bottom")

    # Add labels and title
    plt.xlabel("Time of the day")
    plt.ylabel("Individual contexts")
    plt.title(f"Context Distribution of {user_id} on {date}")
    plt.grid(True)

    # Y-axis
    ax.set_ylim(-1, 4)
    ax.set_yticks(np.arange(4))
    ax.set_yticklabels(["Physical", "Light", "Noise", "App Use"])
    plt.gca().invert_yaxis()

    # X-axis
    ax.set_xlim(0, 24)
    ax.set_xticks(range(0, 24, 1))
    x_labels = [f"{hour}:00" for hour in range(24)]
    ax.set_xticklabels(x_labels, rotation=45)
    ax.xaxis.grid()

    plt.show()


@F.udf(StringType())
def label_app_category(user_id, app_name, is_system_app):
    """
    Retrieves pre-saved app categories and maps input app name to its category.
    Currently support 5 major categories:
    1. Work - work-related communication or tools (e.g., email, document reader)
    2. Social - social communication
    3. Entertainment - e.g., social media apps, games, music
    4. Utilities - other system apps (e.g., maps for navigation, note-taking, alarm)
    5. Others - other personal apps (may include education, finance, navigation, retail)
    """    
    with open(f"{user_id}_contexts.json") as f:
        contexts = json.load(f)
        app_categories = contexts["app_categories"]
        for category in app_categories:
            if app_name in app_categories[category]:
                return category
        if is_system_app == 1:
            return "utilities"
    return "others"

def compute_high_level_personal_contexts(user_id):
    """
    Computes and saves high-level personal contextual information that should be adaptive.
    1. Location clusters and labels
    2. Primary and secondary location clusters based on visits
    3. Primary WiFi devices, normalized based on overall min and max occurrence
    4. Primary Bluetooth devices, normalized based on overall min and max occurrence
    5. Application categories
    6. Threshold of dark environment based on ambient light distribution
    7. Threshold of quiet environment based on ambient noise distribution

    # NOTE:
    1. Other system applications are assumed to be in "utilities" category.

    References:
    1. https://www.sciencedirect.com/science/article/pii/004724849290046C - threshold for audio frequency

    # TODO:
    1. Estimate home based on sleep locations
    2. Compute primary contacts
    3. Automate labeling of location clusters and app categories
    """
    with open(f"{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)

    light_df = process_light_data(user_id)
    brightness_quartiles = light_df.approxQuantile("mean_light_lux", [0.25, 0.50], 0.01)
    contexts["dark_threshold"] = round(brightness_quartiles[1])

    noise_df = process_raw_noise_data(user_id)
    audio_quartiles = noise_df.approxQuantile("double_decibels", [0.25, 0.50], 0.01)
    # silent_threshold = round((audio_quartiles[1]-audio_quartiles[0]) / 2)
    contexts["silent_threshold"] = round(audio_quartiles[1])

    rms_quartiles = noise_df.approxQuantile("double_rms", [0.75], 0.01)
    contexts["rms_threshold"] = round(rms_quartiles[0])
    contexts["frequency_threshold"] = 60

    # Frequently seen WiFi devices with top 95% percentile occurrence
    wifi_df = process_wifi_data(user_id)\
        .withColumn("hour", F.col("hour").cast(IntegerType()))\
        .withColumn("minute", F.col("minute").cast(IntegerType()))\
        .withColumn("wifi_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    wifi_device_occurrence = wifi_df.groupBy("ssid").agg(F.count("wifi_datetime").alias("device_occurrence"))
    wifi_occurrence_quartiles = wifi_device_occurrence.withColumn("device_occurrence", F.col("device_occurrence").cast(DoubleType()))\
        .approxQuantile("device_occurrence", [0.95], 0.01)
    
    # Min-max normalization of device occurrence
    max_occurrence = wifi_device_occurrence.agg(F.max("device_occurrence")).collect()[0][0]
    min_occurrence = wifi_device_occurrence.agg(F.min("device_occurrence")).collect()[0][0]
    wifi_device_occurrence = wifi_device_occurrence.filter(F.col("device_occurrence") >= wifi_occurrence_quartiles[0])\
        .withColumn("weighted_occurrence", (F.col("device_occurrence")-F.lit(min_occurrence))/(max_occurrence-min_occurrence))\
        .drop("device_occurrence").sort("weighted_occurrence", ascending=False)
    contexts["primary_wifi_devices"] = wifi_device_occurrence.rdd.map(lambda row: row.asDict()).collect()

    # Frequently seen Bluetooth devices with top 95% percentile occurrence
    bluetooth_df = process_bluetooth_data(user_id)\
        .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
        .withColumn("bluetooth_datetime", udf_datetime_from_timestamp("timestamp"))
    bluetooth_device_occurrence = bluetooth_df.groupBy("bt_name", "bt_address").agg(F.count("bluetooth_datetime").alias("device_occurrence"))
    bluetooth_occurrence_quartiles = bluetooth_device_occurrence.withColumn("device_occurrence", F.col("device_occurrence").cast(DoubleType()))\
        .approxQuantile("device_occurrence", [0.95], 0.01)
    
    # Min-max normalization of device occurrence
    max_occurrence = bluetooth_device_occurrence.agg(F.max("device_occurrence")).collect()[0][0]
    min_occurrence = bluetooth_device_occurrence.agg(F.min("device_occurrence")).collect()[0][0]
    bluetooth_device_occurrence = bluetooth_device_occurrence.filter(F.col("device_occurrence") >= bluetooth_occurrence_quartiles[0])\
        .withColumn("weighted_occurrence", (F.col("device_occurrence")-F.lit(min_occurrence))/(max_occurrence-min_occurrence))\
        .drop("device_occurrence").sort("weighted_occurrence", ascending=False)
    contexts["primary_bluetooth_devices"] = bluetooth_device_occurrence.rdd.map(lambda row: row.asDict()).collect()

    # Location clusters
    location_df = complement_location_data(user_id)
    unique_ssid_clusters = location_df.select("datetime", "cluster_id", "ssid").distinct().dropna()\
        .groupBy("cluster_id", "ssid").agg(F.count("datetime").alias("WiFi_occurrence"))
    # Major location clusters based on visit datetimes
    cluster_visits = location_df.select("datetime", "cluster_id").distinct()\
        .groupBy("cluster_id").agg(F.count("datetime").alias("cluster_visits"))
    cluster_visits = cluster_visits.filter(F.col("cluster_id") != -1)\
        .union(cluster_visits.filter(F.col("cluster_id") == -1)).toPandas()
    valid_clusters = cluster_visits[cluster_visits["cluster_id"] != -1]\
        .sort_values(by=["cluster_visits"], ascending=False)
    cluster_limit = 2
    if len(valid_clusters) < cluster_limit:
        cluster_limit = len(valid_clusters)
    major_clusters = list(valid_clusters["cluster_id"][:cluster_limit])
    
    # Cluster importance of ssid: min-max normalized based on device occurrence in each cluster
    ssid_cluster_importance = unique_ssid_clusters.groupBy("cluster_id")\
        .agg(F.max("WiFi_occurrence").alias("max_WiFi_occurrence"),\
             F.min("WiFi_occurrence").alias("min_WiFi_occurrence"),\
             F.sum("WiFi_occurrence").alias("total_WiFi_occurrence"))
    unique_ssid_clusters = unique_ssid_clusters.join(ssid_cluster_importance, "cluster_id")\
        .withColumn("weighted_occurrence", ((F.col("WiFi_occurrence")-F.col("min_WiFi_occurrence"))/\
                                            (F.col("max_WiFi_occurrence")-F.col("min_WiFi_occurrence"))))\
        .sort("cluster_id", "weighted_occurrence", ascending=[True, False])
    unique_ssid_clusters = unique_ssid_clusters.filter(F.col("cluster_id") >= 0)\
        .union(unique_ssid_clusters.filter(F.col("cluster_id") < 0))
    

    # For each cluster: keep track of total WiFi device occurrence, max occurrence, min occurrence,
    # and ssids with occurence min-max normalized based on cluster-specific device occurrence
    cluster_ssids_dict = {}
    for _, cluster_row in cluster_visits.iterrows():
        cur_cluster_dict = {"is_primary": False,\
                            "is_secondary": False,\
                            "visits": int(cluster_row["cluster_visits"])}
        try:
            index = major_clusters.index(int(cluster_row["cluster_id"]))
            if index == 0:
                cur_cluster_dict["is_primary"] = True
            else:
                cur_cluster_dict["is_secondary"] = True
        except ValueError:
            pass
        cur_ssids = unique_ssid_clusters.filter(F.col("cluster_id") == int(cluster_row["cluster_id"]))\
            .rdd.map(lambda row: row.asDict()).collect()
        for col in ["total_WiFi_occurrence", "max_WiFi_occurrence", "min_WiFi_occurrence"]:
            cur_cluster_dict[col] = cur_ssids[0][col]
        ssid_occurrences_dict = {}
        for row in cur_ssids:
            ssid_occurrences_dict[row["ssid"]] = row["weighted_occurrence"]
        cur_cluster_dict["ssids"] = ssid_occurrences_dict
        cluster_ssids_dict[int(cluster_row["cluster_id"])] = cur_cluster_dict
    contexts["location_clusters"] = cluster_ssids_dict

    with open(f"{user_id}_contexts.json", "w") as f:
        json.dump(contexts, f)

def get_epoch_from_hour(hour):
    """
    Returns epoch of the day based on input hour.
    4 partitions of the same duration:
    1. Morning - 6AM to 12PM
    2. Afternoon - 12PM to 6PM
    3. Evening - 6PM to 12AM
    4. Night - 12AM to 6AM
    """
    epochs = list(TIME_EPOCHS.keys())
    for index, epoch in enumerate(epochs[:-1]):
        time_range = TIME_EPOCHS[epoch]
        if time_range["min"] <= hour < time_range["max"]:
            return index
    return 3

def extract_cluster_contexts(user_id, cluster_time_range, activity_df, light_df, noise_df, app_usage_df, bluetooth_df, location_df):
    """
    Extracts features for each cluster during each epoch of the day:
    1. Total time duration
    2. Total duration of each activity state, normalized by the duration of stay at the cluster
    3. Total usage of each application category, normalized by the duration of stay at the cluster and unlock duration
    4. Min, max, mean, standard dev of ambient light
    5. Min, max, mean, standard dev of ambient noise
    6. Total duration in dark environment, normalized by the duration of stay at the cluster
    7. Total duration in quiet environment, normalized by the duration of stay at the cluster
    8. Number of unique WiFi and Bluetooth devices
    9. Occurrence of primary WiFi and Bluetooth devices weighted by overall significance, normalized by the number of entries at the cluster
    
    # TODO:
    1. Number of calls and messaging
    2: Occurrence/normalized occurrence of primary contacts

    # References:
    1. https://dl.acm.org/doi/10.1145/3191775 (generate features based on DSM criteria):
        a. Phone lock/unlock events normalized by duration of stay at the location
    
    2. https://ieeexplore.ieee.org/document/9230984:
        a. % of time spent in a specific activity state
        b. Transition time (time spent moving)
    """
    # Retrieves pre-saved contexts
    with open(f"{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)

    cluster_features = {}
    # Time range at the current cluster
    print(f"Time range in this cluster: {cluster_time_range[0]} - {cluster_time_range[1]}")
    total_time_spent = (cluster_time_range[1] - cluster_time_range[0]).total_seconds()
    cluster_features["stay_duration"] = total_time_spent

    wifi_df = location_df.filter(F.col("ssid").isNotNull())
    if wifi_df.count() > 0:
        # Number of unique WiFi devices
        cluster_wifi_entry_count = wifi_df.count()
        cluster_features["unique_wifi_device"] = wifi_df.select("ssid").distinct().count()

        # Retrieves primary WiFi devices and their weighted occurrence
        primary_wifi_devices = [item["ssid"] for item in contexts["primary_wifi_devices"]]
        primary_wifi_weight = [item["weighted_occurrence"] for item in contexts["primary_wifi_devices"]]
        primary_wifi_df = wifi_df.filter(F.col("ssid").isin(*primary_wifi_devices))
        
        if primary_wifi_df.count() > 0:
            # Occurrence of each WiFi device weighted by total WiFi entries in the current cluster
            primary_wifi_occurrence = primary_wifi_df.groupBy("ssid")\
                .agg(F.count("location_datetime").alias("device_occurrence"))\
                .sort("device_occurrence", ascending=False).toPandas()
            # Compute overall weighted occurrence of primary WiFi devices
            wifi_weighted_occurrence = 0
            for _, row in primary_wifi_occurrence.iterrows():
                list_index = primary_wifi_devices.index(row["ssid"])
                wifi_weighted_occurrence += row["device_occurrence"] * primary_wifi_weight[list_index]
            cluster_features["major_wifi_weighted_occurrence"] = wifi_weighted_occurrence
            cluster_features["major_wifi_normalized_weighted_occurrence"] = wifi_weighted_occurrence/cluster_wifi_entry_count
        else:
            cluster_features["major_wifi_weighted_occurrence"] = 0
            cluster_features["major_wifi_normalized_weighted_occurrence"] = 0
    else:
        for col in ["unique_wifi_device", "major_wifi_weighted_occurrence", "major_wifi_normalized_weighted_occurrence"]:
            cluster_features[col] = 0

    if bluetooth_df.count() > 0:
        cluster_bt_entry_count = bluetooth_df.count()
        cluster_features["unique_bluetooth_device"] = bluetooth_df.select("bt_address").distinct().count()

        # Retrieves primary Bluetooth devices and their weighted occurrence
        primary_bt_devices = [item["bt_address"] for item in contexts["primary_bluetooth_devices"]]
        primary_bt_weight = [item["weighted_occurrence"] for item in contexts["primary_bluetooth_devices"]]

        # Occurrence of each Bluetooth device weighted by total Bluetooth entries in the current cluster
        primary_bt_df = bluetooth_df.filter(F.col("bt_address").isin(*primary_bt_devices))
        if primary_bt_df.count() > 0:
            primary_bt_occurrence = primary_bt_df.groupBy("bt_name", "bt_address")\
                .agg(F.count("bt_datetime").alias("device_occurrence"))\
                .sort("device_occurrence", ascending=False).toPandas()
            # Compute overall weighted occurrence of primary WiFi devices
            bt_weighted_occurrence = 0
            for _, row in primary_bt_occurrence.iterrows():
                list_index = primary_bt_devices.index(row["bt_address"])
                bt_weighted_occurrence += row["device_occurrence"] * primary_bt_weight[list_index]
            cluster_features["major_bluetooth_weighted_occurrence"] = bt_weighted_occurrence
            cluster_features["major_bluetooth_normalized_weighted_occurrence"] = bt_weighted_occurrence/cluster_bt_entry_count
        else:
            cluster_features["major_bluetooth_weighted_occurrence"] = 0
            cluster_features["major_bluetooth_normalized_weighted_occurrence"] = 0
    else:
        for col in ["unique_bluetooth_device", "major_bluetooth_weighted_occurrence", "major_bluetooth_normalized_weighted_occurrence"]:
            cluster_features[col] = 0

    activities = ["still", "in_vehicle", "on_bicycle", "on_foot", "tilting", "walking", "running"]
    if activity_df.count() > 0:
        # Total duration and normalized duration (by time spent at the cluster) for each activity state
        activity_duration = activity_df.groupBy("activity_name")\
            .agg(F.sum("duration").alias("total_activity_duration"))\
            .withColumn("normalized_activity_duration", F.col("total_activity_duration")/total_time_spent)\
            .sort("total_activity_duration", ascending=False).toPandas()
        activity_names = activity_duration["activity_name"].to_list()
        durations = activity_duration["normalized_activity_duration"].to_list()
        for act in activities:
            if act in activity_names:
                cluster_features[f"{act}_normalized_duration"] = durations[activity_names.index(act)]
            else:
                cluster_features[f"{act}_normalized_duration"] = 0
    else:
        for act in activities:
            cluster_features[f"{act}_normalized_duration"] = 0

    # Total duration of each application category and normalized by time spent at the cluster and unlock duration
    app_categories = list(contexts["app_categories"].keys()) + ["utilities", "others"]
    if app_usage_df.count() > 0:
        total_phone_use_duration = app_usage_df.agg(F.sum("double_elapsed_device_on")).collect()[0][0]/1000
        cluster_features["normalized_phone_use_duration"] = total_phone_use_duration/total_time_spent
        app_usage_duration = app_usage_df.groupBy("category")\
            .agg((F.sum("usage_duration")/60).alias("total_usage_duration"))\
            .withColumn("cluster_normalized_usage_duration", F.col("total_usage_duration")/total_time_spent)\
            .withColumn("phone_use_normalized_usage_duration", F.col("total_usage_duration")/total_phone_use_duration)\
            .sort("total_usage_duration", ascending=False).toPandas()
        
        cur_apps = app_usage_duration["category"].to_list()
        cluster_normalized_app_durations = app_usage_duration["cluster_normalized_usage_duration"].to_list()
        normalized_app_durations = app_usage_duration["phone_use_normalized_usage_duration"].to_list()
        for category in app_categories:
            if category in cur_apps:
                cluster_features[f"{category}_app_cluster_normalized_duration"] = cluster_normalized_app_durations[cur_apps.index(category)]
                cluster_features[f"{category}_app_phone_use_normalized_duration"] = normalized_app_durations[cur_apps.index(category)]
            else:
                cluster_features[f"{category}_app_cluster_normalized_duration"] = 0
                cluster_features[f"{category}_app_phone_use_normalized_duration"] = 0
    else:
        for col in ["normalized_phone_use_duration"] + [f"{cat}_app_cluster_normalized_duration" for cat in app_categories] +\
            [f"{cat}_app_phone_use_normalized_duration" for cat in app_categories]:
            cluster_features[col] = 0

    stat_functions = [F.min, F.max, F.mean, F.stddev]
    stat_names = ["min", "max", "mean", "std"]
    if light_df.count() > 0:
        # Min, max, mean, and standard deviation of ambient light
        agg_expressions = [stat_functions[index](f"{stat_names[index]}_light_lux").alias(f"{stat_names[index]}_light_lux") for index in range(len(stat_functions))]
        ambient_light = light_df.agg(*agg_expressions).collect()[0]
        for stat_index, stat in enumerate(stat_names):
            cluster_features[f"{stat}_luminance"] = ambient_light[stat_index]
        
        dark_df = light_df.filter(F.col("is_dark") == 1)
        if dark_df.count() > 0:
            cluster_features["normalized_dark_duration"] = dark_df.agg(F.sum("duration")).collect()[0][0]/total_time_spent
        else:
            cluster_features["normalized_dark_duration"] = 0
    else:
        for col in [f"{stat}_luminance" for stat in stat_names] + ["normalized_dark_duration"]:
            cluster_features[col] = 0

    if noise_df.count() > 0:
        # Min, max, mean, and standard deviation of ambient noise
        agg_expressions = [stat_functions[index](f"{stat_names[index]}_decibels").alias(f"{stat_names[index]}_decibels") for index in range(len(stat_functions))]
        ambient_noise = noise_df.agg(*agg_expressions).collect()[0]
        for stat_index, stat in enumerate(stat_names):
            cluster_features[f"{stat}_decibels"] = ambient_noise[stat_index]
        
        quiet_df = noise_df.filter(F.col("is_quiet") == 1)
        if quiet_df.count() > 0:
            cluster_features["normalized_silent_duration"] = quiet_df.agg(F.sum("duration")).collect()[0][0]/total_time_spent
        else:
            cluster_features["normalized_silent_duration"] = 0
    else:
        for col in [f"{stat}_decibels" for stat in stat_names] + ["normalized_silent_duration"]:
            cluster_features[col] = 0

    return cluster_features

def features_vs_mood(user_id):
    """
    Include contextual information with features and investigate their correlation with labels, including sleep quality, sleep duration etc.
    Contextual information (referenced from https://link.springer.com/article/10.1186/s40537-019-0219-y)
    1. Temporal context: hour/epoch of the day, day of the week, weekday vs weekend
    2. Spatial context: cluster ID
    3. Social context: frequently seen Bluetooth devices and contacted individuals
    
    Assumptions:
    1. Mood is prompted in the evening
    """
    mood_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_esms.csv")\
        .withColumn("esm_id", udf_extract_esm_id("esm_json"))\
        .filter(F.col("esm_id") == 1)\
        .groupBy("date").pivot("esm_id").agg(F.first("esm_user_answer"))\
        .withColumnRenamed(str(1), "reported_mood").sort("date")\
        .withColumn("reported_mood", F.col("reported_mood").cast(IntegerType()))\
        .select("date", "reported_mood").toPandas()
    
    dates = mood_df["date"].to_list()
    reported_mood = mood_df["reported_mood"].to_list()

    daily_features = []
    for index, date in enumerate(dates):
        features = extract_daily_features(user_id, date)
        # features["reported_mood"] = reported_mood[index]
        daily_features.append(features)
    
    features_mood_df = pd.DataFrame.from_dict(daily_features)
    corr_df = pd.DataFrame()
    feat1s = []
    feat2s = []
    corrs = []
    p_values = []
    for feat1 in features_mood_df.columns[1:]:
        for feat2 in features_mood_df.columns[1:]:
            if feat1 != feat2:
                feat1s.append(feat1)
                feat2s.append(feat2)
                corr, p_value = pearsonr(features_mood_df[feat1], features_mood_df[feat2])
                corrs.append(corr)
                p_values.append(p_value)
    corr_df['Feature_1'] = feat1s
    corr_df['Feature_2'] = feat2s
    corr_df['Correlation'] = corrs
    corr_df['p_value'] = p_values
    print(corr_df)
    

if __name__ == "__main__":
    # Spark installation: https://phoenixnap.com/kb/install-spark-on-windows-10
    # -- Spark configuration --
    os.environ["HADOOP_HOME"] = "C:/hadoop"
    findspark.init()

    # Find out hadoop version using spark._jvm.org.apache.hadoop.util.VersionInfo.getVersion()
    spark = SparkSession.builder.master("local[*]")\
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")\
        .config("spark.jars.packages", "org.apache.hadoop:hadoop-aws:3.3.2")\
        .config("spark.driver.memory", "50g")\
        .config("spark.executor.memory", "50g")\
        .config("spark.memory.offHeap.size", "50g")\
        .config("spark.memory.offHeap.enabled", "true")\
        .config("spark.driver.maxResultSize", "10G")\
        .config("spark.sql.session.timeZone", "Australia/Melbourne")\
        .getOrCreate()
    sc = spark.sparkContext

    ALL_TABLES = ["applications_foreground", "applications_history", "applications_notifications",\
                  "bluetooth", "light", "locations", "calls", "messages", "screen", "wifi", "esms",\
                    "plugin_ambient_noise", "plugin_device_usage", "plugin_google_activity_recognition"]
    DATA_FOLDER = "user_data"
    TIMEZONE = pytz.timezone("Australia/Melbourne")
    ACTIVITY_NAMES = ["in_vehicle", "on_bicycle", "on_foot", "still", "unknown", "tilting", "", "walking", "running"]
    ACTIVITY_PRIORITIES = ["in_vehicle", "on_bicycle", "running", "walking", "on_foot", "tilting", "still", "unknown", ""]
    SLEEP_QUALITY_RATINGS = ["Poor", "Fair", "Average", "Good", "Excellent"]
    # "min"s are inclusive, "max"s are exclusive
    TIME_EPOCHS = {
        "night": {
            "min": 0,
            "max": 6
        },
        "morning": {
            "min": 6,
            "max": 12
        },
        "afternoon": {
            "min": 12,
            "max": 18
        },
        "evening": {
            "min": 18,
            "max": 0
        }
    }

    udf_datetime_from_timestamp = F.udf(lambda x: datetime.fromtimestamp(x/1000, TIMEZONE), TimestampType())
    udf_generate_datetime = F.udf(lambda d, h, m: TIMEZONE.localize(datetime.strptime(f"{d} {h:02d}:{m:02d}", "%Y-%m-%d %H:%M")), TimestampType())
    udf_get_date_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).date().strftime("%Y-%m-%d"), StringType())
    udf_get_hour_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).time().hour, IntegerType())
    udf_get_minute_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).time().minute, IntegerType())
    udf_string_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M"), StringType())
    udf_unique_wifi_list = F.udf(lambda x: ", ".join(sorted(list(set(x.split(", "))))), StringType())
    udf_round_datetime_to_nearest_minute = F.udf(lambda dt, n: round_time_to_nearest_n_minutes(dt, n), TimestampType())
    udf_map_activity_name_to_type = F.udf(lambda x: ACTIVITY_NAMES.index(x), IntegerType())
    activity_confidence_schema = ArrayType(StructType([
            StructField("activity", StringType(), True),
            StructField("confidence", IntegerType(), True)
        ]))
    udf_extract_max_confidence_activity = F.udf(lambda x: extract_max_confidence_activity(x), activity_confidence_schema)
    udf_extract_esm_id = F.udf(lambda x: json.loads(x.replace("'", "\""))["id"], StringType())
    udf_hours_relative_to_midnight = F.udf(time_to_midnight_hours, FloatType())

    # user_identifier = "pixel3"
    user_identifier = "S3"

    # -- NOTE: Only execute this block when db connection is required --
    # with open("database_config.json", 'r') as file:
    #     db_config = json.load(file)

    # db_connection = mysql.connector.connect(**db_config)
    # db_cursor = db_connection.cursor(buffered=True)

    # # Export data from database to local csv
    # export_user_data(db_cursor, user_identifier)

    # # Remove entries corresponding to invalid devices from all tables
    # # valid_devices = get_valid_device_id(db_cursor)
    # # for table in ALL_TABLES + ["sensor_bluetooth", "sensor_light", "sensor_wifi", "aware_log", "aware_studies"]:
    # #     delete_invalid_device_entries(db_cursor, table, tuple(valid_devices))
    # #     db_connection.commit()

    # # Remove entries corresponding to a single user_id from all tables
    # # device_id = get_user_device(db_cursor, "S1")[0][0]
    # # for table in ALL_TABLES + ["sensor_bluetooth", "sensor_light", "sensor_wifi", "aware_log", "aware_studies", "aware_device"]:
    # #     delete_single_entry(db_cursor, table, device_id)
    # #     db_connection.commit()
    

    # db_cursor.close()
    # db_connection.close()
    # -- End of block --

    # -- NOTE: This block of functions execute the extraction and early processing of sensor data into dataframes
    # process_light_data(user_identifier)
    # process_activity_data(user_identifier)
    # process_raw_noise_data(user_identifier)
    # process_screen_data(user_identifier)
    # process_application_usage_data(user_identifier)
    # process_bluetooth_data(user_identifier)
    # process_location_data(user_identifier)
    # process_wifi_data(user_identifier)

    # Recomputes and saves high-level contextual information (some are hardcoded based on individual's inputs)
    # compute_high_level_personal_contexts(user_identifier)
    # NOTE: Must be executed after computing high-level contexts since conversation estimate depends on the audio thresholds.
    # process_noise_data_with_conv_estimate(user_identifier)
    # -- End of block
    

    # -- NOTE: This block of functions combine multiple sensor information to generate interpretation
    # sleep_df = estimate_sleep(user_identifier)
    # location_df = process_location_data(user_identifier)
    # cluster_df = cluster_locations(user_identifier, location_df, "double_latitude", "double_longitude")
    # complement_location_data(user_identifier)
    # map_overview_estimated_sleep_duration_to_sleep_ema(user_identifier, [3, 4, 1])
    # extract_daily_features(user_identifier)
    # features_vs_mood(user_identifier)
    # -- End of block