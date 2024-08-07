"""
Author: Lin Sze Khoo
Created on: 24/01/2024
Last modified on: 04/08/2024
"""
import collections
import json
import joblib
import math
import os
import haversine as hs
import textwrap
import time
from datetime import datetime, timezone, timedelta
from itertools import product

import findspark
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import mysql.connector
import numpy as np
import pandas as pd
import pytz
import plotly.express as px 
import seaborn as sns
from scipy.stats import pearsonr
from kneed import KneeLocator
from pyspark.ml.feature import QuantileDiscretizer
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType, StructField, 
                               StructType, TimestampType, BooleanType, ArrayType, DoubleType)
from pyspark.sql.window import Window
from scipy.interpolate import interp1d
from scipy.stats import loguniform
from sklearn.cluster import DBSCAN
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import silhouette_score, f1_score, roc_auc_score, make_scorer, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error, r2_score, silhouette_samples
from sklearn.model_selection import RandomizedSearchCV, train_test_split, GridSearchCV, cross_validate, KFold, RepeatedKFold, cross_val_score, StratifiedKFold
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
    csv_filename = f"{DATA_FOLDER}/{user_id}_light.csv"
    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        light_df = spark.read.option("header", True).csv(csv_filename)
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
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def process_raw_noise_data(user_id):
    """
    Ambient audio data without aggregation.
    Used to get estimation of quartiles from the entire distribution for thresholding.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_raw_noise.parquet"
    csv_filename = f"{DATA_FOLDER}/{user_id}_plugin_ambient_noise.csv"
    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        noise_df = spark.read.option("header", True).csv(csv_filename)
        time_cols = ["date", "hour", "minute"]
        noise_df = noise_df.withColumn("double_frequency", F.col("double_frequency").cast(FloatType()))\
            .withColumn("double_decibels", F.col("double_decibels").cast(FloatType()))\
            .withColumn("double_rms", F.col("double_rms").cast(FloatType()))\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        noise_df.write.parquet(parquet_filename)
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def process_noise_data_with_conv_estimate(user_id):
    """
    Ambient audio data with rough estimation of nearby conversation based on frequency, decibels, and rms thresholds.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_noise.parquet"
    if not os.path.exists(parquet_filename):
        raw_noise_df = process_raw_noise_data(user_id)
        if raw_noise_df is not None:
            raw_noise_df = raw_noise_df.withColumn("double_frequency", F.col("double_frequency").cast(FloatType()))\
            .withColumn("double_decibels", F.col("double_decibels").cast(FloatType()))\
            .withColumn("double_rms", F.col("double_rms").cast(FloatType()))\
            .withColumn("second_timestamp", F.round(F.col("timestamp")/1000).cast(TimestampType()))\
            .withColumn("second_timestamp", udf_round_datetime_to_nearest_minute(F.col("second_timestamp"), F.lit(1)))
        
            with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
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

    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

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
    csv_filename = f"{DATA_FOLDER}/{user_id}_plugin_google_activity_recognition.csv"

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        activity_df = spark.read.option("header", True).csv(csv_filename)
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

    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def process_screen_data(user_id):
    """
    (Event-based)
    Screen status data: currently does not attempt to identify missing data
    NOTE: Device usage plugin readily provides usage (screen unlocked) and non-usage duration (screen off)
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_screen.parquet"
    csv_filename = f"{DATA_FOLDER}/{user_id}_screen.csv"

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        screen_df = spark.read.option("header", True).csv(csv_filename)
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
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def process_phone_usage_data(user_id):
    """
    Two versions of obtaining active phone usage:
    1. V1: retrieves directly from device usage plugin.
    2. V2: computes manually from screen status as duration from state 3 (unlocked) to 2 (locked)

    NOTE:
    1. V1 may have overlaps in durations since device was on/off due to race condition during data logging.
    2. Duration is in milliseconds
    """
    # V1
    # phone_usage_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")
    # phone_usage_df = phone_usage_df.filter(F.col("double_elapsed_device_on") > 0)\
    #     .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
    #     .withColumn("start_timestamp", F.col("timestamp") - F.col("double_elapsed_device_on"))\
    #     .withColumnRenamed("timestamp", "end_timestamp")\
    #     .withColumn("duration", round(F.col("double_elapsed_device_on")/1000))

    # V2: Manual computation through screen states that consider duration between state 3 and state 2
    # NOTE: This method does not consider that certain apps might still be used in locked state (value 2) 

    # Compute start and end usage event where state 3 indicates the start timepoint and the first subsequent state 2 indicates the end timepoint.
    screen_df = process_screen_data(user_id)
    if screen_df is not None:
        prev_consecutive_time_window = Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, -1)
        phone_usage_df = screen_df.withColumn("start_usage", F.when(F.col("screen_status") == 3, F.col("timestamp")))\
            .withColumn("end_usage", F.when(F.col("screen_status") == 2, F.col("timestamp")))\
            .withColumn("last_start_usage", F.last("start_usage", True).over(prev_consecutive_time_window))\
            .withColumn("last_end_usage", F.last("end_usage", True).over(prev_consecutive_time_window))\
            .withColumn("start_usage", F.when(F.col("start_usage").isNotNull(), F.col("start_usage"))\
                .otherwise(F.when((F.col("end_usage").isNotNull()) | (F.col("last_start_usage") > F.col("last_end_usage")), F.col("last_start_usage"))))\
            .filter(F.col("last_end_usage") <= F.col("start_usage")).sort("timestamp")

        # Aggregate consecutive start events since the last end event without any new end event in between
        phone_in_use_df = phone_usage_df\
            .groupBy("last_end_usage").agg(F.min("start_usage").alias("start_timestamp"),\
                        F.min("end_usage").alias("end_timestamp"))\
            .withColumn("duration", F.col("end_timestamp")-F.col("start_timestamp"))\
            .select("start_timestamp", "end_timestamp", "duration").dropDuplicates().dropna().sort("start_timestamp")
        return phone_in_use_df
    
    return None

def process_application_usage_data(user_id):
    """
    (Event-based)
    Application usage: does not attempt to identify missing data
    1. Combine information from foreground applications and device usage plugin.
    2. Include: Non-system applications running in the foreground when device screen is on.

    Prerequisites:
    1. Screen data is available.
    2. Phone usage df has been created.
    3. Applications foreground data is available.

    NOTE Assumptions:
    1. Assume that the usage duration of an application is the duration between the current usage timestamp and
    that of the subsequent foreground application or the end of device usage duration.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_app_usage.parquet"
    csv_filename = f"{DATA_FOLDER}/{user_id}_applications_foreground.csv"
    phone_usage_df = process_phone_usage_data(user_id)

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename) and phone_usage_df is not None:
        # Might interleave with system process for rendering UI
        substring_regex_pattern = "|".join(["systemui", "launcher", "biometrics"])
        app_usage_df = spark.read.option("header", True).csv(csv_filename)\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("is_system_app", F.col("is_system_app").cast(IntegerType()))\
            .filter((F.col("is_system_app") == 0) | ~(F.expr(f"package_name rlike '{substring_regex_pattern}'") |\
                    (F.lower(F.col("application_name")).contains("launcher"))))

        # Obtain intersect of phone screen in use and having applications running in foreground
        time_window = Window.orderBy("timestamp")
        in_use_app_df = app_usage_df.join(phone_usage_df, (phone_usage_df["start_timestamp"] <= app_usage_df["timestamp"]) &\
                                        (phone_usage_df["end_timestamp"] >= app_usage_df["timestamp"]))\
            .select("application_name", "timestamp", "start_timestamp", "end_timestamp", "is_system_app", "duration")\
            .dropDuplicates()
        in_use_app_df = in_use_app_df.withColumn("next_timestamp", F.lead(F.col("timestamp")).over(time_window))\
            .withColumn("next_timestamp", F.when((F.col("next_timestamp") <= F.col("end_timestamp")), F.col("next_timestamp")).otherwise(F.col("end_timestamp")))\
            .withColumn("usage_duration", (F.col("next_timestamp") - F.col("timestamp"))/1000)\
            .withColumn("duration", F.col("duration")/1000)\
            .filter(F.col("usage_duration") > 0)\
            .withColumnRenamed("start_timestamp", "start_phone_use_timestamp")\
            .withColumnRenamed("end_timestamp", "end_phone_use_timestamp")\
            .withColumnRenamed("timestamp", "start_timestamp")\
            .withColumnRenamed("next_timestamp", "end_timestamp").sort("start_timestamp")\
            .withColumn("category", label_app_category(F.lit(user_id), F.col("application_name"), F.col("is_system_app")))
    
    #     in_use_app_df.write.parquet(parquet_filename)
        return in_use_app_df
    return None
    
def process_bluetooth_data(user_id):
    """
    (Event-based)
    Bluetooth data: does not attempt to identify missing data
    1. RSSI values indicate signal strength: -100 dBm (weaker) to 0 dBm (strongest)
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_bluetooth.parquet"
    csv_filename = f"{DATA_FOLDER}/{user_id}_bluetooth.csv"

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        bluetooth_df = spark.read.option("header", True).csv(csv_filename)

        # Fill in null names for those with the same address
        known_bt_names = bluetooth_df.select("bt_address", "bt_name").distinct().dropna()\
            .withColumnRenamed("bt_name", "temp_bt_name")
        bluetooth_df = bluetooth_df.join(known_bt_names, "bt_address", "left")\
            .withColumn("bt_name", F.coalesce(F.col("bt_name"), F.col("temp_bt_name")))
        # Filter off instances where Bluetooth was disabled
        bluetooth_df = bluetooth_df.withColumn("bt_rssi", F.col("bt_rssi").cast(IntegerType()))\
            .filter(F.col("bt_rssi") != 0)
        bluetooth_df.write.parquet(parquet_filename)
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def optimize_cluster(loc_df):
    """
    Optimizes epsilon and mininum number of points based on silhouette score of clusters from DBSCAN clustering.
    References:
    1. https://machinelearningknowledge.ai/tutorial-for-dbscan-clustering-in-python-sklearn/
    2. Distance metrics: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html#sklearn.metrics.pairwise.distance_metrics
    TODO: Enforce constraints on clusters based on still activity
    """
    min_points_multiple = 5
    min_points_decrement_factor = 0.75
    min_samples = math.floor(
        len(loc_df) / min_points_multiple) * min_points_multiple
    best_epsilon = 0
    best_min_samples = 0
    best_silhouette_score = 0
    temp_epsilon = 0
    temp_min_samples = 0

    # # Organize into lists of coordinates that should belong in the same cluster 
    # known_clusters = known_cluster_df["coordinates"].to_list()
    # for index, cluster in enumerate(known_clusters):
    #     cluster_coords = [coord.split(', ') for coord in cluster]
    #     cluster_coords = [(float(coord[0]), float(coord[1])) for coord in cluster_coords]
    #     known_clusters[index] = cluster_coords

    # Optimize epsilon value
    while min_samples > 0:
        # Distance from each point to its closest neighbour
        nb_learner = NearestNeighbors(n_neighbors=min_samples)
        nearest_neighbours = nb_learner.fit(loc_df)
        distances, _ = nearest_neighbours.kneighbors(loc_df)
        distances = np.sort(distances[:, min_samples-1], axis=0)

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
            dbscan_cluster = DBSCAN(eps=opt_eps, min_samples=min_samples, metric="haversine")
            dbscan_cluster.fit(loc_df)
            cluster_labels = dbscan_cluster.labels_

            # # Get a mapping of coords to respecitve labels
            # coord_to_label = dict(zip(map(tuple, loc_df), cluster_labels))
            # # Get the originally assigned label and compare with all other coordinates that this coord has been paired with
            # for assigned_coord in coord_to_label.keys():
            #     coord_label = []
            #     # Look for this coord in known clusters and keep the labels assigned to the others
            #     for cluster_coords in known_clusters:
            #         if len(cluster_coords) > 1 and assigned_coord in cluster_coords:
            #         # if assigned_coord in cluster_coords:
            #             for coord in cluster_coords:
            #                 if coord != assigned_coord:
            #                     coord_label.append(coord_to_label.get(coord, -1))
            #     if len(coord_label) > 0:
            #         max_freq_coord_label = collections.Counter(coord_label).most_common(1)[0][0]
            #         coord_to_label[assigned_coord] = max_freq_coord_label
            # # Update labels based on the new mapping
            # adjusted_labels = np.array([coord_to_label.get(tuple(coord), -1) for coord in loc_df])

            # Number of clusters
            n_clusters = len(set(cluster_labels))-(1 if -1 in cluster_labels else 0)
            if n_clusters > 1:
                # Can only calculate silhouette score when there are more than 1 clusters
                score = silhouette_score(loc_df, cluster_labels)
                if score > best_silhouette_score:
                    best_epsilon = opt_eps
                    best_min_samples = min_samples
                    best_silhouette_score = score
            elif n_clusters == 1:
                # Store temporarily in case only 1 cluster can be found after optimization
                temp_epsilon = opt_eps
                temp_min_samples = min_samples

        # Decrease min_samples by 25% each time and round to multiple of 5 for computational efficiency
        new_min_samples = int(round(min_samples*min_points_decrement_factor / min_points_multiple) * min_points_multiple)
        if new_min_samples == min_samples:
            # May reach a plateau where the min_samples could not be decreased further -> normal decrement instead
            min_samples -= min_points_multiple
        else:
            min_samples = new_min_samples

    if best_epsilon == 0:
        best_epsilon = temp_epsilon
        best_min_samples = temp_min_samples

    print(f"Best silhouette score: {best_silhouette_score}")
    print(f"Best epsilon value: {best_epsilon}")
    print(f"Best min points: {best_min_samples}")
    return best_epsilon, best_min_samples

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
        latitude = np.round(np.array(loc_df.select(latitude_col).collect()).flatten(), 5)
        longitude = np.round(np.array(loc_df.select(longitude_col).collect()).flatten(), 5)
        loc_df = np.unique(np.transpose(np.vstack([latitude, longitude])), axis=0)
        epsilon, min_points = optimize_cluster(loc_df)
        dbscan_cluster = DBSCAN(eps=epsilon, min_samples=min_points, metric="haversine")
        dbscan_cluster.fit(loc_df)

        cluster_df_data = []
        for index, (lat, long) in enumerate(loc_df):
            cluster_df_data.append((float(lat), float(long), int(dbscan_cluster.labels_[index])))
        cluster_df = spark.createDataFrame(cluster_df_data, schema=StructType([StructField(latitude_col, FloatType()),\
                                                                            StructField(longitude_col, FloatType()),\
                                                                            StructField("cluster_id", IntegerType())]))
        # temp_cluster_df = cluster_df.toPandas()
        # plt.scatter(temp_cluster_df["double_latitude"], temp_cluster_df["double_longitude"], c=temp_cluster_df["cluster_id"], cmap='Accent')
        # plt.show()
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
    csv_filename = f"{DATA_FOLDER}/{user_id}_locations.csv"

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        location_df = spark.read.option("header", True).csv(csv_filename)
        float_cols = ["timestamp", "double_latitude", "double_longitude", "double_bearing", "double_speed", "double_altitude", "accuracy"]
        for col in float_cols:
            location_df = location_df.withColumn(col, F.col(col).cast(FloatType()))
        time_cols = ["date", "hour", "minute"]
        max_accuracy_location = location_df.groupBy(time_cols).agg(F.max(F.col("accuracy")).alias("accuracy"))
        location_df = location_df.join(max_accuracy_location, time_cols + ["accuracy"])\
            .dropDuplicates()
        location_df.write.parquet(parquet_filename)
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def process_wifi_data(user_id):
    """
    (Event-based)
    WiFi data: does not attempt to identify missing value

    NOTE Assumptions:
    1. Only retain unique ssid for nearby WiFi devices at a given time
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_wifi.parquet"
    csv_filename = f"{DATA_FOLDER}/{user_id}_wifi.csv"

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        wifi_df = spark.read.option("header", True).csv(csv_filename)
        time_cols = ["date", "hour", "minute"]
        wifi_df = wifi_df.filter(F.col("ssid") != "null").select(*time_cols + ["ssid"]).distinct()\
            .sort(time_cols)
        wifi_df.write.parquet(parquet_filename)
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def process_calls_data(user_id):
    """
    Extracts calls data from csv file into dataframe.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_calls.parquet"
    csv_filename = f"{DATA_FOLDER}/{user_id}_calls.csv"

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        call_df = spark.read.option("header", True).csv(csv_filename)
        call_df.write.parquet(parquet_filename)  
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

def process_messages_data(user_id):
    """
    Extracts messages data from csv file into dataframe.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_messages.parquet"
    csv_filename = f"{DATA_FOLDER}/{user_id}_messages.csv"

    if not os.path.exists(parquet_filename) and os.path.exists(csv_filename):
        message_df = spark.read.option("header", True).csv(csv_filename)
        message_df.write.parquet(parquet_filename)  
    
    if os.path.exists(parquet_filename):
        return spark.read.parquet(parquet_filename)
    return None

@F.udf(IntegerType())
def resolve_cluster_id(cluster_ids, current_ts, prev_ts, next_ts, prev_cluster_ids, next_cluster_ids):
    """
    Resolves discrepancies in cluster IDs at the same time point by obtaining the mode cluster ID.
    If more than one mode cluster IDs are involved, use the cluster ID that co-occur at an adjacent time point.
    """
    most_frequent_clusters = None
    if len(cluster_ids) > 0:
        counts = collections.Counter(cluster_ids)
        max_count = np.max(list(counts.values()))
        most_frequent_clusters = []
        for cluster_id, count in counts.items():
            if count == max_count:
                most_frequent_clusters.append(cluster_id)
        most_frequent_clusters = list(set(most_frequent_clusters))

    if most_frequent_clusters is None or len(most_frequent_clusters) > 1:
        prev_distance = abs(current_ts - prev_ts) if prev_ts is not None and len(prev_cluster_ids) > 0 else np.inf
        next_distance = abs(current_ts - next_ts) if next_ts is not None and len(next_cluster_ids) > 0 else np.inf
        adj_clusters = None

        if prev_distance < next_distance:
            adj_clusters = prev_cluster_ids
        elif next_distance < prev_distance:
            adj_clusters = next_cluster_ids
        
        if adj_clusters is not None:
            adj_cluster_counts = collections.Counter(adj_clusters)
            max_adj_count = np.max(list(adj_cluster_counts.values()))
            most_frequent_adj_clusters = []
            for cluster_id, count in adj_cluster_counts.items():
                if count == max_adj_count:
                    most_frequent_adj_clusters.append(cluster_id)
            most_frequent_adj_clusters = list(set(most_frequent_adj_clusters))
            # Make sure that the options for adjacent cluster IDs also exist in the current one
            most_frequent_adj_clusters = [c for c in most_frequent_adj_clusters if most_frequent_clusters is None or c in most_frequent_clusters]
            if len(most_frequent_adj_clusters) > 0:
                most_frequent_clusters = most_frequent_adj_clusters

    if most_frequent_clusters is not None:
        return most_frequent_clusters[0]
    else:
        return None

def resolve_cluster_fluctuations(user_id):
    """
    Resolves and removes short fluctuations (less than 5 mins) in location clusters
    TODO: Join with initial location df removes certain transitions
    """
    time_window = Window.orderBy("datetime")
    main_location_df = cross_check_cluster_with_activity_state(user_id)\
        .withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(time_window))

    cluster_transitions = main_location_df.select("datetime", "cluster_id", "prev_cluster")\
        .dropDuplicates().filter(F.col("cluster_id") != F.col("prev_cluster"))\
        .withColumn("cluster_end_time", F.lead(F.col("datetime")).over(time_window))\
        .withColumn("cluster_duration", F.unix_timestamp("cluster_end_time")-F.unix_timestamp("datetime"))\
        .withColumn("next_transition_cluster", F.lead(F.col("cluster_id")).over(time_window))\
        .withColumn("next_transition_datetime", F.lead(F.col("datetime")).over(time_window))
    
    cluster_transitions = cluster_transitions.withColumn("cluster_id", F.when((F.col("cluster_duration")<=300) &\
        (F.col("prev_cluster") == F.col("next_transition_cluster")), F.col("prev_cluster")).otherwise(F.col("cluster_id")))
    updated_transitions = cluster_transitions.filter(F.col("cluster_id") == F.col("prev_cluster"))\
        .select("datetime", "next_transition_datetime", "cluster_id").dropna()
    for col in updated_transitions.columns:
        updated_transitions = updated_transitions.withColumnRenamed(col, f"temp_{col}")

    main_location_df = main_location_df.join(updated_transitions, ((F.col("datetime") >= F.col("temp_datetime")) &\
        (F.col("datetime") < F.col("temp_next_transition_datetime"))))\
        .withColumn("cluster_id", F.coalesce(F.col("temp_cluster_id"), F.col("cluster_id")))\
        .select(*[col for col in main_location_df.columns]).dropDuplicates().sort("datetime")

    return main_location_df

def cross_check_cluster_with_activity_state(user_id):
    """
    Cross-checks cluster changes with activity states
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_adjusted_combined_location.parquet"
    if not os.path.exists(parquet_filename):
        time_window = Window().orderBy("location_datetime")
        main_location_df = complement_location_data(user_id)
        
        location_with_cluster_df = main_location_df.select("datetime", "double_latitude", "double_longitude", "cluster_id")\
            .dropDuplicates().withColumnRenamed("datetime", "location_datetime").sort("location_datetime")\
            .withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(time_window))\
            .withColumn("prev_location_datetime", F.lag(F.col("location_datetime")).over(time_window))

        # Segregates into moving vs non-moving (tilting is not considered since it will not cause cluster change)
        activity_df = process_activity_data(user_id)\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .filter(F.col("activity_name").isin("in_vehicle", "on_bicycle", "running", "walking", "still"))\
            .withColumn("is_still", F.when(F.col("activity_name") == "still", 1).otherwise(0))
        
        # Compute consecutive moving vs still duration
        time_window = Window().orderBy("datetime")
        consecutive_movement_df = activity_df.withColumn("prev_is_still", F.lag(F.col("is_still")).over(time_window))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
            .withColumn("new_group", (F.col("is_still") != F.col("prev_is_still")).cast("int"))\
            .withColumn("group_id", F.sum("new_group").over(time_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
            .groupBy("group_id", "is_still")\
            .agg(F.min("datetime").alias("start_movement_datetime"),\
                F.max("next_datetime").alias("end_movement_datetime"))\
            .withColumn("movement_duration", F.unix_timestamp("end_movement_datetime")-F.unix_timestamp("start_movement_datetime"))\
            .drop("group_id").sort("start_movement_datetime")
        
        # There might be sampling gaps if the duration of a location transition exceeds two hours
        significant_gap_location_transitions = location_with_cluster_df.filter(((F.col("cluster_id") != F.col("prev_cluster"))) &\
            ((F.unix_timestamp("location_datetime")-F.unix_timestamp("prev_location_datetime"))>7200)) 
        # Cross-check with intersecting duration of continuous non-movement
        intersecting_movement = significant_gap_location_transitions.join(consecutive_movement_df.filter((F.col("is_still") == 1) &\
            (F.col("movement_duration") >= 3600)),\
            ((F.col("prev_location_datetime") < F.col("end_movement_datetime")) &\
            (F.col("location_datetime") > F.col("start_movement_datetime"))))
        
        # Get the longest non-movement duration to update cluster transition datetime
        # 1. If still start time > cluster transition start time: update cluster transition end time earlier to still start time
        # 2. If still start time < cluster transition start time: update cluster transition start time later to still end time
        rank_still_duration_window = Window.partitionBy(*[col for col in significant_gap_location_transitions.columns])\
            .orderBy(F.desc("movement_duration"))
        intersecting_movement = intersecting_movement.withColumn("rank", F.row_number().over(rank_still_duration_window))\
            .filter(F.col("rank") == 1).drop("rank").toPandas()

        with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
            contexts = json.load(f)

        primary_loc = None
        location_clusters = contexts["location_clusters"]
        for key in list(location_clusters.keys()):
            if location_clusters[key]["is_primary"]:
                primary_loc = int(key)
        
        rows_to_insert = []
        for _, row in intersecting_movement.iterrows():
            if row["start_movement_datetime"] > row["prev_location_datetime"]:
                # Insert a new row before this: the cluster stay might have happened earlier
                new_row = {"location_datetime": row["start_movement_datetime"],\
                    "double_latitude": row["double_latitude"],\
                    "double_longitude": row["double_longitude"],\
                    "cluster_id": row["cluster_id"]}
                rows_to_insert.append(new_row)
            elif row["start_movement_datetime"] < row["prev_location_datetime"]:
                # Insert a new row after this: the cluster stay might have ended later
                new_row = {"location_datetime": row["end_movement_datetime"],\
                    "double_latitude": None,\
                    "double_longitude": None,\
                    "cluster_id": row["prev_cluster"]}
                if (row["end_movement_datetime"] - row["prev_location_datetime"]).total_seconds() > 12*3600:
                    if primary_loc is not None:
                        new_row["cluster_id"] = primary_loc
                rows_to_insert.append(new_row)

        # Union updated transitions and duration into previous cluster transitions
        time_window = Window.orderBy("location_datetime")
        for row in rows_to_insert:
            row["location_datetime"] = row["location_datetime"].strftime("%Y-%m-%d %H:%M:%S")
        df_to_insert = spark.createDataFrame(rows_to_insert, schema=StructType(
            [StructField("location_datetime", StringType(), False),\
            StructField("double_latitude", FloatType(), True),
            StructField("double_latitude", FloatType(), True),\
            StructField("cluster_id", IntegerType(), False)])).dropDuplicates()
        df_to_insert = df_to_insert.withColumn("location_datetime", F.col("location_datetime").cast(TimestampType()))
        adjusted_transitions = location_with_cluster_df.select(*["location_datetime", "double_latitude", "double_longitude", "cluster_id"])\
            .union(df_to_insert).sort("location_datetime")\
            .withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(time_window))\
            .withColumn("prev_location_datetime", F.lag(F.col("location_datetime")).over(time_window))
        for col in adjusted_transitions.columns:
            adjusted_transitions = adjusted_transitions.withColumnRenamed(col, f"temp_{col}")

        # Update the initial df to adjust cluster_id and include new rows to take into account new cluster transitions
        updated_location_df = main_location_df.join(adjusted_transitions,\
            (((F.col("datetime") > F.col("temp_prev_location_datetime")) & \
            (F.col("datetime") < F.col("temp_location_datetime"))) |\
            (F.col("datetime") == F.col("temp_location_datetime"))), "outer")\
            .withColumn("datetime", F.coalesce(F.col("datetime"), F.col("temp_location_datetime")))\
            .withColumn("cluster_id", F.when(F.col("datetime") == F.col("temp_location_datetime"), F.col("temp_cluster_id"))\
                .otherwise(F.col("temp_prev_cluster")))
        updated_location_df = updated_location_df.drop(*[col for col in updated_location_df.columns if col.startswith("temp_")])\
            .dropDuplicates().sort("datetime")\
            .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
            .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
            .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))

        updated_location_df.write.parquet(parquet_filename)
        
    if os.path.exists(parquet_filename):
        location_df = spark.read.parquet(parquet_filename)
        return location_df
    return None

def complement_location_data(user_id, adjusted_clusters=True):
    """
    Combines location and WiFi data to provide location semantics.
    Clusters locations and identify WiFi devices available in each cluster.

    NOTE:
    1. Location coordinates are rounded to 5 decimal places (up to 1 metres) - can also be rounded to 4 dp for 10 metres allowance
    2. Fill in cluster IDs based on groups of WiFi devices that have co-occur at clusters with known location coordinates.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_combined_location.parquet"
    location_df = process_location_data(user_id)
    if not os.path.exists(parquet_filename) and location_df is not None:
        time_cols = ["date", "hour", "minute"]
        coordinate_cols = ["double_latitude", "double_longitude"]
        location_df = location_df.withColumn("double_latitude", F.round(F.col("double_latitude"), 5))\
            .withColumn("double_longitude", F.round(F.col("double_longitude"), 5))\
            .withColumn("double_altitude", F.round(F.col("double_altitude"), 5))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .filter((F.col("double_latitude") != 0.0) & (F.col("double_longitude") != 0.0))

        cluster_df = cluster_locations(user_id, location_df, "double_latitude", "double_longitude")
        # Location clusters are derived from all unique coordinates in location_df so there will be no null cluster_id
        location_df = location_df.join(cluster_df, coordinate_cols).dropDuplicates()
        
        wifi_df = process_wifi_data(user_id)
        if wifi_df is not None:
            wifi_df = wifi_df.withColumn("hour", F.col("hour").cast(IntegerType()))\
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
                    .withColumn("resolved_cluster_id", resolve_cluster_id(F.col("cluster_ids"), F.unix_timestamp("datetime"), 
                        F.unix_timestamp("prev_datetime"), F.unix_timestamp("next_datetime"), 
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
        
        else:
            location_df = location_df.withColumn("ssid", F.lit(None).cast(StringType()))
            all_locations = location_df.select(*time_cols + ["datetime", "ssid", "double_latitude", "double_longitude", "double_altitude", "cluster_id"])

        all_locations.write.parquet(parquet_filename)

    if os.path.exists(parquet_filename):
        # if adjusted_clusters:
        #     adjusted_location_df = cross_check_cluster_with_activity_state(user_id)
        #     if adjusted_location_df is not None:
        #         return adjusted_location_df
        location_df = spark.read.parquet(parquet_filename)
        return location_df
    return None

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

    NOTE Adjustments and customizations:
    1. Check consecutive_min (5 mins by default)
    2. dark_threshold and silent_threshold - estimated from overall distribution by default, updated when self-reported sleep is available
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_sleep" + (
        "_optional" if with_allowance else "") + ".parquet"
    if not os.path.exists(parquet_filename):
        time_cols = ["date", "hour", "minute"]
        time_window = Window.partitionBy("date").orderBy(*time_cols)
        consecutive_min = 10
        # Retrieves pre-saved contexts
        with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
            contexts = json.load(f)

        light_df = process_light_data(user_id)
        dark_threshold = contexts["dark_threshold"]
        light_df = light_df.withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("is_dark", F.when((F.col("min_light_lux") <= dark_threshold) | (F.col("mean_light_lux") <= dark_threshold), 1).otherwise(0))\
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
        

        noise_df = process_noise_data_with_conv_estimate(user_id)
        silence_threshold = contexts["silent_threshold"]
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

        # V1: Manual computation through active phone usage inferred from screen states
        # Exclude screen activation caused by notifications or active phone checking
        phone_use_df = process_phone_usage_data(user_id)
        not_in_use_df = phone_use_df.withColumn("prev_end_timestamp", F.lag(F.col("end_timestamp")).over(Window.orderBy("start_timestamp")))\
            .filter(F.col("prev_end_timestamp").isNotNull())\
            .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("prev_end_timestamp")))\
            .withColumn("end_datetime", udf_datetime_from_timestamp("start_timestamp"))\
            .select("start_datetime", "end_datetime")
        
        # V2: Used plugin data directly
        # phone_use_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")
        # not_in_use_df = phone_use_df.filter(F.col("double_elapsed_device_off") > 0)\
        #     .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
        #     .withColumn("start_timestamp", F.col("timestamp") - F.col("double_elapsed_device_off"))\
        #     .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_timestamp")))\
        #     .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("timestamp")))\
        #     .withColumn("consecutive_duration", F.round(F.col("double_elapsed_device_off")/1000).cast(IntegerType()))\
        #     .select("start_datetime", "end_datetime", "consecutive_duration").sort("start_datetime")

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

def haversine_distance_with_altitude(coord1, coord2):
    """
    Haversine (great circle) distance between two coordinates in the form of (latitude, longitude, altitude)
    Referred from package haversine computation.
    """
    # Convert decimal degrees to radians
    lat1, long1, lat2, long2 = map(np.radians, [coord1[0], coord1[1], coord2[0], coord2[1]])
    dlong = long2 - long1
    dlat = lat2 - lat1
    dalt = coord2[2] - coord1[2]
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlong/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Referred from haversine.py
    # mean earth radius - https://en.wikipedia.org/wiki/Earth_radius#Mean_radius
    _AVG_EARTH_RADIUS_KM = 6371.0088
    metres = _AVG_EARTH_RADIUS_KM * c * 1000
    distance = np.sqrt(metres**2 + dalt**2)
    
    return distance

def average_coordinates(coord_series):
    avg_lat = np.mean([coord[0] for coord in coord_series])
    avg_lon = np.mean([coord[1] for coord in coord_series])
    avg_alt = np.mean([coord[2] for coord in coord_series])
    return (avg_lat, avg_lon, avg_alt)

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

def create_agg_features(user_id, group_by_cols, activity_df, light_df, noise_df, app_usage_df, phone_usage_df, bluetooth_df, location_df):
    """
    Extracts functions from input dataframes and aggregates based on columns defined
    """
    with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)

    agg_features = {}

    activities = ["still", "in_vehicle", "on_bicycle", "tilting", "walking", "running"]
    if activity_df.count() > 0:
        # Total duration and normalized duration (by time spent at the cluster) for each activity state
        activity_duration = activity_df.groupBy(*group_by_cols + ["activity_name"])\
            .agg(F.sum("duration").alias("total_activity_duration"))\
            .sort("total_activity_duration", ascending=False).collect()
        for row in activity_duration:
            key = tuple(row[col] for col in group_by_cols)
            if key not in agg_features:
                agg_features[key] = {}
            group_by_dict = agg_features[key]
            if row["activity_name"] in activities:
                group_by_dict[f"{row['activity_name']}_duration"] = row["total_activity_duration"]

    # Total phone usage duration
    if phone_usage_df.count() > 0:
        total_phone_use_duration = phone_usage_df.groupBy(group_by_cols)\
            .agg(F.sum("usage_duration").alias("phone_use_duration")).collect()
        for row in total_phone_use_duration:
            key = tuple(row[col] for col in group_by_cols)
            if key not in agg_features:
                agg_features[key] = {}
            group_by_dict = agg_features[key]
            group_by_dict["phone_use_duration"] = row["phone_use_duration"]

    # Total duration of each application category and normalized by time spent at the cluster and unlock duration
    app_categories = list(contexts["app_categories"].keys()) + ["utilities", "others"]
    if app_usage_df.count() > 0:
        app_usage_duration = app_usage_df.groupBy(*group_by_cols + ["category"])\
            .agg((F.sum("usage_duration")).alias("total_usage_duration"))\
            .sort("total_usage_duration", ascending=False).collect()
        for row in app_usage_duration:
            key = tuple(row[col] for col in group_by_cols)
            if row["category"] in app_categories:
                if key not in agg_features:
                    agg_features[key] = {}
                group_by_dict = agg_features[key]
                group_by_dict[f"app_use_duration_{row['category']}"] = row["total_usage_duration"]

                if "phone_use_duration" in group_by_dict:
                    group_by_dict[f"app_use_normalized_by_phone_use_duration_{row['category']}"] = row["total_usage_duration"]/group_by_dict["phone_use_duration"]

    stat_functions = [F.min, F.max, F.mean, F.stddev]
    stat_names = ["min", "max", "mean", "std"]
    if light_df.count() > 0:
        # Min, max, mean, and standard deviation of ambient light
        agg_expressions = [stat_functions[index](f"{stat_names[index]}_light_lux").alias(f"{stat_names[index]}_light_lux") for index in range(len(stat_functions))]
        ambient_light = light_df.groupBy(*group_by_cols).agg(*agg_expressions).collect()
        for row in ambient_light:
            key = tuple(row[col] for col in group_by_cols)
            if key not in agg_features:
                agg_features[key] = {}
            group_by_dict = agg_features[key]
            for stat in stat_names:
                group_by_dict[f"{stat}_light_lux"] = row[f"{stat}_light_lux"]
        
        dark_df = light_df.filter(F.col("is_dark") == 1)
        if dark_df.count() > 0:
            dark_duration_df = dark_df.groupBy(*group_by_cols).agg(F.sum("duration").alias("dark_duration")).collect()
            for row in dark_duration_df:
                key = tuple(row[col] for col in group_by_cols)
                group_by_dict = agg_features[key]
                group_by_dict["dark_duration"] = row["dark_duration"]

    if noise_df.count() > 0:
        # Min, max, mean, and standard deviation of ambient noise
        agg_expressions = [stat_functions[index](f"{stat_names[index]}_decibels").alias(f"{stat_names[index]}_decibels") for index in range(len(stat_functions))]
        ambient_noise = noise_df.groupBy(*group_by_cols).agg(*agg_expressions).collect()
        for row in ambient_noise:
            key = tuple(row[col] for col in group_by_cols)
            if key not in agg_features:
                agg_features[key] = {}
            group_by_dict = agg_features[key]
            for stat in stat_names:
                group_by_dict[f"{stat}_decibels"] = row[f"{stat}_decibels"]
        
        quiet_df = noise_df.filter(F.col("is_quiet") == 1)
        if quiet_df.count() > 0:
            quiet_duration_df = quiet_df.groupBy(*group_by_cols).agg(F.sum("duration").alias("silent_duration")).collect()
            for row in quiet_duration_df:
                key = tuple(row[col] for col in group_by_cols)
                group_by_dict = agg_features[key]
                group_by_dict["silent_duration"] = row["silent_duration"]

    location_time_window = Window().partitionBy(*group_by_cols).orderBy("datetime")
    location_coordinates = location_df.select(*group_by_cols +\
        ["datetime", "double_latitude", "double_longitude", "cluster_id"]).distinct()
    distance_traveled = location_coordinates\
        .withColumn("next_latitude", F.lead(F.col("double_latitude")).over(location_time_window))\
        .withColumn("next_longitude", F.lead(F.col("double_longitude")).over(location_time_window))\
        .dropna().withColumn("distance", distance(F.col("double_latitude"), F.col("double_longitude"),\
            F.col("next_latitude"), F.col("next_longitude")))

    total_distance = distance_traveled.groupBy(*group_by_cols).agg(F.sum("distance").alias("distance_traveled")).collect()
    for row in total_distance:
        key = tuple(row[col] for col in group_by_cols)
        if key not in agg_features:
            agg_features[key] = {}
        group_by_dict = agg_features[key]
        group_by_dict["distance_traveled"] = row["distance_traveled"]

    # Location variance
    location_variance = location_coordinates.groupBy(*group_by_cols)\
        .agg(F.variance("double_latitude").alias("variance_latitude"),\
             F.variance("double_longitude").alias("variance_longitude"))\
        .withColumn("log_variance", F.log(F.col("variance_latitude") + F.col("variance_longitude"))).collect()
    for row in location_variance:
        key = tuple(row[col] for col in group_by_cols)
        if key not in agg_features:
            agg_features[key] = {}
        group_by_dict = agg_features[key]
        group_by_dict["location_variance"] = row["log_variance"]

    # Count of unique location entries (could have multiple at the same timestamp due to WiFi devices)
    unique_location_entries = location_df.select(*group_by_cols + ["datetime", "cluster_id"] +\
        [col for col in location_df.columns if "double_" in col]).dropDuplicates()
    unique_location_count = unique_location_entries.groupBy(*group_by_cols)\
        .agg(F.count("datetime").alias("unique_location_count")).collect()
    for row in unique_location_count:
        key = tuple(row[col] for col in group_by_cols)
        if key not in agg_features:
            agg_features[key] = {}
        group_by_dict = agg_features[key]
        group_by_dict["unique_location_count"] = row["unique_location_count"]
    
    # Count of unknown locations
    unknown_location_count = unique_location_entries.filter(F.col("cluster_id") == -1)\
        .groupBy(*group_by_cols).agg(F.count("datetime").alias("unknown_location_count")).collect()
    for row in unknown_location_count:
        key = tuple(row[col] for col in group_by_cols)
        group_by_dict = agg_features[key]
        group_by_dict["unknown_location_count"] = row["unknown_location_count"]
        if "unique_location_count" in group_by_dict:
            group_by_dict["unknown_location_count_normalized_by_entries"] = row["unknown_location_count"]/\
                group_by_dict["unique_location_count"]

    # Time spent in cluster
    cluster_time_spent = unique_location_entries\
        .withColumn("next_datetime", F.lead(F.col("datetime")).over(location_time_window)).dropna()\
        .withColumn("duration", F.unix_timestamp("next_datetime")-F.unix_timestamp("datetime"))\
        .groupBy(*group_by_cols + ["cluster_id"]).agg(F.sum("duration").alias("stay_duration")).collect()
    
    overall_location_cluster_info = contexts["location_clusters"]
    overall_clusters = list(overall_location_cluster_info.keys())

    cluster_stay = {}
    for row in cluster_time_spent:
        key = tuple(row[col] for col in group_by_cols)
        if key not in cluster_stay:
            cluster_stay[key] = {}
        cluster_dict = cluster_stay[key]
        cluster_dict[f"cluster{row['cluster_id']}_stay_duration"] = row["stay_duration"]
        # Time spent at primary and secondary clusters
        if overall_location_cluster_info[str(row["cluster_id"])]["is_primary"]:
            cluster_dict["time_spent_primary_cluster"] = row["stay_duration"]
        elif overall_location_cluster_info[str(row["cluster_id"])]["is_secondary"]:
            cluster_dict["time_spent_secondary_cluster"] = row["stay_duration"]
    
    for group_by_key in list(cluster_stay.keys()):
        if group_by_key not in agg_features:
            agg_features[key] = {}
        group_by_dict = agg_features[key]
        cluster_group_by_dict = cluster_stay[group_by_key]
        time_at_cluster = []
        for cluster_key in list(cluster_group_by_dict.keys()):
            group_by_dict[cluster_key] = cluster_group_by_dict[cluster_key]
            if "stay_duration" in cluster_key:
                time_at_cluster.append(cluster_group_by_dict[cluster_key])
        # Add a small negligible value to avoid log(0) when a specific cluster is not visited
        probability = np.array(time_at_cluster)/(24*3600) + 1e-10
        entropy = - np.sum(probability * np.log(probability))
        normalized_entropy = entropy/math.log(len(overall_clusters))
        group_by_dict["location_entropy"] = entropy
        group_by_dict["normalized_location_entropy"] = normalized_entropy

    wifi_df = location_df.filter(F.col("ssid").isNotNull())
    if wifi_df.count() > 0:
        # Number of WiFi entries and unique WiFi devices
        wifi_entry_count = wifi_df.groupBy(*group_by_cols).agg(F.count("ssid").alias("wifi_count")).collect()
        for row in wifi_entry_count:
            key = tuple(row[col] for col in group_by_cols)
            if key not in agg_features:
                agg_features[key] = {}
            group_by_dict = agg_features[key]
            group_by_dict["wifi_count"] = row["wifi_count"]
        unique_wifi_entry_count = wifi_df.groupBy(*group_by_cols).agg(F.count_distinct("ssid").alias("unique_wifi_count")).collect()
        for row in unique_wifi_entry_count:
            key = tuple(row[col] for col in group_by_cols)
            group_by_dict = agg_features[key]
            group_by_dict["unique_wifi_count"] = row["unique_wifi_count"]

        # Retrieves primary WiFi devices and their weighted occurrence
        primary_wifi_devices = [item["ssid"] for item in contexts["primary_wifi_devices"]]
        primary_wifi_weight = [item["weighted_occurrence"] for item in contexts["primary_wifi_devices"]]
        primary_wifi_weighted_by_overall_occurrence = {}
        
        primary_wifi_df = wifi_df.filter(F.col("ssid").isin(*primary_wifi_devices))
        if primary_wifi_df.count() > 0:
            # Occurrence of each WiFi device weighted by total WiFi entries in the current cluster
            primary_wifi_occurrence = primary_wifi_df.groupBy(*group_by_cols+["ssid"])\
                .agg(F.count("datetime").alias("device_occurrence"))\
                .sort("device_occurrence", ascending=False).collect()
            # Compute overall weighted occurrence of primary WiFi devices
            for row in primary_wifi_occurrence:
                key = tuple(row[col] for col in group_by_cols)
                list_index = primary_wifi_devices.index(row["ssid"])
                if key not in primary_wifi_weighted_by_overall_occurrence:
                    primary_wifi_weighted_by_overall_occurrence[key] = 0
                primary_wifi_weighted_by_overall_occurrence[key] = primary_wifi_weighted_by_overall_occurrence[key] +\
                    row["device_occurrence"] * primary_wifi_weight[list_index]
            
            for key in primary_wifi_weighted_by_overall_occurrence:
                group_by_dict = agg_features[key]
                group_by_dict["wifi_overall_weighted_occurrence"] = primary_wifi_weighted_by_overall_occurrence[key]
                if "wifi_count" in group_by_dict:
                    group_by_dict["wifi_group_weighted_occurrence"] = primary_wifi_weighted_by_overall_occurrence[key]/group_by_dict["wifi_count"]

    if bluetooth_df.count() > 0:
        # Number of Bluetooth entries and unique Bluetooth devices
        bt_entry_count = bluetooth_df.groupBy(*group_by_cols).agg(F.count("bt_address").alias("bluetooth_count")).collect()
        for row in bt_entry_count:
            key = tuple(row[col] for col in group_by_cols)
            if key not in agg_features:
                agg_features[key] = {}
            group_by_dict = agg_features[key]
            group_by_dict["bluetooth_count"] = row["bluetooth_count"]
        unique_bt_count = bluetooth_df.groupBy(*group_by_cols).agg(F.count_distinct("bt_address").alias("unique_bluetooth_count")).collect()
        for row in unique_bt_count:
            key = tuple(row[col] for col in group_by_cols)
            group_by_dict = agg_features[key]
            group_by_dict["unique_bluetooth_count"] = row["unique_bluetooth_count"]
        
        # Retrieves primary Bluetooth devices and their weighted occurrence
        primary_bt_devices = [item["bt_address"] for item in contexts["primary_bluetooth_devices"]]
        primary_bt_weight = [item["weighted_occurrence"] for item in contexts["primary_bluetooth_devices"]]
        primary_bt_weighted_by_overall_occurrence = {}

        # Occurrence of each Bluetooth device weighted by total Bluetooth entries in the current cluster
        primary_bt_df = bluetooth_df.filter(F.col("bt_address").isin(*primary_bt_devices))
        if primary_bt_df.count() > 0:
            primary_bt_occurrence = primary_bt_df.groupBy(*group_by_cols + ["bt_name", "bt_address"])\
                .agg(F.count("datetime").alias("device_occurrence"))\
                .sort("device_occurrence", ascending=False).collect()
            # Compute overall weighted occurrence of primary WiFi devices
            for row in primary_bt_occurrence:
                key = tuple(row[col] for col in group_by_cols)
                list_index = primary_bt_devices.index(row["bt_address"])
                if key not in primary_bt_weighted_by_overall_occurrence:
                    primary_bt_weighted_by_overall_occurrence[key] = 0
                primary_bt_weighted_by_overall_occurrence[key] = primary_bt_weighted_by_overall_occurrence[key] +\
                    row["device_occurrence"] * primary_bt_weight[list_index]
            
            for key in primary_bt_weighted_by_overall_occurrence:
                group_by_dict = agg_features[key]
                group_by_dict["bt_overall_weighted_occurrence"] = primary_bt_weighted_by_overall_occurrence[key]
                if "bluetooth_count" in group_by_dict:
                    group_by_dict["bt_group_weighted_occurrence"] = primary_bt_weighted_by_overall_occurrence[key]/group_by_dict["bluetooth_count"]

    return agg_features

@F.udf(ArrayType(TimestampType()))
def generate_rows_for_each_minute(start_datetime, end_datetime):
    """
    Creates a list of rows each representing each minute between input start and end datetimes.
    Used to insert new rows for durations such as phone and app usage for consistency.
    """
    return [(start_datetime + timedelta(minutes=i)) for i in range(0, (end_datetime - start_datetime).seconds // 60 + 1)]


def extract_custom_agg_features(user_id, group_by_cols):
    """
    Prepares dataframes and make triggers to extract features aggregated based on input columns.
    """
    # Retrieves pre-saved contexts
    with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)

    # Filter data for the particular day
    physical_mobility = process_activity_data(user_id)\
        .withColumn("datetime", udf_datetime_from_timestamp(F.col("timestamp").cast(FloatType()))-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))
    
    ambient_light = process_light_data(user_id)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
        .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("is_dark", F.when(F.col("min_light_lux") <= contexts["dark_threshold"], 1).otherwise(0))
    
    ambient_noise = process_noise_data_with_conv_estimate(user_id)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
        .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("is_quiet", F.when(F.col("mean_decibels") <= contexts["silent_threshold"], 1).otherwise(0))
    
    # locations = resolve_cluster_fluctuations(user_id)\
    locations = cross_check_cluster_with_activity_state(user_id)\
        .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .drop("hour", "minute")
    location_df = locations.select(*[col for col in locations.columns if col != "ssid"]).distinct()

    context_dfs = [physical_mobility, ambient_light, ambient_noise, location_df]
    datetime_window = Window().orderBy("datetime")
    for df_index, df in enumerate(context_dfs):
        # Get the last entry of the previous day to get data at 00:00
        day_transition_df = df.withColumn("next_date", F.lead(F.col("date")).over(datetime_window))\
            .filter(F.col("next_date") != F.col("date")).dropna()
        start_of_day_df = day_transition_df.withColumn("date", F.col("next_date"))\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.lit(0), F.lit(0))).drop("next_date")
        end_of_day_df = day_transition_df.withColumn("datetime", udf_generate_datetime(F.col("date"), F.lit(23), F.lit(59))).drop("next_date")
        context_dfs[df_index] = df.union(start_of_day_df).union(end_of_day_df).sort("datetime")\
            .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
            .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))\
            .withColumn("day_of_week", F.dayofweek("datetime"))\
            .withColumn("epoch", get_epoch_from_hour(F.col("hour")))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(datetime_window))\
            .withColumn("duration", F.unix_timestamp("next_datetime") - F.unix_timestamp("datetime"))

    locations = locations.join(context_dfs[3], [col for col in locations.columns if col != "ssid"], "outer")\
        .dropDuplicates().sort("datetime")

    dt_cols = ["start_timestamp", "end_timestamp", "start_phone_use_timestamp", "end_phone_use_timestamp"]
    app_usage = process_application_usage_data(user_id)
    for col in dt_cols:
        app_usage = app_usage.withColumn(f"{col[:col.rfind('_')]}_datetime",\
            udf_datetime_from_timestamp(F.col(col).cast(FloatType()))-timedelta(hours=2))

    # Expand each phone usage instance into multiple rows based on duration
    expanded_app_usage_df = app_usage.withColumn("minute_rows",\
        generate_rows_for_each_minute(F.col("start_datetime"), F.col("end_datetime")))\
        .select(*[col for col in app_usage.columns if "duration" not in col] + \
            [F.col("usage_duration").alias("app_use_instance_duration"),\
            F.col("duration").alias("phone_use_instance_duration"),\
            F.explode("minute_rows").alias("datetime")])\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))\
        .withColumn("day_of_week", F.dayofweek("datetime"))\
        .withColumn("epoch", get_epoch_from_hour(F.col("hour")))\
        .withColumn("usage_duration", F.lit(60)).sort("datetime")

    phone_usage_df = expanded_app_usage_df.select("start_phone_use_datetime",\
        "end_phone_use_datetime", "phone_use_instance_duration").distinct()
    expanded_phone_usage_df = phone_usage_df.withColumn("minute_rows",\
        generate_rows_for_each_minute(F.col("start_phone_use_datetime"), F.col("end_phone_use_datetime")))\
        .select(*[col for col in phone_usage_df.columns] + [F.explode("minute_rows").alias("datetime")])\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))\
        .withColumn("day_of_week", F.dayofweek("datetime"))\
        .withColumn("epoch", get_epoch_from_hour(F.col("hour")))\
        .withColumn("usage_duration", F.lit(60)).sort("datetime")

    bluetooth_df = process_bluetooth_data(user_id).drop("temp_bt_name")\
        .withColumn("datetime", udf_datetime_from_timestamp(F.col("timestamp").cast(FloatType())))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))\
        .withColumn("day_of_week", F.dayofweek("datetime"))\
        .withColumn("epoch", get_epoch_from_hour(F.col("hour"))).sort("datetime")

    # Add location cluster to all other dfs
    cluster_transitions = locations.select(*["datetime", "cluster_id"]).dropDuplicates()\
        .withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(datetime_window))\
        .withColumn("prev_location_datetime", F.lag(F.col("datetime")).over(datetime_window))\
        .filter(F.col("prev_cluster") != F.col("cluster_id")).sort("datetime")\
        .withColumnRenamed("datetime", "location_datetime")
    
    first_transition_row = cluster_transitions.orderBy("location_datetime").first()
    first_transition_location_datetime = first_transition_row["prev_location_datetime"]
    first_cluster_id = first_transition_row["prev_cluster"]
    last_transition_row = cluster_transitions.orderBy(F.desc("location_datetime")).first()
    last_transition_location_datetime = last_transition_row["location_datetime"]
    last_cluster_id = last_transition_row["cluster_id"]
    dfs_to_agg = context_dfs[:-1] + [expanded_app_usage_df, expanded_phone_usage_df, bluetooth_df] + [locations]
    for df_index, df in enumerate(dfs_to_agg[:-1]):
        df = df.join(cluster_transitions, ((F.col("datetime") < F.col("location_datetime")) &\
            (F.col("datetime") >= F.col("prev_location_datetime"))), "left").drop("cluster_id")\
            .withColumn("cluster_id", F.col("prev_cluster"))\
            .withColumn("cluster_id", F.when(F.col("datetime")>=last_transition_location_datetime, F.lit(last_cluster_id))\
                .otherwise(F.col("cluster_id")))\
            .withColumn("cluster_id", F.when(F.col("datetime")<first_transition_location_datetime, F.lit(first_cluster_id))\
                .otherwise(F.col("cluster_id"))).sort("datetime")\
            .drop("prev_cluster", "prev_location_datetime")
        dfs_to_agg[df_index] = df

    agg_features = create_agg_features(user_id, group_by_cols, *dfs_to_agg)

    return agg_features

def extract_day_features(user_id, date=None, visualize_cluster_contexts=False):
    """
    Extracts day-level features within a specific day and returns as a dictionary with 31 keys (date as the first key):
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

    HACK
    1. The cloud database is set to Australia/Melbourne timezone so manual adjustments (-timedelta(hours=2)) have been made throughout this analysis.
    2. However, PySpark has recognized this timezone so certain operations like F.min(datetimes) will automatically convert it again.
    3. Temporary fix: when saving datetimes (after adjustments), make sure to assert the timezone using .astimezone(TIMEZONE) 

    References:
    1. https://peerj.com/articles/2537/ (computation of):
        a. Location variance
        b. Total distance traveled
        c. Entropy and normalized entropy
    """
    time_cols = ["date", "hour", "minute"]
    time_window = Window().orderBy("datetime")

    # Retrieves pre-saved contexts
    with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)
    
    if date is None:
        sleep_df = estimate_sleep(user_id)\
            .withColumn("start_datetime", F.col("start_datetime")-timedelta(hours=2))\
            .withColumn("end_datetime", F.col("end_datetime")-timedelta(hours=2))\
            .withColumn("date", udf_get_date_from_datetime("end_datetime"))\
            .sort("start_datetime")
        cur_day = np.array(sleep_df.select("date").distinct().sort("date").collect()).flatten()[0]
    else:
        cur_day = date
    
    features = {"date": cur_day}

    # Filter data for the particular day
    physical_mobility = process_activity_data(user_id)\
        .withColumn("datetime", udf_datetime_from_timestamp(F.col("timestamp").cast(FloatType()))-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime")))
    # Get the last entry of the previous day to get data at 00:00
    physical_mobility = physical_mobility.filter(F.col("date") < cur_day).orderBy(F.col("datetime").desc()).limit(1)\
        .withColumn("datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0)))\
        .union(physical_mobility.filter(F.col("date") == cur_day)).sort("datetime")
    # Resolve multiple activity entries at the same time point with custom granularity priorities
    # physical_mobility = physical_mobility.groupBy("datetime").agg(F.collect_list("activity_name").alias("activity_list"))\
    #     .withColumn("activity_name", resolve_activity_priority("activity_list"))
    physical_mobility = physical_mobility.withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
        .filter(F.col("next_datetime").isNotNull())\
        .withColumn("duration", F.unix_timestamp("next_datetime") - F.unix_timestamp("datetime"))
    # consecutive_physical_mobility = physical_mobility.withColumn("prev_activity", F.lag(F.col("activity_name")).over(time_window))\
    #     .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
    #     .withColumn("new_group", (F.col("activity_name") != F.col("prev_activity")).cast("int"))\
    #     .withColumn("group_id", F.sum("new_group").over(time_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
    #     .groupBy("group_id", "activity_name", "activity_type")\
    #     .agg(F.min("datetime").alias("start_datetime"),\
    #             F.max("next_datetime").alias("end_datetime"))\
    #     .drop("group_id").sort("start_datetime")\
    #     .withColumn("consecutive_duration", (F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime"))))\
    #     .dropna()

    activity_duration = physical_mobility.groupBy("activity_name")\
        .agg(F.sum("duration").alias("total_activity_duration")).toPandas()

    activity_names = activity_duration["activity_name"].to_list()
    durations = activity_duration["total_activity_duration"].to_list()
    for act in ACTIVITY_PRIORITIES[:7]:
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
        .withColumn("datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0)))\
        .union(ambient_light.filter(F.col("date") == cur_day)).sort("datetime")
    ambient_light = ambient_light.withColumn("is_dark", F.when(F.col("min_light_lux") <= contexts["dark_threshold"], 1).otherwise(0))\
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
        .withColumn("datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0)))\
        .union(ambient_noise.filter(F.col("date") == cur_day)).sort("datetime")
    ambient_noise = ambient_noise.withColumn("is_quiet", F.when(F.col("mean_decibels") <= contexts["silent_threshold"], 1).otherwise(0))\
            .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
            .filter(F.col("next_datetime").isNotNull())\
            .withColumn("duration", F.unix_timestamp("next_datetime") - F.unix_timestamp("datetime"))
    total_quiet_duration = ambient_noise.filter(F.col("is_quiet")==1).agg(F.sum("duration")).collect()[0][0]
    if not total_quiet_duration:
        total_quiet_duration = 0
    features["quiet_duration"] = total_quiet_duration

    dt_cols = ["start_timestamp", "end_timestamp", "start_phone_use_timestamp", "end_phone_use_timestamp"]
    app_usage = process_application_usage_data(user_id)
    for col in dt_cols:
        app_usage = app_usage.withColumn(f"{col[:col.rfind('_')]}_datetime", udf_datetime_from_timestamp(F.col(col).cast(FloatType()))-timedelta(hours=2))        
    app_usage = app_usage.withColumn("date", udf_get_date_from_datetime("start_datetime"))\
        .filter(F.col("date") == cur_day)

    phone_usage = app_usage.select("start_phone_use_datetime", "duration").distinct()
    # phone_usage = process_phone_usage_data(user_id)\
    #     .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_timestamp").cast(FloatType())))\
    #     .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("end_timestamp").cast(FloatType())))\
    #     .withColumn("date", udf_get_date_from_datetime("start_datetime"))\
    #     .filter(F.col("date") == cur_day)
    total_phone_use_duration = phone_usage.agg(F.sum("duration")).collect()[0][0]
    features["phone_use_duration"] = total_phone_use_duration

    app_usage_duration = app_usage.groupBy("category")\
        .agg((F.sum("usage_duration")).alias("total_usage_duration"))\
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
        .withColumn("bt_datetime", udf_datetime_from_timestamp(F.col("timestamp").cast(FloatType())))\
        .filter(F.col("date") == cur_day).sort("bt_datetime")
    features["unique_bluetooth_device"] = bluetooth_df.select("bt_address").distinct().count()

    # locations = complement_location_data(user_id).withColumnRenamed("datetime", "location_datetime")\
    # locations = cross_check_cluster_with_activity_state(user_id).withColumnRenamed("datetime", "location_datetime")\
    locations = resolve_cluster_fluctuations(user_id).withColumnRenamed("datetime", "location_datetime")\
        .withColumn("location_datetime", F.col("location_datetime")-timedelta(hours=2))\
        .withColumn("date", udf_get_date_from_datetime(F.col("location_datetime")))\
        .withColumn("hour", udf_get_hour_from_datetime(F.col("location_datetime")))\
        .withColumn("minute", udf_get_minute_from_datetime(F.col("location_datetime")))\
        .filter(F.col("date") == cur_day).sort("location_datetime")
    # .withColumn("next_location_datetime", F.col("next_location_datetime")-timedelta(hours=2))\

    # locations = locations.filter(F.col("date") < cur_day).orderBy(F.col("location_datetime").desc()).limit(1)\
    #     .withColumn("location_datetime", udf_generate_datetime(F.lit(cur_day), F.lit(0), F.lit(0)))\
    #     .union(locations.filter(F.col("date") == cur_day)).sort("location_datetime")

    wifi_df = locations.filter(F.col("ssid").isNotNull())
    features["unique_wifi_device"] = wifi_df.select("ssid").distinct().count()

    # Total distance travelled
    time_window = Window().orderBy("location_datetime")
    location_coordinates = locations.select("location_datetime", "double_latitude", "double_longitude", "cluster_id")\
        .dropDuplicates().dropna().sort("location_datetime")
    # distance_traveled = location_coordinates\
    #     .withColumn("next_latitude", F.lead(F.col("double_latitude")).over(time_window))\
    #     .withColumn("next_longitude", F.lead(F.col("double_longitude")).over(time_window))\
    #     .filter((F.col("next_latitude").isNotNull()) & (F.col("next_longitude").isNotNull()))\
    #     .withColumn("distance", distance(F.col("double_latitude"), F.col("double_longitude"),\
    #                                      F.col("next_latitude"), F.col("next_longitude")))
    # total_distance = distance_traveled.agg(F.sum("distance")).collect()[0][0]
    # features["total_distance_traveled"] = total_distance

    # # Location variance
    # latitude_variance = location_coordinates.agg(F.variance("double_latitude")).collect()[0][0]
    # longitude_variance = location_coordinates.agg(F.variance("double_longitude")).collect()[0][0]
    # location_variance = math.log(latitude_variance + longitude_variance)
    # features["location_variance"] = location_variance

    # Group WiFi devices at each time point to compute cluster transitions between consecutive time points
    cluster_transitions = locations.groupBy(*[col for col in locations.columns if col != "ssid"])\
        .agg(F.concat_ws(", ", F.collect_set("ssid")).alias("WiFi_devices"))\
        .dropDuplicates().sort("location_datetime")
    cluster_transitions = cluster_transitions.withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(time_window))\
        .withColumn("prev_location_datetime", F.lag(F.col("location_datetime")).over(time_window))\
        .drop(*time_cols).filter(F.col("prev_cluster") != F.col("cluster_id")).sort("location_datetime")

    # V2:
    # cluster_transitions = locations.select("location_datetime", "next_location_datetime", "cluster_id", "next_cluster")\
    #     .dropDuplicates().filter(F.col("cluster_id") != F.col("next_cluster")).sort("location_datetime")
    # visualize_day_contexts(user_id, cur_day, physical_mobility, ambient_light, ambient_noise, app_usage, cluster_transitions)

    # NOTE: (N+1) cluster analysis will be involved for N cluster transition points
    context_dfs = [physical_mobility, ambient_light, ambient_noise, app_usage, bluetooth_df, locations]
    context_df_datetime_cols = ["datetime", "datetime", "datetime", "start_datetime", "bt_datetime", "location_datetime"]

    # First row will always be the first filtered row for the day of interest
    location_transition_datetimes = np.array(cluster_transitions.select("location_datetime").collect()).flatten()
    end_of_day = datetime.strptime(f"{cur_day} 21:59", "%Y-%m-%d %H:%M").astimezone(TIMEZONE)
    # location_transition_datetimes = np.array(cluster_transitions\
    #     .filter(F.col("location_datetime")<=end_of_day)\
    #     .select("next_location_datetime").collect()).flatten()
    if len(location_transition_datetimes) == 0:
        location_clusters = np.array(locations.select("cluster_id").distinct().dropna().collect()).flatten()
        features["cluster_count"] = 1
        features["unique_cluster_count"] = 1
    else:
        first_cluster = np.array(cluster_transitions.select("prev_cluster").collect()).flatten()[0]
        location_clusters = np.append(first_cluster, np.array(cluster_transitions.select("cluster_id").collect()).flatten())
        # first_cluster = np.array(cluster_transitions.select("cluster_id").dropna().collect()).flatten()[0]
        # location_clusters = np.append(first_cluster, np.array(cluster_transitions.select("next_cluster").dropna().collect()).flatten())
        features["cluster_count"] = len(location_clusters)
        features["unique_cluster_count"] = len(list(set(location_clusters)))

    epoch_list = list(TIME_EPOCHS.keys())
    cluster_time_range = []
    cluster_context_dfs = [[] for _ in range(len(context_dfs))]
    for cluster_index, cluster in enumerate(location_clusters):
        if cluster_index == 0:
            if len(location_transition_datetimes) == 0:
                cluster_end_datetime = end_of_day
            else:
                cluster_end_datetime = (location_transition_datetimes[cluster_index]).astimezone(TIMEZONE)
            # Compute start datetime as the minimum of all context dataframes
            min_context_datetime = []
            for context_index, context_df in enumerate(context_dfs):
                cur_df = context_df.filter(F.col(context_df_datetime_cols[context_index]) < cluster_end_datetime)
                if cur_df.count() > 0:
                    min_context_datetime.append(cur_df.agg(F.min(context_df_datetime_cols[context_index])).collect()[0][0])
                cluster_context_dfs[context_index].append(cur_df)
            cluster_start_datetime = np.min(min_context_datetime).astimezone(TIMEZONE)
        elif cluster_index < len(location_clusters)-1:
            cluster_start_datetime = (location_transition_datetimes[cluster_index-1]).astimezone(TIMEZONE)
            cluster_end_datetime = (location_transition_datetimes[cluster_index]).astimezone(TIMEZONE)
            for context_index, context_df in enumerate(context_dfs):
                cluster_context_dfs[context_index].append(context_df\
                    .filter((F.col(context_df_datetime_cols[context_index]) >= cluster_start_datetime) &\
                        (F.col(context_df_datetime_cols[context_index]) < cluster_end_datetime)))
        else:
            cluster_start_datetime = (location_transition_datetimes[-1]).astimezone(TIMEZONE)
            cluster_end_datetime = end_of_day
            for context_index, context_df in enumerate(context_dfs):
                cluster_context_dfs[context_index].append(context_df.filter(F.col(context_df_datetime_cols[context_index]) >= cluster_start_datetime))
        cluster_time_range.append((cluster_start_datetime, cluster_end_datetime))

        # start_hour = cluster_start_datetime.time().hour
        # end_hour = cluster_end_datetime.time().hour
        # start_epoch = get_epoch_from_hour(start_hour)
        # end_epoch =  get_epoch_from_hour(end_hour)
        # # Check if the cluster spans across multiple epochs of the day
        # if start_epoch != end_epoch:
        #     epoch_count = (end_epoch - start_epoch) % len(epoch_list)
        #     epoch_start_time = cluster_start_datetime
        #     for _ in range(epoch_count):
        #         # Compute individual cluster features for each epoch
        #         max_epoch_time = TIME_EPOCHS[epoch_list[start_epoch]]['max']
        #         epoch_end_time = datetime.strptime(f"{cur_day} {max_epoch_time-2 if max_epoch_time > 0 else 24-max_epoch_time-2}:00", "%Y-%m-%d %H:%M").astimezone(TIMEZONE)
        #         cluster_epoch_features = extract_epoch_cluster_contexts(*[user_id, (epoch_start_time, epoch_end_time)] +\
        #             [cluster_context_dfs[i][cluster_index].filter((F.col(context_df_datetime_cols[i]) >= epoch_start_time) &\
        #                                       (F.col(context_df_datetime_cols[i]) < epoch_end_time))\
        #                                         for i in range(len(cluster_context_dfs))])
        #         for key, value in cluster_epoch_features.items():
        #             features[f"cluster{cluster}_{epoch_list[get_epoch_from_hour(epoch_start_time.time().hour)]}_{key}"] = value
        #         epoch_start_time = epoch_end_time
        #         start_epoch = get_epoch_from_hour(epoch_start_time.time().hour)
            
        #     cluster_epoch_features = extract_epoch_cluster_contexts(*[user_id, (epoch_start_time, cluster_end_datetime)] +\
        #         [cluster_context_dfs[i][cluster_index].filter(F.col(context_df_datetime_cols[i]) >= epoch_start_time)\
        #                                     for i in range(len(cluster_context_dfs))])
        #     cluster_feature_keys = list(cluster_epoch_features.keys())
        #     for key, value in cluster_epoch_features.items():
        #         features[f"cluster{cluster}_{epoch_list[get_epoch_from_hour(epoch_start_time.time().hour)]}_{key}"] = value
        # else:
        #     cluster_features = extract_epoch_cluster_contexts(*[user_id, cluster_time_range[cluster_index]] + [df[cluster_index] for df in cluster_context_dfs])
        #     cluster_feature_keys = list(cluster_features.keys())
        #     for key, value in cluster_features.items():
        #         features[f"cluster{cluster}_{epoch_list[start_epoch]}_{key}"] = value
        
        # Visualize context details within each cluster (regardless of time epoch)
        if visualize_cluster_contexts:
            # Update physical activity df to compute consecutive duration
            physical_df = cluster_context_dfs[0][cluster_index]
            time_window = Window().orderBy("datetime")
            if physical_df.count() > 0:
                physical_df = physical_df.withColumn("prev_activity", F.lag(F.col("activity_name")).over(time_window))\
                    .withColumn("next_datetime", F.lead(F.col("datetime")).over(time_window))\
                    .withColumn("new_group", (F.col("activity_name") != F.col("prev_activity")).cast("int"))\
                    .withColumn("group_id", F.sum("new_group").over(time_window.rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
                    .groupBy("group_id", "activity_name", "activity_type")\
                    .agg(F.min("datetime").alias("start_datetime"),\
                            F.max("next_datetime").alias("end_datetime"))\
                    .drop("group_id").sort("start_datetime")\
                    .withColumn("consecutive_duration", (F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime"))))\
                    .dropna()
                cluster_context_dfs[0][cluster_index] = physical_df
            
            # Update application usage df to compute duration normalized by each phone use duration
            app_usage_df = cluster_context_dfs[3][cluster_index]
            if app_usage_df.count() > 0:
                # app_usage_df.sort("start_phone_use_datetime").show()
                app_usage_df = app_usage_df.groupBy("start_phone_use_datetime", "end_phone_use_datetime", "duration", "category")\
                    .agg(F.sum("usage_duration").alias("total_duration"),\
                        F.collect_list(F.col("application_name")).alias("apps"))
                app_usage_df = app_usage_df.withColumn("normalized_usage_duration", F.col("total_duration")/F.col("duration"))
                # Check and cap phone use start and end datetime within the current time window
                app_usage_df = app_usage_df.withColumn("start_phone_use_datetime", F.when(F.col("start_phone_use_datetime") < cluster_start_datetime, F.lit(cluster_start_datetime))\
                    .otherwise(F.col("start_phone_use_datetime")))\
                    .withColumn("end_phone_use_datetime", F.when(F.col("end_phone_use_datetime") > cluster_end_datetime, F.lit(cluster_end_datetime))\
                    .otherwise(F.col("end_phone_use_datetime")))
                cluster_context_dfs[3][cluster_index] = app_usage_df

            arranged_contexts = [cluster_context_dfs[i][cluster_index] for i in [1, 2, 0, 3]]
            # visualize_context_breakdown(user_id, *[df.toPandas() if df.count() > 0 else None for df in arranged_contexts],\
            #     f"Contexts At Location Cluster {cluster}: {cur_day} {pd.to_datetime(cluster_time_range[cluster_index][0]).strftime('%H:%M')} - {pd.to_datetime(cluster_time_range[cluster_index][1]).strftime('%H:%M')}",\
            #     f"{cur_day}_cluster{cluster}_{pd.to_datetime(cluster_time_range[cluster_index][0]).strftime('%H%M')}_contexts")
    
    # Retrieves pre-computed clusters
    # overall_location_info = contexts["location_clusters"]
    # overall_clusters = [int(id) for id in overall_location_info.keys()]
    # # Make sure that cluster features exist for each combination of location cluster and epoch
    # for cluster_id in overall_clusters:
    #     for epoch in epoch_list:
    #         if f"cluster{cluster_id}_{epoch}_{cluster_feature_keys[0]}" not in features:
    #             for col in cluster_feature_keys:
    #                 features[f"cluster{cluster_id}_{epoch}_{col}"] = 0

    # # Count of unknown locations, normalized by total location entries
    # # Locations could have multiple entries at the same timestamp due to multiple WiFi devices
    # unique_location_entries = locations.select(*["location_datetime", "cluster_id"] +\
    #                                            [col for col in locations.columns if "double_" in col]).dropDuplicates()
    # unknown_location_count = unique_location_entries.filter(F.col("cluster_id") == -1).count()
    # normalized_unknown_location_count = unknown_location_count / unique_location_entries.count()
    # features["unknown_location_count"] = unknown_location_count
    # features["normalized_unknown_location_count"] = normalized_unknown_location_count

    # cluster_time_spent = [0 for _ in range(len(overall_clusters))]
    # # Time spent in each cluster
    # for cluster_index, cluster_id in enumerate(location_clusters):
    #     overall_index = overall_clusters.index(cluster_id)
    #     cluster_time_spent[overall_index] += (cluster_time_range[cluster_index][1] - cluster_time_range[cluster_index][0]).total_seconds()
    # # Entropy and normalized entropy
    # # Add a small negligible value to avoid log(0) when a specific cluster is not visited
    # probability = np.array(cluster_time_spent)/(24*3600) + 1e-10
    # entropy = - np.sum(probability * np.log(probability))
    # normalized_entropy = entropy/math.log(len(overall_clusters))
    # features["location_entropy"] = entropy
    # features["normalized_location_entropy"] = normalized_entropy

    # # Time spent at primary clusters
    # time_spent_primary_cluster = 0
    # time_spent_secondary_cluster = 0
    # for index, cluster_id in enumerate(overall_clusters):
    #     features[f"time_spent_cluster{cluster_id}"] = cluster_time_spent[index]
    #     if overall_location_info[str(cluster_id)]["is_primary"]:
    #         time_spent_primary_cluster = cluster_time_spent[index]
    #     elif overall_location_info[str(cluster_id)]["is_secondary"]:
    #         time_spent_secondary_cluster = cluster_time_spent[index]
    # features["time_spent_primary_cluster"] = time_spent_primary_cluster
    # features["time_spent_secondary_cluster"] = time_spent_secondary_cluster

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

def pd_time_to_midnight_hours(t):
    hour = t.hour
    minute = t.minute
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

def retrieve_sleep_ema(user_id, esm_ids=[3, 4, 1]):
    """
    Retrieves EMA responses related to sleep: onset and wake times and sleep quality.
    Computes sleep duration by considering potential errors in reported times.
    Reference of error handling: https://www.jmir.org/2017/4/e118/
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
        .withColumn("sleep_quality_rating", F.col("sleep_quality_rating").cast(IntegerType()))\
        .filter((F.col("reported_sleep_time").isNotNull()) & F.col("reported_wake_time").isNotNull()).toPandas()
    
    # Resolves potential errors in reported times
    date_format = "%Y-%m-%d"
    datetime_format = f"{date_format} %H:%M"

    sleep_durations = []
    adjusted_sleep_times = []
    adjusted_wake_times = []
    adjusted_sleep_datetimes = []
    adjusted_wake_datetimes = []

    for _, row in esm_df.iterrows():
        cur_date = row["date"]
        cur_datetime = pd.to_datetime(cur_date, format=date_format)
        day_before = (cur_datetime - pd.DateOffset(days=1)).strftime(date_format)
        onset_time = row["reported_sleep_time"].hour + row["reported_sleep_time"].minute/60
        wake_time = row["reported_wake_time"].hour + row["reported_wake_time"].minute/60

        # Falls asleep after midnight
        if wake_time > onset_time:
            sleep_duration = wake_time - onset_time

        # Falls asleep before midnight
        elif (onset_time > wake_time):
            sleep_duration = (24 - onset_time) + wake_time
        else:
            # Assumes that a mistake is made: correct to 12 hrs by default.
            if wake_time > 12:
                wake_time = wake_time - 12
            else:
                onset_time = onset_time - 12
            sleep_duration = 12
        
        # Assumes that a mistake is made (e.g., 00:00 was input as 12:00)
        if sleep_duration >= 15:
            onset_time = onset_time - 12
            sleep_duration = sleep_duration - 12        
        if onset_time > wake_time:
            sleep_date = day_before
        else:
            sleep_date = cur_date
        
        # Include updated sleep onset and wake datetimes
        sleep_hour = int(onset_time)
        sleep_minute = round((onset_time - sleep_hour) * 60)
        adjusted_sleep_datetimes.append(pd.to_datetime(sleep_date + " " + str(sleep_hour).zfill(2) +\
        ":" + str(sleep_minute).zfill(2), format=datetime_format))

        wake_hour = int(wake_time)
        wake_minute = round((wake_time - wake_hour) * 60)
        adjusted_wake_datetimes.append(pd.to_datetime(cur_date + " " + str(wake_hour).zfill(2) +\
        ":" + str(wake_minute).zfill(2), format=datetime_format))

        adjusted_sleep_times.append(onset_time)
        adjusted_wake_times.append(wake_time)
        sleep_durations.append(sleep_duration)

    esm_df["adjusted_sleep_time"] = adjusted_sleep_times
    esm_df["adjusted_wake_time"] = adjusted_wake_times
    esm_df["sleep_duration"] = sleep_durations
    esm_df["adjusted_sleep_datetime"] = adjusted_sleep_datetimes
    esm_df["adjusted_wake_datetime"] = adjusted_wake_datetimes

    return esm_df

def map_overview_estimated_sleep_duration_to_sleep_ema(user_id, esm_ids):
    """
    (Overview - multiple days)
    Maps estimated sleep duration to self-reported sleep duration and sleep quality rating.
    Plots vertical bars showing estimated sleep duration colored based on quality rating and overlayed with reported sleep duration.
    """
    esm_df = retrieve_sleep_ema(user_id, esm_ids)
    sleep_df = estimate_sleep(user_id).withColumn("date", udf_get_date_from_datetime("end_datetime"))\
        .withColumn("start_datetime", F.col("start_datetime")-timedelta(hours=2))\
        .withColumn("end_datetime", F.col("end_datetime")-timedelta(hours=2))
    
    # Identify daily sleep duration from the longest sleep duration for each day
    wake_df = sleep_df.groupBy("date").agg(F.min("end_datetime").alias("end_datetime"))\
        .join(sleep_df, ["date", "end_datetime"])
    
    # Map estimated sleep duration to self-reported sleep duration
    combined_sleep_df = wake_df.join(esm_df, "date", "left")
    datetime_cols = ["start_datetime", "end_datetime", "reported_sleep_time", "reported_wake_time"]
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

def visualize_reported_sleep_ema(user_id):
    """
    Plots vertical bars, each showing reported sleep duration colored based on quality rating.
    """
    esm_df = retrieve_sleep_ema(user_id)
    datetime_cols = ["adjusted_sleep_time", "adjusted_wake_time"]
    for col in datetime_cols:
        condition = esm_df[col] > 12
        esm_df.loc[condition, col] = esm_df.loc[condition, col] - 24

    _, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=1, vmax=5)

    for i, row in esm_df.iterrows():
        day_index = i  # x-axis position
        start = row["adjusted_sleep_time"]
        end = row["adjusted_wake_time"]
        
        # Plot estimated sleep duration and overlay with self-reported start and end datetime
        if np.isnan(row["sleep_quality_rating"]):
            bar_color = "grey"
        else:
            bar_color = cmap(norm(int(row["sleep_quality_rating"])))
        ax.bar(day_index, start-end, bottom=end, color=bar_color, edgecolor="black", width=0.5, alpha=0.9)
        
    # X-axis as date
    ax.set_xticks(np.arange(len(esm_df)))
    ax.set_xticklabels(esm_df["date"], rotation=30)

    # Y-axis
    min_y = math.floor(esm_df[datetime_cols].min().min())
    max_y = math.ceil(esm_df[datetime_cols].max().max())
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
    ax.set_ylabel("Reported sleep duration")
    ax.yaxis.grid()
    plt.gca().invert_yaxis()

    # Title and legend
    ax.set_title('Sleep Duration and Quality Rating Across A Week')
    legend_elements = [mpatches.Patch(facecolor="grey", edgecolor="black", label="No rating")]
    for index, rating in enumerate(SLEEP_QUALITY_RATINGS):
        legend_elements.append(mpatches.Patch(facecolor=cmap(norm(index+1)), edgecolor="black", label=rating))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), title="Sleep quality rating")

    plt.show()

def visualize_reported_sleep_ema_with_overlapping_sleep_estimate(user_id):
    """
    Plots estimated sleep windows that overlap with reported sleep duration.
    The y-axis is centred around 12 AM.
    """
    estimated_sleep_df = estimate_sleep(user_id)\
        .withColumn("start_datetime", F.col("start_datetime")-timedelta(hours=2))\
        .withColumn("end_datetime", F.col("end_datetime")-timedelta(hours=2)).sort("start_datetime").toPandas()
    reported_sleep_df = retrieve_sleep_ema(user_id).sort_values(by="adjusted_sleep_datetime")

    for col in ["start_datetime", "end_datetime"]:
        cur_col = f"{col[:-8]}time"
        estimated_sleep_df[cur_col] = estimated_sleep_df[col].dt.hour + estimated_sleep_df[col].dt.minute/60
        condition = estimated_sleep_df[cur_col] > 12
        estimated_sleep_df.loc[condition, cur_col] = estimated_sleep_df.loc[condition, cur_col] - 24

    for col in ["adjusted_sleep_time", "adjusted_wake_time"]:
        condition = reported_sleep_df[col] > 12
        reported_sleep_df.loc[condition, col] = reported_sleep_df.loc[condition, col] - 24

    _, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=1, vmax=5)
    min_y = np.inf
    max_y = -np.inf

    # Plot duration of reported sleep
    for i, row in reported_sleep_df.iterrows():
        day_index = i
        sleep_time = row["adjusted_sleep_time"]
        wake_time = row["adjusted_wake_time"]
        if sleep_time < min_y:
            min_y = sleep_time
        if wake_time > max_y:
            max_y = wake_time

        # Plot estimated sleep duration and overlay with self-reported start and end datetime
        if np.isnan(row["sleep_quality_rating"]):
            bar_color = "grey"
        else:
            bar_color = cmap(norm(int(row["sleep_quality_rating"])))

        ax.bar(day_index, sleep_time-wake_time, bottom=wake_time, color=bar_color, edgecolor="black", width=0.5, alpha=0.9)

        overlapping_estimated_window = estimated_sleep_df[(estimated_sleep_df["start_datetime"]<=row["adjusted_wake_datetime"])&\
                                                          (estimated_sleep_df["end_datetime"]>=row["adjusted_sleep_datetime"])]
        if len(overlapping_estimated_window) > 0:
            for _, inner_row in overlapping_estimated_window.iterrows():
                start = inner_row["start_time"]
                end = inner_row["end_time"]
                ax.plot([i - 0.2, i + 0.2], [start, start], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
                ax.plot([i - 0.2, i + 0.2], [end, end], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
                ax.vlines(i, ymin=start, ymax=end, color="red", linestyle='-')
                
                if start < min_y:
                    min_y = start
                if end > max_y:
                    max_y = end

    # X-axis as date
    ax.set_xticks(np.arange(len(reported_sleep_df["date"])))
    ax.set_xticklabels(reported_sleep_df["date"], rotation=30)

    # Y-axis
    min_y = math.floor(min_y)
    max_y = math.ceil(max_y)
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
    ax.set_ylabel("Sleep Times")
    ax.yaxis.grid()
    plt.gca().invert_yaxis()

    # Title and legend
    ax.set_title("Reported Sleep Times with Overlapping Estimated Sleep Windows")
    legend_elements = []
    legend_elements = [plt.Line2D([0], [0], color="red", lw=2, label="Estimated sleep window")]
    legend_elements.append(plt.Line2D([], [], color="none", label=""))
    legend_elements.append(plt.Line2D([], [], color="none", label="Sleep quality rating"))
    legend_elements.append(mpatches.Patch(facecolor="grey", edgecolor="black", label="No rating"))
    for index, rating in enumerate(SLEEP_QUALITY_RATINGS):
        legend_elements.append(mpatches.Patch(facecolor=cmap(norm(index+1)), edgecolor="black", label=rating))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def visualize_estimated_sleep_windows_against_reported_sleep(user_id):
    """
    Plots all estimated sleep windows with reported sleep duration.
    The y-axis spans for 24 hours of the day to consider daytime naps.
    """
    date_format = "%Y-%m-%d"
    estimated_sleep_df = estimate_sleep(user_id)\
        .withColumn("start_datetime", F.col("start_datetime")-timedelta(hours=2))\
        .withColumn("end_datetime", F.col("end_datetime")-timedelta(hours=2)).sort("start_datetime").toPandas()
    # estimated_sleep_df = estimated_sleep_df[estimated_sleep_df["duration (hr)"] > 1.5]
    
    for col in ["start_datetime", "end_datetime"]:
        estimated_sleep_df[col[:-4]] = pd.to_datetime(estimated_sleep_df[col]).dt.strftime(date_format)
        estimated_sleep_df[col] = estimated_sleep_df[col].dt.hour + estimated_sleep_df[col].dt.minute/60

    reported_sleep_df = retrieve_sleep_ema(user_id)
    for col in ["adjusted_sleep_datetime", "adjusted_wake_datetime"]:
        reported_sleep_df[col[:-4]] = pd.to_datetime(reported_sleep_df[col]).dt.strftime(date_format)
        reported_sleep_df[col] = reported_sleep_df[col].dt.hour + reported_sleep_df[col].dt.minute/60

    # Get unique dates from reported and estimated sleep
    unique_dates = pd.concat([reported_sleep_df["adjusted_sleep_date"], reported_sleep_df["adjusted_wake_date"],\
                              estimated_sleep_df["start_date"], estimated_sleep_df["end_date"]]).unique().tolist()
    unique_dates.sort()

    _, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap('Blues')
    norm = mcolors.Normalize(vmin=1, vmax=5)

    # Plot duration of reported sleep
    for _, row in reported_sleep_df.iterrows():
        start_date = row["adjusted_sleep_date"]
        end_date = row["adjusted_wake_date"]

        # Plot estimated sleep duration and overlay with self-reported start and end datetime
        if np.isnan(row["sleep_quality_rating"]):
            bar_color = "grey"
        else:
            bar_color = cmap(norm(int(row["sleep_quality_rating"])))

        if start_date != end_date:
            start_day_index = unique_dates.index(start_date)
            end_day_index = unique_dates.index(end_date)

            ax.bar(start_day_index, row["adjusted_sleep_time"]-24, bottom=24, color=bar_color, edgecolor="black", width=0.5, alpha=0.9)
            ax.bar(end_day_index, 0-row["adjusted_wake_time"], bottom=row["adjusted_wake_time"], color=bar_color, edgecolor="black", width=0.5, alpha=0.9)
        else:
            day_index = unique_dates.index(row["date"])
            start = row["adjusted_sleep_time"]
            end = row["adjusted_wake_time"]

            ax.bar(day_index, start-end, bottom=end, color=bar_color, edgecolor="black", width=0.5, alpha=0.9)

    # Plot estimated sleep windows
    for index, row in estimated_sleep_df.iterrows():
        start_date = row["start_date"]
        end_date = row["end_date"]

        if start_date != end_date:
            start_day_index = unique_dates.index(start_date)
            end_day_index = unique_dates.index(end_date)

            ax.plot([start_day_index - 0.2, start_day_index + 0.2], [row["start_datetime"], row["start_datetime"]], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
            ax.plot([start_day_index - 0.2, start_day_index + 0.2], [24, 24], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
            ax.vlines(start_day_index, ymin=row["start_datetime"], ymax=24, color="red", linestyle='-')
            
            ax.plot([end_day_index - 0.2, end_day_index + 0.2], [0, 0], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
            ax.plot([end_day_index - 0.2, end_day_index + 0.2], [row["end_datetime"], row["end_datetime"]], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
            ax.vlines(end_day_index, ymin=0, ymax=row["end_datetime"], color="red", linestyle='-')

        else:
            day_index = unique_dates.index(end_date)
            start = row["start_datetime"]
            end = row["end_datetime"]

            ax.plot([day_index - 0.2, day_index + 0.2], [start, start], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
            ax.plot([day_index - 0.2, day_index + 0.2], [end, end], color="red", markersize=10, linestyle='-', linewidth=2, alpha=0.8)
            ax.vlines(day_index, ymin=start, ymax=end, color="red", linestyle='-')
                
    # X-axis as date
    ax.set_xticks(np.arange(len(unique_dates)))
    ax.set_xticklabels(unique_dates, rotation=30)

    # Y-axis
    ax.set_ylim(0, 24)
    ax.set_yticks(range(0, 24, 2))
    y_labels = []
    for time in range(0, 24, 2):
        if time < 12:
            y_labels.append(f"{time} AM")
        else:
            y_labels.append(f"{time} PM")

    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Time of the Day")
    ax.yaxis.grid()
    plt.gca().invert_yaxis()

    # Title and legend
    ax.set_title("Estimated Sleep Windows and Reported Sleep Times")
    legend_elements = []
    legend_elements = [plt.Line2D([0], [0], color="red", lw=2, label="Estimated sleep window")]
    legend_elements.append(plt.Line2D([], [], color="none", label=""))
    legend_elements.append(plt.Line2D([], [], color="none", label="Sleep quality rating"))
    legend_elements.append(mpatches.Patch(facecolor="grey", edgecolor="black", label="No rating"))
    for index, rating in enumerate(SLEEP_QUALITY_RATINGS):
        legend_elements.append(mpatches.Patch(facecolor=cmap(norm(index+1)), edgecolor="black", label=rating))

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.9, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()

def get_unique_apps(user_id):
    """
    Retrieves all unique applications used.
    """
    app_usage_df = process_application_usage_data(user_id)
    print(np.array(app_usage_df.select("application_name").distinct().collect()).flatten())

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
    with open(f"{user_id}/{user_id}_contexts.json") as f:
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
    1. Manually include app categories into contexts.json file after it's being created.
    2. Other system applications are assumed to be in "utilities" category.

    References:
    1. https://www.sciencedirect.com/science/article/pii/004724849290046C - threshold for audio frequency

    # TODO:
    1. Estimate home based on sleep locations
    2. Compute primary contacts
    3. Automate labeling of location clusters and app categories
    """
    context_file = f"{user_id}/{user_id}_contexts.json"
    if os.path.exists(context_file):
        with open(context_file, "r") as f:
            contexts = json.load(f)
    else:
        contexts = {}

    light_df = process_light_data(user_id)
    if light_df is not None:
        brightness_quartiles = light_df.approxQuantile("mean_light_lux", [0.25, 0.50], 0.01)
        contexts["dark_threshold"] = round(brightness_quartiles[1])
    else:
        contexts["dark_threshold"] = None

    noise_df = process_raw_noise_data(user_id)
    if noise_df is not None:
        audio_quartiles = noise_df.approxQuantile("double_decibels", [0.25, 0.50], 0.01)
        # silent_threshold = round((audio_quartiles[1]-audio_quartiles[0]) / 2)
        contexts["silent_threshold"] = round(audio_quartiles[1])
        rms_quartiles = noise_df.approxQuantile("double_rms", [0.75], 0.01)
        contexts["rms_threshold"] = round(rms_quartiles[0])
        contexts["frequency_threshold"] = 60
    else:
        for key in ["silent_threshold", "rms_threshold", "frequency_threshold"]:
            contexts[key] = None

    # Frequently seen WiFi devices with top 95% percentile occurrence
    wifi_df = process_wifi_data(user_id)
    if wifi_df is not None:
        wifi_df = wifi_df.withColumn("hour", F.col("hour").cast(IntegerType()))\
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
    else:
        contexts["primary_wifi_devices"] = None

    # Frequently seen Bluetooth devices with top 95% percentile occurrence
    bluetooth_df = process_bluetooth_data(user_id)
    if bluetooth_df is not None:
        bluetooth_df = bluetooth_df.withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
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
    else:
        contexts["primary_bluetooth_devices"] = None

    # Location clusters
    location_df = complement_location_data(user_id)
    unique_ssid_clusters = location_df.select("datetime", "cluster_id", "ssid").distinct().dropna()\
        .groupBy("cluster_id", "ssid").agg(F.count("datetime").alias("WiFi_occurrence"))
    # Major location clusters based on visit datetimes
    cluster_visits = location_df.select("datetime", "cluster_id").distinct()\
        .groupBy("cluster_id").agg(F.count("datetime").alias("cluster_visits"))
    cluster_visits = cluster_visits.filter(F.col("cluster_id") != -1).sort("cluster_visits", ascending=False)\
        .union(cluster_visits.filter(F.col("cluster_id") == -1)).toPandas()
    valid_clusters = [clust_id for clust_id in list(cluster_visits["cluster_id"]) if clust_id != -1]
    # cluster_limit = 2
    # if len(valid_clusters) < cluster_limit:
    #     cluster_limit = len(valid_clusters)
    # major_clusters = list(valid_clusters["cluster_id"][:cluster_limit])
    
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
            index = valid_clusters.index(int(cluster_row["cluster_id"]))
            if index == 0:
                cur_cluster_dict["is_primary"] = True
            elif index == 1:
                cur_cluster_dict["is_secondary"] = True
        except ValueError:
            pass
        cur_ssids = unique_ssid_clusters.filter(F.col("cluster_id") == int(cluster_row["cluster_id"]))\
            .rdd.map(lambda row: row.asDict()).collect()
        Wifi_cols = ["total_WiFi_occurrence", "max_WiFi_occurrence", "min_WiFi_occurrence"]
        if len(cur_ssids) > 0:
            for col in Wifi_cols:
                cur_cluster_dict[col] = cur_ssids[0][col]
            ssid_occurrences_dict = {}
            for row in cur_ssids:
                ssid_occurrences_dict[row["ssid"]] = row["weighted_occurrence"]
            cur_cluster_dict["ssids"] = ssid_occurrences_dict
        else:
            for col in Wifi_cols:
                cur_cluster_dict[col] = 0
            cur_cluster_dict["ssids"] = {}
        cluster_ssids_dict[int(cluster_row["cluster_id"])] = cur_cluster_dict
    contexts["location_clusters"] = cluster_ssids_dict

    if not os.path.isdir(user_id):
        os.mkdir(user_id)
    with open(f"{user_id}/{user_id}_contexts.json", "w") as f:
        json.dump(contexts, f)

@F.udf(IntegerType())
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

def extract_epoch_cluster_contexts(user_id, cluster_time_range, activity_df, light_df, noise_df, app_usage_df, bluetooth_df, location_df):
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
    with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
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

    activities = ["still", "in_vehicle", "on_bicycle", "tilting", "walking", "running"]
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
        total_phone_use_duration = app_usage_df.select("start_phone_use_timestamp", "duration").distinct()\
            .agg(F.sum("duration")).collect()[0][0]
        cluster_features["normalized_phone_use_duration"] = total_phone_use_duration/total_time_spent
        app_usage_duration = app_usage_df.groupBy("category")\
            .agg((F.sum("usage_duration")).alias("total_usage_duration"))\
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

def day_features_vs_mood(user_id):
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
        features = extract_day_features(user_id, date)
        features["reported_mood"] = reported_mood[index]
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
    corr_df.to_csv(f"{user_id}_day_feature_correlations.csv", index=False)

def process_sleep_data(user_id):
    """
    Analyzes self-reported sleep data and extracts features during reported sleep duration.
    Updates thresholds of dark and quiet environments based on available data.
    """
    pickle_filename = f"{DATA_FOLDER}/{user_id}_sleep_features.pkl"
    if not os.path.exists(pickle_filename):
        reported_sleep_df = retrieve_sleep_ema(user_id)
        physical_mobility_df = process_activity_data(user_id)\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
            .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
            .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
            .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime"))).sort("datetime").toPandas()
        light_df = process_light_data(user_id)\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
            .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
            .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
            .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime"))).sort("datetime").toPandas()
        noise_df = process_noise_data_with_conv_estimate(user_id)\
            .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .withColumn("datetime", F.col("datetime")-timedelta(hours=2))\
            .withColumn("date", udf_get_date_from_datetime(F.col("datetime")))\
            .withColumn("hour", udf_get_hour_from_datetime(F.col("datetime")))\
            .withColumn("minute", udf_get_minute_from_datetime(F.col("datetime"))).sort("datetime").toPandas()
        phone_use_df = process_phone_usage_data(user_id)\
            .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_timestamp"))-timedelta(hours=2))\
            .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("end_timestamp"))-timedelta(hours=2)).toPandas()
        app_usage_df = process_application_usage_data(user_id)\
            .withColumn("start_timestamp", F.col("start_timestamp").cast(FloatType()))\
            .withColumn("start_datetime", udf_datetime_from_timestamp("start_timestamp")-timedelta(hours=2))\
            .withColumn("end_timestamp", F.col("end_timestamp").cast(FloatType()))\
            .withColumn("end_datetime", udf_datetime_from_timestamp("end_timestamp")-timedelta(hours=2))\
            .withColumn("start_phone_use_datetime", udf_datetime_from_timestamp("start_phone_use_timestamp")-timedelta(hours=2))\
            .withColumn("end_phone_use_datetime", udf_datetime_from_timestamp("end_phone_use_timestamp")-timedelta(hours=2))\
            .withColumn("date", udf_get_date_from_datetime("start_datetime"))\
            .sort("start_timestamp").toPandas()
        bluetooth_df = process_bluetooth_data(user_id)\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("bt_datetime", udf_datetime_from_timestamp("timestamp")).sort("bt_datetime").toPandas()
        location_df = complement_location_data(user_id).sort("datetime").toPandas()
        location_with_coordinates = location_df[["datetime", "double_latitude", "double_longitude", "double_altitude"]]\
            .dropna().drop_duplicates().sort_values(by="datetime").reset_index(drop=True)
        wifi_df = location_df[["datetime", "ssid"]].dropna().drop_duplicates().sort_values(by="datetime").reset_index(drop=True)

        context_dfs = [physical_mobility_df, light_df, noise_df, location_with_coordinates, phone_use_df, app_usage_df, bluetooth_df, wifi_df]
        context_df_datetime_cols = ["datetime", "datetime", "datetime",  "datetime", "start_datetime", "start_datetime", "bt_datetime", "datetime"]
        
        daily_sleep_features = []
        readable_daily_sleep_features = []
        for _, row in reported_sleep_df.iterrows():
            sleep_time = row["adjusted_sleep_datetime"]
            wake_time = row["adjusted_wake_datetime"]
            cur_day_dfs = []
            for df_index, df in enumerate(context_dfs):
                dt_col = context_df_datetime_cols[df_index]
                cur_df = df[(df[dt_col]>=sleep_time) & (df[dt_col]<=wake_time)]
                if df_index == 4: # Find overlapping duration of active phone usage
                    cur_df = df[(df["start_datetime"]<=wake_time) & (df["end_datetime"]>=sleep_time)]
                elif df_index <= 3:
                    # Get last entry before the filter window and append to the current df
                    last_entry_before_dt = df[df[dt_col]<sleep_time].iloc[-1:]
                    first_entry_after_dt = df[df[dt_col]>wake_time].iloc[:1]
                    # if len(last_entry_before_dt) > 0:
                    #     last_entry_before_dt.loc[last_entry_before_dt.index[0], dt_col] = sleep_time
                    cur_df = pd.concat([last_entry_before_dt, cur_df, first_entry_after_dt]).sort_values(by=dt_col)
                cur_day_dfs.append(cur_df)
            features_during_sleep = extract_features_during_sleep(*[user_id, sleep_time, wake_time] + cur_day_dfs)

            # -- Saves features in readable form as reference -- 
            # Feature values that are stored separately as dataframes
            df_cols = ["light_df", "noise_df"]
            features_to_save = features_during_sleep.copy()
            for key in features_during_sleep.keys():
                if key in df_cols:
                    col_df = pd.DataFrame(features_during_sleep[key])
                    col_df.to_csv(f"{user_id}/{user_id}_{sleep_time.strftime('%m%d_%H%M')}_{key}.csv")
                    del features_to_save[key]
            readable_daily_sleep_features.append(features_to_save)
            # -- End of block --
            daily_sleep_features.append(features_during_sleep)
        daily_sleep_features = pd.DataFrame(daily_sleep_features)
            
        # NOTE: Update dark and silent thresholds according to self-reported sleep durations
        with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
            contexts = json.load(f)
        for index, element in enumerate(["luminance", "decibels"]):
            element_stats = []
            for stat in ["mean", "std"]:
                stats = np.array([d[f"{stat}_{element}"] for d in daily_sleep_features])
                # Use 95th percentile to detect and remove potential outliers
                Q5 = np.percentile(stats, 5)
                Q95 = np.percentile(stats, 95)
                if Q95 != Q5:   # In case all elements are of the same value
                    stats = stats[stats < Q95]
                element_stats.append(stats)
            overall_mean, overall_std = (*element_stats,)
            threshold = math.ceil(np.median(overall_mean) + 2*np.mean(overall_std))
            if index == 0:
                contexts["dark_threshold"] = threshold
            else:
                contexts["silent_threshold"] = threshold
        
        with open(f"{user_id}/{user_id}_contexts.json", "w") as f:
            json.dump(contexts, f)
        # -- End of block --

        # NOTE: Saves 2 versions of extracted features during reported sleep times
        # 1. Readable CSV form has light and noise dfs saved in separate csv files.
        # 2. Pickle form has all features (including light and noise dfs) for internal use.
        pd.DataFrame(readable_daily_sleep_features).to_csv(f"{user_id}/{user_id}_sleep_features.csv")
        daily_sleep_features.to_pickle(pickle_filename)
        
    daily_sleep_features = pd.read_pickle(pickle_filename)
    return daily_sleep_features

def extract_features_during_sleep(user_id, sleep_time, wake_time, activity_df, light_df, noise_df, location_df, phone_usage_df, app_usage_df, bluetooth_df, wifi_df):
    """
    Extracts features within the duration of input sleep and wake time.
    1. Occurrences of non still physical movement
    2. Min, max, mean, and standard deviation of ambient light and noise during sleep time.
    3. Phone usage frequency, total duration, and average duration.
    4. Sleep location and displacement
    """
    sleep_time_features = {"sleep_time": sleep_time, "wake_time": wake_time}

    # Find occurrences where activity is non-stationary
    grouped_activity_df = None
    if len(activity_df) > 0:
        activity_df = activity_df.sort_values(by="datetime").reset_index(drop=True)
        activity_df["next_datetime"] = activity_df["datetime"].shift(-1)
        activity_df["prev_activity"] = activity_df["activity_name"].shift(1)
        activity_df["new_group"] = (activity_df["activity_name"] != activity_df["prev_activity"]).astype(int)
        activity_df["group_id"] = activity_df["new_group"].cumsum()

        grouped_activity_df = activity_df.groupby(["group_id", "activity_name", "activity_type"]).agg(
            start_datetime=("datetime", "min"),
            end_datetime=("next_datetime", "max")).reset_index()
        grouped_activity_df["consecutive_duration"] = (grouped_activity_df["end_datetime"] -\
                                                    grouped_activity_df["start_datetime"]).dt.total_seconds()

        # Get all occurrences of non-still activity
        non_still_activity = grouped_activity_df.where(grouped_activity_df["activity_name"] != "still").dropna()
        if len(non_still_activity) > 0:
            non_still_occurrences = []
            for _, row in non_still_activity.iterrows():
                non_still_occurrences.append(row[["start_datetime", "activity_name", "activity_type", "consecutive_duration"]].to_dict())
            sleep_time_features["non_still_occurrences"] = non_still_occurrences
    
    if "non_still_occurrences" not in sleep_time_features:
        sleep_time_features["non_still_occurrences"] = []

    # Distribution of ambient light and noise
    stat_functions = ["min", "max", "mean", "std"]

    for stat in stat_functions:
        if len(light_df) > 0:
            sleep_time_features["light_df"] = light_df.to_dict(orient="records")
            # Min, max, mean, and standard deviation of ambient light
            sleep_time_features[f"{stat}_luminance"] = float(light_df[f"{stat}_light_lux"].agg(stat))
        else:
            sleep_time_features["light_df"] = None
            sleep_time_features[f"{stat}_luminance"] = 0

        if len(noise_df) > 0:
            sleep_time_features["noise_df"] = noise_df.to_dict(orient="records")
            sleep_time_features[f"{stat}_decibels"] = float(noise_df[f"{stat}_decibels"].agg(stat))
        else:
            sleep_time_features["noise_df"] = None
            sleep_time_features[f"{stat}_decibels"] = 0
    
    # Phone usage
    if len(phone_usage_df) > 0:
        sleep_time_features["phone_usage"] = phone_usage_df[["start_datetime", "end_datetime", "duration"]]\
            .to_dict(orient="records")
    else:
        sleep_time_features["phone_usage"] = []

    # Duration of application usage for each category
    app_usage = []
    if len(app_usage_df) > 0:
        category_app_usage = app_usage_df.groupby(["start_phone_use_datetime", "end_phone_use_datetime", "duration", "category"])\
            .agg(total_duration=("usage_duration", "sum"),\
                 apps=("application_name", list)).reset_index()
        category_app_usage["normalized_usage_duration"] = category_app_usage["total_duration"]/category_app_usage["duration"]
        app_usage = category_app_usage.to_dict(orient="records")
    sleep_time_features["app_usage"] = app_usage

    # Location and displacement during sleep
    if len(location_df) > 0:
        location_df["coordinates"] = list(zip(location_df["double_latitude"],\
                                                location_df["double_longitude"],\
                                                location_df["double_altitude"]))
        location_df["next_datetime"] = location_df["datetime"].shift(-1)
        location_df["prev_datetime"] = location_df["datetime"].shift(1)
        location_df["prev_coordinates"] = location_df["coordinates"].shift(1)
        location_df["new_group"] = (location_df["coordinates"] != location_df["prev_coordinates"]).astype(int)
        location_df["group_id"] = location_df["new_group"].cumsum()

        # Grouping and aggregating to find consecutive durations
        grouped_location_df = location_df.groupby(["group_id", "coordinates"]).agg(
            start_datetime=("datetime", "min"),
            end_datetime=("next_datetime", "max")
        ).reset_index().dropna()
        grouped_location_df["consecutive_duration"] = (grouped_location_df["end_datetime"] - grouped_location_df["start_datetime"]).dt.total_seconds()
        grouped_location_df = grouped_location_df.drop("group_id", axis=1)

        # Cross-check with activity recognition to determine if displacement is significant
        if grouped_activity_df is not None:
            still_activity = grouped_activity_df.where(grouped_activity_df["activity_name"] == "still").dropna()
            for _, row in still_activity.iterrows():
                still_disp = grouped_location_df.where((row["start_datetime"] <= grouped_location_df["end_datetime"]) & \
                                                        (row["end_datetime"] >= grouped_location_df["start_datetime"])).dropna()
                if not still_disp.empty:
                    overall_disp = {"start_datetime": [np.min(still_disp["start_datetime"])],\
                                    "end_datetime": [np.max(still_disp["end_datetime"])],\
                                    "consecutive_duration": [np.sum(still_disp["consecutive_duration"])],\
                                    "coordinates": [average_coordinates(still_disp["coordinates"])]}
                    overall_disp = pd.DataFrame(overall_disp)
                    grouped_location_df = grouped_location_df.drop(still_disp.index)
                    if grouped_location_df.empty:
                        grouped_location_df = overall_disp
                    else:
                        grouped_location_df = pd.concat([grouped_location_df, overall_disp], ignore_index=True)
        grouped_location_df = grouped_location_df.sort_values(by="start_datetime").reset_index(drop=True)

        # Calculate displacement across location transitions
        if len(grouped_location_df) > 1:
            grouped_location_df["prev_coordinates"] = grouped_location_df["coordinates"].shift(1).dropna()
            grouped_location_df["transition_distance"] = grouped_location_df.apply(lambda row: haversine_distance_with_altitude(row["prev_coordinates"], row["coordinates"]), axis=1)
            sleep_time_features["displacements"] = grouped_location_df[["datetime", "prev_datetime", "transition_distance"]].to_dict(orient="records")
        else:
            sleep_time_features["displacements"] = []

        # Sleep location as the coordinates in which the longest duration was spent
        sleep_location = grouped_location_df.loc[grouped_location_df["consecutive_duration"].idxmax()]
        sleep_time_features["sleep_location"] = sleep_location["coordinates"]

    else:
        sleep_time_features["sleep_location"] = ()
    
    return sleep_time_features

def visualize_events_during_sleep(user_id):
    """
    Plots non-stationary activity, location displacements, and active phone usage during multiple reported sleep times.
    """
    daily_sleep_features = process_sleep_data(user_id)
    daily_sleep_features["date"] = pd.to_datetime(daily_sleep_features["wake_time"]).dt.strftime("%Y-%m-%d")

    min_y = np.inf
    max_y = -np.inf
    _, ax = plt.subplots(figsize=(12, 8))
    
    for day_index, row in daily_sleep_features.iterrows():
        sleep_time = row["sleep_time"].hour + row["sleep_time"].minute/60
        sleep_time = sleep_time - 24 if sleep_time > 12 else sleep_time
        wake_time = row["wake_time"].hour + row["wake_time"].minute/60
        wake_time = wake_time - 24 if wake_time > 12 else wake_time

        if sleep_time < min_y:
            min_y = sleep_time
        if wake_time > max_y:
            max_y = wake_time

        ax.bar(day_index, sleep_time-wake_time, bottom=wake_time, color="tab:blue", edgecolor="tab:blue", width=0.3, alpha=0.5)
        for movement in row["non_still_occurrences"]:
            event_start = movement["start_datetime"].hour + movement["start_datetime"].minute/60
            event_start = event_start - 24 if event_start > 12 else event_start
            ax.plot([day_index - 0.25, day_index + 0.25], [event_start, event_start], color="orange", linestyle='-', linewidth=1, alpha=0.8)

        # Only for those transition greater than 1 meter
        for disp in row["displacements"]:
            event_start = disp["datetime"].hour + disp["datetime"].minute/60
            event_start = event_start - 24 if event_start > 12 else event_start
            ax.plot(day_index, event_start, color="red", marker="x", markersize=6)
        
        for phone_usage in row["phone_usage"]:
            event_start = phone_usage["start_datetime"].hour + phone_usage["start_datetime"].minute/60
            event_start = event_start - 24 if event_start > 12 else event_start
            event_end = phone_usage["end_datetime"].hour + phone_usage["end_datetime"].minute/60
            event_end = event_end - 24 if event_end > 12 else event_end
            rectangle = mpatches.Rectangle((day_index-0.25, event_start), 0.5, event_end-event_start, linewidth=1, edgecolor="black", facecolor="green", alpha=0.3)
            ax.add_patch(rectangle)
            if event_start < min_y:
                min_y = event_start
            if event_end > max_y:
                max_y = event_end

    ax.set_xticks(np.arange(len(daily_sleep_features)))
    ax.set_xticklabels(daily_sleep_features["date"], rotation=30)

    # Y-axis
    min_y = math.floor(min_y)
    max_y = math.ceil(max_y)
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
    ax.set_ylabel("Reported sleep duration")
    ax.yaxis.grid()
    plt.gca().invert_yaxis()

    # Title and legend
    ax.set_title("Events During Reported Sleep Times")
    legend_elements = [mpatches.Patch(facecolor="tab:blue", edgecolor="tab:blue", label="Reported sleep duration", alpha=0.6),\
                       plt.Line2D([0], [0], color="red", marker="x", linestyle=None, lw=0, label="Location displacement"),\
                       plt.Line2D([0], [0], color="orange", lw=1, label=textwrap.fill("Non-stationary activity event", width=25)),\
                       mpatches.Patch(facecolor="green", edgecolor="black", label="Active phone usage", alpha=0.3)]

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    plt.savefig(f"{user_id}/{user_id}_events_during_reported_sleep.png", dpi=300, format="png")
    # plt.show()

def visualize_context_breakdown(user_id, light_df, noise_df, physical_df, app_usage_df, title, filename):
    """
    Visualizes ambient light, ambient noise, occurrences of physical activity, and application usage within a specific window.
    Shared to visualize sleep window and stay at a specific cluster (may span across multiple epochs).
    """
    with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)
    app_categories = list(contexts["app_categories"].keys()) + ["utilities", "others"]

    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(4, 1, height_ratios=[1, 1, 0.75, 1.5]) 
    axs = [plt.subplot(gs[i]) for i in range(4)]
    min_x = np.inf
    max_x = -np.inf

    cur_date = None
    if light_df is None or light_df.empty:
        axs[0].text(0.5, 1.3, "No ambient light data", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=12)
        axs[0].set_yticks([])
        axs[0].set_yticklabels([])
    else:
        light_df["date"] = light_df["datetime"].dt.date
        light_df["datetime"] = light_df["datetime"].dt.hour + light_df["datetime"].dt.minute/60
        cur_date = np.max(light_df["date"])
        condition = light_df["date"] < cur_date
        light_df.loc[condition, "datetime"] = light_df.loc[condition, "datetime"] - 24
        light_df["upper_std"] = light_df["mean_light_lux"] + light_df["std_light_lux"]
        light_df["lower_std"] = light_df["mean_light_lux"] - light_df["std_light_lux"]
        axs[0].plot(light_df["datetime"], light_df["mean_light_lux"], label="Mean luminance", color="green")
        axs[0].fill_between(light_df["datetime"], light_df["min_light_lux"], light_df["max_light_lux"], color='orange', alpha=0.3, label='Min-Max Range')
        axs[0].fill_between(light_df["datetime"], light_df["lower_std"], light_df["upper_std"], color='purple', alpha=0.5, label='Std Dev Range')
        axs[0].set_ylabel("Luminance")
        axs[0].set_title("Ambient Light")
        axs[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        first_light_dt = light_df["datetime"].iloc[0]
        last_light_dt = light_df["datetime"].iloc[-1]
        if first_light_dt < min_x:
            min_x = first_light_dt
        if last_light_dt > max_x:
            max_x = last_light_dt

    if noise_df is None or noise_df.empty:
        axs[1].text(0.5, 0.5, "No ambient noise data", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=12)
        axs[1].set_yticks([])
        axs[1].set_yticklabels([])
    else:
        noise_df["date"] = noise_df["datetime"].dt.date
        noise_df["datetime"] = noise_df["datetime"].dt.hour + noise_df["datetime"].dt.minute/60
        cur_date = np.max(noise_df["date"])
        condition = noise_df["date"] < cur_date
        noise_df.loc[condition, "datetime"] = noise_df.loc[condition, "datetime"] - 24
        noise_df["upper_std"] = noise_df["mean_decibels"] + noise_df["std_decibels"]
        noise_df["lower_std"] = noise_df["mean_decibels"] - noise_df["std_decibels"]
        axs[1].plot(noise_df["datetime"], noise_df["mean_decibels"], label="Mean Decibels", color="green")
        axs[1].fill_between(noise_df["datetime"], noise_df["min_decibels"], noise_df["max_decibels"], color='orange', alpha=0.3, label='Min-Max Range')
        axs[1].fill_between(noise_df["datetime"], noise_df["lower_std"], noise_df["upper_std"], color='purple', alpha=0.5, label='Std Dev Range')
        axs[1].set_ylabel("Decibels")
        axs[1].set_title("Ambient Noise")
        axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        first_noise_dt = noise_df["datetime"].iloc[0]
        last_noise_dt = noise_df["datetime"].iloc[-1]
        if first_noise_dt < min_x:
            min_x = first_noise_dt
        if last_noise_dt > max_x:
            max_x = last_noise_dt

    if physical_df is None or physical_df.empty:
        axs[2].text(0.5, -0.75, "No occurrence of physical movement", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=12)
        axs[2].set_yticks([])
        axs[2].set_yticklabels([])
    else:
        color_map = plt.get_cmap("Accent")
        physical_df["datetime"] = pd.to_datetime(physical_df["start_datetime"])
        physical_df["date"] = physical_df["datetime"].dt.date
        physical_df["datetime"] = physical_df["datetime"].dt.hour + physical_df["datetime"].dt.minute/60
        cur_date = np.max(physical_df["date"])
        condition = physical_df["date"] < cur_date
        physical_df.loc[condition, "datetime"] = physical_df.loc[condition, "datetime"] - 24
        invalid_activity_type = physical_df["activity_type"] > len(ACTIVITY_NAMES)
        physical_df.loc[invalid_activity_type, "activity_type"] = 6
        physical_df = physical_df.where(physical_df["activity_type"] != 6)
        unique_activity = physical_df["activity_name"].drop_duplicates().to_list()
        for _, row in physical_df.iterrows():
            # inner_row_index = unique_activity.index(row["activity_name"])
            # axs[2].barh(inner_row_index, row["consecutive_duration"]/3600 + 0.01, left=row["datetime"], height=0.4,\
            #     color=color_map(int(row["activity_type"])))
            axs[2].barh(1, row["consecutive_duration"]/3600 + 0.01, left=row["datetime"], height=0.4,\
                color=color_map(int(row["activity_type"])))
            end_datetime = row["datetime"] + row["consecutive_duration"]/3600 + 0.01
            if row["datetime"] < min_x:
                min_x = row["datetime"]
            if end_datetime > max_x:
                max_x = end_datetime
        # axs[2].set_yticks(range(len(unique_activity)))
        # axs[2].set_yticklabels(unique_activity)
        axs[2].set_yticks([])
        axs[2].set_yticklabels([])
        axs[2].set_ylabel("Activity state")
        legend_elements = []
        for act in unique_activity:
            index = ACTIVITY_NAMES.index(act)
            legend_elements.append(mpatches.Patch(facecolor=color_map(index), edgecolor=color_map(index), label=act))
        axs[2].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    axs[2].set_title("Physical Movement")

    if app_usage_df is None or app_usage_df.empty:
        axs[3].text(0.5, -2, "No occurrence of active phone usage", horizontalalignment='center', verticalalignment='center', transform=axs[1].transAxes, fontsize=12)
        axs[3].set_yticks([])
        axs[3].set_yticklabels([])
    else:
        color_map = plt.get_cmap("Set2")
        norm_category_color = mcolors.Normalize(vmin=0, vmax=len(app_categories))
        app_usage_df["start_date"] = app_usage_df["start_phone_use_datetime"].dt.date
        app_usage_df["end_date"] = app_usage_df["end_phone_use_datetime"].dt.date
        app_usage_df["start_datetime"] = app_usage_df["start_phone_use_datetime"].dt.hour + app_usage_df["start_phone_use_datetime"].dt.minute/60
        app_usage_df["end_datetime"] = app_usage_df["end_phone_use_datetime"].dt.hour + app_usage_df["end_phone_use_datetime"].dt.minute/60
        if cur_date is None:
            cur_date = np.max(np.concatenate((app_usage_df["start_date"], app_usage_df["end_date"])))
        condition = app_usage_df["start_date"] < cur_date
        app_usage_df.loc[condition, "start_datetime"] = app_usage_df.loc[condition, "start_datetime"] - 24
        condition = app_usage_df["end_date"] < cur_date
        app_usage_df.loc[condition, "end_datetime"] = app_usage_df.loc[condition, "end_datetime"] - 24
        app_usage_df["duration"] = app_usage_df["end_datetime"] - app_usage_df["start_datetime"]
        app_usage_df["normalized_usage_duration"] = app_usage_df["normalized_usage_duration"]*100
        unique_phone_usage = app_usage_df["start_datetime"].drop_duplicates().to_list()
        for start_time in unique_phone_usage:
            cur_app_usage = app_usage_df.where(app_usage_df["start_datetime"] == start_time)\
                .sort_values(by="normalized_usage_duration", ascending=False).dropna()
            overall_normalized_duration = 0
            for _, row in cur_app_usage.iterrows():
                category_index = app_categories.index(row["category"])
                axs[3].bar(start_time, height=row["normalized_usage_duration"], width=row["duration"], bottom=overall_normalized_duration,\
                           color=color_map(norm_category_color(category_index)), align="edge")
                overall_normalized_duration += row["normalized_usage_duration"]
            if start_time < min_x:
                min_x = start_time
            if start_time + row["duration"] > max_x:
                max_x = start_time + row["duration"]
        axs[3].set_ylabel("% time spent on apps")
        legend_elements = []

        # Compile an overall list of apps for each category
        app_usage_df["apps"] = app_usage_df["apps"].apply(lambda x: list(x))
        overall_category_apps = app_usage_df.groupby("category")["apps"].agg(sum).reset_index()
        overall_category_apps["apps"] = overall_category_apps["apps"].apply(lambda x: list(set(x)))
        for _, row in overall_category_apps.iterrows():
            index = app_categories.index(row["category"])
            label = f"{row['category'].capitalize()} - {', '.join(row['apps'])}"
            legend_elements.append(mpatches.Patch(facecolor=color_map(norm_category_color(index)),\
                                                  edgecolor=color_map(norm_category_color(index)),\
                                                  label=textwrap.fill(label, width=20)))
        axs[3].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    axs[3].set_title("Phone and Application Usage")

    if min_x == np.inf:
        min_x = 0
    else:
        min_x = math.floor(min_x)
    if max_x == -np.inf:
        max_x = 0
    else:
         max_x = math.ceil(max_x)
    for i in range(4):
        # Adjust the plots to accommodate legend boxes
        box = axs[i].get_position()
        axs[i].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        axs[i].set_xlim(min_x, max_x)
        axs[i].set_xticklabels([])
    axs[3].set_xticks(range(min_x, max_x, 1))
    x_labels = []
    for hour in range(min_x, max_x, 1):
        if hour == 0:
            x_labels.append("12 AM")
        elif hour < 0:
            x_labels.append(f"{12+hour} PM")
        elif hour == 12:
            x_labels.append(f"{hour} PM")
        elif hour < 12:
            x_labels.append(f"{hour} AM")
        else:
            x_labels.append(f"{hour-12} PM")
    if len(x_labels) >= 20:
        axs[3].set_xticklabels(x_labels, rotation=30)
    else:
        axs[3].set_xticklabels(x_labels)
    axs[3].set_xlabel("Time")

    fig.suptitle(title)
    plt.savefig(f"{user_id}/{user_id}_{filename}.png",
                dpi=300, format="png")
    # plt.show()

def visualize_mobility_across_days(user_id, activity_df, location_df, title, filename):
    """
    Displays vertically-aligned high-level mobility info across multiple days.
    1. Duration of each physical state
    2. Time spent at each location cluster (labeled with total distance traveled and % of noise location points)
    """
    with open(f"{user_id}/{user_id}_contexts.json") as f:
        contexts = json.load(f)
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(3, 1) 
    axs = [plt.subplot(gs[i]) for i in range(3)]

    # A dedicated plot for stationary duration
    axs[0].plot(activity_df["date"], activity_df["still_duration"]/60)
    axs[0].set_ylabel("Duration (mins)")
    axs[0].set_title("Duration in Stationary State")

    # Duration for other activity states
    activity_color_map = plt.get_cmap("Accent")
    legend_elements = []
    for act_col in activity_df.columns[1:]:
        if act_col != "still_duration" and np.max(activity_df[act_col]) > 0:
            activity_df[act_col] = activity_df[act_col]/60
            act_index = ACTIVITY_NAMES.index(act_col[:-9])
            act_color = activity_color_map(act_index)
            axs[1].plot(activity_df["date"], activity_df[act_col], label=act_col, color=act_color)
            legend_elements.append(mpatches.Patch(facecolor=act_color, edgecolor=act_color, label=act_col[:-8]))
    axs[1].set_ylabel("Duration (mins)")
    axs[1].set_title("Duration in Other Activity States")
    axs[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # axs[1].legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))
    
    # Stacked bar plot of time spent in each cluster
    cluster_color_map = plt.get_cmap("Dark2")
    overall_clusters = contexts["location_clusters"].keys()
    for date_index, row in location_df.iterrows():
        overall_duration = 0
        for cluster_index, cluster in enumerate(overall_clusters):
            bar_width = 0.4
            cur_col = f"time_spent_cluster{cluster}"
            axs[2].bar(date_index, height=row[cur_col]/3600, width=bar_width, bottom=overall_duration,\
                           color=cluster_color_map(cluster_index))
            overall_duration += row[cur_col]/3600
        
    cluster_legend_elements = []
    for cluster_index, cluster in enumerate(overall_clusters):
        cluster_legend_elements.append(mpatches.Patch(facecolor=cluster_color_map(cluster_index), edgecolor=cluster_color_map(cluster_index),\
            label=f"cluster {cluster}"))
 
    axs[2].set_ylabel("Duration (hrs)")
    axs[2].set_title("Time Spent at Location Clusters")
    axs[2].legend(handles=cluster_legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    all_dates = activity_df["date"].to_list()
    for i in range(3):
        # Adjust the plots to accommodate legend boxes
        box = axs[i].get_position()
        axs[i].set_position([box.x0, box.y0, box.width * 0.9, box.height])
        axs[i].set_xlim(0, len(all_dates))
        axs[i].set_xticklabels([])
    axs[2].set_xticks(range(len(all_dates)))
    axs[2].set_xticklabels(all_dates, rotation=45)
    axs[2].set_xlabel("Date")

    fig.suptitle(title)
    plt.savefig(f"{user_id}/{user_id}_{filename}.png",
                dpi=300, format="png")
    # plt.show()

def visualize_high_level_day_events(user_id, day_feature_df, title, filename):
    """
    Visualizes high-level distribution of:
    1. Duration in dark vs bright environment
    2. Duration in quiet vs non-quiet environment
    3. Duration in each activity state
    4. Duration of application usage from each category

    Reference:
    1. https://www.geeksforgeeks.org/sunburst-plot-using-plotly-in-python/
    """
    with open(f"{user_id}/{user_id}_contexts.json", "r") as f:
        contexts = json.load(f)
    app_categories = list(contexts["app_categories"].keys()) + ["utilities", "others"]
    overall_clusters = list(contexts["location_clusters"].keys())
    epochs = list(TIME_EPOCHS.keys())

    plot_data = []
    for cluster in overall_clusters:
        for epoch in epochs:
            feature_prefix = f"cluster{cluster}_{epoch}"
            if day_feature_df[f"{feature_prefix}_stay_duration"] > 0:
                for activity in ["still", "in_vehicle", "on_bicycle", "tilting", "walking", "running"]:
                    if day_feature_df[f"{feature_prefix}_{activity}_normalized_duration"] > 0:
                        plot_data.append({"epoch": epoch, "context": "Activity State", "category": activity,\
                                          "duration": day_feature_df[f"{feature_prefix}_{activity}_normalized_duration"]})
                # else:
                #     plot_data.append({"epoch": epoch, "context": "Activity State", "category": activity, "duration": None})
            
                if day_feature_df[f"{feature_prefix}_normalized_phone_use_duration"] > 0:
                    for category in app_categories:
                        if day_feature_df[f"{feature_prefix}_{category}_app_cluster_normalized_duration"] > 0:
                            plot_data.append({"epoch": epoch, "context": "App Usage", "category": category,\
                                            "duration": day_feature_df[f"{feature_prefix}_{category}_app_cluster_normalized_duration"]})
                
                plot_data.append({"epoch": epoch, "context": "Ambient Light", "category": "Dark",\
                                "duration": day_feature_df[f"{feature_prefix}_normalized_dark_duration"]})
                plot_data.append({"epoch": epoch, "context": "Ambient Light", "category": "Non-dark",\
                                "duration": 1-day_feature_df[f"{feature_prefix}_normalized_dark_duration"]})
                plot_data.append({"epoch": epoch, "context": "Ambient Noise", "category": "Silent",\
                                "duration": day_feature_df[f"{feature_prefix}_normalized_silent_duration"]})
                plot_data.append({"epoch": epoch, "context": "Ambient Noise", "category": "Non-silent",\
                                "duration": 1-day_feature_df[f"{feature_prefix}_normalized_silent_duration"]})

        plot_df = pd.DataFrame(plot_data)
        fig = px.sunburst(plot_df, path=["epoch", "context", "category"], color="context", values="duration",\
                          title=f"Time Spent at Cluster{cluster} {title}")
        # fig.show()
        fig.write_image(f"{user_id}/{user_id}_{filename}.png")
        # plt.savefig(f"{user_id}/{user_id}_{filename}.png",
        #             dpi=300, format="png")
        # plt.show()

def prepare_features_during_sleep(user_id):
    """
    Retrieves all extracted sleep features and organizes them for analysis.
    """
    daily_sleep_feature_df = process_sleep_data(user_id)
    with open(f"{user_id}/{user_id}_contexts.json") as f:
        contexts = json.load(f)
    
    organized_features = []
    # Organize features for analysis
    for _, day_record in daily_sleep_feature_df.iterrows():
        day_features = {"sleep_time": pd_time_to_midnight_hours(day_record["sleep_time"]),\
                        "wake_time": pd_time_to_midnight_hours(day_record["wake_time"])}
        sleep_duration = (day_record["wake_time"] - day_record["sleep_time"]).total_seconds()
        day_features["sleep_duration"] = sleep_duration

        # Frequency of occurrence for each of the top 6 non-still activity type
        non_still_frequency = [0 for _ in range(6)]
        non_still_duration = [0 for _ in range(6)]    # Proportion of duration out of the sleep time
        
        # Timing of occurrence: within the first hour after falling asleep, intermediate, or at the last hour before waking up
        non_still_first_hour = [0 for _ in range(6)]
        non_still_first_hour_duration = [0 for _ in range(6)]
        non_still_last_hour = [0 for _ in range(6)]
        non_still_last_hour_duration = [0 for _ in range(6)]
        non_still_intermediate_hour = [0 for _ in range(6)]
        non_still_intermediate_hour_duration = [0 for _ in range(6)]
        non_still_activity_type = ACTIVITY_PRIORITIES[:6]
        non_still_occurrences = day_record["non_still_occurrences"]
        if len(non_still_occurrences) > 0:
            for act in non_still_occurrences:
                try:
                    index = non_still_activity_type.index(act["activity_name"])
                    non_still_frequency[index] += 1
                    non_still_duration[index] += act["consecutive_duration"]
                    # Get the hour in which the occurrence happens
                    occurrence_hour = (act["start_datetime"] - day_record["sleep_time"]).total_seconds()
                    if occurrence_hour < 3600:
                        non_still_first_hour[index] += 1
                        non_still_first_hour_duration[index] += act["consecutive_duration"]
                    elif (day_record["wake_time"] - act["start_datetime"]).total_seconds() < 3600:
                        non_still_last_hour[index] += 1
                        non_still_last_hour_duration[index] += act["consecutive_duration"]
                    else:
                        non_still_intermediate_hour[index] += 1
                        non_still_intermediate_hour_duration[index] += act["consecutive_duration"]
                except ValueError:
                    pass

        for index, act in enumerate(non_still_activity_type):
            day_features[f"{act}_occurrence"] = non_still_frequency[index]
            day_features[f"{act}_first_hour_occurrence"] = non_still_first_hour[index]
            day_features[f"{act}_first_hour_duration"] = non_still_first_hour_duration[index]
            day_features[f"{act}_last_hour_occurrence"] = non_still_last_hour[index]
            day_features[f"{act}_last_hour_duration"] = non_still_last_hour_duration[index]
            day_features[f"{act}_intermediate_hour_occurrence"] = non_still_intermediate_hour[index]
            day_features[f"{act}_intermediate_hour_duration"] = non_still_intermediate_hour_duration[index]
            day_features[f"{act}_normalized_duration"] = non_still_duration[index] / sleep_duration

        for col in ["min_luminance", "max_luminance", "mean_luminance", "std_luminance",\
                    "min_decibels",  "max_decibels", "mean_decibels", "std_decibels"]:
            day_features[col] = day_record[col]
        
        # Timing of occurrence: within the first hour after falling asleep, intermediate, or at the last hour before waking up
        phone_usage_occurrences = day_record["phone_usage"]
        day_features["phone_use_frequency"] = len(phone_usage_occurrences)
        phone_usage_duration = 0
        phone_usage_first_hour = 0
        phone_usage_first_hour_duration = 0
        phone_usage_last_hour = 0
        phone_usage_last_hour_duration = 0
        phone_usage_intermediate_hour = 0
        phone_usage_intermediate_hour_duration = 0
        if len(phone_usage_occurrences) > 0:
            for use in phone_usage_occurrences:
                phone_usage_duration += use["duration"]/1000
                # Get the hour in which the occurrence happens
                occurrence_hour = (use["start_datetime"] - day_record["sleep_time"]).total_seconds()
                if occurrence_hour < 3600:
                    phone_usage_first_hour += 1
                    phone_usage_first_hour_duration += use["duration"]/1000
                elif (day_record["wake_time"] - use["start_datetime"]).total_seconds() < 3600:
                    phone_usage_last_hour += 1
                    phone_usage_last_hour_duration += use["duration"]/1000
                else:
                    phone_usage_intermediate_hour += 1
                    phone_usage_intermediate_hour_duration += use["duration"]/1000

        day_features["phone_use_first_hour_occurrence"] = phone_usage_first_hour
        day_features["phone_use_first_hour_duration"] = phone_usage_first_hour_duration
        day_features["phone_use_last_hour_occurrence"] = phone_usage_last_hour
        day_features["phone_use_last_hour_duration"] = phone_usage_last_hour_duration
        day_features["phone_use_intermediate_hour_occurrence"] = phone_usage_intermediate_hour
        day_features["phone_use_intermediate_hour_duration"] = phone_usage_intermediate_hour_duration
        day_features["phone_use_normalized_duration"] = phone_usage_duration / sleep_duration

        # Frequency of occurrence for app category
        app_category = list(contexts["app_categories"].keys()) + ["utilities", "others"]
        app_frequency = [0 for _ in range(len(app_category))]
        app_duration = [0 for _ in range(len(app_category))]    # Proportion of duration out of the sleep time
        # Proportion of duration out of each phone use
        app_phone_use_normalized_duration = [0 for _ in range(len(app_category))]
        
        # Timing of occurrence: within the first hour after falling asleep, intermediate, or at the last hour before waking up
        app_first_hour = [0 for _ in range(len(app_category))]
        app_first_hour_duration = [0 for _ in range(len(app_category))]
        app_last_hour = [0 for _ in range(len(app_category))]
        app_last_hour_duration = [0 for _ in range(len(app_category))]
        app_intermediate_hour = [0 for _ in range(len(app_category))]
        app_intermediate_hour_duration = [0 for _ in range(len(app_category))]
        app_occurrences = day_record["app_usage"]
        if len(app_occurrences) > 0:
            for app in app_occurrences:
                try:
                    index = app_category.index(app["category"])
                    app_frequency[index] += 1
                    app_duration[index] += app["total_duration"]
                    app_phone_use_normalized_duration[index] += app["normalized_usage_duration"]
                    # Get the hour in which the occurrence happens
                    occurrence_hour = (app["start_phone_use_datetime"] - day_record["sleep_time"]).total_seconds()
                    if occurrence_hour < 3600:
                        app_first_hour[index] += 1
                        app_first_hour_duration[index] += app["total_duration"]
                    elif (day_record["wake_time"] - app["start_phone_use_datetime"]).total_seconds() < 3600:
                        app_last_hour[index] += 1
                        app_last_hour_duration[index] += app["total_duration"]
                    else:
                        app_intermediate_hour[index] += 1
                        app_intermediate_hour_duration[index] += app["total_duration"]
                except ValueError:
                    pass
        
        for index, category in enumerate(app_category):
            day_features[f"{category}_occurrence"] = app_frequency[index]
            day_features[f"{category}_first_hour_occurrence"] = app_first_hour[index]
            day_features[f"{category}_first_hour_duration"] = app_first_hour_duration[index]
            day_features[f"{category}_last_hour_occurrence"] = app_last_hour[index]
            day_features[f"{category}_last_hour_duration"] = app_last_hour_duration[index]
            day_features[f"{category}_intermediate_hour_occurrence"] = app_intermediate_hour[index]
            day_features[f"{category}_intermediate_hour_duration"] = app_intermediate_hour_duration[index]
            day_features[f"{category}_normalized_duration"] = app_duration[index] / sleep_duration
            day_features[f"{category}_normalized_phone_use_duration"] = app_phone_use_normalized_duration[index]

        # Location displacements
        location_displacements = day_record["displacements"]
        day_features["disp_occurrence"] = len(location_displacements)
        disp_distance = 0
        disp_first_hour = 0
        disp_first_hour_distance = 0
        disp_last_hour = 0
        disp_last_hour_distance = 0
        disp_intermediate_hour = 0
        disp_intermediate_hour_distance = 0
        if len(location_displacements) > 0:
            for disp in location_displacements:
                disp_distance += disp["transition_distance"]
                # Get the hour in which the occurrence happens
                occurrence_hour = (disp["datetime"] - day_record["sleep_time"]).total_seconds()
                if occurrence_hour < 3600:
                    disp_first_hour += 1
                    disp_first_hour_distance += disp["transition_distance"]
                elif disp["datetime"] > day_record["wake_time"] or (day_record["wake_time"] - disp["datetime"]).total_seconds() < 3600:
                    disp_last_hour += 1
                    disp_last_hour_distance += disp["transition_distance"]
                else:
                    disp_intermediate_hour += 1
                    disp_intermediate_hour_distance += disp["transition_distance"]

        day_features["displacement_first_hour_occurrence"] = disp_first_hour
        day_features["displacement_first_hour_distance"] = disp_first_hour_distance
        day_features["displacement_last_hour_occurrence"] = disp_last_hour
        day_features["displacement_last_hour_distance"] = disp_last_hour_distance
        day_features["displacement_intermediate_hour_occurrence"] = disp_intermediate_hour
        day_features["displacement_intermediate_hour_distance"] = disp_intermediate_hour_distance
        day_features["displacement_distance"] = disp_distance

        organized_features.append(day_features)

    return pd.DataFrame(organized_features)

def feature_correlation_during_sleep(user_id):
    """
    Analyze the correlations between features extracted during reported sleep times and the sleep duration and quality rating.

    # Interpreting correlation coefficients and p-values:
    1. NaN due to 0 variance (probably all 0s)
    2. 
    """
    organized_feature_df = prepare_features_during_sleep(user_id)

    corr_df = pd.DataFrame()
    feat1s = []
    feat2s = []
    corrs = []
    p_values = []

    all_cols = organized_feature_df.columns
    for index, feat1 in enumerate(all_cols[:-1]):
        for inner_index in range(index+1, len(all_cols)):
            feat2 = all_cols[inner_index]
            feat1s.append(feat1)
            feat2s.append(feat2)
            corr, p_value = pearsonr(organized_feature_df[feat1], organized_feature_df[feat2])
            corrs.append(corr)
            p_values.append(p_value)
    corr_df['Feature_1'] = feat1s
    corr_df['Feature_2'] = feat2s
    corr_df['Correlation'] = corrs
    corr_df['p_value'] = p_values
    print(corr_df)
    corr_df.to_csv(f"{user_id}/{user_id}_feature_during_sleep_correlations.csv", index=False)

def prepare_day_features(user_id, visualize_day_cluster_contexts=False):
    """
    
    """
    pickle_filename = f"{DATA_FOLDER}/{user_id}_day_features.pkl"
    if not os.path.exists(pickle_filename) or visualize_day_cluster_contexts:
        # Based on the days where EMAs were admnistered (retrieve_sleep_ema function already filtered those with EMA responses)
        esm_dates = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_identifier}_esms.csv")\
            .select("date").distinct().sort("date").toPandas()
        esm_dates = esm_dates["date"].to_list()
        all_day_features = []
        for date in esm_dates:
            # day_level_features = extract_day_features(user_identifier, date, visualize_day_cluster_contexts)
            extract_day_features(user_identifier, date, visualize_day_cluster_contexts)
    #         all_day_features.append(day_level_features)
    #     all_day_features = pd.DataFrame(all_day_features)
    #     all_day_features.to_pickle(pickle_filename)
    
    # return pd.read_pickle(pickle_filename)

def fine_tune_gbt(X, y, tuned_model_file, regression=False):
    """
    TODO
    Fine tune Keras GBT classification model using nested cross-validation.
    References:
    1. https://stackabuse.com/gradient-boosting-classifiers-in-python-with-scikit-learn/
    2. https://scikit-learn.org/stable/auto_examples/model_selection/plot_randomized_search.html# sphx-glr-auto-examples-model-selection-plot-randomized-search-py
    3. https://stackoverflow.com/questions/67535904/userwarning-one-or-more-of-the-test-scores-are-non-finite-warning-only-when-a
    4. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html
    NOTE: Check if GBT is a classifier or regressor, update scorer of RandomizedSearchCV if classifier
    """
    if regression:
        gbt_model = GradientBoostingRegressor(random_state=42)
        tuner_scoring = "neg_mean_squared_error"
    else:
        gbt_model = GradientBoostingClassifier(random_state=42)
        tuner_scoring = make_scorer(f1_score, average="micro", zero_division=0)
        # tuner_scoring = make_scorer(roc_auc_score, average="micro")
    
    # Inner CV for fine-tuning hyperparameters
    # Outer CV for evaluating best fine-tuned model for each inner fold
    outer_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    param_dist = {
        "n_estimators": np.arange(20, 200, 2),
        "learning_rate": loguniform(0.05, 1),
        "max_depth": np.arange(2, 20),
        "min_samples_leaf": np.arange(1, 15)
    }

    best_hyperparameters = []
    test_scores = []

    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Inner loop for hyperparameter tuning
        n_iter_search = 200
        random_search = RandomizedSearchCV(gbt_model, param_distributions=param_dist, n_iter=n_iter_search, verbose=1, scoring=tuner_scoring,
                                       return_train_score=True, n_jobs=-1, cv=inner_cv, random_state=42)
        tuning = random_search.fit(X_train, y_train)
        # results = tuning.cv_results_
        # for mean_score in results["mean_test_score"]:
        #     print(f"Mean f1-score: {mean_score}")

        # Store best hyperparameters from inner loop
        best_hyperparameters.append(random_search.best_params_)
        
        # Train the final model with best hyperparameters
        best_model = random_search.best_estimator_
        best_model.fit(X_train, y_train)
        
        # Evaluate on the outer test set
        y_pred = best_model.predict(X_test)

        if regression:
            outer_r2 = r2_score(y_test, y_pred)
            test_scores.append(outer_r2)
            mae = mean_absolute_error(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            rmse = math.sqrt(mean_squared_error(y_test, y_pred))
        else:
            outer_f1 = f1_score(y_test, y_pred, average="micro", zero_division=0)
            test_scores.append(outer_f1)


    # Average performance from outer loop
    average_test_score = np.mean(test_scores)
    print(f'Average test score from nested CV: {average_test_score:.3f}')

    joblib.dump(best_model, tuned_model_file)

def train_gbt(user_id):
    """
    Train GBT model based on input data.
    TODO
    """
    best_model_file = f"tuned_gbt_{user_id}.pkl"
    data_features = prepare_features_during_sleep(user_id)
    ground_truth_data = retrieve_sleep_ema(user_id)
    X = data_features.to_numpy()
    y = ground_truth_data["sleep_quality_rating"].to_numpy()

    start_time = time.time()
    fine_tune_gbt(X, y, best_model_file)
    end_time = time.time()
    print(f"Time elapsed: {end_time - start_time} s")
    repeated_kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    model_final = joblib.load(best_model_file)
    scores = cross_val_score(model_final, X, y, cv=repeated_kf, scoring='accuracy')

    print(f'Mean accuracy from repeated K-Fold: {scores.mean():.3f} ± {scores.std():.3f}')


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

    # -- Dev user IDs --
    # user_identifier = "pixel3"
    # user_identifier = "S3"
    # user_identifier = "S5"

    # -- Pilot user IDs --
    user_identifier = "S07"

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
    # get_unique_apps(user_identifier)
    # NOTE: Must be executed after computing high-level contexts since conversation estimate depends on the audio thresholds.
    # process_noise_data_with_conv_estimate(user_identifier)
    # -- End of block
    

    # -- NOTE: This block of functions combine multiple sensor information to generate interpretation
    # location_df = process_location_data(user_identifier)
    # cluster_df = cluster_locations(user_identifier, location_df, "double_latitude", "double_longitude")
    # complement_location_data(user_identifier)
    # cross_check_cluster_with_activity_state(user_identifier)
    # resolve_cluster_fluctuations(user_identifier)
    # -- End of block


    # -- Sleep-related --
    # sleep_df = estimate_sleep(user_identifier)
    # map_overview_estimated_sleep_duration_to_sleep_ema(user_identifier, [3, 4, 1])
    # visualize_reported_sleep_ema(user_identifier)
    # visualize_reported_sleep_ema_with_overlapping_sleep_estimate(user_identifier)
    # visualize_estimated_sleep_windows_against_reported_sleep(user_identifier)
    # visualize_events_during_sleep(user_identifier)
    # daily_sleep_features = process_sleep_data(user_identifier)
    # datetime_format = "%Y-%m-%d %H:%M"
    # for index, sleep_window_df in daily_sleep_features.iterrows():
    #     if index == len(daily_sleep_features)-1:
    #         sleep_time = pd.to_datetime(sleep_window_df["sleep_time"]).strftime(datetime_format)
    #         wake_time = pd.to_datetime(sleep_window_df["wake_time"]).strftime(datetime_format)
    #         visualize_context_breakdown(user_identifier, pd.DataFrame(sleep_window_df["light_df"]),\
    #                                     pd.DataFrame(sleep_window_df["noise_df"]),\
    #                                     pd.DataFrame(sleep_window_df["non_still_occurrences"]),\
    #                                     pd.DataFrame(sleep_window_df["app_usage"]),\
    #                                     f"Contexts During Sleep Window: {sleep_time} - {wake_time}",\
    #                                     f"{pd.to_datetime(sleep_window_df['sleep_time']).strftime('%m%d_%H%M')}_sleep_contexts")
    # feature_correlation_during_sleep(user_identifier)
    # -- End of block

    # -- Daytime features --
    # Extracts and visualizes contexts within each visited cluster in each day
    # prepare_day_features(user_identifier, True)
    # all_day_features = prepare_day_features(user_identifier)
    # location_cols = ["total_distance_traveled", "location_variance", "cluster_count", "unique_cluster_count",\
    #                  "unknown_location_count", "normalized_unknown_location_count", "location_entropy",\
    #                     "normalized_location_entropy"] + [col for col in all_day_features.columns if "time_spent_cluster" in col]
    # # all_day_activity = all_day_features[["date"] + [f"{act}_duration" for act in ACTIVITY_PRIORITIES[:7]]]
    # # all_day_location = all_day_features[["date"] + location_cols]
    # # visualize_mobility_across_days(user_identifier, all_day_activity, all_day_location, "Physical Mobility Across Days", "mobility_all")

    # for _, day_row in all_day_features.iterrows():
    #     date = pd.to_datetime(day_row["date"]).strftime("%Y-%m-%d")
    #     visualize_high_level_day_events(user_identifier, day_row, f"on {date}",\
    #                                     f"high_level_contexts_{pd.to_datetime(day_row['date']).strftime('%m%d')}")
    # day_features_vs_mood(user_identifier)
    extract_custom_agg_features(user_identifier, ["date", "epoch"])
    # -- End of block

    # -- Other analysis --
    # train_gbt(user_identifier)
    # -- End of block