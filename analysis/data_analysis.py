"""
Author: Lin Sze Khoo
Created on: 24/01/2024
Last modified on: 30/05/2024
"""
import collections
import json
import math
import os
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
from kneed import KneeLocator
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType,
                               StructField, StructType, TimestampType, BooleanType, ArrayType)
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

def delete_emulator_device(cursor, db):
    cursor.execute("DELETE FROM aware_device WHERE device='emu64xaif t is no'")
    db.commit()

def delete_single_entry(cursor, table_name, device_id):
    cursor.execute(f"DELETE FROM {table_name} WHERE device='{device_id}'")

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

def process_noise_data(user_id, data_minute_interval=1):
    """
    (Continuous with intervals)
    Ambient audio data
    
    NOTE:
    1. Latest sensing configuration sets to collect data for 30 seconds every 10 mins
    2. Round timestamp to the nearest 5-minute and compute statistical features over each sampling window
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_noise.parquet"
    if not os.path.exists(parquet_filename):
        noise_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_ambient_noise.csv")
        time_cols = ["date", "hour", "minute"]
        noise_df = noise_df.withColumn("double_frequency", F.col("double_frequency").cast(FloatType()))\
            .withColumn("double_decibels", F.col("double_decibels").cast(FloatType()))\
            .withColumn("double_rms", F.col("double_rms").cast(FloatType()))\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("second_timestamp", F.round(F.col("timestamp")/1000).cast(TimestampType()))\
            .withColumn("second_timestamp", udf_round_datetime_to_nearest_minute(F.col("second_timestamp"), F.lit(data_minute_interval)))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        
        # Obtain statistical descriptors within each 1-minute time window (30 seconds buffer for the 30-second sampling duration)      
        stat_functions = [F.min, F.max, F.mean, F.stddev]
        stat_names = ["min", "max", "mean", "std"]
        agg_expressions = []
        agg_cols = [col for col in noise_df.columns if col.startswith("double_") and col != "double_silence_threshold"]
        for col in agg_cols:
            for index, func in enumerate(stat_functions):
                agg_expressions.append(func(col).alias(f"{stat_names[index]}_{col[7:]}"))
        noise_df = noise_df.withWatermark("second_timestamp", "1 minute")\
            .groupBy(F.window("second_timestamp", "1 minute"))\
            .agg(*agg_expressions)\
            .withColumn("start_timestamp", F.col("window.start"))\
            .withColumn("end_timestamp", F.col("window.end"))\
            .withColumn("date", udf_get_date_from_datetime("start_timestamp"))\
            .withColumn("hour", udf_get_hour_from_datetime("start_timestamp"))\
            .withColumn("minute", udf_get_minute_from_datetime("start_timestamp"))\
            .drop("window", "start_timestamp", "end_timestamp").sort(*time_cols)

        # Fill in missing time points
        # noise_df = fill_df_hour_minute(noise_df, data_minute_interval)
        # noise_df = noise_df.toPandas()
        # feature_ts = noise_df["average_decibels_minute"]

        # # Perform interpolation to fill in nan values
        # feature_ts = [float(t) for t in feature_ts]
        # if np.isnan(np.min(feature_ts)):
        #     feature_ts = nearest_neighbour_interpolation(feature_ts)
        # noise_df["average_decibels_minute"] = feature_ts
        # standard_dev = np.std(feature_ts)
        # mean = np.mean(feature_ts)
        # outlier_threshold = 2 * standard_dev
        # noise_df["potential_outlier"] = noise_df["average_decibels_minute"] - mean > outlier_threshold
        # noise_df = spark.createDataFrame(noise_df.values.tolist(), schema=StructType([StructField("date", StringType()),\
        #                                                                                 StructField("hour", IntegerType()),\
        #                                                                                 StructField("minute", IntegerType()),\
        #                                                                                 StructField("average_decibels_minute", FloatType()),\
        #                                                                                 StructField("potential_outlier", BooleanType())]))
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
    2. Include: Non-system applications running in the foreground when screen is active.

    NOTE Assumptions:
    1. Assume that the usage duration of an application is the duration between the current usage timestamp and
    that of the subsequent foreground application or the end of device usage duration.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_app_usage.parquet"
    if not os.path.exists(parquet_filename):
        phone_use_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")
        phone_in_use = phone_use_df.filter(F.col("double_elapsed_device_on") > 0)\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("start_timestamp", F.col("timestamp") - F.col("double_elapsed_device_on"))\
            .withColumnRenamed("timestamp", "end_timestamp")

        app_usage_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_applications_foreground.csv")\
            .withColumnRenamed("timestamp", "usage_timestamp")
        time_window = Window.partitionBy("start_timestamp")\
            .orderBy("usage_timestamp")
        
        # Obtain intersect of phone screen in use and having applications running in foreground
        in_use_app_df = app_usage_df.join(phone_in_use, (phone_in_use["start_timestamp"] <= app_usage_df["usage_timestamp"]) & (phone_in_use["end_timestamp"] >= app_usage_df["usage_timestamp"]))
        in_use_app_df = in_use_app_df.select(*["package_name", "application_name", "usage_timestamp", "start_timestamp", "end_timestamp"]).dropDuplicates()\
            .sort("start_timestamp", "usage_timestamp")
        in_use_app_df = in_use_app_df.withColumn("next_timestamp", F.lead(F.col("usage_timestamp")).over(time_window))\
            .withColumn("usage_duration (ms)", F.when(F.col("next_timestamp").isNotNull(), F.col("next_timestamp") - F.col("usage_timestamp"))\
                        .otherwise(F.col("end_timestamp") - F.col("usage_timestamp")))\
            .filter(F.col("is_system_app") == 0).sort("usage_timestamp") # Filter off system applications
        in_use_app_df.write.parquet(parquet_filename)
    
    in_use_app_df = spark.read.parquet(parquet_filename)
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
    
    occurrence_df = bluetooth_df.groupBy("hour", "bt_address").agg(F.count("*").alias("occurrence"))\
        .join(bluetooth_df.select("bt_address", "bt_name"), "bt_address")\
        .dropDuplicates()\
        .sort(F.col("occurrence").desc())
    occurrence_df.show()

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
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))
        cluster_df = cluster_locations(user_id, location_df, "double_latitude", "double_longitude")
        # Location clusters are derived from all unique coordinates in location_df so there will be no null cluster_id
        location_df = location_df.join(cluster_df, coordinate_cols).dropDuplicates()
        wifi_df = process_wifi_data(user_id)\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))

        # Combine coordinates and WiFi devices information by date time
        all_locations = location_df.select(*time_cols + coordinate_cols + ["cluster_id"])\
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
            time_window = Window().partitionBy("date").orderBy("hour", "minute")
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


        screen_df = process_screen_data(user_id)
        # Only exclude active usage time because phone screen might still be activated by notifications
        phone_usage_df = screen_df.withColumn("is_in_use", F.when(F.col("prev_status") == 3, 1).otherwise(0))\
            .withColumn("prev_in_use", F.lag(F.col("is_in_use")).over(time_window))\
            .withColumn("new_group", (F.col("prev_in_use") != F.col("is_in_use")).cast("int"))\
            .withColumn("group_id", F.sum("new_group").over(Window.orderBy("timestamp").rowsBetween(Window.unboundedPreceding, Window.currentRow)))\
            .filter(F.col("prev_timestamp").isNotNull())\
            .groupBy("group_id", "is_in_use")\
            .agg(F.min("prev_timestamp").alias("start_datetime"),\
                    F.max("timestamp").alias("end_datetime"))\
            .withColumn("consecutive_duration", F.round((F.col("end_datetime") - F.col("start_datetime"))/1000).cast(IntegerType()))\
            .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_datetime")))\
            .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("end_datetime")))\
            .drop("group_id").sort("start_datetime")

        # With considerations of screen activation caused by notifications or active phone checking
        not_in_use_df = phone_usage_df.filter((F.col("is_in_use") == 0) & (F.col("consecutive_duration") > consecutive_min*60))\
            .drop("is_in_use", "consecutive_duration").sort("start_datetime")

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

def ambient_light(user_id):
    minutes = lambda i: i * 60000 # Timestamp is in milliseconds
    data_folder = "analysis/user_data"
    light_df = spark.read.option("header", True).csv(f"{data_folder}/{user_id}_light.csv")
    # 1 - Apply rolling average within 10-minute window frame to smoothen noise
    light_df = light_df.withColumn("double_light_lux", F.col("double_light_lux").cast(FloatType()))
    smooth_window = (Window()\
        .partitionBy(F.col("date"))\
        .orderBy(F.col("timestamp").cast("long"))\
        .rangeBetween(-minutes(10), 0))
    rolled_light = light_df.withColumn("rolling_average", F.mean("double_light_lux").over(smooth_window))
    rolled_light.show()

    # 2 - Descriptive statistics - mean, median, standard deviation, variance within each hour
    # grouped_light = rolled_light.groupBy("date", "hour")\
    #     .agg(F.mean("rolling_average").alias("average"), \
    #         F.stddev("rolling_average").alias("standard_deviation"), \
    #         F.variance("rolling_average").alias("variance"), \
    #         F.max("rolling_average").alias("max"), \
    #         F.min("rolling_average").alias("min"))
    # grouped_light.show()

    # timestamp_window = Window.partitionBy("date").orderBy("timestamp")
    # rolled_light = rolled_light.withColumn("prev_timestamp", F.lag(F.col("timestamp")).over(timestamp_window))
    # light_df.withColumn("row", F.row_number().over(Window().partitionBy(F.col("date")).orderBy(F.col("timestamp").cast("long"))))\
    #     .withColumn("window_avg", F.mean("double_light_lux").over(average_window))

    # light_df = light_df.withColumn("window_start_time",
    #                            F.expr("timestamp - 600000"))
    # average_window = Window.partitionBy("date", "window_start_time").orderBy("timestamp")
    # rolled_light = light_df.withColumn("custom_window_average", F.avg("double_light_lux").over(average_window))
    # average_window = (Window()\
    #     .partitionBy(F.col("date"))\
    #     .orderBy(F.col("timestamp").cast("long"))\
    #     .rangeBetween(0, minutes(10)))
    # rolled_light = light_df.withColumn("rolling_average", F.mean("double_light_lux").over(average_window))
    
    # light_df.withColumn("row", F.row_number().over(Window().partitionBy(F.col("date")).orderBy(F.col("timestamp").cast("long"))))\
    #     .withColumn("window_avg", F.mean("double_light_lux").over(average_window))
    # df = df.withColumn(
    #     "next_timestamp", F.lead(F.col("timestamp")).over(timestamp_window))
    # df = df.withColumn(
    #     "next_inference", F.lead(F.col(f" {sensing_folder} inference")).over(timestamp_window))
    # df = df.withColumn("duration", F.when(F.col(f" {sensing_folder} inference") == F.col("next_inference"),
    #                                         F.col("next_timestamp") - F.col("timestamp")).otherwise(0))
    # rolled_light = light_df.withColumn("row", F.row_number().over(Window().partitionBy(F.col("date")).orderBy(F.col("timestamp").cast("long"))))\
    #     .withColumn("window_avg", F.mean("double_light_lux").over(average_window))
    # print(rolled_light.count())
    
    # light_viz = rolled_light.filter(F.col("date") == "2024-02-01").toPandas()
    # plt.plot(light_viz["time"], light_viz["rolling_average"])
    # plt.show()
    # grouped_light = light_df.groupby(["date", "time"]).mean()
    # rolling_light = light_df.rolling()


def daily_routine(user_id):
    """
    Combines all information related to an individual's daytime productivity.
    1. Deduce home based on location during estimated sleep duration

    NOTE Assumptions:
    1. Wake time as the minimum "end_datetime" of estimated sleep duration for each day
    2. The most frequently seen cluster at wake time as primary location

    TODO: Map daily routine to mood, productivity or sleep rating

    Group information into hourly or epoch of the day
    1. Range of activity involved during the day
    2. Light environment
    
    Other information:
    1. Map light with noise information - sleep, work vs outdoor environment
    2. People/device around - Bluetooth
    """
    coordinate_cols = ["double_latitude", "double_longitude"]
    time_cols = ["date", "hour", "minute"]
    time_window = Window.partitionBy("date")\
        .orderBy("hour", "minute")

    # Pre-saved unique location clusters and available semantics
    # location_cluster = cluster_locations(user_id, location_df, "double_latitude", "double_longitude")
    # # Number of clusters excluding noise with -1 labels
    # n_clust = cluster_df.select(F.col("cluster_id")).filter(F.col("cluster_id") != -1).distinct().count()
    # print(f"Estimated number of clusters: {n_clust}")

    # Time-based contexts: physical mobility, ambient light and noise
    physical_mobility = process_activity_data(user_id)\
        .withColumn("physical_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    ambient_light = process_light_data(user_id)\
        .withColumn("light_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    ambient_noise = process_noise_data(user_id)\
        .withColumn("noise_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))

    # Sleep estimation
    sleep_df = estimate_sleep(user_id)\
        .withColumn("date", udf_get_date_from_datetime("end_datetime"))
    cur_day = np.array(sleep_df.select("date").distinct().sort("date").collect()).flatten()[0]

    # Wake environment
    wake_env = sleep_df.groupBy("date").agg(F.min("end_datetime").alias("end_datetime"))\
        .join(sleep_df, ["date", "end_datetime"])
    cur_wake_env = wake_env.filter(F.col("date") == cur_day)
    
    # Wake location/cluster as primary location
    all_locations = complement_location_data(user_id)\
        .withColumn("hour", F.col("hour").cast(IntegerType()))\
        .withColumn("minute", F.col("minute").cast(IntegerType()))\
        .withColumn("location_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    wake_locations = wake_env.drop(*[col for col in wake_env.columns if col in time_cols]).\
        join(all_locations.drop(*time_cols), (F.col("start_datetime") <= F.col("location_datetime")) &\
             (F.col("end_datetime") >= F.col("location_datetime")))\
        .sort("start_datetime").dropDuplicates()
    wake_cluster = wake_locations.sort(*["location_datetime", "ssid"]).groupBy("start_datetime", "end_datetime", "cluster_id")\
        .agg(F.count("location_datetime").alias("cluster_count"), \
             F.concat_ws(", ", F.collect_set("ssid")).alias("WiFi_devices"))
    wake_cluster = wake_cluster.groupBy("start_datetime", "end_datetime").agg(F.max("cluster_count").alias("cluster_count"))\
        .join(wake_cluster, ["start_datetime", "end_datetime", "cluster_count"])

    # NOTE: Only work with data in a specific day
    # Include 1 min before wake time to capture transition in contexts
    cur_wake_env = cur_wake_env.withColumn("prev_datetime", F.unix_timestamp(F.col("end_datetime")) - 60)\
        .withColumn("prev_datetime", udf_datetime_from_timestamp(F.col("prev_datetime")*1000))
    day_contexts = physical_mobility.withColumnRenamed("date", "temp_date")\
        .join(cur_wake_env, (F.col("temp_date") == F.col("date")) & \
              (F.col("physical_datetime") >= F.col("prev_datetime")))\
        .select(*time_cols + ["physical_datetime", "activity_name", "activity_type"]).dropDuplicates()
    day_contexts = day_contexts.join(ambient_light.drop(*time_cols), F.col("physical_datetime") == F.col("light_datetime"))\
        .join(ambient_noise.drop(*time_cols), F.col("physical_datetime") == F.col("noise_datetime"))\
        .withColumnRenamed("physical_datetime", "datetime").drop("light_datetime", "noise_datetime")\
        .dropDuplicates().sort("datetime")
    day_contexts = day_contexts.withColumn("prev_activity", F.lag(F.col("activity_type")).over(time_window))\
        .withColumn("prev_light", F.lag(F.col("mean_light_lux")).over(time_window))\
        .withColumn("prev_decibels", F.lag(F.col("min_decibels")).over(time_window))\
        .drop(*time_cols)
    # Remove rows with null previous values (1st row)
    day_contexts = day_contexts.dropna(subset=[col for col in day_contexts.columns if "prev_" in col]).sort("datetime")

    # Locations after wake time
    day_locations = all_locations.withColumnRenamed("date", "temp_date")\
        .join(cur_wake_env, (F.col("temp_date") == F.col("date")) & \
              (F.col("location_datetime") >= F.col("prev_datetime")))\
        .select(*all_locations.columns).dropDuplicates()
    day_locations = day_locations.groupBy(*[col for col in all_locations.columns if col != "ssid"])\
        .agg(F.concat_ws(", ", F.collect_set("ssid")).alias("WiFi_devices"))\
        .dropDuplicates().sort("location_datetime")
    day_locations = day_locations.withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(time_window))\
        .withColumn("prev_location_datetime", F.lag(F.col("location_datetime")).over(time_window))\
        .drop(*time_cols).sort("location_datetime")
    coordinates_WiFi_devices = day_locations.select(*coordinate_cols + ["WiFi_devices"]).distinct().dropna()\
        .filter(F.col("WiFi_devices") != "")
    day_locations = day_locations.join(coordinates_WiFi_devices.withColumnRenamed("WiFi_devices", "temp_WiFi_devices"),\
                                       coordinate_cols, "left")\
                                .withColumn("WiFi_devices", F.when(F.col("WiFi_devices") == "", F.col("temp_WiFi_devices")).otherwise(F.col("WiFi_devices")))\
                                .drop("temp_WiFi_devices").dropDuplicates().sort("location_datetime")\
                                .fillna("", subset=["WiFi_devices"])
    
    # Contexts during location transition: inter cluster travel
    cluster_transitions = day_locations.filter((F.col("prev_cluster") != F.col("cluster_id")) | (F.col("prev_cluster").isNull()))
    inter_cluster_travel = day_contexts.join(cluster_transitions,\
                                             (F.col("datetime") >= F.col("prev_location_datetime")) &\
                                                (F.col("datetime") <= F.col("location_datetime")))\
                                        .drop(*time_cols).dropDuplicates().sort("datetime")
    # inter_cluster_travel.write.csv("inter_cluster_travel.csv", header=True)
    # cluster_transitions.show()

    # Context transitions
    # Duration and contexts in each cluster
    significant_light_transition = 50
    significant_noise_transition = 10
    day_contexts = day_contexts.withColumn("any_transition",\
                                           F.when((F.abs(F.col("mean_light_lux") - F.col("prev_light")) >= significant_light_transition) |\
                                                  (F.abs(F.col("min_decibels") - F.col("prev_decibels")) >= significant_noise_transition) |\
                                                    (F.col("activity_type") != F.col("prev_activity")), True).otherwise(False))
    context_transitions = day_contexts.filter((F.col("any_transition") == True) & (F.col("activity_type") != 4) & (F.col("prev_activity") != 4))\
        .sort("datetime")

    # Context analysis
    app_usage = process_application_usage_data(user_id)
    for col in app_usage.columns:
        if "timestamp" in col:
            app_usage = app_usage.withColumn(col, F.col(col).cast(FloatType()))

    # NOTE: (N+1) cluster analysis will be involved for N cluster transition points
    cluster_contexts_df = None
    cluster_productivity_df = None
    # First row will always be the first filtered row for the day of interest
    location_transition_datetimes = np.array(cluster_transitions.select("location_datetime").collect()).flatten()[1:]
    location_clusters = np.array(cluster_transitions.select("cluster_id").collect()).flatten()
    for index, location_datetime in enumerate(location_transition_datetimes):
        if index == 0:
            cluster_contexts = day_contexts.filter(F.col("datetime") < location_datetime)
            cluster_productivity = app_usage.filter(udf_datetime_from_timestamp(F.col("usage_timestamp")) < location_datetime)
        else:
            cluster_contexts = day_contexts.filter((F.col("datetime") >= location_transition_datetimes[index-1]) & (F.col("datetime") < location_datetime))
            cluster_productivity = app_usage.filter((udf_datetime_from_timestamp(F.col("usage_timestamp")) >= location_transition_datetimes[index-1]) &\
                                                    (udf_datetime_from_timestamp(F.col("usage_timestamp")) < location_datetime))
        
        # context_analysis(cluster_contexts, cluster_productivity)
        cluster_contexts = cluster_contexts.withColumn("cluster_id", F.lit(int(location_clusters[index])))
        cluster_productivity = cluster_productivity.withColumn("cluster_id", F.lit(int(location_clusters[index])))
        if cluster_contexts_df is None:
            cluster_contexts_df = cluster_contexts
            cluster_productivity_df = cluster_productivity
        else:
            cluster_contexts_df = cluster_contexts_df.union(cluster_contexts)
            cluster_productivity_df = cluster_productivity_df.union(cluster_productivity)

    # Include the last cluster context after the last transition point
    cluster_contexts = day_contexts.filter(F.col("datetime") >= location_transition_datetimes[-1])
    cluster_productivity = app_usage.filter(udf_datetime_from_timestamp(F.col("usage_timestamp")) >= location_transition_datetimes[-1])
    cluster_contexts = cluster_contexts.withColumn("cluster_id", F.lit(int(location_clusters[-1])))
    cluster_productivity = cluster_productivity.withColumn("cluster_id", F.lit(int(location_clusters[-1])))
    cluster_contexts_df = cluster_contexts_df.union(cluster_contexts)
    cluster_productivity_df = cluster_productivity_df.union(cluster_productivity)

    # Construct a dataframe for the current day to be exported to CSV file
    cluster_contexts_df.coalesce(1).write.csv(f"{user_id}_{cur_day}_cluster_contexts.csv", header=True)
    cluster_productivity_df.coalesce(1).write.csv(f"{user_id}_{cur_day}_cluster_productivity.csv", header=True)
    return cluster_contexts_df

def context_analysis(time_based_contexts, event_based_productivity):
    """
    Combine time-based contexts (physical mobility, ambient light and noise) with event-based application usage
    1. Intermediate movements
    2. Ambient light and noise
    3. Application usage

    Analyzed information:
    1. (Overall) Total usage duration of each application
    2. (Overall) Total duration of each physical state
    3. Context transitions
    """
    event_based_productivity = event_based_productivity.withColumn("usage_datetime", udf_datetime_from_timestamp(F.col("usage_timestamp")))\
        .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("next_timestamp")))
    app_usage_duration = event_based_productivity.groupBy("application_name")\
        .agg(F.sum("usage_duration (ms)").alias("total_usage_duration (ms)"),
             F.mean("usage_duration (ms)").alias("average_usage_duration (ms)"))
    # app_usage_duration.show()
    physical_mobility_duration = time_based_contexts.groupBy("activity_type")\
        .agg(F.count("activity_name").alias("total_physical_duration (min)"),\
             F.mean("average_lux_minute").alias("average_luminance"),\
                F.mean("average_decibels_minute").alias("average_noise"))
    # physical_mobility_duration.show()

    # Physical states during application usage
    device_usage_events = event_based_productivity.select("start_timestamp", "end_timestamp").distinct()

    # Context transitions
    context_transitions = time_based_contexts.filter(F.col("any_transition") == True)
    physical_transitions = context_transitions.filter((F.col("activity_type") != F.col("prev_activity")) &\
                                                      (F.col("activity_type") != 4) & \
                                                        (F.col("prev_activity") != 4))
    ambient_transitions = context_transitions.filter(F.abs(F.col("average_lux_minute") - F.col("prev_light")) >= 50)

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
    # legend_elements.append(mpatches.Patch(visible=False, label="Sleep quality rating"))
    legend_elements.append(mpatches.Patch(facecolor="grey", edgecolor="black", label="No rating"))
    for index, rating in enumerate(SLEEP_QUALITY_RATINGS):
        legend_elements.append(mpatches.Patch(facecolor=cmap(norm(index+1)), edgecolor="black", label=rating))

    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5))

    plt.show()


def visualize_daytime_contexts(user_id):
    """
    Plots day-level distribution of location
    Combines all information related to an individual's daytime productivity.

    NOTE Assumptions:
    1. Wake time as the minimum "end_datetime" of estimated sleep duration for each day
    2. The most frequently seen cluster at wake time as primary location

    TODO: Map daily routine to mood, productivity or sleep rating

    Group information into hourly or epoch of the day
    1. Range of activity involved during the day
    2. Light environment
    
    Other information:
    1. Map light with noise information - sleep, work vs outdoor environment
    2. People/device around - Bluetooth
    """
    coordinate_cols = ["double_latitude", "double_longitude"]
    time_cols = ["date", "hour", "minute"]
    time_window = Window.partitionBy("date")\
        .orderBy("hour", "minute")

    # Sleep estimation
    sleep_df = estimate_sleep(user_id)\
        .withColumn("date", udf_get_date_from_datetime("end_datetime"))
    cur_day = np.array(sleep_df.select("date").distinct().sort("date").collect()).flatten()[0]

    # NOTE: Filter data for the particular day
    physical_mobility = process_activity_data(user_id).filter(F.col("date") == cur_day)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    ambient_light = process_light_data(user_id).filter(F.col("date") == cur_day)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))   
    ambient_noise = process_noise_data(user_id).filter(F.col("date") == cur_day)\
        .withColumn("datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    app_usage = process_application_usage_data(user_id)\
        .withColumn("datetime", udf_datetime_from_timestamp("usage_timestamp"))\
        .withColumn("date", udf_get_date_from_datetime("datetime"))\
        .filter(F.col("date") == cur_day)
    for col in app_usage.columns:
        if "timestamp" in col:
            app_usage = app_usage.withColumn(col, F.col(col).cast(FloatType()))
    locations = complement_location_data(user_id).filter(F.col("date") == cur_day)\
        .withColumn("hour", F.col("hour").cast(IntegerType()))\
        .withColumn("minute", F.col("minute").cast(IntegerType()))\
        .withColumn("location_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    

    day_locations = locations.groupBy(*[col for col in locations.columns if col != "ssid"])\
        .agg(F.concat_ws(", ", F.collect_set("ssid")).alias("WiFi_devices"))\
        .dropDuplicates().sort("location_datetime")
    day_locations = day_locations.withColumn("prev_cluster", F.lag(F.col("cluster_id")).over(time_window))\
        .withColumn("prev_location_datetime", F.lag(F.col("location_datetime")).over(time_window))\
        .drop(*time_cols).sort("location_datetime")
    coordinates_WiFi_devices = day_locations.select(*coordinate_cols + ["WiFi_devices"]).distinct().dropna()\
        .filter(F.col("WiFi_devices") != "")
    day_locations = day_locations.join(coordinates_WiFi_devices.withColumnRenamed("WiFi_devices", "temp_WiFi_devices"),\
                                       coordinate_cols, "left")\
                                .withColumn("WiFi_devices", F.when(F.col("WiFi_devices") == "", F.col("temp_WiFi_devices")).otherwise(F.col("WiFi_devices")))\
                                .drop("temp_WiFi_devices").dropDuplicates().sort("location_datetime")\
                                .fillna("", subset=["WiFi_devices"])
    
    # Location transition: inter cluster travel
    cluster_transitions = day_locations.filter((F.col("prev_cluster") != F.col("cluster_id")) | (F.col("prev_cluster").isNull()))
    cluster_transitions.show()

    # NOTE: (N+1) cluster analysis will be involved for N cluster transition points
    context_dfs = [physical_mobility, ambient_light, ambient_noise]
    cluster_context_dfs = [None for _ in range(len(context_dfs))]
    cluster_productivity_df = None
    # First row will always be the first filtered row for the day of interest
    location_transition_datetimes = np.array(cluster_transitions.select("location_datetime").collect()).flatten()[1:]
    location_clusters = np.array(cluster_transitions.select("cluster_id").collect()).flatten()
    for cluster_index, location_datetime in enumerate(location_transition_datetimes):
        if cluster_index == 0:
            for context_index, context_df in enumerate(context_dfs):
                cluster_context = context_df.filter(F.col("datetime") < location_datetime)\
                    .withColumn("cluster_id", F.lit(int(location_clusters[cluster_index])))
                cluster_context_dfs[context_index] = cluster_context
            cluster_productivity_df = app_usage.filter(udf_datetime_from_timestamp(F.col("usage_timestamp")) < location_datetime)\
                .withColumn("cluster_id", F.lit(int(location_clusters[cluster_index])))
        else:
            for context_index, context_df in enumerate(context_dfs):
                cluster_context = context_df.filter((F.col("datetime") >= location_transition_datetimes[cluster_index-1]) & (F.col("datetime") < location_datetime))\
                    .withColumn("cluster_id", F.lit(int(location_clusters[cluster_index])))
                cluster_context_dfs[context_index] = cluster_context_dfs[context_index].union(cluster_context)
            cluster_productivity = app_usage.filter((udf_datetime_from_timestamp(F.col("usage_timestamp")) >= location_transition_datetimes[cluster_index-1]) &\
                                                    (udf_datetime_from_timestamp(F.col("usage_timestamp")) < location_datetime))
            cluster_productivity_df = cluster_productivity_df.union(cluster_productivity.withColumn("cluster_id", F.lit(int(location_clusters[cluster_index]))))

    # Include the last cluster context after the last transition point
    for context_index, context_df in enumerate(context_dfs):
        cluster_context = context_df.filter(F.col("datetime") >= location_transition_datetimes[-1])\
            .withColumn("cluster_id", F.lit(int(location_clusters[-1])))
        cluster_context_dfs[context_index] = cluster_context_dfs[context_index].union(cluster_context)
    cluster_productivity = app_usage.filter(udf_datetime_from_timestamp(F.col("usage_timestamp")) >= location_transition_datetimes[-1])
    cluster_productivity_df = cluster_productivity_df.union(cluster_productivity.withColumn("cluster_id", F.lit(int(location_clusters[-1]))))
    

    # # Context transitions
    # # Duration and contexts in each cluster
    # significant_light_transition = 50
    # significant_noise_transition = 10
    # day_contexts = day_contexts.withColumn("any_transition",\
    #                                        F.when((F.abs(F.col("mean_light_lux") - F.col("prev_light")) >= significant_light_transition) |\
    #                                               (F.abs(F.col("min_decibels") - F.col("prev_decibels")) >= significant_noise_transition) |\
    #                                                 (F.col("activity_type") != F.col("prev_activity")), True).otherwise(False))
    # context_transitions = day_contexts.filter((F.col("any_transition") == True) & (F.col("activity_type") != 4) & (F.col("prev_activity") != 4))\
    #     .sort("datetime")
    
    # day_contexts.write.csv(f"{user_id}_{cur_day}_contexts.csv", header=True)

    # # Context analysis



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

    # ALL_TABLES = ["applications_foreground", "applications_history", "applications_notifications",\
    #               "bluetooth", "light", "locations", "calls", "messages", "screen", "wifi", "esms",\
    #                 "plugin_ambient_noise", "plugin_device_usage", "plugin_google_activity_recognition"]
    ALL_TABLES = ["esms"]
    DATA_FOLDER = "user_data"
    TIMEZONE = pytz.timezone("Australia/Melbourne")
    ACTIVITY_NAMES = ["in_vehicle", "on_bicycle", "on_foot", "still", "unknown", "tilting", "", "walking", "running"]
    SLEEP_QUALITY_RATINGS = ["Poor", "Fair", "Average", "Good", "Excellent"]

    # .astimezone(TIMEZONE)
    # .replace(tzinfo=TIMEZONE)
    udf_datetime_from_timestamp = F.udf(lambda x: datetime.fromtimestamp(x/1000, TIMEZONE), TimestampType())
    udf_generate_datetime = F.udf(lambda d, h, m: TIMEZONE.localize(datetime.strptime(f"{d} {h:02d}:{m:02d}", "%Y-%m-%d %H:%M")), TimestampType())
    udf_get_date_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).date().strftime("%Y-%m-%d"), StringType())
    udf_get_hour_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).time().hour, IntegerType())
    udf_get_minute_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).time().minute, IntegerType())
    udf_string_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M"), StringType())
    udf_unique_wifi_list = F.udf(lambda x: ", ".join(sorted(list(set(x.split(", "))))), StringType())
    udf_round_datetime_to_nearest_minute = F.udf(lambda dt, n: round_time_to_nearest_n_minutes(dt, n), TimestampType())
    udf_map_activity_type_to_name = F.udf(lambda x: ACTIVITY_NAMES[x], StringType())
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

    # # # Delete emulator devices from aware_device table
    # # delete_emulator_device(db_cursor, db_connection)

    # # # Remove entries corresponding to invalid devices from all tables
    # # valid_devices = get_valid_device_id(db_cursor)
    # # for table in ALL_TABLES:
    # #     delete_invalid_device_entries(db_cursor, table, tuple(valid_devices))
    # #     # delete_single_entry(db_cursor, "aware_device", "emu64xa")
    # #     db_connection.commit()

    # db_cursor.close()
    # db_connection.close()
    # -- End of block --

    # -- NOTE: This block of functions execute the extraction and early processing of sensor data into dataframes
    # process_light_data(user_identifier)
    # process_activity_data(user_identifier)
    # process_noise_data(user_identifier, 5)
    # process_screen_data(user_identifier)
    # process_application_usage_data(user_identifier)
    # process_bluetooth_data(user_identifier)
    # process_location_data(user_identifier)
    # process_wifi_data(user_identifier)
    # -- End of block

    # -- NOTE: This block of functions combine multiple sensor information to generate interpretation
    # sleep_df = estimate_sleep(user_identifier)
    # location_df = process_location_data(user_identifier)
    # cluster_df = cluster_locations(user_identifier, location_df, "double_latitude", "double_longitude")
    complement_location_data(user_identifier)
    # map_overview_estimated_sleep_duration_to_sleep_ema(user_identifier, [3, 4, 1])
    # daily_routine(user_identifier)
    # visualize_daytime_contexts(user_identifier)
    # -- End of block

    # ambient_light(user_identifier)