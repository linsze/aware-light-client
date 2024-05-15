"""
Author: Lin Sze Khoo
Created on: 24/01/2024
Last modified on: 14/05/2024
"""
import json
import math
import os
from datetime import datetime, timezone
from itertools import product

import findspark
import matplotlib.pyplot as plt
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
    cursor.execute("DELETE FROM aware_device WHERE device='emu64xa'")
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
            if "activities" in table_df.columns:
                table_df["activities"] = [d.replace('"', "'") for d in table_df["activities"]]
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

def fill_df_hour_minute(cur_df):
    time_cols = ["date", "hour", "minute"]
    unique_dates = np.array(cur_df.select("date").distinct().collect()).flatten().tolist()
    minute_range = list(range(0, 60))
    hour_range = list(range(0, 24))
    date_hour_minute = list(product(unique_dates, hour_range, minute_range))
    time_df = spark.createDataFrame(date_hour_minute, schema=StructType(
        [StructField("date", StringType(), False),\
         StructField("hour", IntegerType(), False),
         StructField("minute", IntegerType(), False)]))\
         .sort(time_cols)
    
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

def process_light_data(user_id):
    """
    (Continuous)
    Ambient luminance data: interpolate missing data using nearest neighbour interpolation based on https://dl.acm.org/doi/pdf/10.1145/3510029

    NOTE Assumptions:
    1. Exclude extremely high luminance from analysis
    2. Exclude outliers detected from applying rolling average
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_light.parquet"
    if not os.path.exists(parquet_filename):
        minutes = lambda i: i * 60000 # Timestamp is in milliseconds
        # Light (continuous)
        light_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_light.csv")
        time_cols = ["date", "hour", "minute"]

        # Moving average to smooth the data and identify outliers
        light_df = light_df.withColumn("double_light_lux", F.col("double_light_lux").cast(FloatType()))\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        smooth_window = (Window()\
            .partitionBy(F.col("date"))\
            .orderBy(F.col("timestamp"))\
            .rangeBetween(-minutes(1), 0))
        
        # NOTE: Ignores extremely high lumination occuring at 01/02/2024 23:06 and outliers based on assumed threshold
        light_df = light_df.filter(F.col("double_light_lux") <= 2000)
        light_df = light_df.withColumn("rolling_light_lux", F.mean("double_light_lux").over(smooth_window))
        light_df = light_df.withColumn("residuals", F.col("double_light_lux") - F.col("rolling_light_lux"))

        # NOTE: Exclude outliers based on standard deviation of difference between initial value and rolling average
        outlier_threshold = 2 * light_df.select(F.stddev("residuals")).first()[0]
        light_df = light_df.withColumn("potential_outlier", F.when((F.abs("residuals") > outlier_threshold), True).otherwise(False))
        light_df = light_df.filter(F.col("potential_outlier") == False)

        light_df = light_df.groupBy(*time_cols)\
            .agg(F.mean("double_light_lux").alias("average_lux_minute"))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        
        # Fill in missing time points
        light_df = fill_df_hour_minute(light_df)
        light_df = light_df.toPandas()
        feature_ts = light_df["average_lux_minute"]

        # Perform interpolation to fill in nan values
        feature_ts = [float(t) for t in feature_ts]
        if np.isnan(np.min(feature_ts)):
            # Nearest neighbour interpolation based on 
            time = list(range(len(feature_ts)))
            initial_time = [index for index in range(
                len(feature_ts)) if not math.isnan(feature_ts[index])]
            initial_feature_ts = [
                feat for feat in feature_ts if not math.isnan(feat)]
            interp_func = interp1d(
                initial_time, initial_feature_ts, kind="nearest", fill_value="extrapolate")
            feature_ts = interp_func(time)
        
        light_df["average_lux_minute"] = feature_ts
        light_df = spark.createDataFrame(light_df.values.tolist(), schema=StructType([StructField("date", StringType()),\
                                                                           StructField("hour", IntegerType()),\
                                                                           StructField("minute", IntegerType()),\
                                                                           StructField("average_lux_minute", FloatType())]))
        light_df.write.parquet(parquet_filename)
    
    light_df = spark.read.parquet(parquet_filename)
    # viz_light = light_df
    # # viz_light = light_df.loc[light_df["date"] == first_entry["date"]]
    # viz_light["hour_min"] = viz_light["hour"].astype(str).str.cat(viz_light["minute"].astype(str), sep=" : ")
    # plt.plot(viz_light["hour_min"], viz_light["average_lux_minute"])
    # plt.show()

    return light_df

def process_noise_data(user_id):
    """
    Ambient audio data: interpolate missing data using nearest neighbour interpolation based on https://dl.acm.org/doi/pdf/10.1145/3510029
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_noise.parquet"
    if not os.path.exists(parquet_filename):
        noise_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_ambient_noise.csv")
        time_cols = ["date", "hour", "minute"]
        noise_df = noise_df.withColumn("double_decibels", F.col("double_decibels").cast(FloatType()))\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        
        noise_df = noise_df.groupBy(*time_cols)\
            .agg(F.mean("double_decibels").alias("average_decibels_minute"))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .sort(time_cols)
        
        # Fill in missing time points
        noise_df = fill_df_hour_minute(noise_df)
        noise_df = noise_df.toPandas()
        feature_ts = noise_df["average_decibels_minute"]

        # Perform interpolation to fill in nan values
        feature_ts = [float(t) for t in feature_ts]
        if np.isnan(np.min(feature_ts)):
            # Nearest neighbour interpolation based on 
            time = list(range(len(feature_ts)))
            initial_time = [index for index in range(
                len(feature_ts)) if not math.isnan(feature_ts[index])]
            initial_feature_ts = [
                feat for feat in feature_ts if not math.isnan(feat)]
            interp_func = interp1d(
                initial_time, initial_feature_ts, kind="nearest", fill_value="extrapolate")
            feature_ts = interp_func(time)
        
        noise_df["average_decibels_minute"] = feature_ts
        standard_dev = np.std(feature_ts)
        mean = np.mean(feature_ts)
        outlier_threshold = 2 * standard_dev
        noise_df["potential_outlier"] = noise_df["average_decibels_minute"] - mean > outlier_threshold
        noise_df = spark.createDataFrame(noise_df.values.tolist(), schema=StructType([StructField("date", StringType()),\
                                                                                        StructField("hour", IntegerType()),\
                                                                                        StructField("minute", IntegerType()),\
                                                                                        StructField("average_decibels_minute", FloatType()),\
                                                                                        StructField("potential_outlier", BooleanType())]))
        noise_df.write.parquet(parquet_filename)

    noise_df = spark.read.parquet(parquet_filename)
    # viz_light = noise_df.loc[noise_df["date"] == "2024-02-02"]
    # viz_light["hour_min"] = viz_light["hour"].astype(str).str.cat(viz_light["minute"].astype(str), sep=" : ")
    # plt.plot(viz_light["hour_min"], viz_light["average_decibels_minute"])

    # plt.scatter(viz_light[viz_light["potential_outlier"]]["hour_min"], viz_light[viz_light["potential_outlier"]]["average_decibels_minute"], color='red', marker='*')
    # plt.show()
    return noise_df

def process_activity_data(user_id):
    """
    (Continuous)
    Activity recognition data: interpolate missing values based on the previous row

    NOTE Assumptions:
    1. Fill in missing values with "still" (activity_type 3) with the assumption that the device is static hence no new entries were recorded to save battery.
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_activity.parquet"
    if not os.path.exists(parquet_filename):
        activity_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_google_activity_recognition.csv")
        time_cols = ["date", "hour", "minute"]

        activity_df = activity_df.withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("hour", F.col("hour").cast(IntegerType()))\
            .withColumn("minute", F.col("minute").cast(IntegerType()))\
            .withColumn("day_of_the_week", F.col("day_of_the_week").cast(IntegerType()))\
            .sort(time_cols)

        # Fill in missing values
        activity_df = fill_df_hour_minute(activity_df)
        activity_df = activity_df.withColumn("activity_type", F.when(F.col("activity_type").isNull(), 3).otherwise(F.col("activity_type")))\
            .withColumn("activity_name", F.when(F.col("activity_name").isNull(), "still").otherwise(F.col("activity_name")))\
            .sort(*time_cols)

        # Calculate duration of each continuous activity state
        time_window = Window().partitionBy(F.col("date")).orderBy(*time_cols)
        activity_df = activity_df.withColumn("prev_activity", F.lag(F.col("activity_type")).over(time_window))
        activity_df = activity_df.withColumn("activity_duration", F.when(F.col("activity_type") == F.col("prev_activity"), 1).otherwise(0))\
            .drop("prev_activity")
        activity_df.write.parquet(parquet_filename)
    
    activity_df = spark.read.parquet(parquet_filename)
    # Count the frequency of each activity type
    activity_freq = activity_df.groupBy("date", "activity_name").agg(F.count("activity_name").alias("day_activity_count"))
    viz_activity_freq = activity_freq.toPandas()
    viz_activity_freq = viz_activity_freq.loc[viz_activity_freq["date"] == "2024-02-01"]
    # viz_activity_freq["hour_min"] = viz_activity_freq["hour"].astype(str).str.cat(viz_activity_freq["minute"].astype(str), sep=" : ")
    # plt.bar(viz_activity_freq["activity_name"], viz_activity_freq["day_activity_count"])
    # plt.show()

    # viz_activity = activity_df.toPandas()
    # viz_activity["hour_min"] = viz_activity["date"].str.cat(viz_activity["hour"].astype(str).str.cat(viz_activity["minute"].astype(str), sep=" : "), sep=" ")
    # plt.plot(viz_activity["hour_min"], viz_activity["activity_type"])
    # plt.show()

    return activity_df

def process_screen_data(user_id):
    """
    (Event-based)
    Screen status data: currently does not attempt to identify missing data
    NOTE: Device usage plugin readily provides usage (screen unlocked) and non-usage duration (screen off)
    """
    screen_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_screen.csv")
    usage_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")
    time_cols = ["date", "hour", "minute"]

    screen_df = screen_df.withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
        .withColumn("hour", F.col("hour").cast(IntegerType()))\
        .withColumn("minute", F.col("minute").cast(IntegerType()))\
        .withColumn("day_of_the_week", F.col("day_of_the_week").cast(IntegerType()))\
        .sort(time_cols)
    usage_df = usage_df.withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
        .withColumn("hour", F.col("hour").cast(IntegerType()))\
        .withColumn("minute", F.col("minute").cast(IntegerType()))\
        .withColumn("day_of_the_week", F.col("day_of_the_week").cast(IntegerType()))\
        .withColumn("double_elapsed_device_on", F.col("double_elapsed_device_on").cast(LongType()))\
        .withColumn("double_elapsed_device_off", F.col("double_elapsed_device_off").cast(LongType()))\
        .sort(time_cols)

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


def complement_location_data(user_id):
    """
    Combines location and WiFi data to provide location semantics.
    Clusters locations and identify WiFi devices available in each cluster.

    NOTE:
    1. Location coordinates are rounded to 5 decimal places (up to 1 metres) - can also be rounded to 4 dp for 10 metres allowance
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_combined_location.parquet"
    if not os.path.exists(parquet_filename):
        time_cols = ["date", "hour", "minute"]
        coordinate_cols = ["double_latitude", "double_longitude"]
        location_df = process_location_data(user_id)\
            .withColumn("double_latitude", F.round(F.col("double_latitude"), 5))\
            .withColumn("double_longitude", F.round(F.col("double_longitude"), 5))
        cluster_df = cluster_locations(user_id, location_df, "double_latitude", "double_longitude")
        # Location clusters are derived from all unique coordinates in location_df so there will be no null cluster_id
        location_df = location_df.join(cluster_df, coordinate_cols).dropDuplicates()
        wifi_df = process_wifi_data(user_id)

        # Combine coordinates and WiFi devices information by date time
        all_locations = location_df.select(*time_cols + coordinate_cols + ["cluster_id"])\
            .join(wifi_df, time_cols, "outer").dropDuplicates().sort(*time_cols)
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
            
            # Fill in those that coexist at the same time points with the newly filled WiFi devices
            # remaining_coexist_ssids = all_locations.groupBy(*time_cols)\
            #     .agg(F.first("cluster_id", ignorenulls=True).alias("temp_cluster_id"))
            # all_locations = all_locations.join(remaining_coexist_ssids, time_cols)\
            #     .withColumn("cluster_id", F.coalesce(F.col("cluster_id"), F.col("temp_cluster_id")))\
            #     .drop("temp_cluster_id").sort(*time_cols)
            cur_null_count = all_locations.filter(F.col("cluster_id").isNull()).count()

        # # Unique combinations of available coordinates and list of nearby WiFi devices
        # unique_semantic_csv = f"{DATA_FOLDER}/{user_id}_unique_coordinate_WiFi.csv"
        # if not os.path.exists(unique_semantic_csv):
        #     unique_coordinates_ssid = all_locations.select(*coordinate_cols + ["WiFi_devices"]).distinct().na.drop().sort(*coordinate_cols)
        #     unique_coordinates_ssid.write.csv(unique_semantic_csv, header=True)
        # unique_coordinates_ssid = spark.read.option("header", True).csv(unique_semantic_csv)
    
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


def estimate_sleep(user_id):
    """
    Estimate sleep duration based on information from light, noise, activity and phone screen.

    NOTE Assumptions:
    1. When conditions are fulfilled for consecutive entries within 5 minutes
    2. Environment may not be completely quiet and/or dark
    3. Does not limit to 1 estimated entry per day
    4. Remove those with duration shorter than 1 hour
    """
    parquet_filename = f"{DATA_FOLDER}/{user_id}_sleep.parquet"
    if not os.path.exists(parquet_filename):
        time_cols = ["date", "hour", "minute"]
        time_window = Window.partitionBy("date")\
            .orderBy(*time_cols)
        consecutive_min = 5
        consecutive_window = Window.partitionBy("date")\
            .orderBy(*time_cols)\
            .rowsBetween(0, 4)
        
        dark_threshold = 10 
        light_df = process_light_data(user_id)\
            .withColumn("is_dark", F.when(F.col("average_lux_minute") <= dark_threshold, 1).otherwise(0))\
            .withColumn("consecutive_dark_count", F.sum("is_dark").over(consecutive_window))
        dark_df = light_df.filter(F.col("consecutive_dark_count") >= consecutive_min)\
            .withColumn("dark_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .select("dark_datetime", "average_lux_minute")
        bright_transition_df = light_df.withColumn("prev_is_dark", F.lag(F.col("is_dark")).over(time_window))\
            .filter((F.col("prev_is_dark") == 1) & (F.col("is_dark") == 0))\
            .withColumn("bright_transition_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
        
        noise_df = process_noise_data(user_id)
        audio_quartiles = noise_df.approxQuantile("average_decibels_minute", [0.25, 0.50], 0.01)
        # silence_threshold = round((audio_quartiles[1]-audio_quartiles[0]) / 2)
        silence_threshold = round(audio_quartiles[1])
        noise_df = noise_df\
            .withColumn("is_quiet", F.when(F.col("average_decibels_minute") <= silence_threshold, 1).otherwise(0))\
            .withColumn("consecutive_quiet_count", F.sum("is_quiet").over(consecutive_window))
        quiet_df = noise_df.filter(F.col("consecutive_quiet_count") >= consecutive_min)\
            .withColumn("quiet_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .select("quiet_datetime", "average_decibels_minute")
        noise_transition_df = noise_df.withColumn("prev_is_quiet", F.lag(F.col("is_quiet")).over(time_window))\
            .filter((F.col("prev_is_quiet") == 1) & (F.col("is_quiet") == 0))\
            .withColumn("noise_transition_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))

        activity_df = process_activity_data(user_id)\
            .withColumn("is_still", F.when(F.col("activity_type") == 3, 1).otherwise(0))\
            .withColumn("consecutive_still_count", F.sum("is_still").over(consecutive_window))
        stationary_df = activity_df.filter(F.col("consecutive_still_count") >= consecutive_min)\
            .withColumn("stationary_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))\
            .select("stationary_datetime", "activities")
        movement_transition_df = activity_df.withColumn("prev_is_still", F.lag(F.col("is_still")).over(time_window))\
            .filter((F.col("prev_is_still") == 1) & (F.col("is_still") == 0))\
            .withColumn("movement_transition_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))

        screen_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")
        off_screen = screen_df.filter(F.col("double_elapsed_device_off") > consecutive_min*60000)\
            .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
            .withColumn("start_timestamp", F.col("timestamp") - F.col("double_elapsed_device_off"))\
            .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_timestamp")))\
            .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("timestamp")))\
            .sort("start_datetime")

        # V1
        # time_diff_allowance = 5*60
        # sleep_df = dark_env.join(quiet_env, F.abs(F.unix_timestamp(dark_env["dark_datetime"]) - F.unix_timestamp(quiet_env["quiet_datetime"])) <= time_diff_allowance)
        # sleep_df = sleep_df.join(stationary, ((F.abs(F.unix_timestamp(sleep_df["dark_datetime"]) - F.unix_timestamp(stationary["stationary_datetime"])) <= time_diff_allowance) |\
        #                        (F.abs(F.unix_timestamp(sleep_df["quiet_datetime"]) - F.unix_timestamp(stationary["stationary_datetime"])) <= time_diff_allowance)))
        # sleep_df = sleep_df.join(off_screen, (((F.col("dark_datetime") >= F.col("start_datetime")) & (F.col("dark_datetime") <= F.col("end_datetime"))) |\
        #                                       ((F.col("quiet_datetime") >= F.col("start_datetime")) & (F.col("quiet_datetime") <= F.col("end_datetime"))) | \
        #                                         ((F.col("stationary_datetime") >= F.col("start_datetime")) & (F.col("stationary_datetime") <= F.col("end_datetime")))))

        # V2
        sleep_onset_df = dark_df.join(quiet_df, F.col("dark_datetime") == F.col("quiet_datetime"))\
            .join(stationary_df, F.col("stationary_datetime") == F.col("dark_datetime"))\
            .join(off_screen, (F.col("dark_datetime") >= F.col("start_datetime")) & (F.col("dark_datetime") <= F.col("end_datetime")))\
            .withColumn("onset_datetime", F.greatest(F.col("dark_datetime"), F.col("start_datetime")))\
            .select(*["onset_datetime", "end_datetime", "average_lux_minute", "average_decibels_minute", "activities"])\
            .dropDuplicates().sort("onset_datetime")
        
        sleep_wake_df = off_screen.join(bright_transition_df, (F.col("bright_transition_datetime") >= F.col("start_datetime")))\
            .join(noise_transition_df, (F.col("noise_transition_datetime") >= F.col("start_datetime")))\
            .join(movement_transition_df, (F.col("movement_transition_datetime") >= F.col("start_datetime")))\
            .withColumn("wake_datetime", F.least(F.col("bright_transition_datetime"), F.col("noise_transition_datetime"), F.col("movement_transition_datetime"), F.col("end_datetime")))\
            .select(*["start_datetime", "wake_datetime"])\
            .groupBy("start_datetime").agg(F.min("wake_datetime").alias("wake_datetime"))\
            .dropDuplicates().sort("start_datetime")
        
        sleep_df = sleep_onset_df.join(sleep_wake_df, (F.col("onset_datetime") >= F.col("start_datetime")) & (F.col("wake_datetime") <= F.col("end_datetime")))\
            .withColumn("start_datetime", F.least(F.col("onset_datetime"), F.col("start_datetime")))\
            .withColumn("end_datetime", F.least(F.col("wake_datetime"), F.col("end_datetime")))\
            .drop("onset_datetime", "wake_datetime")\
            .groupBy("start_datetime", "end_datetime").agg(F.min(F.col("average_lux_minute")).alias("minimum_lux_minute"),\
                                                           F.min(F.col("average_decibels_minute")).alias("minimum_decibels_minute"))\
            .dropDuplicates().sort("start_datetime")\
            .withColumn("duration (s)", F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime")))\
            .filter(F.col("duration (s)") >= 3600)\
            .withColumn("duration (hr)", F.col("duration (s)") / 3600)\
            .withColumn("onset_time", udf_string_datetime(F.col("start_datetime")))\
            .withColumn("end_time", udf_string_datetime(F.col("end_datetime")))\
            .sort("start_datetime")
        sleep_df.write.parquet(parquet_filename)
    
    sleep_df = spark.read.parquet(parquet_filename)
    return sleep_df


def phone_productivity(user_id):
    """
    Map screen status to app notifications to infer productivity/distraction.

    NOTE Assumptions:
    1. Remove screen on (status 1) due to notifications since they are not initiated by people themselves
    2. 
    """
    time_cols = ["date", "hour", "minute"]
    screen_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_screen.csv")
    app_notifications_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_applications_notifications.csv")
    notification_screen_on = screen_df.filter(F.col("screen_status") == 1).join(app_notifications_df, time_cols)
    notification_screen_on.show()

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
    2. 

    NOTE Assumptions:
    1. Wake time as the minimum "end_datetime" of estimated sleep duration for each day
    2. The most frequently seen cluster at wake time as primary location
    3. 

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
    cur_day = "2024-02-02"

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
        .withColumn("prev_light", F.lag(F.col("average_lux_minute")).over(time_window))\
        .withColumn("prev_decibels", F.lag(F.col("average_decibels_minute")).over(time_window))\
        .withColumn("prev_decibel_outlier", F.lag(F.col("potential_outlier")).over(time_window))\
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
    # day_locations.write.csv(f"{cur_day}_locations.csv", header=True)
    
    # Contexts during location transition: inter cluster travel
    cluster_transitions = day_locations.filter(F.col("prev_cluster") != F.col("cluster_id"))
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
                                           F.when((F.abs(F.col("average_lux_minute") - F.col("prev_light")) >= significant_light_transition) |\
                                                  (F.abs(F.col("average_decibels_minute") - F.col("prev_decibels")) >= significant_noise_transition) |\
                                                    (F.col("activity_type") != F.col("prev_activity")), True).otherwise(False))
    context_transitions = day_contexts.filter((F.col("any_transition") == True) & (F.col("activity_type") != 4) & (F.col("prev_activity") != 4))\
        .sort("datetime")
    
    # day_contexts.write.csv(f"{cur_day}_contexts.csv", header=True)

    # Context analysis
    app_usage = process_application_usage_data(user_id)
    for col in app_usage.columns:
        if "timestamp" in col:
            app_usage = app_usage.withColumn(col, F.col(col).cast(FloatType()))

    # NOTE: (N+1) cluster analysis will be involved for N cluster transition points
    cluster_contexts_df = None
    cluster_productivity_df = None
    location_transition_datetimes = np.array(cluster_transitions.select("location_datetime").collect()).flatten()
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
    # cluster_contexts_df.write.csv(f"{cur_day}_cluster_contexts.csv", header=True)
    # cluster_productivity_df.write.csv(f"{cur_day}_cluster_productivity.csv", header=True)

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

    ALL_TABLES = ["applications_crashes", "applications_foreground", "applications_history",\
                    "applications_notifications", "bluetooth", "gsm", "light", "locations", "network",\
                    "plugin_ambient_noise", "plugin_device_usage", "plugin_google_activity_recognition",\
                    "screen", "screentext", "sensor_accelerometer", "sensor_bluetooth", "sensor_light",\
                    "sensor_wifi", "telephony", "wifi", "esms"]
    DATA_FOLDER = "user_data"
    TIMEZONE = pytz.timezone("Australia/Melbourne")

    # .astimezone(TIMEZONE)
    # .replace(tzinfo=TIMEZONE)
    udf_datetime_from_timestamp = F.udf(lambda x: datetime.fromtimestamp(x/1000, TIMEZONE), TimestampType())
    udf_generate_datetime = F.udf(lambda d, h, m: TIMEZONE.localize(datetime.strptime(f"{d} {h:02d}:{m:02d}", "%Y-%m-%d %H:%M")), TimestampType())
    udf_get_date_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).date().strftime("%Y-%m-%d"), StringType())
    udf_get_hour_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).time().hour, IntegerType())
    udf_get_minute_from_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).time().minute, IntegerType())
    udf_string_datetime = F.udf(lambda x: x.astimezone(TIMEZONE).strftime("%Y-%m-%d %H:%M"), StringType())
    udf_unique_wifi_list = F.udf(lambda x: ", ".join(sorted(list(set(x.split(", "))))), StringType())

    user_identifier = "pixel3"

    # -- NOTE: Only execute this block when db connection is required --
    # with open("database_config.json", 'r') as file:
    #     db_config = json.load(file)

    # db_connection = mysql.connector.connect(**db_config)
    # db_cursor = db_connection.cursor(buffered=True)

    # Export data from database to local csv
    # export_user_data(db_cursor, user_identifier)

    # # Delete emulator devices from aware_device table
    # delete_emulator_device(db_cursor, db_connection)

    # # Remove entries corresponding to invalid devices from all tables
    # valid_devices = get_valid_device_id(db_cursor)
    # for table in ALL_TABLES:
    #     delete_invalid_device_entries(db_cursor, table, tuple(valid_devices))
    #     # delete_single_entry(db_cursor, "aware_device", "emu64xa")
    #     db_connection.commit()

    # db_cursor.close()
    # db_connection.close()
    # -- End of block --

    # -- NOTE: This block of functions execute the extraction and early processing of sensor data into dataframes
    # process_light_data(user_identifier)
    # process_activity_data(user_identifier)
    # process_noise_data(user_identifier)
    # process_screen_data(user_identifier)
    # process_application_usage_data(user_identifier)
    # process_bluetooth_data(user_identifier)
    # process_location_data(user_identifier)
    # process_wifi_data(user_identifier)
    # -- End of block

    # -- NOTE: This block of functions combine multiple sensor information to generate interpretation
    # location_df = process_location_data(user_identifier)
    # cluster_df = cluster_locations(user_identifier, location_df, "double_latitude", "double_longitude")
    # estimate_sleep(user_identifier)
    # complement_location_data(user_identifier)
    daily_routine(user_identifier)
    # -- End of block

    # ambient_light(user_identifier)
    # phone_productivity(user_identifier)
    
    

    
    

    