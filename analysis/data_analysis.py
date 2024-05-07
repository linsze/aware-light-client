"""
Author: Lin Sze Khoo
Created on: 24/01/2024
Last modified on: 07/05/2024
"""
import json
import math
import os
from datetime import datetime
from itertools import product

import findspark
import matplotlib.pyplot as plt
import mysql.connector
import numpy as np
import pandas as pd
import pytz
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (FloatType, IntegerType, LongType, StringType,
                               StructField, StructType, TimestampType)
from pyspark.sql.window import Window
from scipy.interpolate import interp1d


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
            table_df["datetime"] = [datetime.fromtimestamp(d/1000) for d in table_df["timestamp"]]
            table_df["datetime"] = table_df["datetime"].dt.tz_localize("UTC").dt.tz_convert("Australia/Melbourne")
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
    Interpolate, smoothen, and clean data based on data type.
    1. Nearest neighbour interpolation based on https://dl.acm.org/doi/pdf/10.1145/3510029
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
        
        # NOTE (Assumption): Ignores extremely high lumination occuring at 01/02/2024 23:06 and outliers based on assumed threshold
        light_df = light_df.filter(F.col("double_light_lux") <= 2000)
        light_df = light_df.withColumn("rolling_light_lux", F.mean("double_light_lux").over(smooth_window))
        light_df = light_df.withColumn("residuals", F.col("double_light_lux") - F.col("rolling_light_lux"))

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
        light_df = spark.createDataFrame(light_df)
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
    Interpolate, smoothen, and clean data based on data type.
    1. Nearest neighbour interpolation based on https://dl.acm.org/doi/pdf/10.1145/3510029
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
        noise_df = spark.createDataFrame(noise_df)
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


def estimate_sleep(user_id):
    """
    Estimate sleep duration based on information from light, noise, activity and phone screen.
    Current assumptions:
    1. Does not limit to 1 estimated entry per day
    2. Remove those with duration shorter than 1 hour
    """
    light_df = process_light_data(user_id)
    noise_df = process_noise_data(user_id)
    activity_df = process_activity_data(user_id)
    screen_df = spark.read.option("header", True).csv(f"{DATA_FOLDER}/{user_id}_plugin_device_usage.csv")

    # NOTE: Environment may not be completely quiet and/or dark
    dark_threshold = 10 
    audio_quartiles = noise_df.approxQuantile("average_decibels_minute", [0.25, 0.50], 0.01)
    silence_threshold = round((audio_quartiles[1]-audio_quartiles[0]) / 2)
    dark_env = light_df.filter(F.col("average_lux_minute") <= dark_threshold)
    quiet_env = noise_df.filter(F.col("average_decibels_minute") <= round(audio_quartiles[1]))
    stationary = activity_df.filter(F.col("activity_type") == 3)

    off_screen = screen_df.filter(F.col("double_elapsed_device_off") > 0)\
        .withColumn("timestamp", F.col("timestamp").cast(FloatType()))\
        .withColumn("start_timestamp", F.col("timestamp") - F.col("double_elapsed_device_off"))\
        .withColumn("start_datetime", udf_datetime_from_timestamp(F.col("start_timestamp")))\
        .withColumn("end_datetime", udf_datetime_from_timestamp(F.col("timestamp")))

    dark_env = dark_env.withColumn("dark_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    quiet_env = quiet_env.withColumn("quiet_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))
    stationary = stationary.withColumn("stationary_datetime", udf_generate_datetime(F.col("date"), F.col("hour"), F.col("minute")))

    # V1
    # time_diff_allowance = 10*60
    # sleep_df = dark_env.join(quiet_env, F.abs(F.unix_timestamp(dark_env["dark_datetime"]) - F.unix_timestamp(quiet_env["quiet_datetime"])) <= time_diff_allowance)
    # sleep_df = sleep_df.join(stationary, ((F.abs(F.unix_timestamp(sleep_df["dark_datetime"]) - F.unix_timestamp(stationary["stationary_datetime"])) <= time_diff_allowance) |\
    #                        (F.abs(F.unix_timestamp(sleep_df["quiet_datetime"]) - F.unix_timestamp(stationary["stationary_datetime"])) <= time_diff_allowance)))
    # sleep_df = sleep_df.join(off_screen, (((F.col("dark_datetime") >= F.col("start_datetime")) & (F.col("dark_datetime") <= F.col("end_datetime"))) |\
    #                                       ((F.col("quiet_datetime") >= F.col("start_datetime")) & (F.col("quiet_datetime") <= F.col("end_datetime"))) | \
    #                                         ((F.col("stationary_datetime") >= F.col("start_datetime")) & (F.col("stationary_datetime") <= F.col("end_datetime")))))

    # V2
    sleep_df = off_screen.join(dark_env, (F.col("dark_datetime") >= F.col("start_datetime")) & (F.col("dark_datetime") <= F.col("end_datetime")))\
        .join(quiet_env, (F.col("quiet_datetime") >= F.col("start_datetime")) & (F.col("quiet_datetime") <= F.col("end_datetime")))\
        .join(stationary, (F.col("stationary_datetime") >= F.col("start_datetime")) & (F.col("stationary_datetime") <= F.col("end_datetime")))

    sleep_df = sleep_df.select(*["start_datetime", "end_datetime", "average_lux_minute", "average_decibels_minute"])\
        .dropDuplicates()
    sleep_df = sleep_df.groupBy("start_datetime", "end_datetime").agg(F.mean(F.col("average_decibels_minute")).alias("average_ambient_decibels"), \
                                                                      F.mean(F.col("average_lux_minute")).alias("average_ambient_luminance"))
    sleep_df = sleep_df.withColumn("duration", F.unix_timestamp(F.col("end_datetime")) - F.unix_timestamp(F.col("start_datetime")))\
        .filter(F.col("duration") >= 3600)\
        .sort("start_datetime")
    sleep_df.show()
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
        .getOrCreate()
    sc = spark.sparkContext

    ALL_TABLES = ["applications_crashes", "applications_foreground", "applications_history",\
                  "applications_notifications", "bluetooth", "gsm", "light", "network", "plugin_ambient_noise",\
                    "plugin_device_usage", "plugin_google_activity_recognition", "screen", "screentext",\
                        "sensor_accelerometer", "sensor_bluetooth", "sensor_light", "sensor_wifi", "telephony",\
                            "wifi", "esms"]
    DATA_FOLDER = "analysis/user_data"
    
    udf_datetime_from_timestamp = F.udf(lambda x: datetime.fromtimestamp(x/1000), TimestampType())
    udf_generate_datetime = F.udf(lambda d, h, m: datetime.strptime(f"{d} {h:02d}:{m:02d}", "%Y-%m-%d %H:%M"), TimestampType())
    udf_get_date_from_datetime = F.udf(lambda x: x.date().strftime("%Y-%m-%d"), StringType())

    with open("analysis/database_config.json", 'r') as file:
        db_config = json.load(file)

    db_connection = mysql.connector.connect(**db_config)
    db_cursor = db_connection.cursor(buffered=True)

    user_identifier = "pixel3"
    # export_user_data(db_cursor, user_identifier)
    # export_user_esm(db_cursor, user_identifier)

    # process_light_data(user_identifier)
    # process_activity_data(user_identifier)
    # process_noise_data(user_identifier)
    # process_screen_data(user_identifier)
    # process_app_notifications_data(user_identifier)
    # ambient_light(user_identifier)
    estimate_sleep(user_identifier)

    
    

    # delete_emulator_device(db_cursor, db_connection)

    # valid_devices = get_valid_device_id(db_cursor)
    # for table in ALL_TABLES:
    #     delete_invalid_device_entries(db_cursor, table, tuple(valid_devices))
    #     # delete_single_entry(db_cursor, "aware_device", "emu64xa")
    #     db_connection.commit()
    
    db_cursor.close()
    db_connection.close()
    