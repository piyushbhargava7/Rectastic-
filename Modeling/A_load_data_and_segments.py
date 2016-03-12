# Importing Libraries
import os
import sys
import numpy as np
import pandas as pd
import time

# Path for spark source folder
os.environ['SPARK_HOME']="/Users/piyushbhargava/Downloads/spark-1.6.0-bin-hadoop2.6/"

# Append pyspark  to Python Path
sys.path.append("/Users/piyushbhargava/Downloads/spark-1.6.0-bin-hadoop2.6/python")
sys.path.append("/Users/piyushbhargava/Downloads/spark-1.6.0-bin-hadoop2.6/python/lib/py4j-0.9-src.zip")

from pyspark import SparkContext
from pyspark import SparkConf
from pyspark.sql import SQLContext, Row
from pyspark.sql.types import *
from pyspark.sql.functions import *

print ("Successfully imported Spark Modules")

conf = (SparkConf()
        .setMaster("local")
        .setAppName("My app")
        .set("spark.executor.memory", "1g")
        .set("spark.exeutor.cores", 5))

sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)


################################################################################
# LOADING DATA
################################################################################
# b - business data set
# c - check in data set
# r - review data set
# u - user data set

# Reading the training files as SQL data frame
train_b = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_business.json")
train_c = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_checkin.json")
train_r = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_review.json")
train_u = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_user.json")


# Reading the fianl test files as SQL dataframe
final_b = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_business.json")  # 1205
final_c = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_checkin.json")  # 734
final_r = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_review.json")  # 22956
final_u = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_user.json")  # 5105


# Infer the schema, and register the DataFrame as a table.

train_u.registerTempTable("train_u")
train_r.registerTempTable("train_r")
train_b.registerTempTable("train_b")
train_c.registerTempTable("train_c")

final_u.registerTempTable("final_u")
final_r.registerTempTable("final_r")
final_b.registerTempTable("final_b")
final_c.registerTempTable("final_c")



################################################################################
# EXTRACTING FEATURES - USEFUL, COOL & FUNNY VOTES FROM TRAINING REVIEW DATASETS
################################################################################

votes_RDD_b = train_r.map(lambda x: (x.business_id, x.votes.cool,
                                  x.votes.funny, x.votes.useful))

schema = StructType([StructField("bus_id", StringType(), True),
                     StructField("b_cool", IntegerType(), True),
                     StructField("b_funny", IntegerType(), True),
                     StructField("b_useful", IntegerType(), True)])
votes_df_b_0 = sqlContext.createDataFrame((votes_RDD_b), schema)

votes_RDD_u = train_u.map(lambda x: (x.user_id, x.votes.cool,
                                  x.votes.funny, x.votes.useful))
schema = StructType([StructField("user_id", StringType(), True),
                     StructField("u_cool", IntegerType(), True),
                     StructField("u_funny", IntegerType(), True),
                     StructField("u_useful", IntegerType(), True)])
votes_df_u = sqlContext.createDataFrame((votes_RDD_u), schema)



votes_df_u.registerTempTable("votes_u")

votes_df_b_0.registerTempTable("votes_b_0")
votes_df_b= sqlContext.sql("select bus_id, sum(b_cool)/count(b_cool) as avg_b_cool,"
                        " sum(b_funny)/count(b_funny) as avg_b_funny, sum(b_useful)/count(b_useful) as avg_b_useful"
                        " from votes_b_0 group by bus_id")
votes_df_b.registerTempTable("votes_b")



################################################################################
# EXTRACTING FEATURES - BUSINESS CHECK-INS FROM CHECKINS DATASETS
################################################################################

train_checkin = train_c.map(lambda x: (x.business_id, x.checkin_info["0-0"],
x.checkin_info["0-1"],
x.checkin_info["0-2"],
x.checkin_info["0-3"],
x.checkin_info["0-4"],
x.checkin_info["0-5"],
x.checkin_info["0-6"],
x.checkin_info["1-0"],
x.checkin_info["1-1"],
x.checkin_info["1-2"],
x.checkin_info["1-3"],
x.checkin_info["1-4"],
x.checkin_info["1-5"],
x.checkin_info["1-6"],
x.checkin_info["2-0"],
x.checkin_info["2-1"],
x.checkin_info["2-2"],
x.checkin_info["2-3"],
x.checkin_info["2-4"],
x.checkin_info["2-5"],
x.checkin_info["2-6"],
x.checkin_info["3-0"],
x.checkin_info["3-1"],
x.checkin_info["3-2"],
x.checkin_info["3-3"],
x.checkin_info["3-4"],
x.checkin_info["3-5"],
x.checkin_info["3-6"],
x.checkin_info["4-0"],
x.checkin_info["4-1"],
x.checkin_info["4-2"],
x.checkin_info["4-3"],
x.checkin_info["4-4"],
x.checkin_info["4-5"],
x.checkin_info["4-6"],
x.checkin_info["5-0"],
x.checkin_info["5-1"],
x.checkin_info["5-2"],
x.checkin_info["5-3"],
x.checkin_info["5-4"],
x.checkin_info["5-5"],
x.checkin_info["5-6"],
x.checkin_info["6-0"],
x.checkin_info["6-1"],
x.checkin_info["6-2"],
x.checkin_info["6-3"],
x.checkin_info["6-4"],
x.checkin_info["6-5"],
x.checkin_info["6-6"],
x.checkin_info["7-0"],
x.checkin_info["7-1"],
x.checkin_info["7-2"],
x.checkin_info["7-3"],
x.checkin_info["7-4"],
x.checkin_info["7-5"],
x.checkin_info["7-6"],
x.checkin_info["8-0"],
x.checkin_info["8-1"],
x.checkin_info["8-2"],
x.checkin_info["8-3"],
x.checkin_info["8-4"],
x.checkin_info["8-5"],
x.checkin_info["8-6"],
x.checkin_info["9-0"],
x.checkin_info["9-1"],
x.checkin_info["9-2"],
x.checkin_info["9-3"],
x.checkin_info["9-4"],
x.checkin_info["9-5"],
x.checkin_info["9-6"],
x.checkin_info["10-0"],
x.checkin_info["10-1"],
x.checkin_info["10-2"],
x.checkin_info["10-3"],
x.checkin_info["10-4"],
x.checkin_info["10-5"],
x.checkin_info["10-6"],
x.checkin_info["11-0"],
x.checkin_info["11-1"],
x.checkin_info["11-2"],
x.checkin_info["11-3"],
x.checkin_info["11-4"],
x.checkin_info["11-5"],
x.checkin_info["11-6"],
x.checkin_info["12-0"],
x.checkin_info["12-1"],
x.checkin_info["12-2"],
x.checkin_info["12-3"],
x.checkin_info["12-4"],
x.checkin_info["12-5"],
x.checkin_info["12-6"],
x.checkin_info["13-0"],
x.checkin_info["13-1"],
x.checkin_info["13-2"],
x.checkin_info["13-3"],
x.checkin_info["13-4"],
x.checkin_info["13-5"],
x.checkin_info["13-6"],
x.checkin_info["14-0"],
x.checkin_info["14-1"],
x.checkin_info["14-2"],
x.checkin_info["14-3"],
x.checkin_info["14-4"],
x.checkin_info["14-5"],
x.checkin_info["14-6"],
x.checkin_info["15-0"],
x.checkin_info["15-1"],
x.checkin_info["15-2"],
x.checkin_info["15-3"],
x.checkin_info["15-4"],
x.checkin_info["15-5"],
x.checkin_info["15-6"],
x.checkin_info["16-0"],
x.checkin_info["16-1"],
x.checkin_info["16-2"],
x.checkin_info["16-3"],
x.checkin_info["16-4"],
x.checkin_info["16-5"],
x.checkin_info["16-6"],
x.checkin_info["17-0"],
x.checkin_info["17-1"],
x.checkin_info["17-2"],
x.checkin_info["17-3"],
x.checkin_info["17-4"],
x.checkin_info["17-5"],
x.checkin_info["17-6"],
x.checkin_info["18-0"],
x.checkin_info["18-1"],
x.checkin_info["18-2"],
x.checkin_info["18-3"],
x.checkin_info["18-4"],
x.checkin_info["18-5"],
x.checkin_info["18-6"],
x.checkin_info["19-0"],
x.checkin_info["19-1"],
x.checkin_info["19-2"],
x.checkin_info["19-3"],
x.checkin_info["19-4"],
x.checkin_info["19-5"],
x.checkin_info["19-6"],
x.checkin_info["20-0"],
x.checkin_info["20-1"],
x.checkin_info["20-2"],
x.checkin_info["20-3"],
x.checkin_info["20-4"],
x.checkin_info["20-5"],
x.checkin_info["20-6"],
x.checkin_info["21-0"],
x.checkin_info["21-1"],
x.checkin_info["21-2"],
x.checkin_info["21-3"],
x.checkin_info["21-4"],
x.checkin_info["21-5"],
x.checkin_info["21-6"],
x.checkin_info["22-0"],
x.checkin_info["22-1"],
x.checkin_info["22-2"],
x.checkin_info["22-3"],
x.checkin_info["22-4"],
x.checkin_info["22-5"],
x.checkin_info["22-6"],
x.checkin_info["23-0"],
x.checkin_info["23-1"],
x.checkin_info["23-2"],
x.checkin_info["23-3"],
x.checkin_info["23-4"],
x.checkin_info["23-5"],
x.checkin_info["23-6"]))


final_checkin = final_c.map(lambda x: (x.business_id, x.checkin_info["0-0"],
x.checkin_info["0-1"],
x.checkin_info["0-2"],
x.checkin_info["0-3"],
x.checkin_info["0-4"],
x.checkin_info["0-5"],
x.checkin_info["0-6"],
x.checkin_info["1-0"],
x.checkin_info["1-1"],
x.checkin_info["1-2"],
x.checkin_info["1-3"],
x.checkin_info["1-4"],
x.checkin_info["1-5"],
x.checkin_info["1-6"],
x.checkin_info["2-0"],
x.checkin_info["2-1"],
x.checkin_info["2-2"],
x.checkin_info["2-3"],
x.checkin_info["2-4"],
x.checkin_info["2-5"],
x.checkin_info["2-6"],
x.checkin_info["3-0"],
x.checkin_info["3-1"],
x.checkin_info["3-2"],
x.checkin_info["3-3"],
x.checkin_info["3-4"],
x.checkin_info["3-5"],
x.checkin_info["3-6"],
x.checkin_info["4-0"],
x.checkin_info["4-1"],
x.checkin_info["4-2"],
x.checkin_info["4-3"],
x.checkin_info["4-4"],
x.checkin_info["4-5"],
x.checkin_info["4-6"],
x.checkin_info["5-0"],
x.checkin_info["5-1"],
x.checkin_info["5-2"],
x.checkin_info["5-3"],
x.checkin_info["5-4"],
x.checkin_info["5-5"],
x.checkin_info["5-6"],
x.checkin_info["6-0"],
x.checkin_info["6-1"],
x.checkin_info["6-2"],
x.checkin_info["6-3"],
x.checkin_info["6-4"],
x.checkin_info["6-5"],
x.checkin_info["6-6"],
x.checkin_info["7-0"],
x.checkin_info["7-1"],
x.checkin_info["7-2"],
x.checkin_info["7-3"],
x.checkin_info["7-4"],
x.checkin_info["7-5"],
x.checkin_info["7-6"],
x.checkin_info["8-0"],
x.checkin_info["8-1"],
x.checkin_info["8-2"],
x.checkin_info["8-3"],
x.checkin_info["8-4"],
x.checkin_info["8-5"],
x.checkin_info["8-6"],
x.checkin_info["9-0"],
x.checkin_info["9-1"],
x.checkin_info["9-2"],
x.checkin_info["9-3"],
x.checkin_info["9-4"],
x.checkin_info["9-5"],
x.checkin_info["9-6"],
x.checkin_info["10-0"],
x.checkin_info["10-1"],
x.checkin_info["10-2"],
x.checkin_info["10-3"],
x.checkin_info["10-4"],
x.checkin_info["10-5"],
x.checkin_info["10-6"],
x.checkin_info["11-0"],
x.checkin_info["11-1"],
x.checkin_info["11-2"],
x.checkin_info["11-3"],
x.checkin_info["11-4"],
x.checkin_info["11-5"],
x.checkin_info["11-6"],
x.checkin_info["12-0"],
x.checkin_info["12-1"],
x.checkin_info["12-2"],
x.checkin_info["12-3"],
x.checkin_info["12-4"],
x.checkin_info["12-5"],
x.checkin_info["12-6"],
x.checkin_info["13-0"],
x.checkin_info["13-1"],
x.checkin_info["13-2"],
x.checkin_info["13-3"],
x.checkin_info["13-4"],
x.checkin_info["13-5"],
x.checkin_info["13-6"],
x.checkin_info["14-0"],
x.checkin_info["14-1"],
x.checkin_info["14-2"],
x.checkin_info["14-3"],
x.checkin_info["14-4"],
x.checkin_info["14-5"],
x.checkin_info["14-6"],
x.checkin_info["15-0"],
x.checkin_info["15-1"],
x.checkin_info["15-2"],
x.checkin_info["15-3"],
x.checkin_info["15-4"],
x.checkin_info["15-5"],
x.checkin_info["15-6"],
x.checkin_info["16-0"],
x.checkin_info["16-1"],
x.checkin_info["16-2"],
x.checkin_info["16-3"],
x.checkin_info["16-4"],
x.checkin_info["16-5"],
x.checkin_info["16-6"],
x.checkin_info["17-0"],
x.checkin_info["17-1"],
x.checkin_info["17-2"],
x.checkin_info["17-3"],
x.checkin_info["17-4"],
x.checkin_info["17-5"],
x.checkin_info["17-6"],
x.checkin_info["18-0"],
x.checkin_info["18-1"],
x.checkin_info["18-2"],
x.checkin_info["18-3"],
x.checkin_info["18-4"],
x.checkin_info["18-5"],
x.checkin_info["18-6"],
x.checkin_info["19-0"],
x.checkin_info["19-1"],
x.checkin_info["19-2"],
x.checkin_info["19-3"],
x.checkin_info["19-4"],
x.checkin_info["19-5"],
x.checkin_info["19-6"],
x.checkin_info["20-0"],
x.checkin_info["20-1"],
x.checkin_info["20-2"],
x.checkin_info["20-3"],
x.checkin_info["20-4"],
x.checkin_info["20-5"],
x.checkin_info["20-6"],
x.checkin_info["21-0"],
x.checkin_info["21-1"],
x.checkin_info["21-2"],
x.checkin_info["21-3"],
x.checkin_info["21-4"],
x.checkin_info["21-5"],
x.checkin_info["21-6"],
x.checkin_info["22-0"],
x.checkin_info["22-1"],
x.checkin_info["22-2"],
x.checkin_info["22-3"],
x.checkin_info["22-4"],
x.checkin_info["22-5"],
x.checkin_info["22-6"],
x.checkin_info["23-0"],
x.checkin_info["23-1"],
x.checkin_info["23-2"],
x.checkin_info["23-3"],
x.checkin_info["23-4"],
x.checkin_info["23-5"],
x.checkin_info["23-6"]))


schema_checkin = StructType([StructField("b_id", StringType(), True),
StructField("c_0_00", IntegerType(), True),
StructField("c_1_00", IntegerType(), True),
StructField("c_2_00", IntegerType(), True),
StructField("c_3_00", IntegerType(), True),
StructField("c_4_00", IntegerType(), True),
StructField("c_5_00", IntegerType(), True),
StructField("c_6_00", IntegerType(), True),
StructField("c_0_01", IntegerType(), True),
StructField("c_1_01", IntegerType(), True),
StructField("c_2_01", IntegerType(), True),
StructField("c_3_01", IntegerType(), True),
StructField("c_4_01", IntegerType(), True),
StructField("c_5_01", IntegerType(), True),
StructField("c_6_01", IntegerType(), True),
StructField("c_0_02", IntegerType(), True),
StructField("c_1_02", IntegerType(), True),
StructField("c_2_02", IntegerType(), True),
StructField("c_3_02", IntegerType(), True),
StructField("c_4_02", IntegerType(), True),
StructField("c_5_02", IntegerType(), True),
StructField("c_6_02", IntegerType(), True),
StructField("c_0_03", IntegerType(), True),
StructField("c_1_03", IntegerType(), True),
StructField("c_2_03", IntegerType(), True),
StructField("c_3_03", IntegerType(), True),
StructField("c_4_03", IntegerType(), True),
StructField("c_5_03", IntegerType(), True),
StructField("c_6_03", IntegerType(), True),
StructField("c_0_04", IntegerType(), True),
StructField("c_1_04", IntegerType(), True),
StructField("c_2_04", IntegerType(), True),
StructField("c_3_04", IntegerType(), True),
StructField("c_4_04", IntegerType(), True),
StructField("c_5_04", IntegerType(), True),
StructField("c_6_04", IntegerType(), True),
StructField("c_0_05", IntegerType(), True),
StructField("c_1_05", IntegerType(), True),
StructField("c_2_05", IntegerType(), True),
StructField("c_3_05", IntegerType(), True),
StructField("c_4_05", IntegerType(), True),
StructField("c_5_05", IntegerType(), True),
StructField("c_6_05", IntegerType(), True),
StructField("c_0_06", IntegerType(), True),
StructField("c_1_06", IntegerType(), True),
StructField("c_2_06", IntegerType(), True),
StructField("c_3_06", IntegerType(), True),
StructField("c_4_06", IntegerType(), True),
StructField("c_5_06", IntegerType(), True),
StructField("c_6_06", IntegerType(), True),
StructField("c_0_07", IntegerType(), True),
StructField("c_1_07", IntegerType(), True),
StructField("c_2_07", IntegerType(), True),
StructField("c_3_07", IntegerType(), True),
StructField("c_4_07", IntegerType(), True),
StructField("c_5_07", IntegerType(), True),
StructField("c_6_07", IntegerType(), True),
StructField("c_0_08", IntegerType(), True),
StructField("c_1_08", IntegerType(), True),
StructField("c_2_08", IntegerType(), True),
StructField("c_3_08", IntegerType(), True),
StructField("c_4_08", IntegerType(), True),
StructField("c_5_08", IntegerType(), True),
StructField("c_6_08", IntegerType(), True),
StructField("c_0_09", IntegerType(), True),
StructField("c_1_09", IntegerType(), True),
StructField("c_2_09", IntegerType(), True),
StructField("c_3_09", IntegerType(), True),
StructField("c_4_09", IntegerType(), True),
StructField("c_5_09", IntegerType(), True),
StructField("c_6_09", IntegerType(), True),
StructField("c_0_10", IntegerType(), True),
StructField("c_1_10", IntegerType(), True),
StructField("c_2_10", IntegerType(), True),
StructField("c_3_10", IntegerType(), True),
StructField("c_4_10", IntegerType(), True),
StructField("c_5_10", IntegerType(), True),
StructField("c_6_10", IntegerType(), True),
StructField("c_0_11", IntegerType(), True),
StructField("c_1_11", IntegerType(), True),
StructField("c_2_11", IntegerType(), True),
StructField("c_3_11", IntegerType(), True),
StructField("c_4_11", IntegerType(), True),
StructField("c_5_11", IntegerType(), True),
StructField("c_6_11", IntegerType(), True),
StructField("c_0_12", IntegerType(), True),
StructField("c_1_12", IntegerType(), True),
StructField("c_2_12", IntegerType(), True),
StructField("c_3_12", IntegerType(), True),
StructField("c_4_12", IntegerType(), True),
StructField("c_5_12", IntegerType(), True),
StructField("c_6_12", IntegerType(), True),
StructField("c_0_13", IntegerType(), True),
StructField("c_1_13", IntegerType(), True),
StructField("c_2_13", IntegerType(), True),
StructField("c_3_13", IntegerType(), True),
StructField("c_4_13", IntegerType(), True),
StructField("c_5_13", IntegerType(), True),
StructField("c_6_13", IntegerType(), True),
StructField("c_0_14", IntegerType(), True),
StructField("c_1_14", IntegerType(), True),
StructField("c_2_14", IntegerType(), True),
StructField("c_3_14", IntegerType(), True),
StructField("c_4_14", IntegerType(), True),
StructField("c_5_14", IntegerType(), True),
StructField("c_6_14", IntegerType(), True),
StructField("c_0_15", IntegerType(), True),
StructField("c_1_15", IntegerType(), True),
StructField("c_2_15", IntegerType(), True),
StructField("c_3_15", IntegerType(), True),
StructField("c_4_15", IntegerType(), True),
StructField("c_5_15", IntegerType(), True),
StructField("c_6_15", IntegerType(), True),
StructField("c_0_16", IntegerType(), True),
StructField("c_1_16", IntegerType(), True),
StructField("c_2_16", IntegerType(), True),
StructField("c_3_16", IntegerType(), True),
StructField("c_4_16", IntegerType(), True),
StructField("c_5_16", IntegerType(), True),
StructField("c_6_16", IntegerType(), True),
StructField("c_0_17", IntegerType(), True),
StructField("c_1_17", IntegerType(), True),
StructField("c_2_17", IntegerType(), True),
StructField("c_3_17", IntegerType(), True),
StructField("c_4_17", IntegerType(), True),
StructField("c_5_17", IntegerType(), True),
StructField("c_6_17", IntegerType(), True),
StructField("c_0_18", IntegerType(), True),
StructField("c_1_18", IntegerType(), True),
StructField("c_2_18", IntegerType(), True),
StructField("c_3_18", IntegerType(), True),
StructField("c_4_18", IntegerType(), True),
StructField("c_5_18", IntegerType(), True),
StructField("c_6_18", IntegerType(), True),
StructField("c_0_19", IntegerType(), True),
StructField("c_1_19", IntegerType(), True),
StructField("c_2_19", IntegerType(), True),
StructField("c_3_19", IntegerType(), True),
StructField("c_4_19", IntegerType(), True),
StructField("c_5_19", IntegerType(), True),
StructField("c_6_19", IntegerType(), True),
StructField("c_0_20", IntegerType(), True),
StructField("c_1_20", IntegerType(), True),
StructField("c_2_20", IntegerType(), True),
StructField("c_3_20", IntegerType(), True),
StructField("c_4_20", IntegerType(), True),
StructField("c_5_20", IntegerType(), True),
StructField("c_6_20", IntegerType(), True),
StructField("c_0_21", IntegerType(), True),
StructField("c_1_21", IntegerType(), True),
StructField("c_2_21", IntegerType(), True),
StructField("c_3_21", IntegerType(), True),
StructField("c_4_21", IntegerType(), True),
StructField("c_5_21", IntegerType(), True),
StructField("c_6_21", IntegerType(), True),
StructField("c_0_22", IntegerType(), True),
StructField("c_1_22", IntegerType(), True),
StructField("c_2_22", IntegerType(), True),
StructField("c_3_22", IntegerType(), True),
StructField("c_4_22", IntegerType(), True),
StructField("c_5_22", IntegerType(), True),
StructField("c_6_22", IntegerType(), True),
StructField("c_0_23", IntegerType(), True),
StructField("c_1_23", IntegerType(), True),
StructField("c_2_23", IntegerType(), True),
StructField("c_3_23", IntegerType(), True),
StructField("c_4_23", IntegerType(), True),
StructField("c_5_23", IntegerType(), True),
StructField("c_6_23", IntegerType(), True)
])

train_checkin_df = sqlContext.createDataFrame((train_checkin), schema_checkin)
final_checkin_df = sqlContext.createDataFrame((final_checkin), schema_checkin)

train_checkin_df= train_checkin_df.fillna(0)
final_checkin_df= final_checkin_df.fillna(0)

train_checkin_df.registerTempTable("train_checkin")
final_checkin_df.registerTempTable("final_checkin")


# grouping checkin times

def checkin(t, d, cat):
    a=[]
    for i in t:
        for j in d:
            if i == t[0] and j==d[0] and j<10:
                b = '(c_'+`i`+'_0'+`j`+'+'
            elif i == t[0] and j==d[0] :
                b = '(c_'+`i`+'_'+`j`+'+'
            elif i == t[-1] and j==d[-1] and j<10:
                b = 'c_'+`i`+'_0'+`j`
            elif i == t[-1] and j==d[-1]:
                b = 'c_'+`i`+'_'+`j`
            elif j<10:
                b = 'c_'+`i`+'_0'+`j`+'+'
            else:
                b = 'c_'+`i`+'_'+`j`+'+'
            a.append(b)
    a.append(cat)
    return " ".join(map(str,a))

weekday_early_mrng = checkin([1,2,3,4,5],[0,1,2,3,4,5,6],") as weekday_early_mrng")
weekday_mrng = checkin([1,2,3,4,5],[7,8,9,10],") as weekday_mrng")
weekday_midday = checkin([1,2,3,4,5],[11,12,13],") as weekday_midday")
weekday_afternoon = checkin([1,2,3,4,5],[14,15,16],") as weekday_afternoon")
weekday_evening = checkin([1,2,3,4,5],[17,18,19,20],") as weekday_evening")
weekday_night = checkin([1,2,3,4,5],[21,22,23],") as weekday_night")


weekend_early_mrng = checkin([0,6],[0,1,2,3,4,5,6],") as weekend_early_mrng")
weekend_mrng = checkin([0,6],[7,8,9,10],") as weekend_mrng")
weekend_midday = checkin([0,6],[11,12,13],") as weekend_midday")
weekend_afternoon = checkin([0,6],[14,15,16],") as weekend_afternoon")
weekend_evening = checkin([0,6],[17,18,19,20],") as weekend_evening")
weekend_night = checkin([0,6],[21,22,23],") as weekend_night")


train_checkin_df_new = sqlContext.sql("select b_id, "
                                      + weekday_early_mrng + ","
                                      + weekday_mrng + ","
                                      + weekday_midday + ","
                                      + weekday_afternoon + ","
                                      + weekday_evening + ","
                                      + weekday_night + ","
                                      + weekend_early_mrng + ","
                                      + weekend_mrng + ","
                                      + weekend_midday + ","
                                      + weekend_afternoon + ","
                                      + weekend_evening + ","
                                      + weekend_night +
                                      " from train_checkin")


final_checkin_df_new = sqlContext.sql("select b_id, "
                                      + weekday_early_mrng + ","
                                      + weekday_mrng + ","
                                      + weekday_midday + ","
                                      + weekday_afternoon + ","
                                      + weekday_evening + ","
                                      + weekday_night + ","
                                      + weekend_early_mrng + ","
                                      + weekend_mrng + ","
                                      + weekend_midday + ","
                                      + weekend_afternoon + ","
                                      + weekend_evening + ","
                                      + weekend_night +
                                      " from final_checkin")


train_checkin_df_new.registerTempTable("train_checkin_new")
final_checkin_df_new.registerTempTable("final_checkin_new")




