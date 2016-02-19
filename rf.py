__author__ = 'piyush'

# Importing Libraries
import os
import sys
import numpy as np
import pandas as pd

# Path for spark source folder
os.environ['SPARK_HOME'] = "/Users/piyushbhargava/Downloads/spark-1.6.0-bin-hadoop2.6/"

# Append pyspark  to Python Path
sys.path.append("/Users/piyushbhargava/Downloads/spark-1.6.0-bin-hadoop2.6/python")

try:
    from pyspark import SparkContext

    from pyspark import SparkConf

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

from pyspark import SparkContext

sc = SparkContext('local')

# sc is an existing SparkContext.

# Importing libraries from spark
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import col

from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.mllib.util import MLUtils

from pyspark.ml.feature import VectorAssembler

from pyspark.sql import SQLContext
from pyspark.sql.functions import *

sqlContext = SQLContext(sc)


################################################################################
# LOADING DATA
################################################################################
# b - business dataset
# c - checkin dataset
# r - review dataset
# u - user dataset

# Reading the training files as SQL dataframe
train_b = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_business.json")
train_c = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_checkin.json")
train_r = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_review.json")
train_u = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_training_set/yelp_training_set_user.json")

# Reading the intermediate test files as SQL dataframe
test_b = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_test_set/yelp_test_set_business.json")
test_c = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_test_set/yelp_test_set_checkin.json")
test_r = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_test_set/yelp_test_set_review.json")
test_u = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/yelp_test_set/yelp_test_set_user.json")

# Reading the fianl test files as SQL dataframe
final_b = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_business.json")  # 1205
final_c = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_checkin.json")  # 734
final_r = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_review.json")  # 22956
final_u = sqlContext.read.json(
        "/Users/piyushbhargava/Desktop/MSAN_630/project/yelp/final_test_set/final_test_set_user.json")  # 5105


################################################################################
# PRINTING THE FIELDS IN THE DATA
################################################################################

# Printing fields in business datasets in train, test, final test
# root

# train_b.printSchema()
# test_b.printSchema()
# final_b.printSchema()

#  |-- business_id: string (nullable = true)
#  |-- categories: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- city: string (nullable = true)
#  |-- full_address: string (nullable = true)
#  |-- latitude: double (nullable = true)
#  |-- longitude: double (nullable = true)
#  |-- name: string (nullable = true)
#  |-- neighborhoods: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- open: boolean (nullable = true)
#  |-- review_count: long (nullable = true)
#  |-- stars: double (nullable = true)
#  |-- state: string (nullable = true)
#  |-- type: string (nullable = true)
#
# root
#  |-- business_id: string (nullable = true)
#  |-- categories: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- city: string (nullable = true)
#  |-- full_address: string (nullable = true)
#  |-- latitude: double (nullable = true)
#  |-- longitude: double (nullable = true)
#  |-- name: string (nullable = true)
#  |-- neighborhoods: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- open: boolean (nullable = true)
#  |-- review_count: long (nullable = true)
#  |-- state: string (nullable = true)
#  |-- type: string (nullable = true)
#
# root
#  |-- business_id: string (nullable = true)
#  |-- categories: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- city: string (nullable = true)
#  |-- full_address: string (nullable = true)
#  |-- latitude: double (nullable = true)
#  |-- longitude: double (nullable = true)
#  |-- name: string (nullable = true)
#  |-- neighborhoods: array (nullable = true)
#  |    |-- element: string (containsNull = true)
#  |-- open: boolean (nullable = true)
#  |-- review_count: long (nullable = true)
#  |-- state: string (nullable = true)
#  |-- type: string (nullable = true)

# Printing fields in checkin datasets in train, test, final test
# root
# train_c.printSchema()
# test_c.printSchema()
# final_c.printSchema()


#  |-- business_id: string (nullable = true)
#  |-- checkin_info: struct (nullable = true)
#  |-- type: string (nullable = true)
#
# root
#  |-- business_id: string (nullable = true)
#  |-- checkin_info: struct (nullable = true)
#  |-- type: string (nullable = true)
#
# root
#  |-- business_id: string (nullable = true)
#  |-- checkin_info: struct (nullable = true)
#  |-- type: string (nullable = true)

# Printing fields in review datasets in train, test, final test
# root

# train_r.printSchema()
# test_r.printSchema()
# final_r.printSchema()


#  |-- business_id: string (nullable = true)
#  |-- date: string (nullable = true)
#  |-- review_id: string (nullable = true)
#  |-- stars: long (nullable = true)
#  |-- text: string (nullable = true)
#  |-- type: string (nullable = true)
#  |-- user_id: string (nullable = true)
#  |-- votes: struct (nullable = true)
#  |    |-- cool: long (nullable = true)
#  |    |-- funny: long (nullable = true)
#  |    |-- useful: long (nullable = true)
#
# root
#  |-- business_id: string (nullable = true)
#  |-- type: string (nullable = true)
#  |-- user_id: string (nullable = true)
#
# root
#  |-- business_id: string (nullable = true)
#  |-- review_id: string (nullable = true)
#  |-- type: string (nullable = true)
#  |-- user_id: string (nullable = true)

# Printing fields in user datasets in train, test, final test
# root
# train_u.printSchema()
# test_u.printSchema()
# final_u.printSchema()

#  |-- average_stars: double (nullable = true)
#  |-- name: string (nullable = true)
#  |-- review_count: long (nullable = true)
#  |-- type: string (nullable = true)
#  |-- user_id: string (nullable = true)
#  |-- votes: struct (nullable = true)
#  |    |-- cool: long (nullable = true)
#  |    |-- funny: long (nullable = true)
#  |    |-- useful: long (nullable = true)
#
# root
#  |-- name: string (nullable = true)
#  |-- review_count: long (nullable = true)
#  |-- type: string (nullable = true)
#  |-- user_id: string (nullable = true)
#
# root
#  |-- name: string (nullable = true)
#  |-- review_count: long (nullable = true)
#  |-- type: string (nullable = true)
#  |-- user_id: string (nullable = true)



# Infer the schema, and register the DataFrame as a table.

train_u.registerTempTable("train_u")
train_r.registerTempTable("train_r")
train_b.registerTempTable("train_b")

final_u.registerTempTable("final_u")
final_r.registerTempTable("final_r")
final_b.registerTempTable("final_b")


################################################################################
# EXTRACTING FEATURES - BUSINESS CATEGORIES FROM BUSINESS DATASETS
################################################################################

# Combining the 'categories' field from train and final business datasets into 1 dataset
categories_total = train_b.map(lambda x: (x['business_id'], x['categories'])).union(
        final_b.map(lambda x: (x['business_id'], x['categories'])))

categories_df_total = sqlContext.createDataFrame(categories_total)
categories_pandas_total = categories_df_total.toPandas()

# Creating dummy variables for all categories
categories_pandas_total_dummy = categories_pandas_total['_2'].str.join(sep='*').str.get_dummies(sep='*')

categories_pandas_total_2 = sqlContext.createDataFrame(
        categories_pandas_total.rename(columns={'_1': 'business_id'}).drop('_2', 1).join(categories_pandas_total_dummy))

categories_pandas_total_2.registerTempTable("categories_total")

################################################################################
# COMBINING ALL TRAIN DATA SETS INTO A SINGLE DATASET
################################################################################

# Creating a single train data set containing available user and business information for all reviews

merged_train = sqlContext.sql("SELECT A.review_id, A.stars, B.average_stars as user_avg_stars, "
                              "B.review_count as u_review_count, C.latitude, "
                              "C.longitude, C.open, C.review_count as b_review_count, "
                              "C.stars as business_avg_stars, D.* from "
                              "(Select business_id, user_id, review_id, stars from train_r) A "
                              "LEFT JOIN "
                              "(SELECT user_id, average_stars, review_count from train_u) B "
                              "on A.user_id = B.user_id "
                              "LEFT JOIN "
                              "(SELECT business_id, stars, latitude, longitude, open, review_count from train_b) C "
                              "on A.business_id = C.business_id "
                              "LEFT JOIN "
                              "(SELECT * FROM categories_total) D ON A.business_id = D.business_id ")


########################################################################################################
# COMBINING ALL TRAIN DATA SETS AND CREATING SUBSETS BASED ON AVAILABLE USER AND BUSINESS INFORMATION
########################################################################################################

# Creating a single train data set for reviews that have both user and business information available
# 215879
merged_train_ku_kb = sqlContext.sql("SELECT A.review_id, A.stars, "
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.* from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")

# Creating a single train data set for reviews that have user information available
# 229907
merged_train_ku_ub = sqlContext.sql("SELECT A.review_id, A.stars,"
                                    "B.review_count as u_review_count from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where B.review_count is not NULL")


# Creating a single train data set for reviews that have business information available
# 229907
merged_train_uu_kb = sqlContext.sql("SELECT A.review_id, A.stars,C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.* from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where C.review_count is not NULL")

# Creating a single train data set for reviews that have neither business nor user information available
# 0
merged_train_uu_ub = sqlContext.sql("SELECT A.review_id, A.stars from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where B.review_count is NULL and C.review_count is NULL")


########################################################################################################
# COMBINING ALL FINAL (TEST) DATA SETS AND CREATING SUBSETS BASED ON AVAILABLE USER AND BUSINESS INFORMATION
########################################################################################################

# Creating a single final(test) data set for reviews that have both user and business information available
# 4767
merged_final_ku_kb = sqlContext.sql("SELECT A.review_id, "
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.* from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from final_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from final_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")


# Creating a single final(test) data set for reviews that have user information available but not business info.
# 13434
merged_final_ku_ub = sqlContext.sql("SELECT A.review_id, "
                                    "B.review_count as u_review_count from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from final_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from final_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where B.review_count is not NULL and C.review_count is NULL")


# Creating a single final(test) data set for reviews that have business information available but not user info.
# 4608
merged_final_uu_kb = sqlContext.sql("SELECT A.review_id, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.* from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from final_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from final_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where B.review_count is NULL and C.review_count is not NULL")


# Creating a single final(test) data set for reviews that have neither business information nor user info. available
# 13595
merged_final_uu_ub = sqlContext.sql("SELECT A.review_id from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from final_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from final_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "where B.review_count is NULL and C.review_count is NULL")



########################################################################################################
# Model 0 - CREATING A SINGLE MODEL FOR PREDICTING RATINGS FOR REVIEWS WITH KNOWN USERS AND BUSINESSES
# USING COMPLETE TRAIN DATA SET
########################################################################################################

# Creating a list of features to be used for training the model
removelist_train = set(['stars', 'business_id', 'review_id', 'user_avg_stars', 'business_avg_stars', 'u_review_count'])
newlist_train = [v for i, v in enumerate(merged_train.columns) if v not in removelist_train]

# Putting data in vector assembler form
assembler_train = VectorAssembler(
        inputCols=newlist_train,
        outputCol="features")

transformed_train = assembler_train.transform(merged_train)


# Creating input dataset in the form of labeled point for training the model
data_train = (transformed_train.select("features", "stars")).map(lambda row: LabeledPoint(row.stars, row.features))


# Training the model using Random forest regressor
model_train = RandomForest.trainRegressor(data_train, categoricalFeaturesInfo={},
                                          numTrees=10, featureSubsetStrategy="auto",
                                          impurity='variance', maxDepth=8, maxBins=32)

########################################################################################################
# PREDICTIONS ON FINAL (TEST) DATASET USING DEVELOPED MODEL 'model_train'
########################################################################################################

# Creating a list of features to be used for predictions
removelist_final = set(['business_id', 'review_id', 'u_review_count'])
newlist_final = [v for i, v in enumerate(merged_final_ku_kb.columns) if v not in removelist_final]

# Putting data in vector assembler form
assembler_final = VectorAssembler(
        inputCols=newlist_final,
        outputCol="features")

transformed_final = assembler_final.transform(merged_final_ku_kb)


# Creating input dataset to be used for predictions
data_final = transformed_final.select("features", "review_id")

# Predicting ratings using the developed model
predictions = model_train.predict(data_final.map(lambda x: x.features))
labelsAndPredictions_ku_kb = data_final.map(lambda data_final: data_final.review_id).zip(predictions)



########################################################################################################
# CREATING SEGMENT WISE MODELS USING RESPECTIVE SEGMENTS OF TRAIN DATA SETS
########################################################################################################

########################################################################################################
# MODEL 1 - Known user known business model - Training the model on Segment with known user and business
# information
########################################################################################################

# Creating a list of features to be used for training the model
removelist_train_ku_kb = set(['stars', 'business_id', 'review_id'])
newlist_train_ku_kb = [v for i, v in enumerate(merged_train_ku_kb.columns) if v not in removelist_train_ku_kb]

# Putting data in vector assembler form
assembler_train_ku_kb = VectorAssembler(
        inputCols=newlist_train_ku_kb,
        outputCol="features")

transformed_train_ku_kb = assembler_train_ku_kb.transform(merged_train_ku_kb)

# Creating input dataset in the form of labeled point for training the model
data_train_ku_kb = (transformed_train_ku_kb.select("features", "stars")).map(
        lambda row: LabeledPoint(row.stars, row.features))

# Training the model using Random forest regressor
model_train_ku_kb = RandomForest.trainRegressor(data_train_ku_kb, categoricalFeaturesInfo={},
                                                numTrees=24, featureSubsetStrategy="auto",
                                                impurity='variance', maxDepth=5, maxBins=32)


# MODEL 1 - Predictions using 'Known user known business model' on Final/Test Segment with user and business information

# Creating a list of features to be used for predictions
removelist_final_ku_kb = set(['business_id', 'review_id'])
newlist_final_ku_kb = [v for i, v in enumerate(merged_final_ku_kb.columns) if v not in removelist_final_ku_kb]

# Putting data in vector assembler form
assembler_final_ku_kb = VectorAssembler(
        inputCols=newlist_final_ku_kb,
        outputCol="features")

transformed_final_ku_kb = assembler_final_ku_kb.transform(merged_final_ku_kb)

# Creating input dataset to be used for predictions
data_final_ku_kb = transformed_final_ku_kb.select("features", "review_id")

# Predicting ratings using the developed model
predictions_ku_kb = model_train_ku_kb.predict(data_final_ku_kb.map(lambda x: x.features))
labelsAndPredictions_ku_kb = data_final_ku_kb.map(lambda data_final_ku_kb: data_final_ku_kb.review_id).zip(
        predictions_ku_kb)


########################################################################################################
# MODEL 2 - UnKnown user known business model - Training the model on Segment with known business information
########################################################################################################


# Creating a list of features to be used for training the model
removelist_train_uu_kb = set(['stars', 'business_id', 'review_id'])
newlist_train_uu_kb = [v for i, v in enumerate(merged_train_uu_kb.columns) if v not in removelist_train_uu_kb]

# Putting data in vector assembler form
assembler_train_uu_kb = VectorAssembler(
        inputCols=newlist_train_uu_kb,
        outputCol="features")

transformed_train_uu_kb = assembler_train_uu_kb.transform(merged_train_uu_kb)

# Creating input dataset in the form of labeled point for training the model
data_train_uu_kb = (transformed_train_uu_kb.select("features", "stars")).map(
        lambda row: LabeledPoint(row.stars, row.features))

# Training the model using Random forest regressor
model_train_uu_kb = RandomForest.trainRegressor(data_train_uu_kb, categoricalFeaturesInfo={},
                                                numTrees=24, featureSubsetStrategy="auto",
                                                impurity='variance', maxDepth=5, maxBins=32)


# MODEL 2 - Predictions using 'UnKnown user known business model' on Final/Test Segment with known business information

# Creating a list of features to be used for predictions
removelist_final_uu_kb = set(['business_id', 'review_id'])
newlist_final_uu_kb = [v for i, v in enumerate(merged_final_uu_kb.columns) if v not in removelist_final_uu_kb]

# Putting data in vector assembler form
assembler_final_uu_kb = VectorAssembler(
        inputCols=newlist_final_uu_kb,
        outputCol="features")

transformed_final_uu_kb = assembler_final_uu_kb.transform(merged_final_uu_kb)

# Creating input dataset to be used for predictions
data_final_uu_kb = transformed_final_uu_kb.select("features", "review_id")

# Predicting ratings using the developed model
predictions_uu_kb = model_train_uu_kb.predict(data_final_uu_kb.map(lambda x: x.features))
labelsAndPredictions_uu_kb = data_final_uu_kb.map(lambda data_final_uu_kb: data_final_uu_kb.review_id).zip(
        predictions_uu_kb)


########################################################################################################
# MODEL 3 - Known user Unknown business model - Training the model on Segment with known user information
########################################################################################################


# Creating a list of features to be used for training the model
removelist_train_ku_ub = set(['stars', 'business_id', 'review_id', 'latitude', 'longitude', 'open', 'b_review_count'])
newlist_train_ku_ub = [v for i, v in enumerate(merged_train_ku_ub.columns) if v not in removelist_train_ku_ub]


# Putting data in vector assembler form
assembler_train_ku_ub = VectorAssembler(
        inputCols=newlist_train_ku_ub,
        outputCol="features")

transformed_train_ku_ub = assembler_train_ku_ub.transform(merged_train_ku_ub)

# Creating input dataset in the form of labeled point for training the model
data_train_ku_ub = (transformed_train_ku_ub.select("features", "stars")).map(
        lambda row: LabeledPoint(row.stars, row.features))

# Training the model using Random forest regressor
model_train_ku_ub = RandomForest.trainRegressor(data_train_ku_ub, categoricalFeaturesInfo={},
                                                numTrees=24, featureSubsetStrategy="auto",
                                                impurity='variance', maxDepth=5, maxBins=32)


# MODEL 3 - Predictions using 'Known user Unknown business model' on Final/Test Segment with known user information

# Creating a list of features to be used for predictions
removelist_final_ku_ub = set(['business_id', 'review_id'])
newlist_final_ku_ub = [v for i, v in enumerate(merged_final_ku_ub.columns) if v not in removelist_final_ku_ub]
# newlist_final_ku_ub

# Putting data in vector assembler form
assembler_final_ku_ub = VectorAssembler(
        inputCols=newlist_final_ku_ub,
        outputCol="features")

transformed_final_ku_ub = assembler_final_ku_ub.transform(merged_final_ku_ub)

# Creating input dataset to be used for predictions
data_final_ku_ub = transformed_final_ku_ub.select("features", "review_id")

# Predicting ratings using the developed model
predictions_ku_ub = model_train_ku_ub.predict(data_final_ku_ub.map(lambda x: x.features))
labelsAndPredictions_ku_ub = data_final_ku_ub.map(lambda data_final_ku_ub: data_final_ku_ub.review_id).zip(
        predictions_ku_ub)



# MODEL 4 - Baseline model for reviews with unknow user and business information

# mu is the average rating for all ratings included in the training set
mu = train_r.groupBy().avg('stars').collect()[0][0]
# 3.76

# u is the vector containing average ratings for each user and subtracting mu.
u = train_u.map(lambda x: (x.user_id, x.average_stars - mu))
u = sqlContext.createDataFrame(u, ['user_id', 'u_stars'])

# b is the vector containing average ratings for each business and subtracting mu.
b = train_b.map(lambda x: (x.business_id, x.stars - mu))
b = sqlContext.createDataFrame(b, ['business_id', 'b_stars'])

# Creating a data frame with baseline ratings for the test data set
merged_test = final_r.join(u, on='user_id', how='left').join(b, on='business_id', how='left')
merged_test = merged_test.fillna(0).drop('type')
prediction = merged_test.map(lambda x: (x.review_id, mu + x.b_stars + x.u_stars))
preds = sqlContext.createDataFrame(prediction, ['review_id', 'stars'])
preds = preds.toPandas()
preds.stars = preds.stars.map(lambda x: 0 if x < 0 else x)
preds.stars = preds.stars.map(lambda x: 5 if x > 5 else x)
# preds.to_csv('submission_1702_b.csv', index=None)


#############
# Iteration 1
#############
# Using the single model for prediction - Num of trees = 3, Max Depth of trees =3

p = sqlContext.createDataFrame(labelsAndPredictions_ku_kb, ['review_id', 'pred'])
p = p.toPandas()

preds_final = preds.merge(p, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_1702_a.csv', index=None)


#############
# Iteration 2
#############
# Using the single model for prediction - Num of trees = 10, Max Depth of trees =8

p = sqlContext.createDataFrame(labelsAndPredictions_ku_kb, ['review_id', 'pred'])
p = p.toPandas()

preds_final = preds.merge(p, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_1702_d.csv', index=None)


#############
# Iteration 3
#############
# Using the single model for prediction - Num of trees = 20, Max Depth of trees =15

p = sqlContext.createDataFrame(labelsAndPredictions_ku_kb, ['review_id', 'pred'])
p = p.toPandas()

preds_final = preds.merge(p, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_1702_e.csv', index=None)


#############
# Iteration 4
#############
# integer ratings of baseline ratings
preds_final = preds.merge(p, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.pred = preds_final.apply(lambda x: int(x.pred) + 1, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.stars = preds_final.stars.map(lambda x: 5 if x > 5 else x)
preds_final.to_csv('submission_1702_c.csv', index=None)


#############
# Iteration 5
#############
# Using segment wise models for prediction (Model 1 and Model 2) - Num of trees = 24, Max Depth of trees = 5
p_combined = sqlContext.createDataFrame(labelsAndPredictions_ku_kb.union(labelsAndPredictions_uu_kb),
                                        ['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_1802_a.csv', index=None)


#############
# Iteration 6
#############
# Using segment wise models for prediction (Models 1, 2 and 3) - Num of trees = 24, Max Depth of trees = 5
p_combined = sqlContext.createDataFrame(
        labelsAndPredictions_ku_kb.union(labelsAndPredictions_uu_kb).union(labelsAndPredictions_ku_ub),
        ['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_1802_b.csv', index=None)
