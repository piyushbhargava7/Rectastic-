__author__ = 'piyush'

# Importing Libraries
import os
import sys

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
from pyspark.sql import SQLContext
from pyspark.sql.functions import *

sqlContext = SQLContext(sc)

path = '/Users/piyushbhargava/Desktop/MSAN_630/project/yelp'
################################################################################
# LOADING DATA
################################################################################
# b - business dataset

# Reading the training file as SQL dataframe
train_b = sqlContext.read.json(
        path + "/yelp_training_set/yelp_training_set_business.json")

# Reading the final test file as SQL dataframe
final_b = sqlContext.read.json(
        path + "/final_test_set/final_test_set_business.json")


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



