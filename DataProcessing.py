__author__ = 'sakshi024'
import os
import sys
import pandas as pd
import re
import numpy as np


# Path for spark source folder
os.environ['SPARK_HOME']="/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/spark-1.6.0-bin-hadoop2.6/"

# Append pyspark  to Python Path
sys.path.append("/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/spark-1.6.0-bin-hadoop2.6/python")

try:
    from pyspark import SparkContext

    from pyspark import SparkConf

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)


sc = SparkContext('local')

# sc is an existing SparkContext.

from pyspark.sql import SQLContext
from pyspark import Row
from pyspark.sql.functions import *
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField
sqlContext = SQLContext(sc)

# Set paths of dataset
training_path = "/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/Project/data/yelp_training_set/"
test_path = "/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/Project/data/final_test_set/"

# Import Training Data
business = sqlContext.read.json(training_path + "yelp_training_set_business.json") # 11537
checkin = sqlContext.read.json(training_path + "yelp_training_set_checkin.json") # 8282
review = sqlContext.read.json(training_path + "yelp_training_set_review.json") # 229907
user = sqlContext.read.json(training_path + "yelp_training_set_user.json") #  43873

# Import Test data


test_bus = sqlContext.read.json(test_path + "final_test_set_business.json") #  2797
test_check = sqlContext.read.json(test_path + "final_test_set_checkin.json") # 1796
test_rvw = sqlContext.read.json(test_path + "final_test_set_review.json") # 36404
test_usr = sqlContext.read.json(test_path + "final_test_set_user.json") #   9522



#Clean any bad data, usually by inserting global averages
usr_mean_stars = user.agg({'average_stars': 'mean'}).collect()[0]['avg(average_stars)']
user = user.withColumn('average_stars', when(user.average_stars<1, usr_mean_stars).otherwise(user.average_stars))

usr_mean_review = user.agg({'review_count': 'mean'}).collect()[0]['avg(review_count)']
user = user.withColumn('review_count', when(user.review_count<1, usr_mean_review).otherwise(user.review_count))

bus_mean_stars = business.agg({'stars': 'mean'}).collect()[0]['avg(stars)']
business = business.withColumn('stars', when(business.stars<1, bus_mean_stars).otherwise(business.stars))

bus_mean_review = business.agg({'review_count': 'mean'}).collect()[0]['avg(review_count)']
business = business.withColumn('review_count', when(business.review_count<1, bus_mean_review).otherwise(business.review_count))

# extract cool/funny/useful votes for users in training data
user.registerTempTable("user")
user = sqlContext.sql("SELECT *, votes['cool'] as cool, votes['funny'] as funny, votes['useful'] as useful  from user")

# extract cool/funny/useful votes for reviews in training data
review.registerTempTable("review")
review = sqlContext.sql("SELECT *, votes['cool'] as cool, votes['funny'] as funny, votes['useful'] as useful  from review")





# Rename all of the training data sets


for col in business.columns:
    if col not in ('business_id', 'user_id'):
        business = business.withColumnRenamed(col , 'b_' + col)

for col in review.columns:
    if col not in ('business_id', 'user_id'):
        review = review.withColumnRenamed(col , 'r_' + col)

for col in user.columns:
    if col not in ('business_id', 'user_id'):
        user = user.withColumnRenamed(col , 'u_' + col)


# Rename all of the test data sets

for col in test_bus.columns:
    if col not in ('business_id', 'user_id'):
        test_bus = test_bus.withColumnRenamed(col , 'b_' + col)

for col in test_rvw.columns:
    if col not in ('business_id', 'user_id'):
        test_rvw = test_rvw.withColumnRenamed(col , 'r_' + col)

for col in test_usr.columns:
    if col not in ('business_id', 'user_id'):
        test_usr = test_usr.withColumnRenamed(col , 'u_' + col)


# Extracting Dummy columns for business categories for both train and test data
bus = business.toPandas()
chk = checkin.toPandas()

tbus = test_bus.toPandas()
tchk = test_check.toPandas()

# Extracting all unique categories

categories = list(set(business.flatMap(lambda x: x['b_categories']).collect()).intersection(set(test_bus.flatMap(lambda x: x['b_categories']).collect())))


def Create_DummiesCategories(pd_df, categories):

    for cat in categories:
        pd_df[cat] = np.asarray([1 if cat in row else 0 for row in pd_df['b_categories']])
        cat2 = re.sub(u'[^A-Za-z0-9]', u'', cat)
        pd_df.rename(columns={cat: 'b_categories_'+ cat2.lower()}, inplace=True)

    return pd_df

bus = Create_DummiesCategories(bus, categories)
tbus = Create_DummiesCategories(tbus, categories)


# sum the total number of check-ins
chk['b_sum_checkins'] = map(lambda x: np.sum(filter(None, chk.iloc[x,1])) , range(checkin.count()))
tchk['b_sum_checkins'] = map(lambda x: np.sum(filter(None, tchk.iloc[x,1])) , range(test_check.count()))



# Creating dataframe from Pandas

bus = bus.drop(['b_type', 'b_neighborhoods', 'b_state', 'b_categories', 'b_name'], axis =1)
tbus = tbus.drop(['b_type', 'b_neighborhoods', 'b_state', 'b_categories', 'b_name'], axis =1)
bus = sqlContext.createDataFrame(bus)
tbus = sqlContext.createDataFrame(tbus)
chk = sqlContext.createDataFrame(chk)
tchk = sqlContext.createDataFrame(tchk)




# Dropping unrequired columns from a training and test data

usr = user.drop('u_type').drop('u_votes').drop('u_name')
rev = review.drop('r_type').drop('r_votes')
chk = chk.drop('checkin_info').drop('type')

tusr = test_usr.drop('u_type').drop('u_name')
trev = test_rvw.drop('r_type')
tchk = tchk.drop('checkin_info').drop('type')


# merge all of the training datasets together on user and business ids
fulltr = rev.join(usr, on='user_id', how='left').join(bus, on= 'business_id', how= 'left').join(chk, on = 'business_id', how='left')

# merge all of the test datasets together on user and business ids
fulltst = trev.join(tusr, on='user_id', how='left').join(tbus, on= 'business_id', how= 'left').join(tchk, on = 'business_id', how='left')



################ Exporting Data to csvs ##########################################################################

print "Exporting to CSV"

fulltr = fulltr.toPandas()
fulltst = fulltst.toPandas()

# export to csv
fulltr.to_csv('yelp_training.csv', index=False, encoding='utf-8')
fulltst.to_csv('yelp_test.csv', index=False, encoding='utf-8')

# get just the reviews text from training  and test data
text_train = fulltr.loc[:, ['user_id', 'business_id', 'r_text']]
text_train.to_csv('yelp_review_text.csv', index=False, encoding='utf-8')

print "Finished"
