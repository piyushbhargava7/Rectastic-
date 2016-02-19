__author__ = 'chhavi21'

import os
import sys
# Path for spark source folder
os.environ['SPARK_HOME']="/Users/chhavi21/spark-1.6.0-bin-hadoop2.6/"
# Append pyspark  to Python Path
sys.path.append("/Users/chhavi21/spark-1.6.0-bin-hadoop2.6/python")


from pyspark import SparkConf, SparkContext, Row
from pyspark.sql import SQLContext
from pyspark.sql.functions import *
from pyspark.sql.types import *

conf = SparkConf().setAppName("yelp predicitions")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

PATH = './data/'

################################################################################
# LOAD TRAINING DATA
################################################################################

business = sqlContext.read.json(PATH + "yelp_training_set_business.json").repartition(16) # 11537
checkin = sqlContext.read.json(PATH + "yelp_training_set_checkin.json").repartition(16) # 8282
review = sqlContext.read.json(PATH + "yelp_training_set_review.json").repartition(16) # 229907
user = sqlContext.read.json(PATH + "yelp_training_set_user.json").repartition(16) #  43873

# business.first()
# min review_count = 3 and min stars = 1.0 --> every business has rating
# no duplicate on business_id and on record level
# column 'type' in review, user and business is waste has same values 'user' and 'business'
# neighbourhood column is waste no values


#/Users/chhavi21/Box Sync/USF/Spring-2016-Module-1/AML/project/code
# review.first()
# no duplicates on review

# user.first()
# no duplicate records
# no duplicate on userid column
# only one column has zero average_stars --- zero average_stars means no rating given by user but the user has given review
# average of average_stars = 3.75

# checkin.first()

mapping = sc.textFile("./data/IdLookupTable.csv")
mapping = mapping.filter(lambda x: "user_id" not in x)
mapping = mapping.map(lambda l: Row(user_id=l.split(",")[0], business_id=l.split(",")[1], RecommendationId=l.split(",")[2]))
mapping = sqlContext.createDataFrame(mapping)
# mapping.show()


# business.groupBy("categories").count().show()
# user.printSchema()
# user.describe().show()


################################################################################
# LOAD TEST DATA
################################################################################

test_bus = sqlContext.read.json(PATH + "final_test_set_business.json").repartition(16) # 2797
test_check = sqlContext.read.json(PATH + "final_test_set_checkin.json").repartition(16) # 1796
test_rvw = sqlContext.read.json(PATH + "final_test_set_review.json").repartition(16) # #36404
test_usr = sqlContext.read.json(PATH + "final_test_set_user.json").repartition(16) #  9522

# test_usr.count()
# test_bus.printSchema()
# test_check.printSchema()
# test_rvw.printSchema()
# test_usr.printSchema()

# known businesses
test_rvw.select(['business_id']).rdd.intersection(business.select(['business_id']).rdd).count() #5544

test_rvw.select(['user_id']).rdd.intersection(user.select(['user_id']).rdd).count() #4719

# no common user
# test_usr.select(['user_id']).rdd.intersection(user.select(['user_id']).rdd).collect()

# review.select(['votes']).map(lambda x: x.cool).first()




################################################################################
# DATA CLEANING
################################################################################

# CAUTION: # avoid using the following code as it increses the number of rows

# ADD COLUMNS TO DATA THAT ARE NESTED INSIDE
# votes_RDD = review.map(lambda x: (x.business_id, x.votes.cool,
#                                   x.votes.funny, x.votes.useful))
# schema = StructType([StructField("business_id", StringType(), True),
#                      StructField("b_cool", IntegerType(), True),
#                      StructField("b_funny", IntegerType(), True),
#                      StructField("b_useful", IntegerType(), True)])
# votes_df = sqlContext.createDataFrame(votes_RDD, schema)
# review = review.join(votes_df, on='business_id', how='left').count()
# review = review.drop('votes')
#
# votes_df.count()
#
# votes_RDD = user.map(lambda x: (x.user_id, x.votes.cool,
#                                   x.votes.funny, x.votes.useful))
# schema = StructType([StructField("user_id", StringType(), True),
#                      StructField("u_cool", IntegerType(), True),
#                      StructField("u_funny", IntegerType(), True),
#                      StructField("u_useful", IntegerType(), True)])
# votes_df = sqlContext.createDataFrame(votes_RDD, schema)
# user = user.join(votes_df, on='user_id', how='left')
# user = user.drop('votes')
#

#Clean any bad data, by inserting global averages
usr_mean_stars = user.agg({'average_stars': 'mean'}).collect()[0]['avg(average_stars)']
user = user.withColumn('average_stars', when(user.average_stars<1, usr_mean_stars).otherwise(user.average_stars))

usr_mean_review = user.agg({'review_count': 'mean'}).collect()[0]['avg(review_count)']
user = user.withColumn('review_count', when(user.review_count<1, usr_mean_review).otherwise(user.review_count))


# drop unneccessary columns
user = user.drop('type')
review = review.drop('date').drop('type')
business = business.drop('neighborhoods').drop('type')

# rename columns
business = business.withColumnRenamed('stars', 'b_stars')
business = business.withColumnRenamed('review_count', 'b_review_count')
business = business.withColumnRenamed('name', 'b_name')
review = review.withColumnRenamed('stars', 'r_stars')
user = user.withColumnRenamed('name', 'u_name')
user = user.withColumnRenamed('review_count', 'u_review_count')
user = user.withColumnRenamed('average_stars', 'u_average_stars')

