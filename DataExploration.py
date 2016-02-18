__author__ = 'sakshi024'

#####################################################################
#### Code for Data Exploration
#####################################################################

import os
import sys
import pandas as pd
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

from pyspark import SparkContext

sc = SparkContext('local')

# sc is an existing SparkContext.

from pyspark.sql import SQLContext
from pyspark.sql.functions import *
sqlContext = SQLContext(sc)



# Set paths of dataset
training_path = "/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/Project/data/yelp_training_set/"
test_path = "/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/Project/data/yelp_test_set/"
final_test = "/Users/sakshi024/Desktop/University Docs/Advanced Machine Learning/Project/data/final_test_set/"

# Training Data import
train_bus = sqlContext.read.json(training_path + "yelp_training_set_business.json") # 11537
train_check = sqlContext.read.json(training_path + "yelp_training_set_checkin.json") # 8282
train_rvw = sqlContext.read.json(training_path + "yelp_training_set_review.json") # 229907
train_usr = sqlContext.read.json(training_path + "yelp_training_set_user.json") #  43873


# Data for cold start

cs_bus = sqlContext.read.json(test_path+ "yelp_test_set_business.json") # 1205
cs_check = sqlContext.read.json(test_path + "yelp_test_set_checkin.json") # 734
cs_rvw = sqlContext.read.json(test_path + "yelp_test_set_review.json") # 22956
cs_usr = sqlContext.read.json(test_path + "yelp_test_set_user.json") #  5105

# Final test data

test_bus = sqlContext.read.json(final_test + "final_test_set_business.json") #  2797
test_check = sqlContext.read.json(final_test + "final_test_set_checkin.json") # 1796
test_rvw = sqlContext.read.json(final_test + "final_test_set_review.json") # 36404
test_usr = sqlContext.read.json(final_test + "final_test_set_user.json") #   9522

# Rating Set: {1, 2, 3, 4, 5}
# Training Set Records: 229,907
# Testing Set Records: 22,956
# Final Test Records: 36,404
# Training Set Average Rating: 3.7667230662833231
# Training Set User Count (In review file): 45,981
# Testing Set User Count (In review file): 11,926
# Training Set Item Count (In review file): 11,537
# Testing Set Item Count (In review file): 5,585
# Testing Set Known User Count: 6,611 # 6216
# Testing Set Known Item Count: 4,380
# Testing Set Cold Start User Count: 5,315 # 5710
# Testing Set Cold Start Item Count: 1,205

# Testing Set (known user, known item) Pair Count: 6,168 
# Testing Set (known user, new item) Pair Count: 7,679 
# Testing Set (new user, known item) Pair Count: 6,086
# Testing Set (new user, new item) Pair Count: 3,023


# Final Test User Count: 15001
# Final Test Item Count: 8341
# Final Test Known User: 4719
# Final Test Known Item: 5544
# Final Test Cold Start User: 10282
# Final Test Cold Start Item: 2797
# Final Test (known user, known item) Pair Count: 12078 (33.2%)
# Final Test (known user, new item) Pair Count: 4086 (11.2%)
# Final Test (new user, known item) Pair Count: 14951 (41.1%)
# Final Test (new user, new item) Pair Count: 5289 (14.5%)

train_usr.registerTempTable("truser")
train_bus.registerTempTable("trbus")
train_check.registerTempTable("trchk")
train_rvw.registerTempTable("trrvw")

cs_bus.registerTempTable("csbus")
cs_usr.registerTempTable("csusr")
cs_check.registerTempTable("cschk")
cs_rvw.registerTempTable("csrvw")

test_bus.registerTempTable("tbus")
test_usr.registerTempTable("tusr")
test_check.registerTempTable("tchk")
test_rvw.registerTempTable("trvw")


# Known User

known_user = len(set(cs_rvw.select('user_id').collect()).intersection(set(train_usr.select('user_id').collect())))
known_user_final = len(set(test_rvw.select('user_id').collect()).intersection(set(train_usr.select('user_id').collect())))


# Known Item/ Business

known_business = len(set(cs_rvw.select('business_id').collect()).intersection(set(train_bus.select('business_id').collect())))
known_bus_final = len(set(test_rvw.select('business_id').collect()).intersection(set(train_bus.select('business_id').collect())))


# Calculating unique number of users in test review

tst_usr = sqlContext.sql("select distinct user_id from trvw")
tst_usr.count()

# Calculating unique number of business in test review
tst_bus = sqlContext.sql("select distinct business_id from trvw")
tst_bus.count()

# Calculating Testing Set (known user, known item) Pair Count

tst_usr_item = sqlContext.sql("select a.user_id from csrvw a inner join truser b  inner join trbus c on a.user_id = b.user_id and a.business_id = c.business_id")
tst_usr_item.count()

final_usr_item = sqlContext.sql("select a.user_id from trvw a inner join truser b  inner join trbus c on a.user_id = b.user_id and a.business_id = c.business_id")
final_usr_item.count()


# Calculating Testing Set (known user, new item) Pair Count


ptrrv = train_rvw.toPandas()
pcsrv = cs_rvw.toPandas()
ptsrv = test_rvw.toPandas()
ptrusr = train_usr.toPandas()
pcsusr = cs_usr.toPandas()
ptsusr = test_usr.toPandas()
ptrbus = train_bus.toPandas()
pcsbus = cs_bus.toPandas()
ptsbus = test_bus.toPandas()

# merge all of the datasets together on user and business ids
full = pd.merge(pcsrv, ptrusr, on='user_id', how='inner')
print "Merged coldstart reviews & user datasets for known users"
full = pd.merge(full, ptrbus, on='business_id', how='left')
print "Merged in business data"
full['stars'].isnull().sum()

# merge all of the datasets together on user and business ids
full = pd.merge(ptsrv, ptrusr, on='user_id', how='inner')
print "Merged coldstart reviews & user datasets for known users"
full = pd.merge(full, ptrbus, on='business_id', how='left')
print "Merged in business data"
full['stars'].isnull().sum()


# Calculating Testing Set (new user, known item) Pair Count

# merge all of the datasets together on user and business ids
full = pd.merge(pcsrv, ptrbus, on='business_id', how='inner')
print "Merged coldstart reviews & business datasets for known items"
full = pd.merge(full, ptrusr, on='user_id', how='left')
print "Merged in user data"
full['average_stars'].isnull().sum()

# merge all of the datasets together on user and business ids
full = pd.merge(ptsrv, ptrbus, on='business_id', how='inner')
print "Merged coldstart reviews & business datasets for known items"
full = pd.merge(full, ptrusr, on='user_id', how='left')
print "Merged in user data"
full['average_stars'].isnull().sum()

###############################################################################################################
#### Data Exploration for each Dataset
###############################################################################################################

business = train_bus
user = train_usr
review = train_rvw
checkin = train_check



################# Summary of User Table ###########################
user.describe().show()
sqlContext.registerDataFrameAsTable(user, "user")
# no duplicate records
# no duplicate on userid column
# only one column has zero average_stars --- zero average_stars means no rating given by user but the user has given review
# average of average_stars = 3.75
user.printSchema()
t = user.dropDuplicates(['user_id'])
t.count()

df2 = sqlContext.sql("SELECT count(review_count) as count from user where review_count = 0.0 ")

# For test data
test_usr.describe().show()
# min review count is one
test_usr.printSchema()
t = test_usr.dropDuplicates(['user_id'])
t.count()
# no average_stars and votes info

################# Summary of Business Table ###########################
business.describe().show()
# min review_count = 3 and min stars = 1.0 --> every business has rating
# no duplicate on business_id and on record level
# column 'type' in review, user and business is waste has same values 'user' and 'business'
# neighbourhood column is waste no values
#

t = business.dropDuplicates(['business_id'])
t.count()

business.printSchema()
business.select(business['business_id'], business['categories'], business['city'], business['name']).show(10)
business.select(business['business_id'], business['name'], business['neighborhoods'], business['stars'], business['state'], business['type']).show(10)
sqlContext.registerDataFrameAsTable(business, "bus")
df2 = sqlContext.sql("SELECT categories, count(categories) as countcat from bus group by categories order by 2 desc")
df2.collect()
business.select("neighborhoods").show()
business.groupBy("categories").count().show()

# For test data
test_bus.printSchema()
test_bus.describe().show()
t = test_bus.dropDuplicates(['business_id'])
t.count()
test_bus.count()
sqlContext.registerDataFrameAsTable(test_bus, "tbus")
df2 = sqlContext.sql("SELECT count(categories) from tbus where categories is null")
df2.collect()
# no null review count


################# Summary of Review Table ###########################
review.describe().show()
review.printSchema()
review.show(10)
t = review.dropDuplicates(['review_id'])
t.count()
# stars are not null...min - 1
# no duplicates on review_id
sqlContext.registerDataFrameAsTable(review, "review")
df2 = sqlContext.sql("SELECT count(user_id) from review where business_id is null")
df2.collect()

# average stars
df2 = sqlContext.sql("SELECT avg(stars) from review")
df2.collect()

# Number of records with 0 stars = 0
df2 = sqlContext.sql("SELECT count(review_id) from review where stars = 0")
df2.collect()


##################### Summary of checkin Table #################################
checkin.describe().show()
checkin.printSchema()
checkin.show(10)
sqlContext.registerDataFrameAsTable(review, "review")
df2 = sqlContext.sql("SELECT count(user_id) from review where business_id is null")
df2.collect()

# Join business and user table


# 14028 users have reviews but no information in usr table
# all business have information in usr table

rvw_usr = review.join(business , review['business_id'] == business['business_id'], 'inner')
rvw_usr.count()
sqlContext.registerDataFrameAsTable(rvw_usr, "rvw_usr")
rvw_usr_bus = sqlContext.sql("SELECT a.*, b.average_stars, b.review_count as usr_rvw_cnt, b.votes as usr_votes,"
                         "c.categories, c.latitude, c.longitude, "
                         " from review a left JOIN user b left join bus c on a.user_id = b.user_id"
                         " and a.business_id = c.business_id")
rvw_usr_bus.take(1)

rvw_usr_bus.printSchema()


