
__author__ = 'chhavi21'

from __future__ import division
from load_data import *
import pandas as pd

user.printSchema()

# Global Average
#mu is the average rating for all ratings included in the training set
mu = review.groupBy().avg('r_stars').collect()[0][0]
# 3.76

# review.printSchema()
# u is the vector containing average ratings for each user and subtracting mu.

def normed_rating(rating, mu):
	if rating < 0.01: return 0
	else: return rating - mu

# User average - global average
u = user.map(lambda x: (x.user_id, normed_rating(x.u_average_stars, mu)))
u = sqlContext.createDataFrame(u, ['user_id', 'u_stars'])

# Business average - global average
b = business.map(lambda x: (x.business_id, normed_rating(x.b_stars, mu)))
b = sqlContext.createDataFrame(b, ['business_id', 'b_stars'])

# add the ratings to get baseline predictions
merged_test = test_rvw.join(u, on='user_id', how='left').join(b, on='business_id', how='left')
merged_test = merged_test.fillna(0).drop('type')
prediction = merged_test.map(lambda x: (x.review_id, mu + x.b_stars + x.u_stars))
prediction.take(5)
preds = sqlContext.createDataFrame(prediction, ['review_id', 'stars'])

#get data in pandas to write it to csv. spark does not have any inbulit function for this
preds = preds.toPandas()
preds.stars = preds.stars.map(lambda x: 1 if x<1 else x)
preds.stars = preds.stars.map(lambda x: 5 if x>5 else x)

sample_pred = pd.read_csv('/Users/chhavi21/Box Sync/USF/Spring-2016-Module-1/AML/project/code/sampleSubmissionFinal.csv')
preds = sample_pred.drop('stars', axis=1).merge(preds, on=['review_id'], how='left')
preds.to_csv('submission.csv', index=None)

# Baseline
# RMSE =1.29726


