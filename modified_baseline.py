__author__ = 'chhavi21'

## this files was not used as the results produced were poor.
from __future__ import division
from load_data import *
from build_features import *
import pandas as pd
import numpy as np


user.printSchema()

# mu is the average rating for all ratings included in the training set
mu = review.groupBy().avg('r_stars').collect()[0][0]
# 3.76

# review.printSchema()
# u is the vector containing average ratings for each user and subtracting mu.

def normed_rating(rating, mu):
    if rating < 0.01: return 0
    else: return rating - mu

def get_rating(mu, b_stars, u_stars, review_score_user, review_score_business):
    d = mu
    if b_stars is not np.nan:
        d += b_stars-review_score_business
    if u_stars is not np.nan:
        d += u_stars-review_score_user
    return d



u = user.map(lambda x: (x.user_id, normed_rating(x.u_average_stars, mu)))
u = sqlContext.createDataFrame(u, ['user_id', 'u_stars'])

b = business.map(lambda x: (x.business_id, normed_rating(x.b_stars, mu)))
b = sqlContext.createDataFrame(b, ['business_id', 'b_stars'])

business_zipcode = business_zipcode.map(lambda x: (x[0], normed_rating(x[1], mu)))
business_zipcode = sqlContext.createDataFrame(business_zipcode, ['business_id', 'zip_stars'])

# merged_test = test_rvw.join(u, on='user_id', how='left').join(b, on='business_id', how='left')
# merged_test = merged_test.join(business_zipcode, on='business_id', how='left')

merged_test = merged_test.drop('type')
merged_test = merged_test.fillna(0)

# RMSE1.73
# used sentiment scores of business and users.
# merged_test = merged_test.join(review_score_user, on='user_id', how='left')
# merged_test = merged_test.join(review_score_business, on='business_id', how='left')
# prediction = merged_test.map(lambda x: (x.review_id,
#                                         (mu + x.b_stars + x.u_stars +
#                                                 - x.review_score_user
#                                                 - x.review_score_business)))

# RMSE 1.33561
# prediction = merged_test.map(lambda x: (x.review_id,
#                                         (mu + x.zip_stars + x.u_stars)))
prediction = prediction.map(lambda x: (x[0], float(x[1])))
preds = sqlContext.createDataFrame(prediction, ['review_id', 'stars'])
preds = preds.toPandas()
preds.stars = preds.stars.map(lambda x: 1 if x<1 else x)
preds.stars = preds.stars.map(lambda x: 5 if x>5 else x)

sample_pred = pd.read_csv('/Users/chhavi21/Box Sync/USF/Spring-2016-Module-1/AML/project/code/sampleSubmissionFinal.csv')
preds = sample_pred.drop('stars', axis=1).merge(preds, on=['review_id'], how='left')
preds.to_csv('submission.csv', index=None)


