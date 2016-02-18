from load_data import *
from baseline_model import *
import numpy as np
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

################################################################################
## CREATE MAPPING
################################################################################

business_id_mapping = dict()
business_id = review.select(["business_id"]).map(lambda x: x.business_id).collect()
business_id = list(set(business_id))
for i in range(len(business_id)):
    business_id_mapping[business_id[i]] = i


user_id_mapping = dict()
user_id = review.select(["user_id"]).map(lambda x: x.user_id).collect()
user_id = list(set(user_id))
for i in range(len(user_id)):
    user_id_mapping[user_id[i]] = i

################################################################################
## CREATE INVERSE MAPPING
################################################################################
inv_business_id_mapping = dict()
for i in business_id_mapping:
    inv_business_id_mapping[business_id_mapping[i]] = i

inv_user_id_mapping = dict()
for i in user_id_mapping:
    inv_user_id_mapping[user_id_mapping[i]] = i

################################################################################
## CREATE TRAIN DATA
################################################################################

als_data = review.select(['user_id', "business_id", 'r_stars']).\
    map(lambda x: Rating(user_id_mapping[x[0]], business_id_mapping[x[1]], x[2]-mu))
als_data.first()

################################################################################
## BUILD MODEL
################################################################################

rank = 100
numIterations = 25
model = ALS.train(als_data, rank, numIterations, lambda_ = 0.3, seed=10)

################################################################################
## PREDICT
################################################################################

def clip(x):
    if x<1: return 1.0
    elif x>5: return 5.0
    return x

test_data = review.select(['user_id', "business_id"]).\
            map(lambda x: (user_id_mapping[x[0]], business_id_mapping[x[1]]))
predictions = model.predictAll(test_data).map(lambda r: ((r[0], r[1]), clip(r[2]+mu)))
predictions = predictions.map(lambda x: (x[0], clip(x[1])))
predictions.collect()

# getting the training data in the right format for comparison
train1 = review.select(['user_id', "business_id", 'r_stars']).\
    map(lambda x: Row(user_id_mapping = user_id_mapping[x[0]],
                      business_id_mapping = business_id_mapping[x[1]],
                      rating = x[2]))
train1 = sqlContext.createDataFrame(train1)

# getting the predictions in the right format for comparison
train2 = predictions.map(lambda x: Row(user_id_mapping = x[0][0],
                                      business_id_mapping = x[0][1],
                                      pred = x[1]))
train2 = sqlContext.createDataFrame(train2)

################################################################################
## COMPUTE RMSE ON TRAINING DATA
################################################################################

joined = train1.join(train2, on=["user_id_mapping", "business_id_mapping"])
se = joined.map(lambda x: (x.rating - x.pred)**2).reduce(lambda a,b: a+b)
n = joined.count()
rmse = np.sqrt(se/n)
# 0.64672825592371597

################################################################################
## GET COMMON TEST DATA
################################################################################

# try to improve this code by not taking stuff out of rdd
known_business = test_rvw.select(['business_id']).rdd.intersection(business.select(['business_id']).rdd)
known_business = known_business.map(lambda x: x.business_id).collect()
known_business = set(known_business)

known_user = test_rvw.select(['user_id']).rdd.intersection(user.select(['user_id']).rdd)
known_user = known_user.map(lambda x: x.user_id).collect()
known_user = set(known_user)

#12078
known_user_and_know_business = test_rvw.drop('type').\
                        map(lambda x: (x.user_id, x.business_id, x.review_id)).\
                        filter(lambda x: (x[0] in known_user)
                                         and (x[1] in known_business))

#user_id, business_if, review_id
known_user_and_know_business = known_user_and_know_business.\
                                            map(lambda x: (user_id_mapping[x[0]],
                                                business_id_mapping[x[1]], x[2]))

test_pred = model.predictAll(known_user_and_know_business.
                            map(lambda x: (x[0], x[1]))).\
                            map(lambda r: ((r[0], r[1]), clip(r[2]+mu)))

test_pred = test_pred.map(lambda x: Row(user_id_mapping = x[0][0],
                                      business_id_mapping = x[0][1],
                                      pred = x[1]))
test_pred = sqlContext.createDataFrame(test_pred)


schema = StructType([StructField("user_id_mapping", StringType(), True),
                     StructField("business_id_mapping", StringType(), True),
                     StructField("review_id", StringType(), True)])
known_user_and_know_business = sqlContext.createDataFrame(known_user_and_know_business, schema)

test_pred = test_pred.join(known_user_and_know_business, on=['user_id_mapping',
                                                 'business_id_mapping'])\
                     .drop('business_id_mapping')\
                     .drop('user_id_mapping')

p = test_pred.toPandas()


preds_final = preds.merge(p, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission.csv', index=None)


# k=10 RMSE 1.38587
# k=8 RMSE 1.36976
# k=20 RMSE 1.33906
# k=25 RMSE 1.33433
# k=70 RMSE 1.30318
# k=100 RMSE 1.29951

#k=100 and lambda=0.5 RMSE = 1.29316
#k=100 and lambda=0.2 RMSE = 1.29096
#k=100 and lambda=0.3 RMSE =1.29070
#k=100 and lambda=0.1 RMSE =1.29322

#lambda is not helping at all
# number of iterations is limited by 25 beacuse of spark version. Need to try on AWS for final predicition
