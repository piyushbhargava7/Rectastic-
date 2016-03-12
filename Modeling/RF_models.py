################################################################################
# APPLYING RANDOM FORESTALGORITHM
################################################################################

from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import RandomForest, RandomForestModel
from pyspark.ml.feature import VectorAssembler
import matplotlib.pyplot as plt

################################################################################
#FUNCTIONS TO PERFORM VALIDATION TO TUNE a) Num of Trees c) Max depth of trees
################################################################################

# Function to create Train and Test segments
def test_train_data(overall_segment):
    removelist_train= set(['stars', 'business_id', 'bus_id', 'b_id','review_id', 'user_id'])
    newlist_train = [v for i, v in enumerate(overall_segment.columns) if v not in removelist_train]

    # Putting data in vector assembler form
    assembler_train = VectorAssembler(inputCols=newlist_train, outputCol="features")

    transformed_train = assembler_train.transform(overall_segment.fillna(0))

    # Creating input dataset in the form of labeled point for training the model
    data_train= (transformed_train.select("features", "stars")).map(lambda row: LabeledPoint(row.stars, row.features))

    (trainingData, testData) = sc.parallelize(data_train.collect(),5).randomSplit([0.7, 0.3])
    return (trainingData, testData)


# Function to divide data into 3 random segments for performing 3-fold validation
def test_train_data_3fold(overall_segment):
    removelist_train= set(['stars', 'business_id', 'bus_id', 'b_id','review_id', 'user_id'])
    newlist_train = [v for i, v in enumerate(overall_segment.columns) if v not in removelist_train]

    # Putting data in vector assembler form
    assembler_train = VectorAssembler(inputCols=newlist_train, outputCol="features")

    transformed_train = assembler_train.transform(overall_segment.fillna(0))

    # Creating input dataset in the form of labeled point for training the model
    data_train= (transformed_train.select("features", "stars")).map(lambda row: LabeledPoint(row.stars, row.features))

    (Data_1, Data_2, Data_3) = sc.parallelize(data_train.collect(),5).randomSplit([0.33, 0.33, 0.34])

    return (Data_1, Data_2, Data_3)


# Function to Perform One fold cross validation
def validation_rf(trainingData,testData, num_trees, depth):
    # Training the model using Random forest regressor
    model_train = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                                    numTrees=num_trees, featureSubsetStrategy="auto",
                                                    impurity='variance', maxDepth=depth, maxBins=32)


    # Evaluate model on test instances and compute test error
    predictions = model_train.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(testData.count())
    return testMSE


# Function to Perform 3-fold validation
def cross_validation_rf(Data_1,Data_2,Data_3,num_trees, depth):
    # Training the model using Random forest regressor
    model_train_1 = RandomForest.trainRegressor(Data_1.union(Data_2), categoricalFeaturesInfo={},
                                                    numTrees=num_trees, featureSubsetStrategy="auto",
                                                    impurity='variance', maxDepth=depth, maxBins=32)


    # Evaluate model on test instances and compute test error
    predictions_1 = model_train_1.predict(Data_3.map(lambda x: x.features))
    labelsAndPredictions_1 = Data_3.map(lambda lp: lp.label).zip(predictions_1)
    testMSE_1 = labelsAndPredictions_1.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(Data_3.count())

    # Training the model using Random forest regressor
    model_train_2 = RandomForest.trainRegressor(Data_2.union(Data_3), categoricalFeaturesInfo={},
                                                    numTrees=num_trees, featureSubsetStrategy="auto",
                                                    impurity='variance', maxDepth=depth, maxBins=32)

    # Evaluate model on test instances and compute test error
    predictions_2 = model_train_2.predict(Data_1.map(lambda x: x.features))
    labelsAndPredictions_2 = Data_1.map(lambda lp: lp.label).zip(predictions_2)
    testMSE_2 = labelsAndPredictions_2.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(Data_1.count())

    # Training the model using Random forest regressor
    model_train_3 = RandomForest.trainRegressor(Data_3.union(Data_1), categoricalFeaturesInfo={},
                                                    numTrees=num_trees, featureSubsetStrategy="auto",
                                                    impurity='variance', maxDepth=depth, maxBins=32)

    # Evaluate model on test instances and compute test error
    predictions_3 = model_train_3.predict(Data_2.map(lambda x: x.features))
    labelsAndPredictions_3 = Data_2.map(lambda lp: lp.label).zip(predictions_3)
    testMSE_3 = labelsAndPredictions_3.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(Data_2.count())

    return (testMSE_1+testMSE_2+testMSE_3)/3


################################################################################
# Function to build final model on complete segment with the selected parameters
################################################################################

def seg_model_rf(train_data, test_data, num_trees, depth):
    removelist_train= set(['stars', 'business_id', 'bus_id', 'b_id','review_id', 'user_id'])
    newlist_train = [v for i, v in enumerate(train_data.columns) if v not in removelist_train]

    # Putting data in vector assembler form
    assembler_train = VectorAssembler(inputCols=newlist_train, outputCol="features")

    transformed_train = assembler_train.transform(train_data.fillna(0))

    # Creating input dataset in the form of labeled point for training the model
    data_train= (transformed_train.select("features", "stars")).map(lambda row: LabeledPoint(row.stars, row.features))

    # Training the model using Random forest regressor
    model_train = RandomForest.trainRegressor(sc.parallelize(data_train.collect(),5), categoricalFeaturesInfo={},
                                                    numTrees=num_trees, featureSubsetStrategy="auto",
                                                    impurity='variance', maxDepth=depth, maxBins=32)


    # MODEL - Predictions using 'UnKnown user known business model' on Final/Test Segment with known business information

    # Creating a list of features to be used for predictions
    removelist_final = set(['business_id', 'bus_id', 'b_id','review_id', 'user_id'])
    newlist_final = [v for i, v in enumerate(test_data.columns) if v not in removelist_final]

    # Putting data in vector assembler form
    assembler_final = VectorAssembler(inputCols=newlist_final,outputCol="features")

    transformed_final= assembler_final.transform(test_data.fillna(0))

    # Creating input dataset to be used for predictions
    data_final = transformed_final.select("features", "review_id")

    # Predicting ratings using the developed model
    predictions = model_train.predict(data_final.map(lambda x: x.features))
    labelsAndPredictions = data_final.map(lambda data_final: data_final.review_id).zip(predictions)
    return labelsAndPredictions

# Building models for 5 different segments
########################################################################################################
# MODEL 1 - Known user known business model - Training the model using user and business information
# (including business stars and user stars)
########################################################################################################
########################################################################################################
# MODEL 2 - Known user known business model - Training the model using user and business information
# (but including business stars only)
########################################################################################################
########################################################################################################
# MODEL 3 - Known user known business model - Training the model using user and business information
# (but including users stars only)
########################################################################################################
########################################################################################################
# MODEL 4 - Known user known business model - No stars- Training the model using user and business
# information but no stars
########################################################################################################
########################################################################################################
# MODEL 5 - Known business model - No stars - Training the model using business information only and no stars
########################################################################################################



########################################################################################################
#  FEW ITERATIONS
########################################################################################################
start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 20, 6)
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 20, 6)
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 20, 6)
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 20, 6)
model_5_rf  = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 20, 6)
stop= time.time()

# 1110

p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

p_combined.columns = ['review_id', 'stars']
p_combined.to_csv('submission_2702_b.csv', index=None)


start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 20, 4)
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 20, 4)
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 20, 4)
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 20, 4)
model_5_rf  = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 20, 4)
stop= time.time()
# 805.0176439285278


# 1.27
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_2702_e.csv', index=None)


# Baseline model for reviews with unknow user and business information

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


# submission
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2702_g.csv', index=None)


# submission
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2702_i.csv', index=None)


start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 10, 4)
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 10, 4)
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 10, 4)
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 10, 4)
model_5_rf  = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 10, 4)
stop= time.time()

# 702.272332906723

p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_2702_e.csv', index=None)
# 1.27


start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 10, 5)
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 10, 5)
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 10, 5)
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 10, 5)
model_5_rf  = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 10, 5)
stop= time.time()

p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_2702_ee.csv', index=None)


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2702_eee.csv', index=None)



# with votes
start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 10, 5)
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 10, 5)
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 10, 5)
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 10, 5)
model_5_rf  = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 10, 5)
stop= time.time()


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_2801_a.csv', index=None)


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2802_aa.csv', index=None)


start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 10, 4)
print "1 done"
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 10, 4)
print "2 done"
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 10, 4)
print "3 done"
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 10, 4)
print "4 done"
model_5_rf  = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 10, 4)
stop= time.time()


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_2801_c.csv', index=None)



p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2802_d.csv', index=None)


start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 20, 4)
print "1 done"
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 20, 4)
print "2 done"
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 20, 4)
print "3 done"
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 20, 4)
print "4 done"
model_5_rf = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 20, 4)
stop= time.time()


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_2801_e.csv', index=None)


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2802_f.csv', index=None)


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2802_g.csv', index=None)


########################################################################################################
# PERFORMING VALIDATION TO SELECT BEST PARAMETERS FOR EACH OF THE 5 SEGMENTS
########################################################################################################
# Choosing the best num_trees and depth for each segment
# 1
start= time.time()
(trainingData_1, testData_1) = test_train_data(merged_train_ku_kb_all_stars)
trainingData_1.persist()
testData_1.persist()
printMSE =[]
num_trees = [10, 20, 25, 30, 35, 40]
depth = [2, 4, 6, 8, 10, 12]
for i in num_trees:
    for j in depth:
        testMSE = validation_rf(trainingData_1, testData_1, i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j
trainingData_1.unpersist()
testData_1.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start
#
# ['For num_trees = 10 and depth = 2,Test Mean Squared Error is 1.163760',
#  'For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.025663',
#  'For num_trees = 10 and depth = 8,Test Mean Squared Error is 0.938606',
#  'For num_trees = 10 and depth = 10,Test Mean Squared Error is 0.932328',
#  'For num_trees = 10 and depth = 12,Test Mean Squared Error is 0.932471',
#  'For num_trees = 20 and depth = 2,Test Mean Squared Error is 1.165537',
#  'For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.031275',
#  'For num_trees = 20 and depth = 8,Test Mean Squared Error is 0.937368',
#  'For num_trees = 20 and depth = 10,Test Mean Squared Error is 0.929129']
# 'For num_trees = 25 and depth = 2,Test Mean Squared Error is 1.158876',
#  'For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.046854',
#  'For num_trees = 25 and depth = 8,Test Mean Squared Error is 0.944675',
#  'For num_trees = 25 and depth = 10,Test Mean Squared Error is 0.941260'
# 'For num_trees = 30 and depth = 2,Test Mean Squared Error is 1.205827',
#  'For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.037277',
#  'For num_trees = 30 and depth = 8,Test Mean Squared Error is 0.947304',
#  'For num_trees = 30 and depth = 10,Test Mean Squared Error is 0.936009',
#  'For num_trees = 35 and depth = 2,Test Mean Squared Error is 1.193255',
#  'For num_trees = 35 and depth = 4,Test Mean Squared Error is 1.041924',
#  'For num_trees = 35 and depth = 8,Test Mean Squared Error is 0.948508']

# 2
start= time.time()
(trainingData_2, testData_2) = test_train_data(merged_train_ku_kb_b_stars)
trainingData_2.persist()
testData_2.persist()
printMSE =[]
num_trees = [10, 20, 25, 30, 35, 40]
depth = [2, 4, 6, 8, 10, 12]
for i in num_trees:
    for j in depth:
        testMSE = validation_rf(trainingData_2, testData_2, i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
trainingData_2.unpersist()
testData_2.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start

# ['For num_trees = 10 and depth = 2,Test Mean Squared Error is 1.285801',
#  'For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.236305',
#  'For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.174789',
#  'For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.177489',
#  'For num_trees = 10 and depth = 12,Test Mean Squared Error is 1.183602',
#  'For num_trees = 20 and depth = 2,Test Mean Squared Error is 1.301577',
#  'For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.204385',
#  'For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.174458',
#  'For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.177259',
#  'For num_trees = 20 and depth = 12,Test Mean Squared Error is 1.184283',
#  'For num_trees = 25 and depth = 2,Test Mean Squared Error is 1.289427',
#  'For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.212468',
#  'For num_trees = 25 and depth = 8,Test Mean Squared Error is 1.172535',
#  'For num_trees = 25 and depth = 10,Test Mean Squared Error is 1.175969',
#  'For num_trees = 25 and depth = 12,Test Mean Squared Error is 1.184226',
#  'For num_trees = 30 and depth = 2,Test Mean Squared Error is 1.276774',
#  'For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.215837',
#  'For num_trees = 30 and depth = 8,Test Mean Squared Error is 1.175235',
#  'For num_trees = 30 and depth = 10,Test Mean Squared Error is 1.175731',
#  'For num_trees = 30 and depth = 12,Test Mean Squared Error is 1.182200',
#  'For num_trees = 35 and depth = 2,Test Mean Squared Error is 1.286011',
#  'For num_trees = 35 and depth = 4,Test Mean Squared Error is 1.207638',
#  'For num_trees = 35 and depth = 8,Test Mean Squared Error is 1.176146',
#  'For num_trees = 35 and depth = 10,Test Mean Squared Error is 1.176028']


# 3
start= time.time()
(trainingData_3, testData_3) = test_train_data(merged_train_ku_kb_u_stars)
trainingData_3.persist()
testData_3.persist()
printMSE =[]
num_trees = [10, 20, 25, 30, 35, 40]
depth = [2, 4, 6, 8, 10, 12]
for i in num_trees:
    for j in depth:
        testMSE = validation_rf(trainingData_3, testData_3, i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j
trainingData_3.unpersist()
testData_3.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start
# For num_trees = 10 and depth = 2,Test Mean Squared Error is 1.411094
# For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.228510
# For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.186330
# For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.177401
# For num_trees = 10 and depth = 12,Test Mean Squared Error is 1.182353
# For num_trees = 20 and depth = 2,Test Mean Squared Error is 1.287318
# For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.241353
# For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.187085
# For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.172512
# For num_trees = 20 and depth = 12,Test Mean Squared Error is 1.177468
# For num_trees = 25 and depth = 2,Test Mean Squared Error is 1.348741
# For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.237880
# For num_trees = 25 and depth = 8,Test Mean Squared Error is 1.184157
# For num_trees = 25 and depth = 10,Test Mean Squared Error is 1.170389
# For num_trees = 25 and depth = 12,Test Mean Squared Error is 1.175948
# For num_trees = 30 and depth = 2,Test Mean Squared Error is 1.310122
# For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.228972
# For num_trees = 30 and depth = 8,Test Mean Squared Error is 1.172382
# For num_trees = 30 and depth = 10,Test Mean Squared Error is 1.173742
# For num_trees = 30 and depth = 12,Test Mean Squared Error is 1.182180
# For num_trees = 35 and depth = 2,Test Mean Squared Error is 1.305088
# For num_trees = 35 and depth = 4,Test Mean Squared Error is 1.221234
# For num_trees = 35 and depth = 8,Test Mean Squared Error is 1.173932
# For num_trees = 35 and depth = 10,Test Mean Squared Error is 1.174905
# For num_trees = 35 and depth = 12,Test Mean Squared Error is 1.179516

# For num_trees = 10 and depth = 2,Test Mean Squared Error is 1.341000
# For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.216139
# For num_trees = 10 and depth = 6,Test Mean Squared Error is 1.196000
# For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.178757
# For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.176049
# For num_trees = 10 and depth = 12,Test Mean Squared Error is 1.181646
# For num_trees = 20 and depth = 2,Test Mean Squared Error is 1.331145
# For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.218233
# For num_trees = 20 and depth = 6,Test Mean Squared Error is 1.207220
# For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.177333
# For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.184662
# For num_trees = 20 and depth = 12,Test Mean Squared Error is 1.185240
# For num_trees = 25 and depth = 2,Test Mean Squared Error is 1.360392
# For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.244506
# For num_trees = 25 and depth = 6,Test Mean Squared Error is 1.213846
# For num_trees = 25 and depth = 8,Test Mean Squared Error is 1.178546
# For num_trees = 25 and depth = 10,Test Mean Squared Error is 1.176488
# For num_trees = 25 and depth = 12,Test Mean Squared Error is 1.181746
# For num_trees = 30 and depth = 2,Test Mean Squared Error is 1.331727
# For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.239921
# For num_trees = 30 and depth = 6,Test Mean Squared Error is 1.190447
# For num_trees = 30 and depth = 8,Test Mean Squared Error is 1.179991
# For num_trees = 30 and depth = 10,Test Mean Squared Error is 1.175872
# For num_trees = 30 and depth = 12,Test Mean Squared Error is 1.187010
# For num_trees = 35 and depth = 2,Test Mean Squared Error is 1.293755
# For num_trees = 35 and depth = 4,Test Mean Squared Error is 1.228460
# For num_trees = 35 and depth = 6,Test Mean Squared Error is 1.194631
# For num_trees = 35 and depth = 8,Test Mean Squared Error is 1.175893
# For num_trees = 35 and depth = 10,Test Mean Squared Error is 1.174044
# For num_trees = 35 and depth = 12,Test Mean Squared Error is 1.181204
# For num_trees = 40 and depth = 2,Test Mean Squared Error is 1.336627
# For num_trees = 40 and depth = 4,Test Mean Squared Error is 1.248273
# For num_trees = 40 and depth = 6,Test Mean Squared Error is 1.201251
# For num_trees = 40 and depth = 8,Test Mean Squared Error is 1.181799
# For num_trees = 40 and depth = 10,Test Mean Squared Error is 1.170667
# For num_trees = 40 and depth = 12,Test Mean Squared Error is 1.179056

# 4
start= time.time()
(trainingData_4, testData_4) = test_train_data(merged_train_ku_kb_no_stars)
trainingData_4.persist()
testData_4.persist()
printMSE =[]
num_trees = [10, 20, 40, 70, 100]
depth = [2, 4, 8, 10, 12]
for i in num_trees:
    for j in depth:
        testMSE = validation_rf(trainingData_4, testData_4, i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j
trainingData_4.unpersist()
testData_4.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start


# 'For num_trees = 10 and depth = 2,Test Mean Squared Error is 1.455449',
#  'For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.436047',
#  'For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.387521',
#  'For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.368191',
#  'For num_trees = 10 and depth = 12,Test Mean Squared Error is 1.351622',
#  'For num_trees = 20 and depth = 2,Test Mean Squared Error is 1.456451',
#  'For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.433128',
#  'For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.389121',
#  'For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.366834',
#  'For num_trees = 20 and depth = 12,Test Mean Squared Error is 1.345053',
#  'For num_trees = 40 and depth = 2,Test Mean Squared Error is 1.456983',
#  'For num_trees = 40 and depth = 4,Test Mean Squared Error is 1.433166',
#  'For num_trees = 40 and depth = 8,Test Mean Squared Error is 1.386145',
#  'For num_trees = 40 and depth = 10,Test Mean Squared Error is 1.364809',
#  'For num_trees = 40 and depth = 12,Test Mean Squared Error is 1.343023',
#  'For num_trees = 70 and depth = 2,Test Mean Squared Error is 1.456446',
#  'For num_trees = 70 and depth = 4,Test Mean Squared Error is 1.433437',
#  'For num_trees = 70 and depth = 8,Test Mean Squared Error is 1.388591']

# 5
start= time.time()
(trainingData_5, testData_5) = test_train_data(merged_train_kb_only_no_stars)
trainingData_5.persist()
testData_5.persist()
printMSE =[]
num_trees = [10, 20, 40, 70, 100]
depth = [2, 4, 8, 10, 12]
for i in num_trees:
    for j in depth:
        testMSE = validation_rf(trainingData_5, testData_5, i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
trainingData_5.unpersist()
testData_5.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start


# 'For num_trees = 10 and depth = 2,Test Mean Squared Error is 1.451861',
#  'For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.431951',
#  'For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.393318',
#  'For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.368951',
#  'For num_trees = 10 and depth = 12,Test Mean Squared Error is 1.347106',
#  'For num_trees = 20 and depth = 2,Test Mean Squared Error is 1.452395',
#  'For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.432170',
#  'For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.385441',
#  'For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.362783',
#  'For num_trees = 20 and depth = 12,Test Mean Squared Error is 1.342133',
#  'For num_trees = 40 and depth = 2,Test Mean Squared Error is 1.452327',
#  'For num_trees = 40 and depth = 4,Test Mean Squared Error is 1.431597',
#  'For num_trees = 40 and depth = 8,Test Mean Squared Error is 1.385234',
#  'For num_trees = 40 and depth = 10,Test Mean Squared Error is 1.365102',
#  'For num_trees = 40 and depth = 12,Test Mean Squared Error is 1.344080',
#  'For num_trees = 70 and depth = 2,Test Mean Squared Error is 1.452118',
#  'For num_trees = 70 and depth = 4,Test Mean Squared Error is 1.431162',
#  'For num_trees = 70 and depth = 8,Test Mean Squared Error is 1.386323',
#  'For num_trees = 70 and depth = 10,Test Mean Squared Error is 1.360827'



start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 20, 10)
print "1 done"
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 20, 8)
print "2 done"
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 25, 10)
print "3 done"
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 20, 12)
print "4 done"
model_5_rf = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 20, 12)
stop= time.time()
# 1505.7540740966797

# rmse - 1.27
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_0203_a.csv', index=None)

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


p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0203_b.csv', index=None)


# rmse - 1.28
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0203_c.csv', index=None)


########################################################################################################
# PERFORMING 3-FOLD CROSS VALIDATION TO SELECT BEST PARAMETERS  FOR EACH OF THE 5 SEGMENTS
########################################################################################################
# 3-fold Cross validation
# 1
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_all_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

# num_trees = [5, 10, 15, 20, 25]
# depth = [4, 6, 8, 10, 12]

printMSE =[]
# num_trees = [5, 10, 15, 20, 25, 30]
# depth = [4, 6, 8, 10]
num_trees = [15]
depth = [2, 12]
for i in num_trees:
    for j in depth:
        testMSE = cross_validation_rf(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start

#
# 'For num_trees = 5 and depth = 4,Test Mean Squared Error is 1.062701',
#  'For num_trees = 5 and depth = 6,Test Mean Squared Error is 0.985325',
#  'For num_trees = 5 and depth = 8,Test Mean Squared Error is 0.956877',
#  'For num_trees = 5 and depth = 10,Test Mean Squared Error is 0.946259',
#  'For num_trees = 5 and depth = 12,Test Mean Squared Error is 0.950529',
#  'For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.056830',
#  'For num_trees = 10 and depth = 6,Test Mean Squared Error is 0.974309',
#  'For num_trees = 10 and depth = 8,Test Mean Squared Error is 0.947391',
#  'For num_trees = 10 and depth = 10,Test Mean Squared Error is 0.939234',
#  'For num_trees = 10 and depth = 12,Test Mean Squared Error is 0.939809',
#  'For num_trees = 15 and depth = 4,Test Mean Squared Error is 1.039497',
#  'For num_trees = 15 and depth = 6,Test Mean Squared Error is 0.969214',
#  'For num_trees = 15 and depth = 8,Test Mean Squared Error is 0.944835',
#  'For num_trees = 15 and depth = 10,Test Mean Squared Error is 0.933909']
# ['For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.050824',
#  'For num_trees = 20 and depth = 6,Test Mean Squared Error is 0.968478',
#  'For num_trees = 20 and depth = 8,Test Mean Squared Error is 0.944972',
#  'For num_trees = 20 and depth = 10,Test Mean Squared Error is 0.933811',
#  'For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.049525',
#  'For num_trees = 25 and depth = 6,Test Mean Squared Error is 0.974561',
#  'For num_trees = 25 and depth = 8,Test Mean Squared Error is 0.944179']
# 'For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.033333',
#  'For num_trees = 30 and depth = 6,Test Mean Squared Error is 0.977108',
#  'For num_trees = 30 and depth = 8,Test Mean Squared Error is 0.942252',
#  'For num_trees = 30 and depth = 10,Test Mean Squared Error is 0.931893'
# For num_trees = 15 and depth = 2,Test Mean Squared Error is 1.175171
# For num_trees = 15 and depth = 12,Test Mean Squared Error is 0.935836
# 1739.46924186
# For num_trees = 15 and depth = 14,Test Mean Squared Error is 0.942030'

# 2
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_b_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]

num_trees = [15]
depth = [2, 12]
# num_trees = [5, 10, 15, 20, 25, 30]
# depth = [4, 6, 8, 10]
for i in num_trees:
    for j in depth:
        testMSE = cross_validation_rf(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start

# For num_trees = 5 and depth = 4,Test Mean Squared Error is 1.258292
# For num_trees = 5 and depth = 6,Test Mean Squared Error is 1.193004
# For num_trees = 5 and depth = 8,Test Mean Squared Error is 1.179462
# For num_trees = 5 and depth = 10,Test Mean Squared Error is 1.178267
# For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.213228
# For num_trees = 10 and depth = 6,Test Mean Squared Error is 1.189101
# For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.172981
# For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.176611
# For num_trees = 15 and depth = 4,Test Mean Squared Error is 1.224030
# For num_trees = 15 and depth = 6,Test Mean Squared Error is 1.181784
# For num_trees = 15 and depth = 8,Test Mean Squared Error is 1.176044
# For num_trees = 15 and depth = 10,Test Mean Squared Error is 1.175161
# For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.220479
# For num_trees = 20 and depth = 6,Test Mean Squared Error is 1.181956
# For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.174923
# For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.175404

# 5341.6423769
# For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.215637
# For num_trees = 25 and depth = 6,Test Mean Squared Error is 1.183168
# For num_trees = 25 and depth = 8,Test Mean Squared Error is 1.175086
# For num_trees = 25 and depth = 10,Test Mean Squared Error is 1.175180
# For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.216973
# For num_trees = 30 and depth = 6,Test Mean Squared Error is 1.184604
# For num_trees = 30 and depth = 8,Test Mean Squared Error is 1.172898
# For num_trees = 30 and depth = 10,Test Mean Squared Error is 1.174990
# 4656.04610085

# For num_trees = 15 and depth = 2,Test Mean Squared Error is 1.326541
# For num_trees = 15 and depth = 12,Test Mean Squared Error is 1.182987
# 1424.1393342

# For num_trees = 15 and depth = 14,Test Mean Squared Error is 1.194052
# For num_trees = 15 and depth = 16,Test Mean Squared Error is 1.206677
# 2797.38185215


# 3
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_u_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
# num_trees = [5, 10, 15, 20, 25]
# depth = [4, 6, 8, 10]
num_trees = [15]
depth = [2, 12]
for i in num_trees:
    for j in depth:
        testMSE = cross_validation_rf(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start
# 20 10
# For num_trees = 5 and depth = 4,Test Mean Squared Error is 1.230617
# For num_trees = 5 and depth = 6,Test Mean Squared Error is 1.237591
# For num_trees = 5 and depth = 8,Test Mean Squared Error is 1.186682
# For num_trees = 5 and depth = 10,Test Mean Squared Error is 1.180045
# For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.235228
# For num_trees = 10 and depth = 6,Test Mean Squared Error is 1.201211
# For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.185425
# For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.181728
# For num_trees = 15 and depth = 4,Test Mean Squared Error is 1.233656
# For num_trees = 15 and depth = 6,Test Mean Squared Error is 1.200014
# For num_trees = 15 and depth = 8,Test Mean Squared Error is 1.180983
# For num_trees = 15 and depth = 10,Test Mean Squared Error is 1.179031
# For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.231883
# For num_trees = 20 and depth = 6,Test Mean Squared Error is 1.201361
# For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.178096
# For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.176391
# For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.245413
# For num_trees = 25 and depth = 6,Test Mean Squared Error is 1.194224
# For num_trees = 25 and depth = 8,Test Mean Squared Error is 1.179123
# For num_trees = 25 and depth = 10,Test Mean Squared Error is 1.173484
# For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.237806
# For num_trees = 30 and depth = 6,Test Mean Squared Error is 1.196399
# For num_trees = 30 and depth = 8,Test Mean Squared Error is 1.185740
# For num_trees = 30 and depth = 10,Test Mean Squared Error is 1.180536
# 563.598444939
# For num_trees = 15 and depth = 2,Test Mean Squared Error is 1.329311
# For num_trees = 15 and depth = 12,Test Mean Squared Error is 1.183647
# 167.342327118
# For num_trees = 15 and depth = 14,Test Mean Squared Error is 1.194079
# For num_trees = 15 and depth = 16,Test Mean Squared Error is 1.216863
# 1228.89851403

# 4
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_no_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
# num_trees = [5, 10, 15, 20, 25, 30]
# depth = [4, 6, 8, 10]
num_trees = [15]
depth = [16]
for i in num_trees:
    for j in depth:
        testMSE = cross_validation_rf(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start

# For num_trees = 5 and depth = 4,Test Mean Squared Error is 1.435563
# For num_trees = 5 and depth = 6,Test Mean Squared Error is 1.415436
# For num_trees = 5 and depth = 8,Test Mean Squared Error is 1.390047
# For num_trees = 5 and depth = 10,Test Mean Squared Error is 1.371740
# For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.433432
# For num_trees = 10 and depth = 6,Test Mean Squared Error is 1.412191
# For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.389026
# For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.367040
# For num_trees = 15 and depth = 4,Test Mean Squared Error is 1.433507
# For num_trees = 15 and depth = 6,Test Mean Squared Error is 1.412850
# For num_trees = 15 and depth = 8,Test Mean Squared Error is 1.388784
# For num_trees = 15 and depth = 10,Test Mean Squared Error is 1.366395
# For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.432932
# For num_trees = 20 and depth = 6,Test Mean Squared Error is 1.411688
# For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.387575
# For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.365248
# For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.433032
# For num_trees = 25 and depth = 6,Test Mean Squared Error is 1.411161
# For num_trees = 25 and depth = 8,Test Mean Squared Error is 1.387108
# For num_trees = 25 and depth = 10,Test Mean Squared Error is 1.364764
# For num_trees = 30 and depth = 4,Test Mean Squared Error is 1.432523
# For num_trees = 30 and depth = 6,Test Mean Squared Error is 1.410693
# For num_trees = 30 and depth = 8,Test Mean Squared Error is 1.387540
# For num_trees = 30 and depth = 10,Test Mean Squared Error is 1.363711
# For num_trees = 30 and depth = 12,Test Mean Squared Error is 1.344449
# 4748.52658606
# For num_trees = 15 and depth = 2,Test Mean Squared Error is 1.456897
# For num_trees = 15 and depth = 12,Test Mean Squared Error is 1.346992
# 880.484043121
# For num_trees = 15 and depth = 14,Test Mean Squared Error is 1.329891
# 942.905301809
# For num_trees = 15 and depth = 16,Test Mean Squared Error is 1.322268
# 2174.27639389
# For num_trees = 15 and depth = 18,Test Mean Squared Error is 1.316599

# 5
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_kb_only_no_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
# num_trees = [5, 10, 15, 20, 25]
# depth = [4, 6, 8, 10]
num_trees = [15]
depth = [16]
for i in num_trees:
    for j in depth:
        testMSE = cross_validation_rf(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For num_trees = %s " %i + "and depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start
# For num_trees = 5 and depth = 4,Test Mean Squared Error is 1.437214
# For num_trees = 5 and depth = 6,Test Mean Squared Error is 1.416764
# For num_trees = 5 and depth = 8,Test Mean Squared Error is 1.395110
# For num_trees = 5 and depth = 10,Test Mean Squared Error is 1.377657
# For num_trees = 10 and depth = 4,Test Mean Squared Error is 1.438164
# For num_trees = 10 and depth = 6,Test Mean Squared Error is 1.413225
# For num_trees = 10 and depth = 8,Test Mean Squared Error is 1.390269
# For num_trees = 10 and depth = 10,Test Mean Squared Error is 1.368717
# For num_trees = 15 and depth = 4,Test Mean Squared Error is 1.436279
# For num_trees = 15 and depth = 6,Test Mean Squared Error is 1.413778
# For num_trees = 15 and depth = 8,Test Mean Squared Error is 1.390908
# For num_trees = 15 and depth = 10,Test Mean Squared Error is 1.370173
# For num_trees = 20 and depth = 4,Test Mean Squared Error is 1.436336
# For num_trees = 20 and depth = 6,Test Mean Squared Error is 1.412283
# For num_trees = 20 and depth = 8,Test Mean Squared Error is 1.391707
# For num_trees = 20 and depth = 10,Test Mean Squared Error is 1.370468
# For num_trees = 25 and depth = 4,Test Mean Squared Error is 1.436109
# For num_trees = 25 and depth = 6,Test Mean Squared Error is 1.413666
# For num_trees = 25 and depth = 8,Test Mean Squared Error is 1.389934
# For num_trees = 25 and depth = 10,Test Mean Squared Error is 1.369087
# 5505.47098899
# For num_trees = 15 and depth = 2,Test Mean Squared Error is 1.459687
# For num_trees = 15 and depth = 12,Test Mean Squared Error is 1.352678
# 955.24980092
# For num_trees = 15 and depth = 14,Test Mean Squared Error is 1.331665
# For num_trees = 15 and depth = 16,Test Mean Squared Error is 1.320540
# 1407.02827501
# For num_trees = 15 and depth = 18,Test Mean Squared Error is 1.315042
# For num_trees = 15 and depth = 20,Test Mean Squared Error is 1.309315

csfont = {'fontname':'Times New Roman'}
hfont = {'fontname':'Times New Roman'}
depth_trees = [2,4,6,8,10,12, 14, 16]
seg_1 = [1.175171, 1.039497,0.969214, 0.944835, 0.933909, 0.935836, 0.942030, 0.94910]
seg_2 = [1.326541, 1.224030,1.181784, 1.176044, 1.175161, 1.182987, 1.194052, 1.206677]
seg_3 = [1.329311, 1.233656,1.200014, 1.180983, 1.179031, 1.183647, 1.194079, 1.216863]
seg_4 = [1.456897, 1.433507,1.412850,1.388784, 1.366395, 1.346992, 1.32989, 1.322268]
seg_5 = [1.459687, 1.436279,1.413778,1.390908,1.370173, 1.352678, 1.331665, 1.320540]
fig = plt.figure()
ax = fig.add_subplot(111)
# plt.plot(depth_trees, seg_1, color='b', label='Segment 1 - Known User, Known business')
# plt.plot(depth_trees, seg_2, color='g', label='Segment 2 - Unknown User, Known business')
# plt.plot(depth_trees, seg_3, color='r', label='Segment 3 - Known User, Unknown business')
# plt.plot(depth_trees, seg_4, color='y', label='Segment 4 - Known User, Known business in Test only')
# plt.plot(depth_trees, seg_5, color='m', label='Segment 5 - Unknown User, Known business in Test only')
plt.plot(depth_trees, seg_1, color='b', label='Segment 1')
plt.plot(depth_trees, seg_2, color='g', label='Segment 2')
plt.plot(depth_trees, seg_3, color='r', label='Segment 3')
plt.plot(depth_trees, seg_4, color='y', label='Segment 4')
plt.plot(depth_trees, seg_5, color='m', label='Segment 5')
plt.title("Cross Validation across segments", fontsize=16, **csfont)
plt.xlabel("Maximum depth of trees",fontsize=14,**hfont)
plt.ylabel("MSE - Mean Squared Error",fontsize=14,**hfont)
# plt.legend(loc='right', fontsize=12)
# plt.legend(bbox_to_anchor=(0, 1), loc='upper left', ncol=1)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)
plt.legend(loc='upper center', bbox_to_anchor=(0.85, 0.45),fontsize=12,
          ncol=1, fancybox=True, shadow=True)
plt.grid()
plt.savefig("1.pdf", format="pdf")
plt.show()




########################################################################################################
# BUILDING FINAL MODELS AND DOING PREDICTIONS ON TEST DATA
########################################################################################################

start= time.time()
model_1_rf = seg_model_rf(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 15, 10)
model_2_rf = seg_model_rf(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 15, 10)
model_3_rf = seg_model_rf(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 15, 10)
model_4_rf = seg_model_rf(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 15, 10)
model_5_rf  = seg_model_rf(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 15, 10)
stop= time.time()


# rmse - 1.267/1.27
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_0503_e_2.csv', index=None)

# Baseline
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

# Blended model - 1 - 1.273
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0503_f.csv', index=None)


# Blended model - 2 - 1.279
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0503_g_2.csv', index=None)


# Blended model - 3 - 1.284
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0503_h.csv', index=None)



# Blended model - 4 - 1.288
p_combined = sqlContext.createDataFrame(model_1_rf,['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0503_i.csv', index=None)