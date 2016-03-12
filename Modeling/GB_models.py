################################################################################
# APPLYING GRADIENT BOOSTING ALGORITHM
################################################################################

from pyspark.mllib.tree import GradientBoostedTrees, GradientBoostedTreesModel
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler


################################################################################
#FUNCTIONS TO PERFORM VALIDATION TO TUNE a) Loss type b) Num of Iterations c) Max depth of trees
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


# Function to Perform One fold validation
def validation_gb(trainingData,testData, loss_type, num_iter, maxDepth):
    # Training the model using Gradient Boosted Trees regressor
    model_train = GradientBoostedTrees.trainRegressor(trainingData, categoricalFeaturesInfo={},
                                                      loss=loss_type,
                                                      numIterations=num_iter, maxDepth=maxDepth)

    # Evaluate model on test instances and compute test error
    predictions = model_train.predict(testData.map(lambda x: x.features))
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(testData.count())
    return testMSE


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


# Function to Perform 3-fold cross validation
def cross_validation_gb(Data_1,Data_2,Data_3,loss_type, num_iter, maxDepth):
    # Training the model using Gradient Boosted Trees regressor
    model_train_1 = GradientBoostedTrees.trainRegressor(Data_1.union(Data_2), categoricalFeaturesInfo={},
                                                      loss=loss_type,
                                                      numIterations=num_iter, maxDepth=maxDepth)

    # Evaluate model on test instances and compute test error
    predictions_1 = model_train_1.predict(Data_3.map(lambda x: x.features))
    labelsAndPredictions_1 = Data_3.map(lambda lp: lp.label).zip(predictions_1)
    testMSE_1 = labelsAndPredictions_1.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(Data_3.count())

    model_train_2 = GradientBoostedTrees.trainRegressor(Data_2.union(Data_3), categoricalFeaturesInfo={},
                                                      loss=loss_type,
                                                      numIterations=num_iter, maxDepth=maxDepth)

    # Evaluate model on test instances and compute test error
    predictions_2 = model_train_2.predict(Data_1.map(lambda x: x.features))
    labelsAndPredictions_2 = Data_1.map(lambda lp: lp.label).zip(predictions_2)
    testMSE_2 = labelsAndPredictions_2.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(Data_1.count())

    model_train_3 = GradientBoostedTrees.trainRegressor(Data_3.union(Data_1), categoricalFeaturesInfo={},
                                                      loss=loss_type,
                                                      numIterations=num_iter, maxDepth=maxDepth)

    # Evaluate model on test instances and compute test error
    predictions_3 = model_train_3.predict(Data_2.map(lambda x: x.features))
    labelsAndPredictions_3 = Data_2.map(lambda lp: lp.label).zip(predictions_3)
    testMSE_3 = labelsAndPredictions_3.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(Data_2.count())

    return (testMSE_1+testMSE_2+testMSE_3)/3

################################################################################
# Function to build final model on complete segment with the selected paramEters
################################################################################

def seg_model_gb(train_data, test_data, loss_type, num_iter, maxDepth):
    removelist_train= set(['stars', 'business_id', 'bus_id', 'b_id','review_id', 'user_id'])
    newlist_train = [v for i, v in enumerate(train_data.columns) if v not in removelist_train]

    # Putting data in vector assembler form
    assembler_train = VectorAssembler(inputCols=newlist_train, outputCol="features")

    transformed_train = assembler_train.transform(train_data.fillna(0))

    # Creating input dataset in the form of labeled point for training the model
    data_train= (transformed_train.select("features", "stars")).map(lambda row: LabeledPoint(row.stars, row.features))

    # Training the model using Gradient Boosted Trees regressor
    model_train = GradientBoostedTrees.trainRegressor(sc.parallelize(data_train.collect(),5), categoricalFeaturesInfo={},
                                                      loss=loss_type,
                                                      numIterations=num_iter, maxDepth=maxDepth)

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
# MODEL 1 - Known user known business model - Business and user stars - Training the model on Segment with known user and business
# information
########################################################################################################
########################################################################################################
# MODEL 2 - UnKnown user known business model - with business stars Training the model on Segment with known business information
########################################################################################################
########################################################################################################
# MODEL 3 - Known user Unknown business model - with user stars - Training the model on Segment with known user information
########################################################################################################
########################################################################################################
# MODEL 4 - Known user known business model - No stars- Training the model on Segment with known user and business
# information
########################################################################################################
########################################################################################################
# MODEL 5 - UnKnown user known business model - No stars - Training the model on Segment with known business information
########################################################################################################



########################################################################################################
#  FEW ITERATIONS
########################################################################################################

start= time.time()
gb_model_1 = seg_model_gb(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 'leastSquaresError', 10)
gb_model_2 = seg_model_gb(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 'leastSquaresError', 10)
gb_model_3 = seg_model_gb(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 'leastSquaresError', 10)
gb_model_4 = seg_model_gb(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 'leastSquaresError', 10)
gb_model_5 = seg_model_gb(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 'leastSquaresError', 10)
stop= time.time()

p_combined = sqlContext.createDataFrame(gb_model_1.union(gb_model_2).union(gb_model_3).union(gb_model_4).union(gb_model_5),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.stars = p_combined.stars.map(lambda x: 5 if x > 5 else x)
p_combined.to_csv('submission_2702_h.csv', index=None)


# 1.29
start= time.time()
gb_model_1 = seg_model_gb(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 'logLoss', 10)
gb_model_2 = seg_model_gb(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 'logLoss', 10)
gb_model_3 = seg_model_gb(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 'logLoss', 10)
gb_model_4 = seg_model_gb(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 'logLoss', 10)
gb_model_5 = seg_model_gb(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 'logLoss', 10)
stop= time.time()

p_combined = sqlContext.createDataFrame(gb_model_1.union(gb_model_2).union(gb_model_3).union(gb_model_4).union(gb_model_5),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_2702_f.csv', index=None)

start= time.time()
gb_model_1 = seg_model_gb(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 'leastSquaresError', 20)
print "1 done"
gb_model_2 = seg_model_gb(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 'leastSquaresError', 20)
print "2 done"
gb_model_3 = seg_model_gb(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 'leastSquaresError', 20)
print "3 done"
gb_model_4 = seg_model_gb(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 'leastSquaresError', 20)
print "4 done"
gb_model_5 = seg_model_gb(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 'leastSquaresError', 20)
stop= time.time()

# 3577.229332923889


p_combined = sqlContext.createDataFrame(gb_model_1.union(gb_model_2).union(gb_model_3).union(gb_model_4).union(gb_model_5),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.stars = p_combined.stars.map(lambda x: 5 if x > 5 else x)
p_combined.to_csv('submission_2802_b.csv', index=None)



# Baseline
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
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_2702_g.csv', index=None)



########################################################################################################
# PERFORMING VALIDATION TO SELECT BEST PARAMETERS  FOR EACH OF THE 5 SEGMENTS
########################################################################################################
# Choosing the best loss_type and num_iter for each segment

# 1
start= time.time()
(trainingData_1, testData_1) = test_train_data(merged_train_ku_kb_all_stars)
trainingData_1.persist()
testData_1.persist()
printMSE =[]
loss_type = ['logLoss' ,'leastSquaresError', 'leastAbsoluteError']
num_iter = [10, 20, 40, 70, 100]
maxDepth= [7, 9]
for i in loss_type:
    for j in maxDepth:
        for k in num_iter :
            testMSE = validation_gb(trainingData_1, testData_1, i, k, j)
            printMSE.append(str("For loss_type = %s, " %i + "num_iter = %s," %k + "and max depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
            print i,j,k
trainingData_1.unpersist()
testData_1.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start

# 'For loss_type = logLoss, num_iter = 10,and max depth = 1,Test Mean Squared Error is 1.287799',
#  'For loss_type = logLoss, num_iter = 10,and max depth = 3,Test Mean Squared Error is 1.065381',
#  'For loss_type = logLoss, num_iter = 20,and max depth = 1,Test Mean Squared Error is 1.288007',
#  'For loss_type = logLoss, num_iter = 20,and max depth = 3,Test Mean Squared Error is 1.065281',
#  'For loss_type = logLoss, num_iter = 30,and max depth = 1,Test Mean Squared Error is 1.288214',
#  'For loss_type = logLoss, num_iter = 30,and max depth = 3,Test Mean Squared Error is 1.067665',
#  'For loss_type = logLoss, num_iter = 40,and max depth = 1,Test Mean Squared Error is 1.288421',
#  'For loss_type = logLoss, num_iter = 40,and max depth = 3,Test Mean Squared Error is 1.068945',
#  'For loss_type = logLoss, num_iter = 70,and max depth = 1,Test Mean Squared Error is 1.289039',
#  'For loss_type = logLoss, num_iter = 70,and max depth = 3,Test Mean Squared Error is 1.072497',
#  'For loss_type = leastSquaresError, num_iter = 10,and max depth = 1,Test Mean Squared Error is 1.081912',
#  'For loss_type = leastSquaresError, num_iter = 10,and max depth = 3,Test Mean Squared Error is 0.972440',
#  'For loss_type = leastSquaresError, num_iter = 20,and max depth = 1,Test Mean Squared Error is 1.024450'
#  ['For loss_type = leastSquaresError, num_iter = 10,and max depth = 1,Test Mean Squared Error is 1.076844',
#  'For loss_type = leastSquaresError, num_iter = 20,and max depth = 1,Test Mean Squared Error is 1.020518',
#  'For loss_type = leastSquaresError, num_iter = 30,and max depth = 1,Test Mean Squared Error is 0.995659',
#  'For loss_type = leastSquaresError, num_iter = 10,and max depth = 3,Test Mean Squared Error is 0.971276',
#  'For loss_type = leastSquaresError, num_iter = 20,and max depth = 3,Test Mean Squared Error is 0.958013',
#  'For loss_type = leastSquaresError, num_iter = 30,and max depth = 3,Test Mean Squared Error is 0.946056',
#  'For loss_type = leastSquaresError, num_iter = 10,and max depth = 5,Test Mean Squared Error is 0.936725',
#  'For loss_type = leastSquaresError, num_iter = 20,and max depth = 5,Test Mean Squared Error is 0.934203']
#  For loss_type = leastSquaresError, num_iter = 10,and max depth = 7,Test Mean Squared Error is 0.934459
# For loss_type = leastSquaresError, num_iter = 15,and max depth = 7,Test Mean Squared Error is 0.932651
# For loss_type = leastSquaresError, num_iter = 20,and max depth = 7,Test Mean Squared Error is 0.931830
# For loss_type = leastSquaresError, num_iter = 10,and max depth = 9,Test Mean Squared Error is 0.941851
# For loss_type = leastSquaresError, num_iter = 15,and max depth = 9,Test Mean Squared Error is 0.943833
# For loss_type = leastSquaresError, num_iter = 20,and max depth = 9,Test Mean Squared Error is 0.945348
# For loss_type = leastAbsoluteError, num_iter = 10,and max depth = 7,Test Mean Squared Error is 0.948413
# For loss_type = leastAbsoluteError, num_iter = 15,and max depth = 7,Test Mean Squared Error is 0.955288
# For loss_type = leastAbsoluteError, num_iter = 20,and max depth = 7,Test Mean Squared Error is 0.961710
# For loss_type = leastAbsoluteError, num_iter = 10,and max depth = 9,Test Mean Squared Error is 0.957059
# For loss_type = leastAbsoluteError, num_iter = 15,and max depth = 9,Test Mean Squared Error is 0.962150
# For loss_type = leastAbsoluteError, num_iter = 20,and max depth = 9,Test Mean Squared Error is 0.968085

# 2
start= time.time()
(trainingData_2, testData_2) = test_train_data(merged_train_ku_kb_b_stars)
trainingData_2.persist()
testData_2.persist()
printMSE =[]
loss_type = ['logLoss' ,'leastSquaresError', 'leastAbsoluteError']
num_iter = [10, 20, 40, 70, 100]
for i in loss_type:
    for j in num_iter:
        testMSE = validation_gb(trainingData_2, testData_2, i, j, k)
        printMSE.append(str("For loss_type = %s " %i + "and num_iter = %s," %j + "Test Mean Squared Error is %f" %testMSE))
trainingData_2.unpersist()
testData_2.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start


# 3
start= time.time()
(trainingData_3, testData_3) = test_train_data(merged_train_ku_kb_u_stars)
trainingData_3.persist()
testData_3.persist()
printMSE =[]
loss_type = ['logLoss' ,'leastSquaresError', 'leastAbsoluteError']
num_iter = [10, 20, 40, 70, 100]
for i in loss_type:
    for j in num_iter:
        testMSE = validation_gb(trainingData_3, testData_3, i, j, k)
        printMSE.append(str("For loss_type = %s " %i + "and num_iter = %s," %j + "Test Mean Squared Error is %f" %testMSE))
trainingData_3.unpersist()
testData_3.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start


# 4
start= time.time()
(trainingData_4, testData_4) = test_train_data(merged_train_ku_kb_no_stars)
trainingData_4.persist()
testData_4.persist()
printMSE =[]
loss_type = ['logLoss' ,'leastSquaresError', 'leastAbsoluteError']
num_iter = [10, 20, 40, 70, 100]
for i in loss_type:
    for j in num_iter:
        testMSE = validation_gb(trainingData_4, testData_4, i, j, k)
        printMSE.append(str("For loss_type = %s " %i + "and num_iter = %s," %j + "Test Mean Squared Error is %f" %testMSE))
trainingData_4.unpersist()
testData_4.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start


# 5
start= time.time()
(trainingData_5, testData_5) = test_train_data(merged_train_kb_only_no_stars)
trainingData_5.persist()
testData_5.persist()
printMSE =[]
loss_type = ['logLoss' ,'leastSquaresError', 'leastAbsoluteError']
num_iter = [10, 20, 40, 70, 100]
for i in loss_type:
    for j in num_iter:
        testMSE = validation_gb(trainingData_5, testData_5, i, j, k)
        printMSE.append(str("For loss_type = %s " %i + "and num_iter = %s," %j + "Test Mean Squared Error is %f" %testMSE))
trainingData_5.unpersist()
testData_5.unpersist()
for i in printMSE:
    print i
stop= time.time()
print stop - start


########################################################################################################
# PERFORMING 3-FOLD CROSS VALIDATION TO SELECT BEST PARAMETERS FOR EACH OF THE 5 SEGMENTS
########################################################################################################

# 3-fold Cross validation
# 1
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_all_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
loss_type = ['leastSquaresError']
num_iter = [10, 15, 20]
maxDepth= [3, 5, 7]
for i in loss_type:
    for j in maxDepth:
        for k in num_iter :
            testMSE = cross_validation_gb(Data_1, Data_2, Data_3,i, k, j)
            printMSE.append(str("For loss_type = %s, " %i + "num_iter = %s," %k + "and max depth = %s," %j + "Test Mean Squared Error is %f" %testMSE))
            print i,j,k
Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start



########################################################################################################
# BUILDING FINAL MODELS AND DOING PREDICTIONS ON TEST DATA
########################################################################################################


start= time.time()
gb_model_1 = seg_model_gb(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 'leastSquaresError', 20, 7)
print "1 done"
gb_model_2 = seg_model_gb(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 'leastSquaresError', 20, 7)
print "2 done"
gb_model_3 = seg_model_gb(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 'leastSquaresError', 20, 7)
print "3 done"
gb_model_4 = seg_model_gb(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 'leastSquaresError', 20, 7)
print "4 done"
gb_model_5 = seg_model_gb(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 'leastSquaresError', 20, 7)
stop= time.time()


p_combined = sqlContext.createDataFrame(gb_model_1.union(gb_model_2).union(gb_model_3).union(gb_model_4).union(gb_model_5),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.stars = p_combined.stars.map(lambda x: 5 if x > 5 else x)
p_combined.to_csv('submission_0303_a.csv', index=None)




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
p_combined = sqlContext.createDataFrame(gb_model_1.union(gb_model_2).union(gb_model_3),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0303_b.csv', index=None)


# submission
p_combined = sqlContext.createDataFrame(gb_model_1.union(gb_model_2).union(gb_model_3).union(gb_model_4),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0303_c.csv', index=None)
