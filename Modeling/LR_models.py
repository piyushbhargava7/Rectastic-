################################################################################
# APPLYING LOGISTIC REGRESSION CLASSIFIER
################################################################################

from pyspark.mllib.classification import LogisticRegressionWithLBFGS
from pyspark.mllib.regression import LabeledPoint
from pyspark.ml.feature import VectorAssembler


################################################################################
#FUNCTIONS TO PERFORM VALIDATION TO TUNE a) Regularization type b) Num of Iterations
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
def validation_lr(trainingData,testData, regType, num_iter):
    # Training the model using Logistic Regression Classifier

    model_train =LogisticRegressionWithLBFGS.train(trainingData, regType =regType, iterations=num_iter, numClasses=5)

    # Evaluate model on test instances and compute test error
    predictions = model_train.predict(testData.map(lambda x: x.features))

    testMSE_1 = labelsAndPredictions_1.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(testData.count())
    labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
    testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() /\
        float(testData.count())
    return testMSE_1,testMSE


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
def cross_validation_lr(Data_1,Data_2,Data_3,regType, num_iter):
    # Training the model using Logistic Regression Classifier
    model_train_1 =LogisticRegressionWithLBFGS.train(Data_1.union(Data_2),
                                                     regType =regType, iterations=num_iter, numClasses=5)

    # Evaluate model on test instances and compute test error
    predictions_1 = model_train_1.predict(Data_3.map(lambda x: x.features))
    labelsAndPredictions_1 = Data_3.map(lambda lp: lp.label).zip(predictions_1)
    testMSE_1 = labelsAndPredictions_1.map(lambda (v, p): (v +0.5 - p) * (v +0.5- p )).sum() /\
        float(Data_3.count())


    model_train_2 =LogisticRegressionWithLBFGS.train(Data_2.union(Data_3),
                                                     regType =regType, iterations=num_iter, numClasses=5)

    # Evaluate model on test instances and compute test error
    predictions_2 = model_train_2.predict(Data_1.map(lambda x: x.features))
    labelsAndPredictions_2 = Data_1.map(lambda lp: lp.label).zip(predictions_2)
    testMSE_2 = labelsAndPredictions_2.map(lambda (v, p): (v +0.5- p) * (v +0.5- p )).sum() /\
        float(Data_1.count())


    model_train_3 =LogisticRegressionWithLBFGS.train(Data_3.union(Data_1),
                                                     regType =regType, iterations=num_iter, numClasses=5)


    # Evaluate model on test instances and compute test error
    predictions_3 = model_train_3.predict(Data_2.map(lambda x: x.features))
    labelsAndPredictions_3 = Data_2.map(lambda lp: lp.label).zip(predictions_3)
    testMSE_3 = labelsAndPredictions_3.map(lambda (v, p): (v +0.5- p ) * (v +0.5- p)).sum() /\
        float(Data_2.count())

    return (testMSE_1+testMSE_2+testMSE_3)/3

################################################################################
# Function to build final model on complete segment with the selected paramEters
################################################################################

def seg_model_lr(train_data, test_data, regType, num_iter):
    removelist_train= set(['stars', 'business_id', 'bus_id', 'b_id','review_id', 'user_id'])
    newlist_train = [v for i, v in enumerate(train_data.columns) if v not in removelist_train]

    # Putting data in vector assembler form
    assembler_train = VectorAssembler(inputCols=newlist_train, outputCol="features")

    transformed_train = assembler_train.transform(train_data.fillna(0))

    # Creating input dataset in the form of labeled point for training the model
    data_train= (transformed_train.select("features", "stars")).map(lambda row: LabeledPoint(row.stars, row.features))

    # Training the model using Logistic regression Classifier
    model_train = LogisticRegressionWithLBFGS.train(sc.parallelize(data_train.collect(),5),
                                                    regType =regType, iterations=num_iter, numClasses=5)

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
# PERFORMING 3-FOLD CROSS VALIDATION TO SELECT BEST PARAMETERS  FOR EACH OF THE 5 SEGMENTS
########################################################################################################
# 3-fold Cross validation
# 1

(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_all_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
RegType = ['l1' ,'l2']
num_iter = [20, 40, 60, 80, 100]
for i in RegType:
    for j in num_iter:
        start= time.time()
        testMSE = cross_validation_lr(Data_1, Data_2, Data_3, i, j)
        time_taken= time.time() -start
        printMSE.append(str("For regularization type = %s " %i +
                            "and iterations = %s," %j + "Test Mean Squared Error is %f." %testMSE +
                            " Time taken is %s s" %time_taken))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start

# For regularization type = l1 and iterations = 20,Test Mean Squared Error is 1.552672. Time taken is 146.4558599 s
# For regularization type = l1 and iterations = 40,Test Mean Squared Error is 1.552672. Time taken is 97.6601657867 s
# For regularization type = l1 and iterations = 60,Test Mean Squared Error is 1.552672. Time taken is 124.995703936 s
# For regularization type = l1 and iterations = 80,Test Mean Squared Error is 1.552672. Time taken is 89.2161040306 s
# For regularization type = l1 and iterations = 100,Test Mean Squared Error is 1.552672. Time taken is 107.553398132 s
# For regularization type = l2 and iterations = 20,Test Mean Squared Error is 1.593535. Time taken is 94.2097308636 s
# For regularization type = l2 and iterations = 40,Test Mean Squared Error is 1.296503. Time taken is 137.395787954 s
# For regularization type = l2 and iterations = 60,Test Mean Squared Error is 1.255664. Time taken is 114.47420311 s
# For regularization type = l2 and iterations = 80,Test Mean Squared Error is 1.227196. Time taken is 193.718127966 s
# For regularization type = l2 and iterations = 100,Test Mean Squared Error is 1.225198. Time taken is 316.26069212 s
# 200 - 1.1732548696292138
# 400 - 1.1732548696292138

 # 2
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_b_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
RegType = ['l1' ,'l2']
num_iter = [20, 40, 60, 80, 100]
for i in RegType:
    for j in num_iter:
        testMSE = cross_validation_lr(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For regularization type = %s " %i + "and iterations = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start

# For regularization type = l1 and iterations = 20,Test Mean Squared Error is 1.552269
# For regularization type = l1 and iterations = 40,Test Mean Squared Error is 1.552269
# For regularization type = l1 and iterations = 60,Test Mean Squared Error is 1.552269
# For regularization type = l1 and iterations = 80,Test Mean Squared Error is 1.552269
# For regularization type = l1 and iterations = 100,Test Mean Squared Error is 1.552269
# For regularization type = l2 and iterations = 20,Test Mean Squared Error is 1.504588
# For regularization type = l2 and iterations = 40,Test Mean Squared Error is 1.401797
# For regularization type = l2 and iterations = 60,Test Mean Squared Error is 1.392566
# For regularization type = l2 and iterations = 80,Test Mean Squared Error is 1.392566
# For regularization type = l2 and iterations = 100,Test Mean Squared Error is 1.392566
# 1767.73312497

# 3
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_u_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
RegType = ['l1' ,'l2']
num_iter = [20, 40, 60, 80, 100]
for i in RegType:
    for j in num_iter:
        testMSE = cross_validation_lr(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For regularization type = %s " %i + "and iterations = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start

# l2 100
# For regularization type = l1 and iterations = 20,Test Mean Squared Error is 1.738607
# For regularization type = l1 and iterations = 40,Test Mean Squared Error is 1.738607
# For regularization type = l1 and iterations = 60,Test Mean Squared Error is 1.738607
# For regularization type = l1 and iterations = 80,Test Mean Squared Error is 1.738607
# For regularization type = l1 and iterations = 100,Test Mean Squared Error is 1.738607
# For regularization type = l2 and iterations = 20,Test Mean Squared Error is 1.724397
# For regularization type = l2 and iterations = 40,Test Mean Squared Error is 1.724232
# For regularization type = l2 and iterations = 60,Test Mean Squared Error is 1.724232
# For regularization type = l2 and iterations = 80,Test Mean Squared Error is 1.724232
# For regularization type = l2 and iterations = 100,Test Mean Squared Error is 1.724232
# 593.250842094

# 4
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_ku_kb_no_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
RegType = ['l1' ,'l2']
num_iter = [20, 40, 60, 80, 100]
for i in RegType:
    for j in num_iter:
        testMSE = cross_validation_lr(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For regularization type = %s " %i + "and iterations = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start

# For regularization type = l1 and iterations = 20,Test Mean Squared Error is 1.553870
# For regularization type = l1 and iterations = 40,Test Mean Squared Error is 1.553870
# For regularization type = l1 and iterations = 60,Test Mean Squared Error is 1.553870
# For regularization type = l1 and iterations = 80,Test Mean Squared Error is 1.553870
# For regularization type = l1 and iterations = 100,Test Mean Squared Error is 1.553870
# For regularization type = l2 and iterations = 20,Test Mean Squared Error is 1.689184
# For regularization type = l2 and iterations = 40,Test Mean Squared Error is 1.587286
# For regularization type = l2 and iterations = 60,Test Mean Squared Error is 1.589703
# For regularization type = l2 and iterations = 80,Test Mean Squared Error is 1.589703
# For regularization type = l2 and iterations = 100,Test Mean Squared Error is 1.589703
# 1080.04374003


# 5
start= time.time()
(Data_1, Data_2, Data_3) = test_train_data_3fold(merged_train_kb_only_no_stars)
Data_1.persist()
Data_2.persist()
Data_3.persist()

printMSE =[]
RegType = ['l1' ,'l2']
num_iter = [20, 40, 60, 80, 100]
for i in RegType:
    for j in num_iter:
        testMSE = cross_validation_lr(Data_1, Data_2, Data_3,i, j)
        printMSE.append(str("For regularization type = %s " %i + "and iterations = %s," %j + "Test Mean Squared Error is %f" %testMSE))
        print i,j

Data_1.unpersist()
Data_2.unpersist()
Data_3.unpersist()

for i in printMSE:
    print i
stop= time.time()
print stop - start
start= time.time()
# l2 100
# For regularization type = l1 and iterations = 20,Test Mean Squared Error is 1.552199
# For regularization type = l1 and iterations = 40,Test Mean Squared Error is 1.552199
# For regularization type = l1 and iterations = 60,Test Mean Squared Error is 1.552199
# For regularization type = l1 and iterations = 80,Test Mean Squared Error is 1.552199
# For regularization type = l1 and iterations = 100,Test Mean Squared Error is 1.552199
# For regularization type = l2 and iterations = 20,Test Mean Squared Error is 1.583571
# For regularization type = l2 and iterations = 40,Test Mean Squared Error is 1.558588
# For regularization type = l2 and iterations = 60,Test Mean Squared Error is 1.562613
# For regularization type = l2 and iterations = 80,Test Mean Squared Error is 1.562613
# For regularization type = l2 and iterations = 100,Test Mean Squared Error is 1.562613
# 2259.461941




########################################################################################################
# BUILDING FINAL MODELS AND DOING PREDICTIONS ON TEST DATA
########################################################################################################


start= time.time()
lr_model_1 = seg_model_lr(merged_train_ku_kb_all_stars, merged_final_ku_kb_all_stars, 'l2', 100)
print "1 done"
lr_model_2 = seg_model_lr(merged_train_ku_kb_b_stars, merged_final_ku_kb_only_b_stars, 'l2', 100)
print "2 done"
lr_model_3 = seg_model_lr(merged_train_ku_kb_u_stars, merged_final_ku_kb_only_u_stars, 'l2', 100)
print "3 done"
lr_model_4 = seg_model_lr(merged_train_ku_kb_no_stars, merged_final_ku_kb_no_stars, 'l1', 100)
print "4 done"
lr_model_5 = seg_model_lr(merged_train_kb_only_no_stars, merged_final_uu_kb_no_stars, 'l1', 100)
stop= time.time()


p_combined = sqlContext.createDataFrame(lr_model_1.union(lr_model_2).union(lr_model_3).union(lr_model_4).union(lr_model_5),['review_id', 'stars'])
p_combined = p_combined.toPandas()
p_combined.to_csv('submission_0503_a.csv', index=None)


# Baseline model

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
# preds.to_csv('submission_0503_d.csv', index=None)


# Blended model - 1
p_combined = sqlContext.createDataFrame(lr_model_1.union(lr_model_2),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0503_b.csv', index=None)


# Blended model - 2
p_combined = sqlContext.createDataFrame(lr_model_1,['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
preds_final.to_csv('submission_0503_c.csv', index=None)
