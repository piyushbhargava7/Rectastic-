
# Reading ALS predictions
als_data = sc.textFile("submission.csv")

def parse_data(line):
    pieces = line.split(",")
    review_id = pieces[1]
    rating = float(pieces[0])
    return {"review_id": review_id, "pred": rating}

parsed_als = als_data.map(parse_data)
p_als = sqlContext.createDataFrame(parsed_als, [ 'pred','review_id'])
p_als = p_als.toPandas()


# Prioritising ALS predcitions over random forest where applicable
# 1 - 1.276
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ).union(model_5_rf),['review_id', 'stars'])
p_combined = p_combined.toPandas()

preds_ensemble = p_combined.merge(p_als, on=['review_id'], how='outer')
preds_ensemble.pred = preds_ensemble.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_ensemble.drop('stars', axis=1, inplace=True)
preds_ensemble.columns = ['review_id', 'stars']
preds_ensemble.to_csv('submission_0503_e_1.csv', index=None)




# 2 1.276
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf).union(model_4_rf ),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
# preds_final.to_csv('submission_0503_f.csv', index=None)

preds_ensemble = preds_final.merge(p_als, on=['review_id'], how='outer')
preds_ensemble.pred = preds_ensemble.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_ensemble.drop('stars', axis=1, inplace=True)
preds_ensemble.columns = ['review_id', 'stars']
preds_ensemble.to_csv('submission_0503_f_1.csv', index=None)


# 3 - 1.282
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf).union(model_3_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
# preds_final.to_csv('submission_0503_g.csv', index=None)

preds_ensemble = preds_final.merge(p_als, on=['review_id'], how='outer')
preds_ensemble.pred = preds_ensemble.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_ensemble.drop('stars', axis=1, inplace=True)
preds_ensemble.columns = ['review_id', 'stars']
preds_ensemble.to_csv('submission_0503_g_1.csv', index=None)


# 4 - 1.287
p_combined = sqlContext.createDataFrame(model_1_rf.union(model_2_rf),['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
# preds_final.to_csv('submission_0503_h.csv', index=None)

preds_ensemble = preds_final.merge(p_als, on=['review_id'], how='outer')
preds_ensemble.pred = preds_ensemble.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_ensemble.drop('stars', axis=1, inplace=True)
preds_ensemble.columns = ['review_id', 'stars']
preds_ensemble.to_csv('submission_0503_h_1.csv', index=None)



# 5 - 1.29
p_combined = sqlContext.createDataFrame(model_1_rf,['review_id', 'pred'])
p_combined = p_combined.toPandas()

preds_final = preds.merge(p_combined, on=['review_id'], how='outer')
preds_final.pred = preds_final.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_final.drop('stars', axis=1, inplace=True)
preds_final.columns = ['review_id', 'stars']
# preds_final.to_csv('submission_0503_i.csv', index=None)


preds_ensemble = preds_final.merge(p_als, on=['review_id'], how='outer')
preds_ensemble.pred = preds_ensemble.apply(lambda x: x.stars if np.isnan(x.pred) else x.pred, axis=1)
preds_ensemble.drop('stars', axis=1, inplace=True)
preds_ensemble.columns = ['review_id', 'stars']
preds_ensemble.to_csv('submission_0503_i_1.csv', index=None)