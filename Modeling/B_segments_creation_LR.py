from load_data_and_segments import *

################################################################################
# EXTRACTING FEATURES - BUSINESS CATEGORIES FROM BUSINESS DATASETS
################################################################################

# Combining the 'categories' field from train and final business datasets into 1 dataset
categories_total = train_b.map(lambda x: (x['business_id'], x['categories'])).union(
        final_b.map(lambda x: (x['business_id'], x['categories'])))

categories_df_total = sqlContext.createDataFrame(categories_total)
# schema_categories = StructType([StructField("business_id", StringType(), True),
#                                 StructField("categories", StringType(), True)])
# categories_df_total = sqlContext.createDataFrame(categories_total, schema_categories)

categories_pandas_total = categories_df_total.toPandas()

# categories_df_total.select(["categories"]).map(lambda x: x.categories).collect()
# categories_df_total.select(["categories"]).map(lambda x: [(v,1) for k,v in enumerate(x.categories)]).collect()
#
# cf = dict()
# cf =categories_df_total.map(lambda x: (x["business_id"],(x.select(["categories"]).map(lambda x: ()) : (x.categories,1)))).collect()


# Creating dummy variables for all categories
categories_pandas_total_dummy = categories_pandas_total['_2'].str.join(sep='*').str.get_dummies(sep='*')

categories_pandas_total_2 = sqlContext.createDataFrame(
        categories_pandas_total.rename(columns={'_1': 'business_id'}).drop('_2', 1).join(categories_pandas_total_dummy))

categories_pandas_total_2.registerTempTable("categories_total")

# Infer the schema, and register the DataFrame as a table.

########################################################################################################
# COMBINING ALL TRAIN DATA SETS AND CREATING SUBSETS BASED ON AVAILABLE USER AND BUSINESS INFORMATION
########################################################################################################

# Creating a single train data set for reviews that have both user and business information available
# 215879
merged_train_ku_kb_all_stars = sqlContext.sql("SELECT A.review_id, A.stars - 1 as stars, B.average_stars - 1 as u_stars, "
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, C.stars - 1  as b_stars,"
                                    "D.*, E.*, F.* , G.* from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count, average_stars from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count, stars "
                                    "from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_b) E ON A.business_id = E.bus_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_u) F ON A.user_id = F.user_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM train_checkin_new) G ON A.business_id = G.b_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")


# 229907
merged_train_ku_kb_b_stars = sqlContext.sql("SELECT A.review_id, A.stars - 1 as stars, "
                                    "C.latitude, C.longitude, C.open, C.review_count as b_review_count, "
                                            "C.stars - 1 as b_stars,"
                                    "D.*, E.*, G.* from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count, stars "
                                    "from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_b) E ON A.business_id = E.bus_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM train_checkin_new) G ON A.business_id = G.b_id "
                                    "where C.review_count is not NULL")

# 215879
merged_train_ku_kb_u_stars = sqlContext.sql("SELECT A.review_id, A.stars - 1 as stars, B.average_stars - 1 as u_stars,"
                                            "B.review_count as u_review_count, F.* from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count, average_stars from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count, stars "
                                    "from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_u) F ON A.user_id = F.user_id "
                                    "where B.review_count is not NULL")

# 215879
merged_train_ku_kb_no_stars = sqlContext.sql("SELECT A.review_id, A.stars - 1 as stars,  "
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.*, G.* from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count "
                                    "from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM train_checkin_new) G ON A.business_id = G.b_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")

# 229907
merged_train_kb_only_no_stars = sqlContext.sql("SELECT A.review_id, A.stars - 1 as stars,C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.*, G.* from "
                                    "(Select business_id, user_id, review_id, stars from train_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM train_checkin_new) G ON A.business_id = G.b_id "
                                    "where C.review_count is not NULL")



########################################################################################################
# COMBINING ALL FINAL (TEST) DATA SETS AND CREATING SUBSETS BASED ON AVAILABLE USER AND BUSINESS INFORMATION
########################################################################################################


# 12078
merged_final_ku_kb_all_stars = sqlContext.sql("SELECT A.review_id, B.average_stars - 1 as u_stars,"
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "C.stars - 1 as b_stars, D.*, E.*, F.*, G.* from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count, average_stars from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count, stars from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_b) E ON A.business_id = E.bus_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_u) F ON A.user_id = F.user_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM train_checkin_new) G ON A.business_id = G.b_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")


# 14951
merged_final_ku_kb_only_b_stars = sqlContext.sql("SELECT A.review_id, "
                                    "C.latitude, C.longitude, C.open, C.review_count as b_review_count, "
                                    "C.stars - 1 as b_stars, D.*, E.*, G.* from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count, average_stars from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count, stars from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_b) E ON A.business_id = E.bus_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM train_checkin_new) G ON A.business_id = G.b_id "
                                    "where B.review_count is NULL and C.review_count is not NULL ")


# 4086
merged_final_ku_kb_only_u_stars = sqlContext.sql("SELECT A.review_id, B.average_stars - 1 as u_stars,"
                                    "B.review_count as u_review_count, F.* from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count, average_stars from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count, stars from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_u) F ON A.user_id = F.user_id "
                                    "where B.review_count is not NULL and C.review_count is NULL ")


# 4767
merged_final_ku_kb_no_stars = sqlContext.sql("SELECT A.review_id, "
                                    "E.review_count as u_review_count, F.latitude, "
                                    "F.longitude, F.open, F.review_count as b_review_count, "
                                    "D.*, G.* from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from final_u) E "
                                    "on A.user_id = E.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from final_b) F "
                                    "on A.business_id = F.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM final_checkin_new) G ON A.business_id = G.b_id "
                                    "where B.review_count is NULL and C.review_count is NULL and"
                                              " E.review_count is not NULL and F.review_count is not NULL")

# 522
merged_final_uu_kb_no_stars = sqlContext.sql("SELECT A.review_id, F.latitude, "
                                    "F.longitude, F.open, F.review_count as b_review_count, "
                                    "D.*, G.* from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM categories_total) D ON A.business_id = D.business_id "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count from final_u) E "
                                    "on A.user_id = E.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count from final_b) F "
                                    "on A.business_id = F.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM final_checkin_new) G ON A.business_id = G.b_id "
                                    "where B.review_count is NULL and C.review_count is NULL and"
                                              " E.review_count is NULL and F.review_count is not NULL")



