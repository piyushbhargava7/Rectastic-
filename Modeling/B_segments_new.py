
from A_load_data_and_segments import *

############## Importing in Spark Environment ##############################
userf_newfeatures = sc.textFile("yelp_usr_trtst1.csv")

busf_newfeatures = sc.textFile('yelp_bus_trtst1.csv')

############### Parsing user features ###################################
def parse_user(line):
    pieces = line.strip().split(',')
    male = int(pieces[2])
    female = int(pieces[3])
    return {"user_id": pieces[0], "u_name": pieces[1], "u_male":male, "u_female": female}

userf_newfeatures = userf_newfeatures.filter(lambda x: "user_id" not in x)
userf_newfeatures = userf_newfeatures.map(lambda line:parse_user(line))


schema = StructType([StructField("user_id", StringType(), True),
                     StructField("u_name", StringType(), True),
                     StructField("u_male", IntegerType(), True),
                     StructField("u_female", IntegerType(), True)])

user_newf= sqlContext.createDataFrame(userf_newfeatures, schema)

user_newf= user_newf.distinct()

################## Parsing business features ###################################

def parse_bus(line):
    pieces = line.strip().split(',')
    clus_ll = float(pieces[1])
    clus_zip = float(pieces[2])
    cat_avg = float(pieces[3])
    east = int(pieces[4])
    west = int(pieces[5])
    north = int(pieces[6])
    south = int(pieces[7])
    return {"business_id": pieces[0], "avg_stars_cluster_ll": clus_ll,
            "avg_stars_cluster_zip":clus_zip, "b_cat_avg": cat_avg,
            "b_strt_east":east, "b_strt_west": west, "b_strt_north": north,
            "b_strt_south": south}

busf_newfeatures = busf_newfeatures.filter(lambda x: "business_id" not in x)
busf_newfeatures = busf_newfeatures.map(lambda line:parse_bus(line))


schema = StructType([StructField("business_id", StringType(), True),
                     StructField("avg_stars_cluster_ll", FloatType(), True),
                     StructField("avg_stars_cluster_zip", FloatType(), True),
                     StructField("b_cat_avg", FloatType(), True),
                     StructField("b_strt_east", IntegerType(), True),
                     StructField("b_strt_west", IntegerType(), True),
                     StructField("b_strt_north", IntegerType(), True),
                     StructField("b_strt_south", IntegerType(), True)])

bus_newf= sqlContext.createDataFrame(busf_newfeatures, schema)


bus_newf.registerTempTable("bus_newf")
user_newf.registerTempTable("user_newf")




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

########################################################################################################
# SEGMENT 1 - Known user and known business in training data - contains both 'user' and 'business' 'stars'
########################################################################################################

# Creating a single train data set for reviews that have both user and business information available
# 215879
merged_train_ku_kb_all_stars = sqlContext.sql("SELECT A.review_id, A.stars, B.average_stars as u_stars, "
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, C.stars as b_stars,"
                                    "D.*, E.*, F.* , G.*, H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")



########################################################################################################
# SEGMENT 2 - Known user and known business in training data - contains 'business' 'stars' only
########################################################################################################

# 229907
merged_train_ku_kb_b_stars = sqlContext.sql("SELECT A.review_id, A.stars, "
                                    "C.latitude, C.longitude, C.open, C.review_count as b_review_count, "
                                            "C.stars as b_stars,"
                                    "D.*, E.*, G.*, H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where C.review_count is not NULL")


########################################################################################################
# SEGMENT 3 - Known user and known business in training data - contains 'user' 'stars' only
########################################################################################################

# 215879
merged_train_ku_kb_u_stars = sqlContext.sql("SELECT A.review_id, A.stars, B.average_stars as u_stars,"
                                            "B.review_count as u_review_count, F.*,"
                                            "  H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is not NULL")


########################################################################################################
# SEGMENT 4 - Known user and known business in training data - contains no 'stars' features
########################################################################################################

# 215879
merged_train_ku_kb_no_stars = sqlContext.sql("SELECT A.review_id, A.stars,  "
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.*, G.*, H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")


########################################################################################################
# SEGMENT 5 - Known business in training data - contains no 'stars' features
########################################################################################################

# 229907
merged_train_kb_only_no_stars = sqlContext.sql("SELECT A.review_id, A.stars,C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "D.*, G.*, H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where C.review_count is not NULL")



########################################################################################################
# COMBINING ALL FINAL (TEST) DATA SETS AND CREATING SUBSETS BASED ON AVAILABLE USER AND BUSINESS INFORMATION
########################################################################################################

########################################################################################################
# SEGMENT 1 - Known user and known business in training data and test data
########################################################################################################

# 12078
merged_final_ku_kb_all_stars = sqlContext.sql("SELECT A.review_id, B.average_stars as u_stars,"
                                    "B.review_count as u_review_count, C.latitude, "
                                    "C.longitude, C.open, C.review_count as b_review_count, "
                                    "C.stars as b_stars, D.*, E.*, F.*, G.*,"
                                              " H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is not NULL and C.review_count is not NULL")



########################################################################################################
# SEGMENT 2 - UnKnown user known business in training data and test data
########################################################################################################

# 14951
merged_final_ku_kb_only_b_stars = sqlContext.sql("SELECT A.review_id, "
                                    "C.latitude, C.longitude, C.open, C.review_count as b_review_count, "
                                    "C.stars as b_stars, D.*, E.*, G.*,"
                                                 " H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is NULL and C.review_count is not NULL ")





########################################################################################################
# SEGMENT 3 - Known user Unknown business in training data and test data
########################################################################################################
# 4086
merged_final_ku_kb_only_u_stars = sqlContext.sql("SELECT A.review_id, B.average_stars as u_stars,"
                                    "B.review_count as u_review_count, F.*, "
                                                 "H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
                                    "(Select business_id, user_id, review_id from final_r) A "
                                    "LEFT JOIN "
                                    "(SELECT user_id, review_count, average_stars from train_u) B "
                                    "on A.user_id = B.user_id "
                                    "LEFT JOIN "
                                    "(SELECT business_id, latitude, longitude, open, review_count, stars from train_b) C "
                                    "on A.business_id = C.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM votes_u) F ON A.user_id = F.user_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is not NULL and C.review_count is NULL ")



########################################################################################################
# SEGMENT 4 - Known user known business in test data only
########################################################################################################
# 4767
merged_final_ku_kb_no_stars = sqlContext.sql("SELECT A.review_id, "
                                    "E.review_count as u_review_count, F.latitude, "
                                    "F.longitude, F.open, F.review_count as b_review_count, "
                                    "D.*, G.*, H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is NULL and C.review_count is NULL and"
                                              " E.review_count is not NULL and F.review_count is not NULL")



########################################################################################################
# SEGMENT 5 - UnKnown user known business in test data only
########################################################################################################

# 522
merged_final_uu_kb_no_stars = sqlContext.sql("SELECT A.review_id, F.latitude, "
                                    "F.longitude, F.open, F.review_count as b_review_count, "
                                    "D.*, G.*, H.avg_stars_cluster_ll, H.avg_stars_cluster_zip, "
                                    "H.b_cat_avg, H.b_strt_east, H.b_strt_west, H.b_strt_north, H.b_strt_south,"
                                    "I.u_male, I.u_female from "
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
                                    "LEFT JOIN "
                                    "(SELECT * FROM bus_newf) H ON A.business_id = H.business_id "
                                    "LEFT JOIN "
                                    "(SELECT * FROM user_newf) I ON A.user_id = I.user_id "
                                    "where B.review_count is NULL and C.review_count is NULL and"
                                              " E.review_count is NULL and F.review_count is not NULL")
