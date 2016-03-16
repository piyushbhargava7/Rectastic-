__author__ = 'chhavi21'

from load_data import *
import string
import re
from collections import Counter
from collections import defaultdict

################################################################################
# SENTIMENT SCORE
################################################################################

stopWords = ['able', 'about', 'across', 'after', 'all', 'almost', 'also',
             'among', 'and', 'any', 'are', 'because', 'been', 'but', 'can',
             'cannot', 'could', 'dear', 'did', 'does', 'either', 'else',
             'ever', 'every', 'for', 'from', 'get', 'got', 'had', 'has',
             'have', 'her', 'here' 'hers', 'him', 'his', 'how', 'however',
             'into', 'its', 'just', 'least', 'let', 'like', 'likely', 'may',
             'might', 'most', 'must', 'neither', 'nor', 'not', 'off', 'often',
             'only', 'other', 'our', 'own', 'put', 'rather', 'said', 'say',
             'says', 'she', 'should', 'since', 'some', 'such', 'than', 'that',
             'the', 'their', 'them','then', 'there', 'these', 'they', 'this',
             'tis', 'too', 'twas', 'wants', 'was', 'were', 'what', 'when',
             'where', 'which', 'while', 'who', 'whom', 'why', 'will', 'with',
             'would', 'yet', 'you', 'your', 'www', 'http', 'women', 'males',
             'each', 'done', 'see', 'before', 'each', 'irs', 'ira', 'hal', 'ham', 'isn']

stopWords = Counter(stopWords)

def clean_lines(line):
    line = re.sub(re.compile(r'[^a-z]'), ' ', line.lower())
    line = line.split()
    line = [l for l in line if l not in stopWords and len(l) > 2]
    return line

from collections import Counter
scores = Counter()

with open(PATH+'AFINN-111.txt', 'r') as file:
    for line in file:
        data = line.split('\t')
        scores[data[0]] = int(data[1])


def getscore(text):
    text = clean_lines(text)
    text_score = 0
    cnt = 0
    for w in text:
        if w in scores:
            text_score += scores[w]
            cnt += 1
    return text_score/float(cnt) if cnt != 0 else 0


review_score_user = review.map(lambda x: (x.user_id, getscore(x.text)))
review_score_user = sqlContext.createDataFrame(review_score_user,
                                               ['user_id', 'review_score_user'])
review_score_user = review_score_user.groupBy('user_id').agg({'review_score_user': 'mean'}).map(lambda x: (x[0], x[1]))
review_score_user = sqlContext.createDataFrame(review_score_user, ['user_id', 'review_score_user'])

review_score_business = review.map(lambda x: (x.business_id, getscore(x.text)))
review_score_business = sqlContext.createDataFrame(review_score_business,
                                                   ['business_id', 'review_score_business'])
review_score_business = review_score_business.groupBy('business_id').agg({'review_score_business': 'mean'}).map(lambda x: (x[0], x[1]))
review_score_business = sqlContext.createDataFrame(review_score_business, ['business_id', 'review_score_business'])


################################################################################
# BUILD FEATURES
################################################################################


from collections import defaultdict
from collections import Counter

def get_cat_score(lst, stars, calc_categories_counts):
    d = defaultdict(int)
    for l in lst:
        if l in calc_categories_counts:
            d[l] += stars
    return d


def dict_red(x,y):
    x = dict(x)
    y = dict(y)
    return {k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y)}


def top_categories(x,y):
    return {k: x.get(k, 0)/y.get(k, 1) for k in set(x) & set(y)}



################
# TOP CATEGORIES
################
merged_Train = business.join(review, on='business_id').join(user, on='user_id').repartition(16)
merged_Train.printSchema()
# merged_Train.count() #24254911
# review.count() #25770065


calc_categories_counts = merged_Train.map(lambda x: Counter(x.categories)).reduce(lambda a,b: a+b)
calc_categories_counts = dict(calc_categories_counts.most_common(421))
top_categories_stars = merged_Train.map(lambda x: get_cat_score(x.categories,
                                                                x.r_stars,
                                                                calc_categories_counts))
top_categories_stars = top_categories_stars.reduce(dict_red)
top_categories = top_categories(top_categories_stars, calc_categories_counts) # gets avg score for each category


########################
# TOP CATEGORIES BUS AVG
########################
## Iterate through every record in the training data set and if the category matches a top category,
# then add the review stars to that category's total stars
calc_categories_counts_avg = business.map(lambda x: Counter(x.categories)).reduce(lambda a,b: a+b)
top_categories_stars_avg = business.map(lambda x: get_cat_score(x.categories,
                                                            x.b_stars,
                                                            calc_categories_counts_avg))
top_categories_stars_avg = top_categories_stars_avg.reduce(dict_red)
top_categories_bus_avg = top_categories(top_categories_stars_avg, calc_categories_counts_avg)

top_categories_stars_avg.first()


########################
# map user gender
########################

names_map = {}
with open('data/mf.txt','r') as file:
    lines = file.readlines()
    for line in lines:
        line  = line.replace('\r\n', '').split('\t')
        names_map[line[0]] = line[1]

def get_gender(name):
    if name in names_map:
        return names_map[name]
    else: return 'f'


u_names = user.map(lambda x: (get_gender(x.u_name.upper()), x.u_average_stars)).reduceByKey(lambda a,b: a+b)
u_count = user.map(lambda x: (get_gender(x.u_name.upper()),1)).reduceByKey(lambda a,b: a+b)
u_names = u_names.collectAsMap()
u_count = u_count.collectAsMap()

# the gender means are very similar so of no use in baseline model
# {'f': 3.7494911888020552, 'm': 3.7363023360794476}
gender_means = {'f': u_names['f']/u_count['f'], 'm': u_names['m']/u_count['m']}

########################
# map business zipcode
########################


business.printSchema()

business_sum = business.map(lambda x: ( (x.full_address.split(' ')[-1]
                        if x.full_address.split(' ')[-1].isdigit() else 'unknown'),
                         x.b_stars)).reduceByKey(lambda a,b: a+b)

business_cnt = business.map(lambda x: ( (x.full_address.split(' ')[-1]
                        if x.full_address.split(' ')[-1].isdigit() else 'unknown'),
                         1)).reduceByKey(lambda a,b: a+b)

business_zipcode = business_sum.join(business_cnt).map(lambda x: (x[0], x[1][0]/x[1][1]))
business_zipcode = business.map(lambda x: ((x.full_address.split(' ')[-1]
                        if x.full_address.split(' ')[-1].isdigit() else 'unknown'),
			 			(x.b_stars,x.business_id))).join(business_zipcode).map(lambda x: (x[1][0][1], x[1][1]))

