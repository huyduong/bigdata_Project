import sys
#import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.functions import array_contains
from pyspark.sql.functions import udf
from pyspark.sql.functions import struct
from pyspark.sql.types import *
import random
import time
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
from pyspark.sql.functions import monotonically_increasing_id

clicks_portion = 0.00001
reg = 0

start_time = time.time()
clicks_portion = float(sys.argv[1])
reg = int(sys.argv[2])
"""
 ALS Algoirthms
"""
# clicks sampling and splitting
spark = SparkSession.builder.appName("Lab3").master("local").config("spark.some.config.option", "some-value").getOrCreate()
sc = spark.sparkContext

clean_clicks = spark.read.csv("./data/clicks_train.csv", header=True, mode="DROPMALFORMED")\
    .select(col("display_id").cast("int"), col("ad_id").cast("int"), col("clicked").cast("int"))

clean_events = spark.read.csv("./data/events.csv", header=True, mode="DROPMALFORMED")\
    .select(col("display_id").cast("int"), col("uuid").alias("uuid_str"))


all_display_id = clean_clicks.select("display_id").distinct().sample(False, clicks_portion, 123)

test_display_id = all_display_id.sample(False, 0.2, 123)
# ---
train_display_id = all_display_id.subtract(test_display_id)

clicks = clean_clicks.join(train_display_id, "display_id", "inner")
events = clean_events.join(train_display_id, "display_id", "inner")

int_user_id = events.select("uuid_str").distinct().rdd.zipWithIndex()\
                .map(lambda (row, index) : Row(uuid_str=row["uuid_str"], uuid=index))\
                .toDF()

events = events.join(int_user_id, "uuid_str", "inner").drop("uuid_str")

user_ad_click = clicks.where(col("clicked") == 1).join(events, "display_id", "inner")\
                .select("uuid", "ad_id", "display_id")\
                .join(all_display_id.select("display_id"), "display_id", "inner")


all_users = events.select("uuid").distinct()
all_ad_ids = clicks.select("ad_id").distinct()

train_ratings = user_ad_click.rdd\
                .map(lambda row: Rating( int(row["uuid"]), int(row["ad_id"]), float(1.0)))\
                .cache()

pred_input = all_users.crossJoin(all_ad_ids)\
            .subtract(user_ad_click.select("uuid", "ad_id")).rdd\
            .map(lambda row: (row["uuid"], row["ad_id"]))\
            .cache()

rank = 5
numIterations = 10
model = ALS.train(train_ratings, rank, numIterations)

pred_user_ad_id_prob = model.predictAll(pred_input)

all_ctr = train_ratings.union(pred_user_ad_id_prob)\
                    .map(lambda r : (r[1], (r[2], 1)))\
                    .reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))\
                    .map(lambda x: (x[0], x[1][0]/x[1][1]))\
                    .collect()


#mapk(test_list)

display_items_set = clean_clicks.join(test_display_id, "display_id", "inner")\
                    .select("display_id", struct("ad_id", "clicked").alias("newcol")) \
                    .groupBy('display_id').agg(collect_list("newcol").alias("items"))

def srt(items):
    ad_ids = [ [x["ad_id"], x["clicked"]] for x in items]
    ad_ids.sort(key=lambda ad_id: get_prob(ad_id))
    return ad_ids

def get_prob(k):
    ad_id = int(k[0])
    for x in all_ctr:
        if x[0] == ad_id:
            return x[1]
    return 0

sort_udf = udf(srt)
ordered_set = display_items_set.withColumn("sorted_ad_id", sort_udf("items"))

ordered_set.show(10)

def mapk(order_ad_click):
    #print(order_ad_click)
    k = 12
    actual = []
    predicted = []
    for ad_click in order_ad_click:
        if ad_click[1] == 1:
            actual.append(ad_click[0])
        predicted.append(ad_click[0])
        
    if len(predicted)>k:
        predicted = predicted[:k]
        
    score = 0.0
    num_hits = 0.0
    
    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)
        
    if not actual:
        return 0.0
    
    min_len = len(actual)
    if min_len > k:
        min_len = k
    
    return score / (min_len)

mapk_udf = udf(mapk)
ordered_set_mapk = ordered_set.withColumn("mapk", mapk_udf(col("sorted_ad_id")))
sum_cnt = ordered_set_mapk.rdd.map(lambda x: (float(x["mapk"]), 1.0))\
    .reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))

print("MAPK() = " + str(sum_cnt[0]/sum_cnt[1]))












# # calculation of appearances and clicked events of each ad_id
# clicks_on_ad = train_set.where(train_set.clicked == 1).groupBy("ad_id").agg(count("display_id").alias("clicks"))
# apps_on_ad = train_set.groupBy("ad_id").agg(count("display_id").alias("views"))
# ctr_ad = apps_on_ad.join(clicks_on_ad, "ad_id", "right_outer").withColumn("ctr", col("clicks")/col("views"))


# # load promoted content and document meta, then join
# promoted_content = spark.read.csv("./data/promoted_content.csv", header=True, mode="DROPMALFORMED")\
#     .select(col("ad_id").cast("int"), col("document_id").cast("int")\
#             , col("campaign_id").cast("int"), col("advertiser_id").cast("int"))

# documents_meta = spark.read.csv("./data/documents_meta.csv", header=True, mode="DROPMALFORMED")\
#     .select(col("source_id").cast("int"), col("document_id").cast("int")\
#             , col("publisher_id").cast("int"))

# ad_prof = promoted_content.join(documents_meta, "document_id", "left_outer")

# ad_prof_with_ctr = ad_prof.join(ctr_ad, "ad_id", "right_outer")\
#             .drop("views", "clicks")



# # Make predict array
# unique_user_id = 1


# # collect clean datas


# # ALS prediction
# rank = 5
# numIterations = 10
# model = ALS.train(train_ratings, rank, numIterations)

# pred_set = train_set.select("ad_id").distinct().union(valid_set.select("ad_id").distinct())\
#                 .subtract(ad_prof_with_ctr.select("ad_id")).rdd\
#                 .map(lambda x: Rating(unique_user_id, x["ad_id"]))\
#                 .cache()

# pred_ctr_ad_id = model.predictAll(pred_set) 

# all_ctr = train_ratings.union(pred_ctr_ad_id).collect()

# ad_prof = ad_prof.collect()
# ad_prof_with_ctr = ad_prof_with_ctr.collect()
# # valid the prediction
# def mapk(order_ad_click):
#     #print(order_ad_click)
#     k = 12
#     actual = []
#     predicted = []
#     for ad_click in order_ad_click:
#         if ad_click[1] == 1:
#             actual.append(ad_click[0])
#         predicted.append(ad_click[0])
        
#     if len(predicted)>k:
#         predicted = predicted[:k]
        
#     score = 0.0
#     num_hits = 0.0
    
#     for i,p in enumerate(predicted):
#         if p in actual and p not in predicted[:i]:
#             num_hits += 1.0
#             score += num_hits / (i+1.0)
        
#     if not actual:
#         return 0.0
    
#     min_len = len(actual)
#     if min_len > k:
#         min_len = k
    
#     return score / (min_len)

# #mapk(test_list)

# display_items_set = clicks.joint(test_display_id, "display_id", "inner")\
#                     .select("display_id", struct("ad_id", "clicked").alias("newcol")) \
#                     .groupBy('display_id').agg(collect_list("newcol").alias("items"))

# def srt(items):
#     ad_ids = [ [x["ad_id"], x["clicked"]] for x in items]
#     ad_ids.sort(key=lambda ad_id: get_prob(ad_id))
#     return ad_ids

# def get_prob(k):
#     ad_id = int(k[0])
#     for x in all_ctr:
#         if x[0] == ad_id:
#             return x[1]
#     return 0

# sort_udf = udf(srt)
# ordered_set = display_items_set.withColumn("sorted_ad_id", sort_udf("items"))

# ordered_set.show(10)

# mapk_udf = udf(mapk)
# ordered_set_mapk = ordered_set.withColumn("mapk", mapk_udf(col("sorted_ad_id")))
# sum_cnt = ordered_set_mapk.rdd.map(lambda x: (float(x["mapk"]), 1.0))\
#     .reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))

# print("MAPK() = " + str(sum_cnt[0]/sum_cnt[1]))
# #print(ordered_set_mapk.rdd.map(lambda x: float(x["mapk"])).reduce(lambda x,y: x+y))
  
