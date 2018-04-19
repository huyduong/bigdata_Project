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

clicks_portion = 0.001

start_time = time.time()
clicks_portion = float(sys.argv[1])
reg = int(sys.argv[2])
"""
 Simple Algoirthms
"""
# clicks sampling and splitting
spark = SparkSession.builder.appName("Lab3").master("local").config("spark.some.config.option", "some-value").getOrCreate()
sc = spark.sparkContext
clicks = spark.read.csv("./data/clicks_train.csv", header=True, mode="DROPMALFORMED")\
    .select(col("display_id").cast("int"), col("ad_id").cast("int"), col("clicked").cast("int"))

all_display_id = clicks.select("display_id").distinct().sample(False, clicks_portion, 123)
test_display_id = [_.display_id for _ in all_display_id.sample(False, 0.01, 123).collect()]
# ---
train_display_id = [_.display_id for _ in all_display_id.filter(~col("display_id").isin(test_display_id)).collect()]
# ---
train_set = clicks.filter(col("display_id").isin(train_display_id))
# ---
valid_set = clicks.filter(col("display_id").isin(test_display_id))

# calculation of appearances and clicked events of each ad_id
clicks_on_ad = train_set.where(train_set.clicked == 1).groupBy("ad_id").agg(count("display_id").alias("clicks"))
apps_on_ad = train_set.groupBy("ad_id").agg(count("display_id").alias("views"))
crt_ad = apps_on_ad.join(clicks_on_ad, "ad_id", "right_outer").withColumn("crt", col("clicks")/(col("views")+reg))


# load promoted content and document meta, then join
promoted_content = spark.read.csv("./data/promoted_content.csv", header=True, mode="DROPMALFORMED")\
    .select(col("ad_id").cast("int"), col("document_id").cast("int")\
            , col("campaign_id").cast("int"), col("advertiser_id").cast("int"))

documents_meta = spark.read.csv("./data/documents_meta.csv", header=True, mode="DROPMALFORMED")\
    .select(col("source_id").cast("int"), col("document_id").cast("int")\
            , col("publisher_id").cast("int"))

ad_prof = promoted_content.join(documents_meta, "document_id", "left_outer")

ad_prof_with_crt = ad_prof.join(crt_ad, "ad_id", "right_outer")\
            .drop("views", "clicks").collect()

ad_prof = ad_prof.collect()

# valid the prediction
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

#mapk(test_list)

display_items_set = valid_set.select("display_id", struct("ad_id", "clicked").alias("newcol")) \
    .groupBy('display_id').agg(collect_list("newcol").alias("items"))


def srt(items):
    ad_ids = [ [x["ad_id"], x["clicked"]] for x in items]
    ad_ids.sort(key=lambda ad_id: get_prob(ad_id))
    return ad_ids

def get_prob(k):
    ad_id = int(k[0])
    this_ad_id_info = [x for x in ad_prof_with_crt if x["ad_id"] == ad_id]
    #print(this_ad_id_info)
    if len(this_ad_id_info) == 0:
        return 0
    if this_ad_id_info[0]["crt"] > 0:
        return this_ad_id_info[0]["crt"]
    selected_ad = this_ad_id_info[0]
    fea_keys = selected_ad.keys()
    similar_list = []
    for x in ad_prof_with_crt:
        similar_list.append(0)
        for k in fea_keys:
            if x[k] == selected_ad[k]:
                similar_list[-1] += 1
        similar_list[-1] = (similar_list[-1], x["crt"])
    #
    similar_list = sorted(similar_list, key = lambda x : x[0])
    if len(similar_list) > 10:
        similar_list = similar_list[0:10]
    #
    sum_crt = 0
    for x in similar_list:
        sum_crt += x[1]
    #
    return sum_crt / len(similar_list)


sort_udf = udf(srt)
ordered_set = display_items_set.withColumn("sorted_ad_id", sort_udf("items"))

ordered_set.show(10)

mapk_udf = udf(mapk)
ordered_set_mapk = ordered_set.withColumn("mapk", mapk_udf(col("sorted_ad_id")))
sum_cnt = ordered_set_mapk.rdd.map(lambda x: (float(x["mapk"]), 1.0))\
    .reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))

print("MAPK() = " + str(sum_cnt[0]/sum_cnt[1]))
#print(ordered_set_mapk.rdd.map(lambda x: float(x["mapk"])).reduce(lambda x,y: x+y))
  
