import sys
#import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.functions import array_contains
from pyspark.sql.functions import udf
import random

import time
start_time = time.time()
clicks_portion = float(sys.argv[1])
reg = int(sys.argv[2])
"""
 Simple Algoirthms
"""
# clicks sampling
spark = SparkSession.builder.appName("Lab3").master("local").config("spark.some.config.option", "some-value").getOrCreate()
sc = spark.sparkContext
clicks = spark.read.csv("./data/clicks_train.csv", header=True, mode="DROPMALFORMED")\
    .select(col("display_id").cast("int"), col("ad_id").cast("int"), col("clicked").cast("int"))

all_display_id = clicks.select("display_id").distinct().sample(False, clicks_portion, 123)
test_display_id = [_.display_id for _ in all_display_id.sample(False, 0.2, 123).collect()]
# print("len(test_display_id) = ", len(test_display_id) )
# print("len(all_display_id) = ", all_display_id.count())
# ---
train_display_id = [_.display_id for _ in all_display_id.filter(~col("display_id").isin(test_display_id)).collect()]
# ---
train_set = clicks.filter(col("display_id").isin(train_display_id))
# ---
valid_set = clicks.filter(col("display_id").isin(test_display_id))

# calculation of appearances and clicked events of each ad_id
clicks_on_ad = train_set.where(train_set.clicked == 1).groupBy("ad_id").agg(count("display_id"))
clicks_on_ad = map(lambda x: x.asDict(), clicks_on_ad.collect())
# ---
clicks_on_ad = {ad_count['ad_id']: ad_count['count(display_id)'] for ad_count in clicks_on_ad}
# ---
apps_on_ad = train_set.groupBy("ad_id").agg(count("display_id"))
apps_on_ad = map(lambda x: x.asDict(), apps_on_ad.collect())
apps_on_ad = {ad_apps['ad_id']: ad_apps['count(display_id)'] for ad_apps in apps_on_ad}

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
    #ad_ids = map(int, x.split())
    #print(items)
    ad_ids = [ [x["ad_id"], x["clicked"]] for x in items]
    ad_ids.sort(key=lambda ad_id: get_prob(ad_id))
    #print(items, ad_ids)
    return ad_ids

def get_prob(k):
    ad_id = k[0]
    if ad_id not in clicks_on_ad.keys():
        return 0
    #print(k, ad_id, clicks_on_ad[ad_id]/(float(apps_on_ad[ad_id])))
    return clicks_on_ad[ad_id]/(float(apps_on_ad[ad_id] + reg))

sort_udf = udf(srt)
ordered_set = display_items_set.withColumn("sorted_ad_id", sort_udf("items"))
#ordered_set.show(10)

mapk_udf = udf(mapk)
ordered_set_mapk = ordered_set.withColumn("mapk", mapk_udf(col("sorted_ad_id")))
sum_cnt = ordered_set_mapk.rdd.map(lambda x: (float(x["mapk"]), 1.0))\
    .reduce(lambda x,y: (x[0]+y[0], x[1]+y[1]))

print("MAPK() = " + str(sum_cnt[0]/sum_cnt[1]))
#print(ordered_set_mapk.rdd.map(lambda x: float(x["mapk"])).reduce(lambda x,y: x+y))
print("--- %s seconds ---" % (time.time() - start_time))
