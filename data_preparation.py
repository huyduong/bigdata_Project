import sys
#import pyspark
from pyspark import SparkContext, SparkConf
from pyspark.sql import *
from pyspark.sql import Row
from pyspark.sql.functions import *
from pyspark.sql.functions import array_contains
import random

# page_views_sample
page_views_sample = spark.read.csv("./data/page_views_sample.csv", header=True, mode="DROPMALFORMED")
page_views_sample.show(10)

# unique user id
number_uuid = page_views_sample.select("uuid").distinct().count()
print("# views : " + str(page_views_sample.count()))
print("# uuid : " + str(number_uuid))

# clicks sampling
clicks = spark.read.csv("./data/clicks_train.csv", header=True, mode="DROPMALFORMED")
number_displays = clicks.select("display_id").distinct().count()
print("# display_id : " + str(number_displays))
number_ads = clicks.select("ad_id").distinct().count()
print("# ad_id : " + str(number_ads))
clicks.show(10)


# events 
events = spark.read.csv("./data/events.csv", header=True, mode="DROPMALFORMED")
number_display_in_events = events.select("display_id").distinct().count()
print("# number_display_in_events : " + str(number_display_in_events))