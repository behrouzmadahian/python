#!/usr/bin/env python
from pyspark import SparkContext
def main():
   sc = SparkContext(appName='SparkWordCount')
   #reading from local. if from hdfs remove 'file://'
   f = sc.textFile('file:///home/cloudera/Spark_with_python/w?.txt')
   counts = f.flatMap(lambda line : line.split()).map(lambda word: (word,1)).reduceByKey(lambda a,b:a+b).sortBy(lambda (word, count): count).coalesce(1)
   #counts.collect()
   counts.saveAsTextFile('file:///home/cloudera/Spark_with_python/wCount.out')
   sc.stop()

if __name__ == '__main__':
    main()

#execute it using:
#$spark-submit --master local wordCount.py
