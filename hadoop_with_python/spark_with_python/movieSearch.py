#!/usr/bin/env python
from pyspark import SparkContext
import sys
def main():
   #ensuring the user provides keyword to search:
   if len(sys.argv) !=2:
       sys.stderr.write('Usage: {} <search_term>'.format(sys.argv[0]))
       sys.exit
   sc = SparkContext(appName='SparkMovieSearch')
   #reading from local. if from hdfs remove 'file://'
   f = sc.textFile('file:///home/cloudera/Spark_with_python/MovieDB.txt')
   #broadcasting te keyword to all worker nodes:
   requested_keyword= sc.broadcast(sys.argv[1].lower())
   titles = f.map(lambda line : line.split('|')[1].lower())
   #titles1=f.flatMap(lambda line: line.split('|').lower())
   matches =titles.filter(lambda x: requested_keyword.value in x).distinct()
   matches1=matches.map(lambda x: [x])
   for title in matches1.toLocalIterator():
       print title
   matches.saveAsTextFile('file:///home/cloudera/Spark_with_python/movieSearch.out')
   sc.stop()

if __name__ == '__main__':
    main()

#execute it using:
#$spark-submit --master local wordCount.py >wordcount.log
