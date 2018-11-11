#!/usr/bin/env python
import sys
curr_word=None
total=0
for line in sys.stdin:
     key=line.split('\t')[0]
     if curr_word !=key:
         if curr_word:
             print '%s\t%d'%(curr_word,total)
         curr_word=key
         total=1
     else:
         total+=1
#outputting the count for the last word:
print '%s\t%d'%(curr_word,total)
##############################################
#testing the mapper and reducer locally:
#make the files executable:
#$chmod +x wordCountMapper.py
#$chmod +x wordCountReducer.py
#cat w1.txt |./wordCountMapper.py | sort |./wordCountReducer.py 
#################################################################
#running on Hadoop.: when we ensure both mapper and reducers work properly we can run them on cluster using
#hadoop streaming utility as follows:
#$hadoop jar /usr/lib/hadoop-mapreduce/hadoop-streaming.jar -input /user/cloudera/someText -output /user/cloudera/someTextW_count -mapper /home/cloudera/snakebite/wordCountMapper.py -reducer /home/cloudera/snakebite/wordCountReducer.py 

#if output directory exists we get an error!
