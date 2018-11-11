#!/usr/bin/env python
import sys
#read each line from stdin
for line in sys.stdin:
   words=line.split()
#generate the count for each word:
   for word in words:
         #write the key-value pair to stdout to be processed by the reducer
         #the key is anything before the first tap and the value is anything after the first tab character!!!
         print '%s\t%d'%(word,1)
         #OR:
         #print '{0}\t{1}'.format(word,1)
#check the mapper is OK from commandline:
#make the .py file executable!  :  $chmod +x wordCountMapper.py
#now:
#cat testfile.txt |./wordcountMapper.py| sort
       
