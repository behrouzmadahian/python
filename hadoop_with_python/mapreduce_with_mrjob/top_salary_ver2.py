#! /user/bin/env python
from mrjob.job import MRJob
#import csv
#cols='Name,lastname,JobTitle,AgencyID,Agency,HireDate,AnnualSalary,GrossPay'
#some of the records do not have the gross income! so when we split it has 7 enrtry instead of 8!
#for some records there is , at the end of the record which causes an empty string in the end of the line.split list
class salaryMax(MRJob): 
    def mapper(self,_,line):
           row=line.strip().split(',')
           if len(row)==8:
               yield 'salary', float(row[6][1:]) #first element is dollar sign
               try:
                  yield 'gross',float(row[7][1:])
               except ValueError:
                  self.increment_counter('warn','missing gross',1)
           if len(row)==9:
               yield 'salary', float(row[7][1:]) #first element is dollar sign
               try:
                  yield 'gross',float(row[8][1:])
               except ValueError:
                  self.increment_counter('warn','missing gross',1)
           if len(row)==7:
              yield 'salary',float(row[6][1:])
    def reducer(self, key,values):
            topten=[]
            #for salary and gross compute the top 10:
            for p in values:
                topten.append(p)
                topten.sort()
                topten=topten[-10:]
            for p in topten:
                 yield key,p
    #combiner=reducer
if __name__=='__main__':
    salaryMax.run()
