#! /user/bin/env python
from mrjob.job import MRJob
import csv
#cols='Name,lastname,JobTitle,AgencyID,Agency,HireDate,AnnualSalary,GrossPay'
#some of the records do not have the gross income! so when we split it has 7 enrtry instead of 8!
#for some records there is , at the end of the record which causes an empty string in the end of the line.split list
class salaryMax(MRJob): 
    def mapper(self,_,line):
           row=[a.strip() for a in csv.reader([line]).next()]
           #yield the salary:
           yield 'salary', round(float(row[5][1:]),2) #first element is dollar sign
           try:
               yield 'gross',round(float(row[6][1:]),2)
           except ValueError:
               self.increment_counter('warn','missing gross',1)
#          
    def reducer(self, key,values):
            topten=[]
            #for salary and gross compute the top 10:
            for p in values:
                topten.append(p)
                topten.sort()
                topten=topten[-10:]
            p=str(topten).strip()
            #for p in topten:
            yield key,p
    #combiner=reducer
if __name__=='__main__':
    salaryMax.run()
