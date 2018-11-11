from mrjob.job import MRJob
from mrjob.step import MRStep
import csv
cols='Name,JobTitle,AgencyID,Agency,HireDate,AnnualSalary,GrossPay'.split(',')
class salaryMax(MRJob): 
    def mapper(self,_,line):
           #convert eachline into a dictonary:
           row=dict(zip(cols,[a.strip() for a in csv.reader([line]).next()]))
           #yield the salary:
           yield 'salary', (float(row['AnnualSalary'][1:]),line) #first element is dollar sign!
#           yield gross pay:
           try:
               yield 'gross',(float(row['GrossPay'][1:]),line)
           except ValueError:
               self.increment_counter('warn','missing gross',1)
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
#note : here we only have two keys going into reducer: 1- salary 2- gross! and for each key it calculates the top 10 .

