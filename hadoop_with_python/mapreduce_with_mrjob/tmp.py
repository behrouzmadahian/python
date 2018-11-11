#! /usr/bin/env python
import csv,sys
cols='Name,JobTitle,AgencyID,Agency,HireDate,AnnualSalary,GrossPay'.split(',')
i=1
for line in sys.stdin:
    row= [a.strip() for a in csv.reader([line]).next()]
    print row
    if i==2:
        break
    i +=1

           
