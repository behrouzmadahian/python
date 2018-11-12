import numpy as np
import pandas as pd
#  pandas primarily uses np.nan to represent missing data. it is by default not included in computations.
#  reindexing allows you to change/add/delete the index on a specified axis. This returns a copy of data!
dates = pd.date_range('20130201', periods=50)
myArr = np.random.rand(len(dates), 4)
print(myArr.shape)
df = pd.DataFrame(myArr, index=dates, columns=list('ABCD'))
print(df.head(5), '\n')
print('re-indexing...')
df1 = df.reindex(index=dates[0:4], columns=list(df.columns[:2])+['E'])
print(df1.shape)
df1.loc['20130201':'20130202', 'E'] = 1
print(df1)
print('Dropping NaN values:')
df2 = df1.dropna(how='any', axis=0)
print(df2.shape)
print('Filling NaN values with 500')
df2 = df1.fillna(value=500)
print(df2.head(), '\n')
df2.loc['20130201':'20130202', 'E'] = None
print(df2, '\n')
print('Boolean mask for nan values positions- The following are equivalent:!')
print(pd.isnull(df1))
print(df1.isnull(), '++++++', '\n')
print('Setting null values in data frame with 250:')
df1[pd.isnull(df1)] = 250
print(df1, '\n')
######

print('Data Frame from list of dictionaries:')
dict_list = [{'a': 1, 'b': 2}, {'b': 3, 'c': 4}]
print(dict_list)
df3 = pd.DataFrame(dict_list)
print(df3)
