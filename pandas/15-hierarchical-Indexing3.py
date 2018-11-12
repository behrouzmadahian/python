import numpy as np
import pandas as pd
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data, '\n')
print('Heart rate of Bob:\n', health_data['Bob', 'HR'], '\n')
'''
as with the single-index case, we can use the loc, iloc, and ix indexers 
'''
print('Simple Indexing: \n', health_data.iloc[:2, :2], '\n')
print('each individual index in loc or iloc can be passed a tuple of multiple indices.\n ')
# following Does not do the job!!!
print(health_data.loc[:, ['Bob', 'HR']])
print('Year 2013 Bob HeartRate:\n')
print(health_data.loc[2013, ['Bob', 'HR']], '\n')
'''
Working with Slices is not necessarily convenient -> use pd.IndexSlice
As we can see it gives all columns for Bob in previous attempt!!
'''
print('Index Slice: ')
idx = pd.IndexSlice
print(idx, '\n')
print('Heart rate all yers all visits for bob:')
print(health_data.loc[idx[:, :], idx['Bob', 'HR']], '\n')
print('All years data for visit=1 and column of interest = All individuals heart rate')
print(health_data.loc[idx[:, 1], idx[:, 'HR']], '\n')
print('Heart rate of everyone for year 2013, visit 1!')
print(health_data.loc[idx[2013, 1], idx[:, 'HR']], '\n')
'''
Rearranging Multi-Indices.
One of the keys to working with multiply indexed data is knowing how to effectively transform the data.
There are a number of operations that will preserve all the information in the dataset,
but rearrange it for the purposes of various computations.
'''
print('Sorted and unsorted indices: \n')
index = pd.MultiIndex.from_product([['a', 'b', 'c'], [2, 1]])
data = pd.Series(np.random.rand(6), index=index, name='CL1')
data.index.names = ['Group', 'Level']
print(data, '\n')
try:
    print(data['a':'b'])
except:
    print('Need to sort indices to be able to perform partial indexing!!!')
'''
If you try: data['a':'b'] you will get an error!!!
Although it is not entirely clear from the error message, this is the result of the MultiIndex not being sorted.
For various reasons, partial slices and other similar operations require the levels
in the MultiIndex to be in sorted (i.e., lexographical) order. 
Pandas provides a number of convenience routines to perform this type of sorting; examples are:
the sort_index() and sortlevel() methods of the DataFrame. We'll use the simplest, sort_index()
'''
print('====')
data = data.sort_index()
print(data['a':'b'], '\n')
'''
Stacking and unStaking Indices:
'''
print('Unstacking data:level=0:\n', data.unstack(level=0), '\n\n')
print('Unstacking data:level=1:\n', data.unstack(level=1), '\n\n')
print('Unstack and stack back! \n', data.unstack(level=0).stack(), '\n')
'''
Flattening the frame using reset_index()
'''
flat_data = data.reset_index()
print('flattened data:\n', flat_data, '\n')
print('Building MultiIndex From Columns: \n')
data = flat_data.set_index(['Group', 'Level'])
print(data, '\n\n')

print('Data Aggregation on Multi-Indices:\n')
'''
We've previously seen that Pandas has built-in data aggregation methods, such as mean(), sum(), and max(). 
For hierarchically indexed data, these can be passed a level parameter 
that controls which subset of the data the aggregate is computed on.
'''
index = pd.MultiIndex.from_product([[2013, 2014], [1, 2]],
                                   names=['year', 'visit'])
columns = pd.MultiIndex.from_product([['Bob', 'Guido', 'Sue'], ['HR', 'Temp']],
                                     names=['subject', 'type'])

# mock some data
data = np.round(np.random.randn(4, 6), 1)
data[:, ::2] *= 10
data += 37

# create the DataFrame
health_data = pd.DataFrame(data, index=index, columns=columns)
print(health_data, '\n')
data_mean = health_data.mean(axis=0, level='year')
print('Mean : level=year:\n', data_mean, '\n')
data_mean = health_data.mean(axis=0, level='visit')
print('Mean : level=visit:\n', data_mean, '\n')
print('By further making use of the axis keyword, we can take the mean among levels on the columns as well:\n')
print('Mean along columns and column level=type:\n', health_data.mean(axis=1, level='type'), '\n')
print('Operation on the mean df:Mean along columns and column level=subject:\n',
      data_mean.mean(axis=1, level='subject'), '\n')
print('Mean along columns and column level=type:\n', data_mean.mean(axis=1, level='type'), '\n')

ind = pd.IndexSlice
print(health_data.loc[ind[2013, 2], ind['Bob', 'HR']])