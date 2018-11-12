import numpy as np
import pandas as pd
'''
1- The most straightforward way to construct a multiply indexed Series or DataFrame
is to simply pass a <<list of two or more index arrays>> to the constructor. For example:
'''
df = pd.DataFrame(np.random.rand(4, 2),
                  index=[['a', 'b', 'c', 'd'], [1, 2, 1, 2]],
                  columns=['data1', 'data2'])
print(df, '\n')
'''
2-if you pass a dictionary with appropriate tuples as keys, 
Pandas will automatically recognize this and use a MultiIndex by default.
'''
data = {('California', 2000): 33871648,
        ('California', 2010): 37253956,
        ('Texas', 2000): 20851820,
        ('Texas', 2010): 25145561,
        ('New York', 2000): 18976457,
        ('New York', 2010): 19378102}
data = pd.Series(data)
print(data, '\n')
'''
3-Explicit MultiIndex Constructor:
'''
myMultiIndex = pd.MultiIndex.from_arrays([['a', 'a', 'b', 'b'], [1, 2, 1, 2]])
print(myMultiIndex, '\n')
'''
4-Constructing Multi Index from list of tuples:
'''
myMultiIndex = pd.MultiIndex.from_tuples([('a', 1), ('a', 2), ('b', 1), ('b', 2)])
print(myMultiIndex, '\n')
'''
5- Constructing MultiIndex from Cartesian product of single indices:
'''
myMultiIndex = pd.MultiIndex.from_product([['a', 'b'], [1, 2]])
print(myMultiIndex)
mydf = pd.DataFrame(np.random.rand(4, 2), index=myMultiIndex, columns=['cl1', 'cl2'])
print(mydf, '\n')
'''
MultiIndex Level Names:
'''
mydf.index.names = ['state', 'year']
print(mydf, '\n')

'''
6- MultiIndex for Columns:
In a DataFrame, the rows and columns are completely symmetric, and just as the rows 
can have multiple levels of indices, the columns can have multiple levels as well. 
Consider the following, which is a mock-up of some (somewhat realistic) medical data:
This is fundamentally four Dimensional data!
where the dimensions are the subject, the measurement type, the year, and the visit number.
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
print(health_data['Guido'], '\n')

'''
7-Indexing and Slicing a MultiIndex:
'''
print('Multiply indexed Series:')
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
index = pd.MultiIndex.from_tuples(index)

pop = pd.Series(populations, index=index)
print(pop, '\n')
print('We can access single elements by indexing with multiple terms:')
print(pop['California', 2000], '\n')
print(pop['California'], '\n')
print(pop.loc['California': 'New York'], '\n')
print('partial indexing can be performed on lower levels by passing an empty slice in the first index:')
print(pop[:, 2000], '\n')
print(pop[pop > 22000000], '\n')
print(pop[['California', 'Texas']], '\n')
print(pop[['California', 'Texas']][:, 2000], '\n')
