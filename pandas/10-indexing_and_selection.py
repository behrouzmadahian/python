import pandas as pd
data = pd.Series([0.25, 0.5, 0.75, 1.0],
                 index=['a', 'b', 'c', 'd'])
print(data)
print(data['b'], '\n')
#  using dictionary like python expression and methods to examine the keys/indices and values
print('a' in data)
print('Dictionary like operation: Getting indices using pd.keys(): ', data.keys(), '\n')
print('Dict like operation: Getting index value pairs. pd.items(): ', list(data.items()), '\n')
#  Series objects can even be modified with a dictionary-like syntax.
data['e'] = 1.5
print(data, '\n')
# A Series builds on this dictionary-like interface and provides array-style item
# selection via the same basic mechanisms as NumPy arrays
print('slicing explicit index: final index is included in the slice:')
print(data['a':'c'], '\n')
print('slicing implicit integer index final index not included:')
print(data[:2], '\n')

print('masking...')
print(data[(data > 0.3) & (data < 0.8)], '\n')

print('fancy indexing')
print(data[['a', 'e']], '\n')
####################################
# Indexers: loc, iloc, and ix
# slicing as above can be source of confusion -> thus its preferred to use these functions!!

# First, the loc attribute allows indexing and slicing that always references the explicit index
print('Using loc function: data.loc["a"]: ', data.loc['a'], '\n')
print('Using loc- explicit indexing using index of series \n', data.loc['b':'c'])
# The iloc attribute allows indexing and slicing that always references the implicit Python-style index!!
print('Using iloc: python style indexing \n', data.iloc[1:3], '\n')
################################################################
print('Data frame as dictionary')
area = pd.Series({'California': 423967, 'Texas': 695662,
                  'New York': 141297, 'Florida': 170312,
                  'Illinois': 149995})
pop = pd.Series({'California': 38332521, 'Texas': 26448193,
                 'New York': 19651127, 'Florida': 19552860,
                 'Illinois': 12882135})
data = pd.DataFrame({'area': area, 'pop': pop})
print(data)
print(data['area'])
print(data.area)  # the above is preferred!
data['density'] = data['pop'] / data['area']
print(data, '\n')

# data frame with two dimensional array!
print(data.values)
print('Transposing the data frame')
print(data.T)

'''
Using the iloc indexer, we can index the underlying array as 
if it is a simple NumPy array (using the implicit Python-style index)
'''
print('*'*100)
print('iloc indexer:')
print(data.iloc[1:3], '\n')
print(data.iloc[1:3, 2:])

'''
using the loc indexer we can index the underlying data in an 
array-like style but using the explicit index and column names:
'''
print('='*20)
print('explicit indexing .loc')
print(data.loc[:'Illinois', :'pop'], '\n\n')
print(data.loc[data['density'] > 100, ['pop', 'density']], '\n')
print('Any of these indexing conventions may also be used to set or modify values:')
data.iloc[0, 2] = 90
print(data, '\n')
print('while indexing refers to columns, slicing refers to rows:')
print(data['Florida':'Illinois'], '==', '\n')
#  is equivalent to:
print(data.loc['Florida':'Illinois'], '\n')
print('Direct masking row wise- not specifying columns!')
print(data[data.density > 100])