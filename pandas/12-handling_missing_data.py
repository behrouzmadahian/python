import pandas as pd
import numpy as np
# Python uses None to represent null values
# pandas uses NaN : not an integer: missing a numerical data! in addition to None,
# Converts between them when appropriate

'''
If there is a None value in an array and you want to do operations with it, it will throw an error!
use np.nan instead to avoid it.
np.nan is floating point type!!
if in an array of type int one value is changed to np.nan, then the type automatically changes to floating point!
Pandas automatically converts None to NaN.

'''
'''
Operating on Null values:
isnull(): Generate a boolean mask indicating missing values
notnull(): Opposite of isnull()
dropna(): Return a filtered version of the data
fillna(): Return a copy of the data with missing values filled or imputed
'''
data = pd.Series([1, np.nan, 'hello', None])
print(data.isnull())
print(data[data.isnull()])
print(data.notnull())
print(data[data.notnull()])
print('Dropping null values')
print(data.dropna(), '\n')
df = pd.DataFrame([[1,      np.nan, 2],
                   [2,      3,      5],
                   [np.nan, 4,      6]],
                  columns=list('abc'))
print(df)
print('===')
# removes all rows having NaN values!!
print('removing rows having NaN:\n', df.dropna(axis=0))

# Alternatively, you can drop NA values along a different axis; axis=1 drops all columns containing a null value:
print('Dropping columns having NaN values')
print(df.dropna(axis=1))
print('you might rather be interested in dropping rows or columns with all NA values, or a majority of NA values.')
df['E'] = np.nan
print(df)
print('drop columns with ALL null values:')
print(df.dropna(axis='columns', how='all'), '\n')
print('minimum number of non-null values for the row/column to be kept: ')
print(df.dropna(axis='rows', thresh=2))
#################################################################################
print('Filling Null Values')
data = pd.Series([1, np.nan, 2, None, 3], index=list('abcde'))
print(data)
print(data.fillna(0))
print('We can specify a forward-fill to propagate the previous value forward:')
print(data.fillna(method='ffill'))
print('we can specify a back-fill to propagate the next values backward:')
print(data.fillna(method='bfill'))
print(df, '\n')
print(df.fillna(method='ffill', axis=0))
print(df.fillna(method='ffill', axis=1))