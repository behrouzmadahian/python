import numpy as np
import pandas as pd
df = pd.DataFrame({'A': ['foo', 'foo', 'bar', 'bar', 'bar', 'goo', 'goo', 'goo'],
                   'B': ['one', 'one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'C': [1, 2, 1, 1]+[20, 4, 1, 5],
                   'D': np.random.randn(8)})
print(df, '\n')
print('Grouping and then Applying a function:- applies the mean function to Numerical Columns!')
rs = df.groupby('B').mean()
print(rs, '\n')
print('scaling by group SUM!!! for columns!!')
print(df[['C', 'B']].groupby('B').apply(lambda x: x/np.sum(x)), '\n')
print(df[['C', 'A', 'B']].groupby(['A', 'B']).apply(lambda x: x/np.sum(x)), '\n')

print('Grouping by Multiple Columns:')
rs = df.groupby(['A', 'B']).sum()
print(rs, '\n')
#########
print('Cross Tabulation')
print(pd.crosstab(df['A'], df['B']))
print('Cross tab with normalized -> frequency!')
print(pd.crosstab(df.A, df.B, normalize=True))
print('Cross tab with normalized columns! and adding total rows and columns!')
print(pd.crosstab(df.A, df.B, normalize='columns'), '\n')
print(pd.crosstab(df.A, df.B, normalize=True, margins=True))
print('#'*100)
print('Cutting continuous variable into bins- categories!')

ages = np.array([0, 10, 15, 13, 12, 23, 25, 28, 59, 60, 70, 74])
print(pd.cut(ages, bins=3,  labels=['adolescence', 'young', 'midage'], include_lowest=False, right=True))
print('When we define bins, upper bound is included!, LOWER bound is NOT! results in NaN value!')
print(pd.cut(ages, bins=[0, 18, 35, 70], labels=None, include_lowest=False, right=True), '\n')

df = pd.DataFrame({'feature': list('bbacab')})
print(df)
# returns the one hot encoding of the categorical feature.
# drop_first: whether to model a categorical variable with k levels with k-1 variables!
# dummy_na=: add a binary column for NA as a level or not!
print(pd.get_dummies(df['feature'], dummy_na=False, drop_first=False))

#  this function is often used alongside pd.cut!
print('Binning a continuous data and converting to dummy variables!')
values = np.random.randn(10)
print(pd.get_dummies(pd.cut(values, bins=3,
                            labels=['A', 'B', 'C'],
                            include_lowest=False, right=True),
                     drop_first=False))
