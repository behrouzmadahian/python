import numpy as np
import pandas as pd
print('creating a data frame using a date range as index!:')
dates = pd.date_range('20130201', periods=50)
print(dates)
# uniform random variables!
myArr = np.random.rand(len(dates), 4)
print(myArr.shape)
df = pd.DataFrame(myArr, index=dates, columns=list('ABCD'))
print(df.head(5))
print('Getting a slice of data frame by the row indices!!! -> date range!')
# this is explicit indexing by index of data frame and NOT numpy style indexing!
# in explicit indexing the upper limit is INCLUDED!
print(df.loc['20130201':'20130210'])
# converting a column to the rank values:
print('##')
print('Ranking Columns of data frame! maximum gets the rank 1!')
df['A_Rank'] = df['A'].rank(ascending=False)
print(df.head(), '\n')

print(' a quick statistics summary of data!')
print(df.describe(), '\n\n')
#####
print('Transposing the data Frame!')
df_T = df.T
print(df_T[:5])
print(df_T.shape, '\n')
#######
print('tail of data frame:', df.tail(5), '\n')
print('Sorting data frame by index:')
df = df.sort_index(ascending=True)
print(df.head(5), '\n')
##################
print('Sorting a data frame by values of a column!')
df = df.sort_values(by='B', ascending=False)
print(df.head(5), '\n')
##################
#  data Selection:
#  While Standard python/ Numpy expressions for selection and setting are intuitive and come in handy for
#  interactive work, for production code we recommend the optimized pandas data access methods!
#  .at, .iat, .loc, .iloc, and .ix
############################
print('Selecting a single column:\n')
###########################
print(df['A'].iloc[:3])
print('First 3 rows: \n')
print(df[:3])
##########################
#  select by label
#########################
print('='*200)
df = df.sort_index(ascending=True)
print(df.head(5), '\n')
print(dates[0], dates[3], '\n')
print('Getting specific columns on a date range(label range!)')
print(df.loc['20130202':'20130205', ['A', 'B']], '\n')
print('Another way to achieve the same thing:\n')
print(df.loc['20130202':'20130205'][['A', 'B']], '\n')
##############
print('='*100)
print('Boolean Indexing!')
print('Using a single column values to select data!')
print(df[df['A'] > 0.8].head(), '\n')
print('lets add a categorical column to our data frame and select!!')
#  creating a new df and copying df to it!
df2 = df.copy()
df2['E'] = ['One', 'Two', 'Three', 'Four', 'Five']*10
print(df2.head(5))
print('='*100)
print('Use of pd.isin([values]) for row selection:')
print(df2[df2['E'].isin(['Two', 'Four'])].head())
print('='*50)
print('Setting values by label:\n')
df2.at['20130201', 'A'] = -200
print(df2.head(5), '\n')
print('setting a whole column:')
df2.loc[:, 'D'] = np.array([5]*len(df))
print(df2.head(5))
df2['D'] = np.array([-5]*len(df))
print(df2.head(5))
print('='*100)
print('Where Operation!')
df2 = df2.drop(['E'], axis=1)
print(df2.shape, df2[df2['A'] > 0].shape)
print(df2[df2 > 0].head())
df2[df2 > 0] = df2*2
print(df2.head(5))
