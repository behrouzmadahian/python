import numpy as np
import pandas as pd
#  OPERATIONS:
dates = pd.date_range('20130101', periods=6)
myArr = np.random.rand(len(dates), 4)
print(myArr.shape)
df = pd.DataFrame(myArr, index=dates, columns=list('ABCD'))
print(df.head(5))
print('column means..')
print(df.mean(axis=0), '\n')
print('Row means!!')
print(df.mean(axis=1), '\n')

print('applying functions to df columns: commulative sum:')
df = df.apply(np.cumsum, axis=0)
print(df, '\n')

print('applying functions to df ROWs commulative sum: \n')
df = df.apply(np.cumsum, axis=1)
print(df, '\n')
print('User defined function to apply to df:')
print('Returns column ranges!')
df1 = df.apply(lambda x: x.max()-x.min(), axis=0)
print(df1)
df['E'] = df['A'].apply(lambda x: x**2)
print(df)
