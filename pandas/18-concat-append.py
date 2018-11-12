import numpy as np
import pandas as pd
'''
join:
1- outer: union of the two frames
2- inner: intersection of the two! along one of the axes!
'''
print('concatenation of numpy arrays:')
x = np.array([[1, 2, 3]])
y = np.array([[4, 5, 6]])
z = np.array([[7, 8, 9]])
print(x.shape)
row_concat = np.concatenate([x, y, z], axis=0)
col_concat = np.concatenate([x, y, z], axis=1)
print('Row concatenation: axis=0:')
print(row_concat, row_concat.shape)
print('Col concat axis=1:')
print(col_concat, col_concat.shape)

# pd.concat:
ser1 = pd.Series(['A', 'B', 'C'], index=[1, 2, 3])
ser2 = pd.Series(['D', 'E', 'F'], index=[4, 5, 6])
print(pd.concat([ser1, ser2], axis=0, join='outer'), '\n')
print('concatenating two data frames with outer join and axis=0:')
# it does not matter if indices exist in the other, it just adds the row after the first frame!
df1 = pd.DataFrame(np.random.rand(3, 4), columns=list('abcd'), index=[10, 20, 30])
df2 = pd.DataFrame(np.random.rand(4, 4), columns=list('abef'), index=[10, 30, 40, 50])
print(df1, '\n', df2, '\n')
print(pd.concat([df1, df2], axis=0, join='outer', sort=True), '\n')
print('Concatenation of data frames with axis=1 and join=outer: ')
print(' Columns duplicate or not are added to the first data frame, ')
print('For Common indices in two frames, the data for that index,'
      ' is added at the end  of the first frame at exact index')
concat_by_columns = pd.concat([df1, df2], axis=1, join='outer', sort=True)
print(concat_by_columns)
print(concat_by_columns.shape, concat_by_columns.columns.tolist(), '\n')
'''Pandas concatenation preserves indices, even if the result will have duplicate indices! '''
print('In order to avoid having duplicate indices, use: ignore_index=True- creates new set of indices')
print(pd.concat([df1, df2], axis=0, join='outer', ignore_index=True, sort=True), '\n')
print('Another option is to specify MultiIndex with keys!!')
print(pd.concat([df1, df2], axis=0, join='outer', sort=True, keys=['x', 'y']), '\n')
print('Concatenation with INNER Joins: \n')
print('Inner join with axis=0: returns all the <common> columns for all rows in two frames')
print(pd.concat([df1, df2], join='inner', axis=0), '\n')
print('Inner join axis=1: joins in row indices, returns concateneted rows with common row indices')
print('Inner join axis=1: can have duplicate columns')
print('Can have duplicate columns!!')
print(pd.concat([df1, df2], join='inner', axis=1), '\n')
print('Joining two data frame and specifying the columns we want returned, when axis=0')
print(df1.columns)
print(pd.concat([df1, df2], join='outer', axis=0, join_axes=[pd.Index(['a', 'b', 'f'])]), '\n')
print(pd.concat([df1, df2], join='inner', axis=0, join_axes=[pd.Index(['a'])]), '\n')
print('Joining to data frame and specifying the  rows we want returned  axis=1!')
print(pd.concat([df1, df2], join='inner', axis=1, join_axes=[pd.Index([10, 30])]), '\n')
