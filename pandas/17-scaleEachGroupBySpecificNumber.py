import numpy as np
import pandas as pd
import copy


def test_scaler(test_data, train_data, group_names, groupColumn, feat):
    for gr in group_names:
        print('Group= ', gr)
        tr1 = train_data[train_data[groupColumn] == gr][feat]
        fmin, fmax = tr1.min(), tr1.max()
        print(fmin, fmax)
        vals = (test_data[test_data[groupColumn] == gr][feat].values - fmin) / (fmax - fmin)
        test_data.loc[test_data[groupColumn] == gr, feat] = vals
    return test_data


df = pd.DataFrame({'B': ['one', 'one', 'one', 'one', 'two', 'two', 'two', 'two'],
                   'C': [1, 2, 1, 1]+[20, 4, 1, 5],
                   'D': np.random.randn(8)})

df1 = copy.deepcopy(df)
df2 = copy.deepcopy(df)
print(df)
print('Grouping and then Applying a function:')
df2.loc[:, 'C'] = df[['C', 'B']].groupby('B').apply(lambda x: (x-x.min())/(x.max() - x.min()))
print(df2)
print('*********************')

td = test_scaler(df1, df, ['one', 'two'], 'B', 'C')
print(td)


