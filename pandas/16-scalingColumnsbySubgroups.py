import pandas as pd
import numpy as np
import seaborn as sns
'''
Here I want to scale the columns of a data frame by the statistics of subgroups!!!
'''
titanic = sns.load_dataset('titanic')
titanic.index = np.arange(1000, 1000+titanic.shape[0], 1)
colnames = titanic.columns.values
print(colnames, '\n')
colnames_numerics_only = titanic.select_dtypes(include=[np.number]).columns.tolist()

# lets get only numeric columns and sex column for ease of representation!:
titanic = titanic[colnames_numerics_only+['sex']]
print(titanic.head(), '\n\n')
print('='*100)
print('Globally scaling each column:')
titanic_global_scale = titanic[colnames_numerics_only].apply(lambda x: x/np.mean(x))
print(titanic_global_scale.head(), '\n\n')
# axis =0: apply the function to every column!
print('Scaling age column by Sex:')
print(titanic[['sex', 'age']].groupby('sex').apply(lambda x: x/np.mean(x)).head(), '\n')

print('Mean age by Gender two different approaches:\n')
print(titanic.groupby('sex').apply(lambda x: np.mean(x))[['age']], '\n')
print(titanic[['sex', 'age']].groupby('sex').apply(lambda x: np.mean(x)), '\n')
print(titanic[['sex', 'age']].groupby('sex').mean(), '===')
titanic.loc[:, 'age'] = titanic[['sex', 'age']].groupby('sex').apply(lambda x: x/np.mean(x))
print(titanic.head())

