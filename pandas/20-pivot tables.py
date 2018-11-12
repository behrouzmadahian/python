import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
'''
pivot_table() by default acts like group by a column and then calculate mean for one column!!!
The pivot table takes simple column-wise data as input, and groups the entries 
into a two-dimensional table that provides a multidimensional summarization of the data.
think of pivot tables as essentially a multidimensional version of GroupBy aggregation. 
you split-apply-combine, but both the split and the combine happen across 
not a one-dimensional index, but across a two-dimensional grid.
https://www.dataquest.io/blog/pandas-pivot-table/
'''
titanic = sns.load_dataset('titanic')
print(titanic.shape, '\n')
print(titanic.info(), '\n')
colnames = titanic.columns.tolist()
print(colnames, '\n')
colnames_numerics_only = titanic.select_dtypes(include=[np.number]).columns.tolist()
print('NUMERIC COLUMNS:', colnames_numerics_only, '\n')
print('Fraction of people survived by gender:')
print(titanic.groupby('sex').apply(lambda x: np.mean(x))[['survived']], '\n')
print('THE SAME AS ABOVE-SLIGHTLY DIFFERENT SYNTAX: \n',
      titanic[['sex', 'survived']].groupby('sex').apply(lambda x: np.mean(x)), '\n')
print('How about Survival by both gender and class?\n')
print(titanic.groupby(['sex', 'class']).apply(lambda x: np.mean(x))[['survived']].unstack(), '\n')
'''
as we can see  the code is starting to look a bit garbled< hard to read!>
This two-dimensional GroupBy is common enough that Pandas includes a convenience routine,
pivot_table, which succinctly handles this type of multi-dimensional aggregation.
'''
print('Pivot Table equivalent of survival by gender and class\n')
print(titanic.pivot_table('survived', index='sex', columns='class', aggfunc='mean'), '\n')
titanic.pivot_table('survived', index='sex', columns='class', aggfunc='mean').plot(kind='bar')
plt.ylabel('Survival rate')
plt.show()
'''
multi-level pivot tables:
Just as in the GroupBy, the grouping in pivot tables can be specified
with multiple levels, and via a number of options.
For example, we might be interested in looking at age as a third dimension. 
'''
titanic['age'] = titanic['age'].fillna(0)
print(titanic.head())
age = pd.cut(titanic['age'], bins=[-1, 18, 80], include_lowest=True, labels=['teenager', 'The rest'])
print(age)
print('\nmulti-level pivot tables:\n')
print(titanic.pivot_table('survived', index=['sex', age], columns='class', aggfunc='mean'), '\n')
'''
We can apply the same strategy when working with the columns as well; 
let's add info on the fare paid using pd.qcut to automatically compute quantiles.
'''
print('We can apply the same strategy when working with the columns as well:\n')
fare = pd.qcut(titanic['fare'], 2)
print(titanic.pivot_table('survived', index=['sex', age],
                          columns=[fare, 'class'], aggfunc='mean', dropna=True), '\n')
titanic.pivot_table('survived', index=['sex', age],
                    columns=[fare, 'class'], aggfunc='mean', dropna=True).plot(kind='bar')
plt.ylabel('Survival by gender, age Group, fare group and class')
plt.show()
titanic.pivot_table('survived', index=['sex', age],
                    columns='class', aggfunc='mean', dropna=True).plot(kind='bar')
plt.ylabel('Survival by gender,age Group, class')
plt.show()

print('Adding margins to pivot table results:\n')
print(titanic.pivot_table('survived', index='sex', columns='class', margins=True, aggfunc='mean'), '\n')

births = pd.read_csv('./births.csv')
print(births.head(), '\n')
births['decade'] = 10 * (births['year'] // 10)
print(births.pivot_table('births', index='year', columns='gender', aggfunc='sum').head(), '\n')

sns.set()  # use Seaborn styles
births.pivot_table('births', index='year', columns='gender', aggfunc='sum').plot(kind='line')
plt.ylabel('total births per year')
plt.show()


def mycustom_agg_func(s):
    '''takes a series( one col of DF and returns a single number as output'''
    return np.mean(s)


print(births.pivot_table('births', index='decade', aggfunc=['mean', mycustom_agg_func]).head())
