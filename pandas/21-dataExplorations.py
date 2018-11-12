import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
birthsDF = pd.read_csv('./births.csv', delimiter=',')
'''
Removing outliers caused by misspelled dates(eg June 31st) or missing values(eg. June 99th).
One easy way to remove these all at once is to cut outliers; we'll do this via a robust sigma-clipping operation:
 we can use the query() method. Dont read too much into their method of removing outliers!!!
 The query function used here will be explained later in high performance pandas!
'''
print(birthsDF.shape)
quartiles = np.percentile(birthsDF['births'], q=[25, 50, 75])
print(quartiles)
mu = quartiles[1]
sig = 0.74 * (quartiles[2] - quartiles[1])  # estimate of mean!
print('Shape of birth data before removing ros with outliers= ', birthsDF.shape)
birthsDF = birthsDF.query('(births > @mu - 5 * @sig) & (births < @mu + 5 *@sig)')
print('Shape of birth data after removing rows with outliers= ', birthsDF.shape, '\n')
print('Next we set the day column to integers; previously it '
      'had been a float because some columns in the dataset contained the value null:\n')
print(birthsDF.head())
print(birthsDF.info(), '\n')
birthsDF['day'] = birthsDF['day'].astype(int)
print('We can combine the day, month, and year to create a Date index:\n')
# birthsDF.index = pd.to_datetime(10000*birthsDF['year'] + 100 * birthsDF['month'] + birthsDF['day'], format='%Y%m%d')
birthsDF.index = [pd.datetime(y, m, d) for y, m, d in zip(birthsDF['year'], birthsDF['month'], birthsDF['day'])]
print(birthsDF.head(), '\n')
print('Now we can quickly calculate day of the week!')
birthsDF['dayofweek'] = birthsDF.index.dayofweek
print(birthsDF.head(), '\n')
print('Using this we can plot births by weekday for several decades:\n')
birthsDF['decade'] = 10 * (birthsDF['year'] // 10)
print(birthsDF.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean'), '\n')
sns.set()  # use Seaborn styles
birthsDF.pivot_table('births', index='dayofweek', columns='decade', aggfunc='mean').plot(kind='line')
plt.gca().set_xticks(np.arange(0, 7))
plt.gca().set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.ylabel('mean births by day')
plt.show()
'''
Another interesting view is to plot the mean number of births by the day of the year. 
'''
print('births by the day of the year:\n')
births_by_date = birthsDF.pivot_table('births', index=[birthsDF.index.month, birthsDF.index.day], aggfunc='mean')
print(births_by_date[:33], births_by_date.shape, '\n\n')
'''
The result is a multi-index over months and days. To make this easily plottable, 
let's turn these months and days into a date by associating them with a dummy year variable
 (making sure to choose a leap year so February 29th is correctly handled!)
'''
births_by_date.index = [pd.datetime(2012, month, day)
                        for (month, day) in births_by_date.index]
print(births_by_date.head())
fig, ax = plt.subplots(figsize=(12, 4))
births_by_date.plot(ax=ax, kind='line')
plt.show()
birthsDF.pivot_table('births', index=birthsDF.index.month, columns=birthsDF.index.day, aggfunc='mean').plot(kind='line')
plt.show()