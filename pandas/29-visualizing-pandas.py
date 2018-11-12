import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
data = pd.read_csv('fremontBridge.csv', index_col='Date', parse_dates=True)
# lets shorten the column names
data.columns = ['West', 'East']
print(data.head())

data['Total'] = data.eval('West + East')
print(data.dropna().describe())
data.plot(kind='line', style=':')
plt.ylabel('Hourly Bicycle Count')
plt.show()

'''
The ~25,000 hourly samples are far too dense for us to make much sense of. 
We can gain more insight by resampling the data to a coarser grid.
'''
weekly = data.resample('W').sum()
weekly.plot(kind='line', style=[':', '--', '-'])
plt.ylabel('Weekly bicycle count')
plt.show()

'''
Another way that comes in handy for aggregating the data is to use a rolling mean, 
utilizing the pd.rolling_mean() function. Here we'll do a 30 day rolling mean of our data, 
making sure to center the window.
'''
print('Data resampled daily:')
daily = data.resample('D').sum()
print(daily.head())
print('\n\n\n')
print('Rolling 30 day mean of data:\n')
print(daily.rolling(30, center=True).mean().head())
daily.rolling(30, center=True).mean().plot(style=[':', '--', '-'])
plt.ylabel('Monthly mean of data')
plt.show()
'''
The jaggedness of the result is due to the hard cutoff of the window. We can get a smoother version
of a rolling mean using a window function–for example, a Gaussian window. The following code specifies
both the width of the window (we chose 50 days) and the width of the Gaussian within the window (we chose 10 days):
'''
daily.rolling(50, center=True,
              win_type='gaussian').mean(std=10).plot(style=[':', '--', '-'])
plt.title('Gaussian smoothing of rolling window')
plt.show()
'''
While these smoothed data views are useful to get an idea of the general trend in the data, they hide much
of the interesting structure. For example, we might want to look at the average traffic as a function 
of the time of day. We can do this using the GroupBy functionality discussed in Aggregation and Grouping:
'''
by_time = data.groupby(data.index.time).mean()
print(by_time)
hourly_ticks = 4 * 60 * 60 * np.arange(6)  # we give the ticks at exact seconds!
print(hourly_ticks)
by_time.plot(xticks=hourly_ticks, style=[':', '--', '-'])
plt.show()

by_weekday = data.groupby(data.index.dayofweek).mean()
by_weekday.index = ['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun']
print(by_weekday)
fig, ax = plt.subplots(1)
by_weekday.plot(ax=ax, style=[':', '--', '-'])
ax.set_xticks(np.arange(7))
ax.set_xticklabels(['Mon', 'Tues', 'Wed', 'Thurs', 'Fri', 'Sat', 'Sun'])
plt.show()

weekend = np.where(data.index.weekday < 5, 'Weekday', 'Weekend')
by_time = data.groupby([weekend, data.index.time]).mean()
print(by_time)
by_time.plot(style=[':', '--', '-'])
plt.show()
'''
ix: A primarily label-location based indexer, with integer position fallback.
'''
fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
by_time.loc['Weekday'].plot(ax=ax[0], title='Weekdays',
                            xticks=hourly_ticks, style=[':', '--', '-'])
by_time.loc['Weekend'].plot(ax=ax[1], title='Weekends',
                            xticks=hourly_ticks, style=[':', '--', '-'])
plt.show()