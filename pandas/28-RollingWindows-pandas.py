import pandas as pd
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
# robinhood only provides one year of daily data!
goog = data.DataReader('GOOG', start='2004', end='2016', data_source='yahoo')
goog = goog['Close']

'''
Rolling statistics are a third type of time series-specific operation implemented by Pandas. These can be accomplished
via the rolling() attribute of Series and DataFrame objects, which returns a view similar to what we saw 
with the groupby operation (see Aggregation and Grouping). This rolling view makes available
a number of aggregation operations by default.
As with group-by operations, the aggregate() and apply() methods can be used for custom rolling computations.
'''
'''
winodw:Size of the moving window. This is the number of observations used for 
calculating the statistic. Each window will be a fixed size.
min_periods: Minimum number of observations in window required to have a value (otherwise result is NA).
center: set the labels at the center of the window.
'''
print('size of the window is 360 and shifts forward 1 value:\n')
rolling = goog.rolling(window=365, center=True)
print(rolling)
data = pd.DataFrame({'input': goog, 'one-year rolling mean': rolling.mean(), 'one-year rolling sd': rolling.std()})
print(data.head())
ax = data.plot(style=['-', ':', '--'])
#ax.lines[0].set_alpha(0.3)
plt.show()