import pandas as pd
from matplotlib import pyplot as plt
import seaborn
seaborn.set()
'''
Because Pandas was developed largely in a finance context, it includes some very specific tools for financial data.
For example, the accompanying pandas-datareader package (installable via conda install pandas-datareader),
knows how to import financial data from a number of available sources, 
including Yahoo finance, Google Finance, and others.
'''
# a fix that in future will be resolved:
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
# robinhood only provides one year of daily data!
goog = data.DataReader('GOOG', start='2004', end='2016', data_source='yahoo')
print(goog.head(), goog.shape)
goog = goog['Close']
goog.plot()
plt.show()
print('Resampling and converting frequencies:\n')
'''
One common need for time series data is resampling at a higher or lower frequency. 
This can be done using the resample() method, or the much simpler asfreq() method. 
The primary difference between the two is that resample() is 
fundamentally a data aggregation, while asfreq() is fundamentally a data selection.
'''
goog.plot(alpha=0.5, style='-')  # making color a bit lighter
print('Business year means:\n')
print(goog.resample('BA').mean())
goog.resample('BA').mean().plot(style='o')
print('Business year selection: using asfreq\n')
print(goog.asfreq('BA'))
goog.asfreq('BA').plot(style='o')
plt.legend(['input', 'resample', 'asfreq'])
plt.show()
'''
resample reports the average of the  year, while asfreq reports the value at the end of the year.
'''
'''
For up-sampling, resample() and asfreq() are largely equivalent, though resample has many more options available. 
In this case, the default for both methods is to leave the up-sampled points empty, that is, filled with NA values. 
Just as with the pd.fillna() function discussed previously, asfreq() accepts a method argument to
specify how values are imputed. Here, we will resample the
business day data at a daily frequency (i.e., including weekends):
'''
fig, ax = plt.subplots(2, sharex=True)
data = goog.iloc[:10]
print(data.asfreq('D', method='bfill'))

data.asfreq('D').plot(ax=ax[0], marker='o')
data.asfreq('D', method='bfill').plot(ax=ax[1], style='-o')
data.asfreq('D', method='ffill').plot(ax=ax[1], style='-o')
ax[1].legend(["back-fill", "forward-fill"])
plt.show()