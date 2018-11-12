import pandas as pd
import seaborn as sns
sns.set()
from matplotlib import pyplot as plt
'''
Another common time series-specific operation is shifting of data in time. Pandas has two closely related methods 
for computing this: shift() and tshift() In short, the difference between them is that shift() shifts the data, 
while tshift() shifts the index. In both cases, the shift is specified in multiples of the frequency.
'''
pd.core.common.is_list_like = pd.api.types.is_list_like
from pandas_datareader import data
# robinhood only provides one year of daily data!
goog = data.DataReader('GOOG', start='2004', end='2016', data_source='yahoo')
goog = goog['Close']
fig, ax = plt.subplots(3, sharey=True)
# apply a frequency to the data:
# 'pad' / 'ffill': propagate last valid observation forward to next valid
goog = goog.asfreq('D', method='pad')
goog.plot(ax=ax[0])
goog.shift(900).plot(ax=ax[1])
goog.tshift(900).plot(ax=ax[2])
# legends and annotations:
local_max = pd.to_datetime('2007-11-05')
offset = pd.Timedelta(900, 'D')
ax[0].legend(['input'], loc=2)
ax[0].get_xticklabels()[2].set(weight='heavy', color='red')
ax[0].axvline(local_max, alpha=0.3, color='red')

ax[1].legend(['shift(900)'], loc=2)
ax[1].get_xticklabels()[2].set(weight='heavy', color='red')
ax[1].axvline(local_max+offset, alpha=0.3, color='red')

ax[2].legend(['tshift(900)'], loc=2)
ax[2].get_xticklabels()[2].set(weight='heavy', color='red')
ax[2].axvline(local_max+offset, alpha=0.3, color='red')
plt.show()

'''
A common context for this type of shift is in computing differences over time. For example, 
we use shifted values to compute the one-year return on investment for Google stock over the course of the dataset:
goog.tshift(-360) / goog:
division is performed by matching by date.  goog.shift(-360), makes the first index that matched the goog actually be
the value 360 days ahead
example:
jan 01-01-2005 of google becomes the index 01-01-2004 in the tshift frame and gets divided by goog price at
 01-01-2004 
'''
print(goog.tshift(-360).head())
ROI = 100 * (goog.tshift(-360) / goog - 1)
print(ROI.head())
ROI.plot()
plt.ylabel('% Return on Investment')
plt.show()