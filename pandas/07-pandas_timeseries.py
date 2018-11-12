import numpy as np
import pandas as pd
'''
Working with time series will be covered later on. This is just a gentle introduction.
We introduce asfreq and resample im more details later on.
asfreq: is a selection method.
resample: is an aggregation method- think about it as functioning like groupby.
'''
# 72 hours starting with midnight Jan 1st 2011:
rng = pd.date_range('1/1/2011', periods=72, freq='H')
print(rng[:10])
# pd.Series:
# One dimensional ndarray with axis labels(including time series)
# labels need not to be unique but must be a hashable type
ts = pd.Series(np.random.randn(len(rng)), index=rng)
print(ts.head(), '\n')
#
#  change frequency and fill gaps:
print('Change Frequency and fill gaps..')
#  to 120 minute frequency and forward fill
converted = ts.asfreq('45Min', method='pad')
print(converted[:6], '\n')
print('45 min frequency!')
print(ts.asfreq('45Min', method='pad').head(), '\n')

print('Daily Means:')
print(ts.resample('D').mean())
