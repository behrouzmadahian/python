import pandas as pd
import numpy as np
import datetime
'''
Dates and times in pandas: BEST OF THE BOTH WORLDS!!!
Pandas builds upon all the tools just discussed to provide a Timestamp object, which combines the ease of use of 
datetime and dateutil with the efficient storage and vectorized interface of numpy.datetime64
From a group of these Timestamp objects, Pandas can construct a
DatetimeIndex that can be used to index data in a Series or DataFrame
'''
print('for EXAMPLE: We can parse a flexibly formatted string date,'
      ' and use format codes to output the day of the week:\n')
date = pd.to_datetime('4th of July, 2015')
print(date)
print(date.strftime('%A'))
print('we can do NumPy-style vectorized operations directly on this same object:\n')
print('Time delta:\n')
print(pd.to_timedelta(np.arange(12), 'D'), '\n')
print('Adding time delta to a date to obtain a date range:\n')
print(date + pd.to_timedelta(np.arange(12), 'D'))

print('Pandas time Series- Indexing by Time:')
index = pd.DatetimeIndex(['2014-07-04', '2014-08-04',
                          '2015-07-04', '2015-08-04'])
data = pd.Series([0, 1, 2, 3], index=index)
print(data)
print('Obtainig data for a specific year:\n')
print(data['2015'], '\n\n')
'''
A closer look at the available time series data structures.:
For time stamps, Pandas provides the Timestamp type. 
As mentioned before, it is essentially a replacement for Python's native datetime, 
but is based on the more efficient numpy.datetime64 data type. 
The associated Index structure is DatetimeIndex.

For time Periods, Pandas provides the Period type.
This encodes a fixed-frequency interval based on numpy.datetime64. 
The associated index structure is PeriodIndex.

For time deltas or durations, Pandas provides the Timedelta type. 
Timedelta is a more efficient replacement for Python's native datetime.timedelta type, 
and is based on numpy.timedelta64. The associated index structure is TimedeltaIndex.

The most fundamental of these date/time objects are the Timestamp and DatetimeIndex objects. 

Passing a single date to pd.to_datetime() yields a Timestamp; 
passing a series of dates by default yields a DatetimeIndex
'''
dates = pd.to_datetime([datetime.datetime(2015, 7, 3), '4th of July, 2015',
                       '2015-Jul-6', '07-07-2015', '20150708'])
print(dates)
print('Any DatetimeIndex can be converted to a PeriodIndex '
      'with the to_period() function with the addition of a frequency code:\n')
print(dates.to_period('D'), '\n')
print('A TimedeltaIndex is created, for example, when a date is subtracted from another:\n')
print(dates-dates[0])
print('Regular Sequences: pd.date_range():\n\n')
'''
To make the creation of regular date sequences more convenient, Pandas offers a few functions for this purpose:
 pd.date_range() for timestamps, pd.period_range() for periods, and pd.timedelta_range() for time deltas. 
'''
print(pd.date_range(start='01-01-2018', end='01-12-2018', freq='D'))
print(pd.date_range(start='01-01-2018', periods=12, freq='D'))
print(pd.date_range(start='01-01-2018', periods=12, freq='H'), '\n\n\n')
'''
To create regular sequences of Period or Timedelta values, 
the very similar pd.period_range() and pd.timedelta_range() functions are useful.
'''
print(pd.period_range(start='2015-07', periods=12, freq='D'))
print(pd.period_range(start='2015-07', periods=12, freq='H'), '\n')
print('And a sequence of durations increasing by an hour:')
print(pd.timedelta_range(0, periods=10, freq='H'), '\n')
'''
Frequencies and offsets:
Code	Description     	Code	Description
D	    Calendar day	    B	    Business day
W	    Weekly		
M	    Month end	        BM	    Business month end
Q	    Quarter end	        BQ	    Business quarter end
A	    Year end	        BA	    Business year end
H	    Hours	            BH	    Business hours
T	    Minutes		
S	    Seconds		
L	    Milliseonds		
U	    Microseconds		
N	    nanoseconds		
The monthly, quarterly, and annual frequencies are all marked at the end of the specified period. 
By adding an S suffix to any of these, they instead will be marked at the beginning
Code	Description		Code	Description
MS	    Month start		BMS	    Business month start
QS	    Quarter start	BQS	    Business quarter start
AS	    Year start		BAS	    Business year start
'''
print('time delta range by 2hr and 30 min frequency!')
print(pd.timedelta_range(0, periods=9, freq='2H30T'))
print('Business days, weekend not included!!:')
print(pd.date_range('2015-07-01', periods=7, freq='B'), '\n')
print('Business hours:')
print(pd.date_range('2015-07-01', periods=10, freq='BH'), '\n')

print('Business months:- stamps end of the month')
print(pd.date_range('2015-01-01', periods=13, freq='BM'))
print('Business month, stamps end begining of the month:\n')
print(pd.date_range('2015-01-01', periods=13, freq='BMS'))
