import numpy as np
from datetime import datetime
from dateutil import parser
'''
Pandas was developed in the context of financial modeling, so as you might expect, it contains a fairly extensive
set of tools for working with dates, times, and time-indexed data. Date and time data comes in a 
few flavors, which we will discuss here:

1. Time stamps reference particular moments in time (e.g., July 4th, 2015 at 7:00am).
2. Time intervals and periods reference a length of time between a particular beginning and end point;
 for example,the year 2015.
Periods usually reference a special case of time intervals in which each interval is of uniform
length and does not overlap (e.g., 24 hour-long periods comprising days).
3. Time deltas or durations reference an exact length of time (e.g., a duration of 22.56 seconds).

we will introduce how to work with each of these types of date/time data in Pandas.
'''
print('Naiive Python dates and times: datetime and dateutil')
print(datetime(year=2015, month=7, day=4, hour=1, minute=25, second=3))
date = parser.parse('4th of July, 2015')
print(date)
print('Once you have a datetime object, you can do things like printing the day of the week:\n')
print(date.strftime('%A'), '\n')
'''
Where they break down is when you wish to work with large arrays of dates and times!!
lists of Python datetime objects are suboptimal compared to typed arrays of encoded dates!!
'''
print('Typed Arrays of times: Numpy"s datetime64:\n')
'''
The weaknesses of Python's datetime format inspired the NumPy team to add a set of 
native time series data type to NumPy. The datetime64 dtype encodes dates as 64-bit integers,
and thus allows arrays of dates to be represented very compactly. 
        <<<<The datetime64 requires a very specific input format>>>>
'''
date = np.array('2015-07-04', dtype=np.datetime64)
print(date, '\n')
print('Now that we have this date formatted, we can quickly do vectorized operations on it.')
myDateRange = date + np.arange(12)
print('a date range starting from 2015-07-04:\n')
print(myDateRange, '\n')
'''
Because of the uniform type in NumPy datetime64 arrays, this type of operation can be accomplished
much more quickly than if we were working directly with Python's datetime objects, especially as arrays get large
One detail of the <<datetime64>> and <<timedelta64>> objects is that they are built on a fundamental time unit. 
Because the datetime64 object is limited to 64-bit precision, the range of encodable times is $2^{64}$ times
this fundamental unit. 
In other words, datetime64 imposes a trade-off between time resolution and maximum time span.
For example, if you want a time resolution of one nanosecond, you only have enough information 
to encode a range of $2^{64}$ nanoseconds, or just under 600 years.
NumPy will infer the desired unit from the input!!
'''
print('Inferring datetime unit from the input:\n')
print(np.datetime64('2015-07-04'))
# timezone automatically set to local time!!
print((np.datetime64('2015-04-07 12:00')))
print(np.datetime64('2015-07-04 12:59:59.50', 'ns'))
'''
list of the available format codes along with the relative and absolute time spans that they can encode:
Code Meaning	    Time span (relative)	Time span (absolute)
Y	Year	        ± 9.2e18 years	        [9.2e18 BC, 9.2e18 AD]
M	Month	        ± 7.6e17 years	        [7.6e17 BC, 7.6e17 AD]
W	Week	        ± 1.7e17 years	        [1.7e17 BC, 1.7e17 AD]
D	Day	            ± 2.5e16 years	        [2.5e16 BC, 2.5e16 AD]
h	Hour	        ± 1.0e15 years	        [1.0e15 BC, 1.0e15 AD]
m	Minute	        ± 1.7e13 years	        [1.7e13 BC, 1.7e13 AD]
s	Second	        ± 2.9e12 years	        [ 2.9e9 BC, 2.9e9 AD]
ms	Millisecond	    ± 2.9e9 years	        [ 2.9e6 BC, 2.9e6 AD]
us	Microsecond 	± 2.9e6 years	        [290301 BC, 294241 AD]
ns	Nanosecond	    ± 292 years         	[ 1678 AD, 2262 AD]
ps	Picosecond	    ± 106 days	            [ 1969 AD, 1970 AD]
fs	Femtosecond 	± 2.6 hours	            [ 1969 AD, 1970 AD]
as	Attosecond	    ± 9.2 seconds	        [ 1969 AD, 1970 AD]
'''