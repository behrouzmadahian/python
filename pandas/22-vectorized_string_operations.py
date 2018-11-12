import pandas as pd
'''
One strength of Python is its relative ease in handling and manipulating string data. 
Pandas builds on this and provides a comprehensive set of vectorized string operations
that become an essential piece of the type of munging required when working with (read: cleaning up) real-world data.
In this section, we'll walk through some of the Pandas string operations, and then take a 
look at using them to partially clean up a very messy dataset of recipes collected from the Internet. 
pandas provide facilities to apply some function to several string types at the same time
and handles missing values as well!
'''
data = ['peter', 'Paul', None, 'MARY', 'gUIDO']
names = pd.Series(data)
print(names)
print('Capitalizing all the entries in our series data:')
print(names.str.capitalize(), '\n')
'''
If you have a good understanding of string manipulation in Python, most of Pandas string 
syntax is intuitive enough that it's probably sufficient to just list a table of available methods.
Methods similar to Python string methods:
Nearly all Python's built-in string methods are mirrored by a Pandas 
vectorized string method. Here is a list of Pandas str methods:

len()	    lower() 	    translate()	    islower()
ljust()	    upper() 	    startswith()	isupper()
rjust()	    find()	        endswith()	    isnumeric()
center()	rfind()	        isalnum()	    isdecimal()
zfill()	    index() 	    isalpha()	    split()
strip()	    rindex()	    isdigit()	    rsplit()
rstrip()	capitalize()	isspace()	    partition()
lstrip()	swapcase()	    istitle()	    rpartition()
'''
monte = pd.Series(['Graham Chapman', 'John Cleese', 'Terry Gilliam',
                   'Eric Idle', 'Terry Jones', 'Michael Palin'])
print('str.lower():')
print(monte.str.lower(), '\n')
print('Returns the length of each entry:\n\n', monte.str.len())
print('Returns true false: starting with T:\n')
print(monte.str.startswith('T'), '\n')
print('splitting each entries if its has space in the string:\n')
print(monte.str.split(), '\n')

'''
Methods using regular Expressions:
Method	    Description
match()	    Call re.match() on each element, returning a boolean.
extract()	Call re.match() on each element, returning matched groups as strings.
findall()	Call re.findall() on each element
replace()	Replace occurrences of pattern with some other string
contains()	Call re.search() on each element, returning a boolean
count()	    Count occurrences of pattern
split()	    Equivalent to str.split(), but accepts regexps
rsplit()	Equivalent to str.rsplit(), but accepts regexps
'''
print("extract the first name from each by asking "
      "for a contiguous group of characters at the beginning of each element:\n")
# if expand=True: returns dataframe if false returns series
print(monte.str.extract('([A-Za-z]+)', expand=False), '\n')
print('finding all names that start and end with a consonant\n')
print(monte.str.findall(r'^[^AEIOU].*[^aeiou]$'), '\n')
'''
Other methods:
Method	        Description
get()	        Index each element
slice()	        Slice each element
slice_replace()	Replace slice in each element with passed value
cat()	        Concatenate strings
repeat()	    Repeat values
normalize()	    Return Unicode form of string
pad()	        Add whitespace to left, right, or both sides of strings
wrap()	        Split long strings into lines with length less than a given width
join()	        Join strings in each element of the Series with passed separator
get_dummies()	extract dummy variables as a dataframe.
'''
'''
The get() and slice() operations, in particular, enable vectorized element access from each array
we can get a slice of the first three characters of each array using str.slice(0, 3)
Note that this behavior is also available through Python's normal indexing syntax–for example,
df.str.slice(0, 3) is equivalent to df.str[0:3]
Indexing via df.str.get(i) and df.str[i] is likewise similar.
'''
print('Vectorized item access and slicing:\n')
print(monte.str[0:3], '\n')
print(monte.str.slice(0, 3), '\n')
print('Indexing using Get:\n')
print('get the first character of each entry!!')
print(monte.str.get(0), '\n')
print('Getting the last name:\n')
print(monte.str.split().str.get(1))
'''
Indicator Variables:
Another method that requires a bit of extra explanation is the get_dummies() method. 
This is useful when your data has a column containing some sort of coded indicator. 
'''
full_monte = pd.DataFrame({'name': monte,
                           'info': ['B|C|D', 'B|D', 'A|C',
                                    'B|D', 'B|C', 'B|C|D']})
print(full_monte, '\n')
print('The get_dummies() routine lets you quickly split-out these indicator variables into a DataFrame:\n')
print(full_monte['info'].str.get_dummies('|'))