import pandas as pd
'''One essential feature offered by Pandas is its high-performance, in-memory join and merge operations. '''
# 1 to 1 join:
print('One to One JOINs')
df1 = pd.DataFrame({'employee': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'group': ['Accounting', 'Engineering', 'Engineering', 'HR']}, index=[10, 20, 30, 40])
df2 = pd.DataFrame({'employee': ['Lisa', 'Bob', 'Jake', 'Sue'],
                    'hire_date': [2004, 2008, 2012, 2014]}, index=[50, 60, 70, 80])
print(df1, '\n')
print(df2, '\n')
df3 = pd.merge(df1, df2, on='employee', how='inner')
print(df3, '\n')
print('keep in mind that the merge in general discards the index, except in the special case of merges by index \n\n')
# Many to One joins:
print('Many to one joins: ')

'''
Many-to-one joins are joins in which one of the two key columns contains duplicate entries.
 For the many-to-one case, the resulting DataFrame will preserve those duplicate entries as appropriate. 
'''
df4 = pd.DataFrame({'group': ['Accounting', 'Engineering', 'HR'],
                    'supervisor': ['Carly', 'Guido', 'Steve']})
print(df3, '\n')
print(df4, '\n')
print(pd.merge(df3, df4, on='group', how='inner'), '\n')
# many to many joins:
print('Many to Many joins:')
'''
Many-to-many joins are a bit confusing conceptually, but are nevertheless well defined. 
If the key column in both the left and right array contains duplicates, then the result is a many-to-many merge.
'''
df5 = pd.DataFrame({'group': ['Accounting', 'Accounting',
                              'Engineering', 'Engineering', 'HR', 'HR'],
                    'skills': ['math', 'spreadsheets', 'coding', 'linux',
                               'spreadsheets', 'organization']})
print(df1, '\n')
print(df5, '\n')
print(pd.merge(df1, df5, on='group', how='inner'), '\n')
'''
left_on right_on keywords:
At times you may wish to merge two datasets with different column names; for example, 
we may have a dataset in which the employee name is labeled as "name" rather than "employee". 
In this case, we can use the left_on and right_on keywords to specify the two column names:
'''
print('\n\n\n')
df3 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'salary': [70000, 80000, 120000, 90000]})
print(df1, '\n')
print(df3, '\n')
print(pd.merge(df1, df3, left_on="employee", right_on="name", how='inner'), '\n\n\n')

'''
The left_index and right_index keywords:
Sometimes, rather than merging on a column, you would instead like to merge on an index
'''
print('Joining on Index:\n')
df1a = df1.set_index('employee')
df2a = df2.set_index('employee')
print(df1a, '\n')
print(df2a, '\n')
print(pd.merge(df1a, df2a, left_index=True, right_index=True, how='inner'), '\n\n')
print('Joining using index in first table and one column on the other!\n')
print(df3, '\n')
print(pd.merge(df1a, df3, left_index=True, right_on='name', how='inner'))

'''Overlapping Column Names: The suffixes Keyword
Because the output would have two conflicting column names, the merge function automatically 
appends a suffix _x or _y to make the output columns unique. If these defaults are inappropriate,
 it is possible to specify a custom suffix using the suffixes keyword:
'''
df8 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'name1': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [1, 2, 3, 4]})
df9 = pd.DataFrame({'name': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'name1': ['Bob', 'Jake', 'Lisa', 'Sue'],
                    'rank': [3, 1, 4, 2]})
print(df8, '\n')
print(df9, '\n')
print(pd.merge(df8, df9, how='inner', on='name'), '\n')
print('Using suffixes keyword:')
print(pd.merge(df8, df9, on='name', how='inner', suffixes=['-t1', '-t2']), '\n')
