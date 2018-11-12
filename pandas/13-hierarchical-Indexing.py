import pandas as pd
'''
Up to this point we've been focused primarily on one-dimensional and two-dimensional data, 
stored in Pandas Series and DataFrame objects, respectively. Often it is useful to go beyond
these and store higher-dimensional data–that is, data indexed by more than one or two keys. 
'''
print('Representing two-dimensional data with one-dimensional series:\n')
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
pop = pd.Series(populations, index=index)
print(pop, '\n')

index = pd.MultiIndex.from_tuples(index)
'''
Notice that the MultiIndex contains multiple levels of indexing–in this case,
the state names and the years, as well as multiple labels for each data point which encode these levels.
'''
print('Building MultiIndex:\n', index)
pop = pop.reindex(index)
print('Our Series with multiIndex:\n')
print(pop)
print('Accessing data for year=2010:\n')
print(pop[:, 2010])
'''
We'll now further discuss this sort of indexing operation on hierarchically indexed data.
'''
# 2-MultiIndex as extra dimension:
print('\nThe pandas built-in unstack() method will quickly convert a multiply '
      'indexed Series into a conventionally indexed DataFrame:')
pop_df = pop.unstack()
print(pop_df)

print('\nstack() method provides the opposite operation: ')
pop_df = pop_df.stack()
print(pop_df)
#########################################################################################
print('\nMultiple indexing Data Frame- 2 Dimensional data:')
populations = [33871648, 37253956,
               18976457, 19378102,
               20851820, 25145561]
index = [('California', 2000), ('California', 2010),
         ('New York', 2000), ('New York', 2010),
         ('Texas', 2000), ('Texas', 2010)]

index = pd.MultiIndex.from_tuples(index)

pop_df = pd.DataFrame({'total': populations,
                       'under18': [9267089, 9284094,
                                   4687374, 4318033,
                                   5906301, 6879014],
                       },
                      index=index)
print(pop_df)
'''
This allows us to easily and quickly manipulate and explore even high-dimensional data.
'''
print('\nAll the ufuncs and other functionality discussed in Operating on Data in Pandas work with hierarchical'
      ' indices as well. Here we compute the fraction of people under 18 by year in each state, given the above data:')
f_u18 = pop_df['under18']/pop_df['total']
print(f_u18, '\n')
f_u18 = f_u18.unstack()
print('Unstacked results:\n', f_u18)
