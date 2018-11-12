import pandas as pd
# a Pandas Series is a one dimensional array of indexed data. It can be created from a list or array
data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data, data.values, data.index.tolist())
print('Accessing part of the Series by indexing')
print(data[:2], '\n')
# Series, a Generalized Numpy array:
# the index can be anything and can be used to give more capabilities than numpy array has!
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data, data['b'], data.loc['b'])
print('='*20)
# Series as a specialized dictionary
population_dict = {'California': 38332521,
                   'Texas': 26448193,
                   'New York': 19651127,
                   'Florida': 19552860,
                   'Illinois': 12882135}
population = pd.Series(population_dict)
print(population, '\n')
print(population['California'])
print('Array Style slicing:')
print(population['California': 'New York'])
print('A Series of length 3 filled with 5:')
s1 = pd.Series(5, index=[100, 200, 300])
print(s1, '\n')

print('building a series object from dictionary but setting index explicitly:')
s2 = pd.Series({2: 'a', 1: 'b', 3: 'c'}, index=[3, 2])
print(s2, '\n')
print('When creating DataFrame, if data contains only ONE ROW, We NEED to provide the index !!!\n')
d1 = pd.DataFrame(population_dict, index=[1])
print(d1, '\n\n', 'Index: ', d1.index.tolist(), '\n')
