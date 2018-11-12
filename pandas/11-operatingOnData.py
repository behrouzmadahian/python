import numpy as np
import pandas as pd
ser = pd.Series(np.random.randint(0, 10, size=4))
print(ser)
df = pd.DataFrame(np.random.randint(0, 10, size=(3, 4)), columns=['A', 'B', 'C', 'D'])

print('If we apply a NumPy ufunc<universal function> on either of these objects, the result'
      ' will be another Pandas object with the indices preserved:')
print(np.exp(ser))
print(np.exp(df))
print(np.sin(df*np.pi/4))

'''
For binary operations on two Series or DataFrame objects, Pandas will align indices
in the process of performing the operation. This is very convenient when working
with incomplete data, as we'll see in some of the examples that follow.
'''

'''
As an example, suppose we are combining two different data sources, 
and find only the top three US states by area and the top three US states by population:
The resulting array contains the union of indices of the two input arrays. filled with NaN in places that
data is missing on one source!
'''
area = pd.Series({'Alaska': 1723337, 'Texas': 695662,
                  'California': 423967}, name='area')
print('*'*100)
print(area, '\n')
population = pd.Series({'California': 38332521, 'Texas': 26448193,
                        'New York': 19651127}, name='population')
print('Population Density: \n', population / area, '\n')
print('Another Example:\n')
A = pd.Series([2, 4, 6], index=[0, 1, 2])
B = pd.Series([1, 3, 5], index=[1, 2, 3])
print(A + B)
print('If using NaN values is not the desired behavior, the fill value '
      'can be modified using appropriate object methods in place of the operators. ')
'''
allows optional explicit specification of the fill value for any elements in A or B that might be missing
""""""""""BEFORE"""""""""" performing addition!
'''
print('Specifying fill value: \n', A.add(B, fill_value=0), '\n')
#############
print('index alignment in DataFrame:')
A = pd.DataFrame(np.random.randint(0, 20, size=(2, 2)),
                 columns=list('AB'))
B = pd.DataFrame(np.random.randint(0, 10, size=(3, 3)),
                 columns=list('BAC'))

print(A)
print(B)
print('Aligns by row and column and then performs addition and fills with NaN when necessary!')
print(A+B)

'''
Here we'll fill with the mean of all values in A (computed by first stacking the rows of A)
'''
print('Stack function: ')
print(A, '\n')
print(A.stack())
fill = A.stack().mean()
print(A.add(B, fill_value=fill), '\n')
'''
The following table lists Python operators and their equivalent Pandas object methods
+	add()
-	sub(), subtract()
*	mul(), multiply()
/	truediv(), div(), divide()
//	floordiv()
%	mod()
**	pow()
'''
######################
print('Operations Between DataFrame and Series:\n\n')
'''
When performing operations between a DataFrame and a Series, the index and column alignment is similarly maintained.
Operations between a DataFrame and a Series are similar to 
operations between a two-dimensional and one-dimensional NumPy array.

Consider one common operation, where we find the difference of a two-dimensional array and one of its rows.
According to NumPy's broadcasting rules, subtraction between a two-dimensional 
array and one of its rows is applied row-wise.
'''
A = np.random.randint(10, size=(3, 4))
print(A, '\n')
print(A - A[0])

df = pd.DataFrame(A, columns=list('QRST'))
print('*****')
print(df)
print('*****')
print(df - df.iloc[0])
print('If you would instead like to operate column-wise, you can '
      'use the object methods mentioned earlier, while specifying the axis keyword')
print(df.subtract(df['R'], axis=0))
'''
Note that these DataFrame/Series operations, like the operations discussed above,
will automatically align indices between the two elements:
'''
print('-----')
halfrow = df.iloc[0, ::2]
print(halfrow)
print(df)
print(df - halfrow)

'''
This preservation and alignment of indices and columns means that operations on data in Pandas
will always maintain the data context, which prevents the types of silly errors that might
come up when working with heterogeneous and/or misaligned data in raw NumPy arrays.
'''