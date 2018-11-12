import pandas as pd
import numpy as np
import numexpr
import time, timeit
'''
we cover eval() and query() functions.
As we've already seen in previous sections, the power of the PyData stack is built upon the ability of NumPy 
and Pandas to push basic operations into C via an intuitive syntax: examples are vectorized/broadcasted operations
in NumPy, and grouping-type operations in Pandas. While these abstractions are efficient and effective for many
common use cases, they often rely on the creation of temporary intermediate objects, which can cause
undue overhead in computational time and memory use.
As of version 0.13 (released January 2014), Pandas includes some experimental tools that allow you to directly 
access C-speed operations without costly allocation of intermediate arrays. 
These are the eval() and query() functions, which rely on the Numexpr package.
'''
print('Compound Expressions:\n')
rng = np.random.RandomState(42)
x = rng.rand(100000)
y = rng.rand(100000)
max_expr = numexpr.evaluate('(x > 0.5) & (y < 0.5)')
print(max_expr, '\n')
'''
The benefit here is that Numexpr evaluates the expression in a way that does not use full-sized
temporary arrays, and thus can be much more efficient than NumPy, especially for large arrays. 
'''
print('pandas.eval() for Efficient Operations:\n')
nrows, ncols = 100000, 100
df1, df2, df3, df4 = (pd.DataFrame(rng.rand(nrows, ncols))
                      for i in range(4))
t1 = time.time()
stmt = df1 + df2 + df3 + df4
t2 = time.time()
print('Total time taken = ', t2-t1)

t1 = time.time()
stmt1 = pd.eval('df1 + df2 + df3 + df4')
t2 = time.time()
print('Total time taken pd.eval= ', t2-t1)
# numpy.allclose: returns true if two arrays are element-wise equal within a tolerance!
print('Operations supported by pd.eval(): supports a wide range of operations. \n')
results = pd.eval('df1 < df2 <= df3 != df4')
print('Object attributes and indices')
print('pd.eval() supports access to object attributes via the obj.attr syntax, and indexes via the obj[index] syntax:\n')
results2 = pd.eval('df2.T[0] + df3.iloc[1]')
'''
Other operations such as function calls, conditional statements, loops, 
and other more involved constructs are currently not implemented in pd.eval().
'''
print('DataFrame.eval() for Column-Wise Operations:\n')
print('The benefit of the myDf.eval() method is that columns can be referred to by name. ')
df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
print(df.head(), '\n\n')

rslt2 = df.eval("(A + B) / (C - 1)")
rslt3 = (df['A'] + df['B']) / (df['C'] - 1)
print(np.allclose(rslt2, rslt3), '\n\n')
print('Assignment in DataFrame.eval():\n')
df.eval('D = (A + B) / C', inplace=True)
print(df.head(), '\n')
print('In the same way, any existing column can be modified:\n\n')
df.eval('D = (A - B) / C', inplace=True)
print(df.head(), '\n')
print('Local variables in DataFrame.eval():\n\n')
'''
The DataFrame.eval() method supports an additional syntax that lets it
work with local Python variables. Consider the following:
'''
col_mean = df.mean()*1000
print(col_mean)
result = df.eval('A = A + @col_mean[0]', inplace=False)
print(df.head(), '\n')
print(result.head(), '\n')
result = df.eval('A + @col_mean[0]', inplace=False)
print(result, '\n')
