import pandas as pd
import numpy as np
'''
The DataFrame has another method based on evaluated strings, called the query() method. Consider the following:
'''
print('Consider the following:\n\n')
rng = np.random.RandomState(42)

df = pd.DataFrame(rng.rand(1000, 3), columns=['A', 'B', 'C'])
result1 = df[(df.A < 0.5) & (df.B < 0.5)]
result2 = pd.eval('df[(df.A < 0.5) & (df.B < 0.5)]')
print(np.allclose(result1, result2))

'''
As with the example used in our discussion of DataFrame.eval(), this is an expression involving
columns of the DataFrame. It cannot be expressed using the DataFrame.eval() syntax, however!
Instead, for this type of filtering operation, you can use the query() method:
This is Row selection, and can not be performed using df.eval()
 -> USE df.query()
can be pefromed using p.eval() also!
'''
results3 = df.query('A < 0.5 and B < 0.5')
print(np.allclose(result1, results3))
print(result1.shape, results3.shape)
'''
In addition to being a more efficient computation, compared to the masking expression this is much easier
to read and understand. Note that the query() method also accepts the @ flag to mark local variables.
'''
cmean = df['C'].mean()
rs1 = df[(df['A'] < cmean) & (df['B'] < cmean)]
rs2 = df.query('(A < @cmean) and (B < @cmean)')
print(np.allclose(rs1, rs2))
'''
When considering whether to use these functions, there are two considerations: computation time and memory use.
Memory use is the most predictable aspect. As already mentioned, every compound expression involving NumPy arrays
or Pandas DataFrames will result in implicit creation of temporary arrays
If the size of the temporary DataFrames is significant compared to your available system memory 
(typically several gigabytes) then it's a good idea to use an eval() or query() expression. 
You can check the approximate size of your array in bytes using this:

df.values.nbytes
'''
print('Size of the data frame: ', df.values.nbytes)
'''
On the performance side, eval() can be faster even when you are not maxing-out your system memory. 
The issue is how your temporary DataFrames compare to the size of the L1 or L2 CPU cache on your system 
(typically a few megabytes in 2016); if they are much bigger, then eval() can avoid some 
potentially slow movement of values between the different memory caches. 
'''

'''
In practice, I find that the difference in computation time between the traditional methods and the eval/query
method is usually not significant–if anything, the traditional method is faster for smaller arrays!
The benefit of eval/query is mainly in the saved memory, and the sometimes cleaner syntax they offer.
'''