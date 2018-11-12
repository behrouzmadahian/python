import numpy as np
import scipy.stats
'''
Calculates the T-test for the means of TWO INDEPENDENT samples of scores.
If True (default), perform a standard independent 2 sample test that assumes equal population variances.
 If False, perform Welch’s t-test, which does not assume equal population variance

'''
x = np.random.rand(100)
y = np.random.rand(100)
results = scipy.stats.ttest_ind(x, y, equal_var=True)
print(list(results))

'''
ONE sample ttest
'''
a = scipy.stats.ttest_1samp(x, 1, axis=0)
print(a)
''' 
for one sided ttest, just divide the p value by half and pay attention to the sign of the t stat
'''
