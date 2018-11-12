import numpy as np
import scipy.stats
import pandas as pd
from matplotlib import pyplot as plt
'''
The one-way ANOVA tests the null hypothesis that two or more groups have the same population mean.
 The test is applied to samples from two or more groups, possibly with differing sizes.
 The ANOVA test has important assumptions that must be satisfied in order for the associated p-value to be valid.
Assumptions
1. The samples are independent.
2. Each sample is from a normally distributed population.
3. The population standard deviations of the groups are all equal. This property is known as homoscedasticity.

If these assumptions are not true for a given set of data, it may still be possible to use the Kruskal-Wallis H-test 
'''
tillamook = [0.0571, 0.0813, 0.0831, 0.0976, 0.0817, 0.0859, 0.0735,
             0.0659, 0.0923, 0.0836]
newport = [0.0873, 0.0662, 0.0672, 0.0819, 0.0749, 0.0649, 0.0835, 0.0725]
petersburg = [0.0974, 0.1352, 0.0817, 0.1016, 0.0968, 0.1064, 0.105]
magadan = [0.1033, 0.0915, 0.0781, 0.0685, 0.0677, 0.0697, 0.0764, 0.0689]
tvarminne = [0.0703, 0.1026, 0.0956, 0.0973, 0.1039, 0.1045]
anova = scipy.stats.f_oneway(tillamook, petersburg,newport, magadan, tvarminne)
print(anova)

from statsmodels.stats.multicomp import pairwise_tukeyhsd
groups = np.array(['one']*len(tillamook) + ['two'] * len(petersburg) + ['three'] * len(newport) +
                  ['four'] * len(magadan) + ['five'] * len(tvarminne))
data = np.concatenate([tillamook, newport, petersburg, magadan, tvarminne], axis=0)
tukey = pairwise_tukeyhsd(data, groups, alpha=0.05)
print(tukey)
# lines are confidence intervals for each estimate. we want groups that do not overlap!
tukey.plot_simultaneous()
plt.show()
print('\n\n')





races = ["asian", "black", "hispanic", "other", "white"]

# Generate random data
voter_race = np.random.choice(a=races,
                              p=[0.05, 0.15 ,0.25, 0.05, 0.5],
                              size=500)

voter_age = scipy.stats.poisson.rvs(loc=18,
                                    mu=30,
                                    size=500)
voter_random = np.random.rand(500)

# Group age data by race
voter_frame = pd.DataFrame({"race": voter_race, "age": voter_age, 'random':voter_random})
print(voter_frame.groupby('race'))
groups = voter_frame.groupby("race").groups
print(groups.keys, '===')
asian = voter_frame['age'][groups["asian"]]

print(asian)
