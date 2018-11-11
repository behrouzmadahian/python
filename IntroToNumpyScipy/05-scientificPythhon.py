from scipy.stats import norm
from scipy.stats import expon
import numpy as np

# import sys
# sys.path.append("C:/Python27/Lib/site-packages/scipy/stats")
'''
#scipy
#SciPy greatly extends the functionality of the NumPy routines.
#sub-package scipy.stats:
#all the distributions are accompanied with help docs:
#print(((norm.__doc__)

#To find the support, i.e., upper and lower bound of the distribution, call:
print('bound of the distribution lower: %s, upper: %s' %(norm.a,norm.b))
#We can list all methods and properties of the distribution with dir(norm)
print(dir(norm))
#rvs: Random Variates
#pdf: Probability Density Function
#cdf: Cumulative Distribution Function
#sf: Survival Function (1-CDF)
#ppf: Percent Point Function (Inverse of CDF)
#isf: Inverse Survival Function (Inverse of SF)
# stats: Return mean, variance, (Fisher's) skew, or (Fisher's) kurtosis
#moment: non-central moments of the distribution
'''
print('Normal distribution CDF X<=0: ', norm.cdf(0))
a = np.array([-1, 0, 1])
print(norm.cdf(a))
print(norm.ppf(0.05))
print(norm.ppf(0.025))
# generating random normal variates:
r = norm.rvs(0, 1, size=10)
print(r)
######################
# shifting and scaling:
# All continuous distributions take loc and scale as keyword
# parameters to adjust the location and scale of the distribution
# for example for normal dist, location is the mean and scale is standard deviation
# stat keyword gives moment of the distribution
print('\n', 'Shifting and Scaling:')
a = norm.stats(loc=3, scale=4, moments="mv")
print(a)
aVec = norm.stats(loc=[0, 1, 2], scale=[1, 2, 4], moments="mv")
print('vectorized: ')
print(aVec)
# random numbers from multivariate normal
# does not give Multivariate normal!
print('MULTI?! Does not give multivariate Normal!')
rvs = norm.rvs(loc=np.array([0, 1]), scale=np.array([[1, 0], [0, 1]]))
print(rvs)
print('\n\n')
#############################################################
#############################################################
# In general the standardized distribution for a random variable X is
# obtained through the transformation (X - loc) / scale. The
# default values are loc = 0 and scale = 1.
print('exponential dist')
print(expon.stats())
# for exponential dist scale=1/lambda
print(expon.mean(scale=3))

print('UNIFORM!!!:')
# This distribution is constant between loc and loc + scale.
from scipy.stats import uniform
print(uniform.cdf([0, 1, 2, 34, 5], loc=1, scale=4))
print(np.mean(norm.rvs(5, size=500)))
# shape parameters:
from scipy.stats import gamma
print('GAMMA')
print(gamma(a=1, scale=2).stats(moments="mv"))
print(gamma.stats(a=1, scale=2, moments="mv"))

###################################################
# freezing a distribution:
rv = gamma(a=1, scale=2)
print(rv.mean(), rv.std())
from scipy.stats import t
#help(t.__doc__)
######
#isf gives the critical value of the dist for the given tail
print('T-dsitribution Critical value:')
print('level= 0.05:  ', t.isf(0.05, df=20))
print('level = 0.025', t.isf(0.025, df=20))

'''
#specific points for discrete distributions:
#Discrete distribution have mostly the same basic methods as the
#continuous distributions. However pdf is replaced the probability
#mass function pmf, no estimation methods, such as fit, are available,
#and scale is not a valid keyword parameter
######################################################################
#ppf(q) = min{x : cdf(x) >= q, x integer}
'''
from scipy.stats import hypergeom
[M, n, N] = [20, 7, 12]
x = np.arange(4)*2
print(x)
print('Hyper Geometric dist:')
prb = hypergeom.cdf(x, M, n, N)
print(prb)

# now getting the points back!
points = hypergeom.ppf(prb, M, n, N)
print(points)
'''
#fitting distributions:
#The main additional methods of the not frozen distribution are related
#to the estimation of distribution parameters:
#fit: maximum likelihood estimation of distribution
#parameters, including location and scale


#fit_loc_scale: estimation of location and scale when shape
#parameters are given
#nnlf: negative log likelihood function
#expect: Calculate the expectation of a function against the pdf or pmf
'''
#########################################################
# multivariate normal distribution:
from scipy.stats import multivariate_normal
multi = multivariate_normal.rvs([0, 0], [[1, .5], [.5, 2]])
print('multivariate normal: ')
print(multi)
