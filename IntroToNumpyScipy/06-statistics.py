__author__ = 'bmdahian'
import numpy as np
a = np.array([1, 4, 3, 8, 9, 2, 3])
print(np.median(a), np.mean(a), np.var(a), np.std(a))
a = np.array([[1, 2, 1, 3], [5, 3, 1, 8]])
print(np.corrcoef(a))
print(np.cov(a))
#############
# random numbers:
# NumPy uses a particular algorithm called the Mersenne Twister to generate pseudorandom numbers
# setting the seed:
np.random.seed(245678)
# Any program that starts with the same seed will generate exactly the same sequence of random numbers
#  each time it is run.
print('1-uniform(0,1) random numbers:')
print(np.random.rand(5))
print(' we can shape the array as well!:')
print(np.random.rand(2, 3))
print(np.random.rand())  # single random number
print('2-random integers: between two numbers given [min,max)')
print(np.random.randint(5, 10))
# there are several other distributions as well:
print('poission with lambda=6:')
print(np.random.poisson(6, 3))  # generate 3 numbers from poi(6)
print(np.random.normal(0, 1, 5))  # five random numbers from normal
# randomly shuffling the order of items in a list
l = np.arange(10)
print(l)
np.random.shuffle(l)  # in place shuffling
print(l)