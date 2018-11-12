import random
'''
generate a random permutation of numbers between 0-n
use as few calls to random number generator as possible
'''
def random_permutation(n):
    a = list(range(n))
    print(a)
    for i in range(n):
        ind = random.randint(i, n - 1)
        a[i], a[ind] = a[ind], a[i]
    return a

for i in range(1):
    print(random_permutation(10))