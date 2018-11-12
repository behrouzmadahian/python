import random
import numpy as np
'''
write a program that maintains a sample of size k of online data.
assume n data points are out, we maintain a random sample of size k.
now data n+1 comes, we need to incorporate that such that we maintain a random subset of data as
it comes.
note: probability of new sample belonging to the subset of size k out of all samples (n) is:
k/ (n + 1).
In other words, probability of any number being removed from subset to make room for new
sample is this.
we could generate a random int from 0-n+1, if it was less than k, replace the sample[idx] with new
sample
'''
def random_sample(A, k):
    if k <= len(A)//2:
        for i in range(k):
            ind = random.sample(i, len(A) - 1)
            A[i], A[ind] = A[ind], A[i]
        return A[:k]
    else:
        k_prime = len(A) - k
        for i in range(k_prime):
            ind = random.randint(i, len(A) - 1)
            A[i], A[ind] = A[ind], A[i]
        return A[k_prime:]

def incorporate_online_sample(curr_sample, total_data_n, b):
    p = np.random.rand()
    if p >= len(curr_sample) / (total_data_n + 1):
        ind = random.randint(0, len(curr_sample))[0]
        curr_sample[ind] = b
