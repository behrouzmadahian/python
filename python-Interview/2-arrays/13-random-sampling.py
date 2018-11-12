import random
'''
Given an array of length n, randomly sample k elements.
we want time complexity to be O(k) and space complexity O(1). i.e., put all the selected
elements at the begining of the array.
note: if k> n/2, we could modify the algorithm to remove n-k elements randomly to improve time
complexity.
This is sampling without replacement!
'''
def random_sample(A, k):
    for i in range(k):
        ind = random.randint(i, len(A) - 1)
        A[i], A[ind] = A[ind], A[i]

    return A[:k]

print(random_sample([1,2,3,4,5,6,7,8, 5], 5))

# now the version that k is greater than n/2:
def random_sample1(A, k):
    k_prime = len(A) - k
    for i in range(k_prime):
        ind = random.randint(i, len(A) - 1)
        A[i], A[ind] = A[ind], A[i]

    return A[k_prime:]

print(random_sample1([1,2,3,4,5,6,7,8, 5], 7))
