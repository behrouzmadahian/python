'''
given an integer, n, and subset size k, generate a random subset of size k of integers from 0-n
replace =false!
brute force:
call the random number generator as long as needed, generate a random number if is not
already in the set, add it. continue until the size is k
time complexity O(n), space complexity O(k)
'''
import random
def random_subset_bruteforce(n, k):
    random.seed(123)
    subset =[]
    while  len(subset) < k:
        i = random.randint(0, n)
        if not i in subset:
            subset.append(i)

    return subset
print(random_subset_bruteforce(10, 3))

# we want to do better!
# can we do in O(k) time complexity?
# lets keep a hash table"
# it has O(k)
def random_subset_hash(n, k):
    random.seed(123)
    subset1 = {}
    for i in range(k):
        idx = random.randint(0, n)
        idx_mapped = subset1.get(idx, idx)
        i_mapped = subset1.get(i, i) # return default value of i if key not in dict
        print(idx,'*')
        subset1[idx] = i_mapped
        subset1[i] = idx_mapped
        print(subset1)

    return [subset1[i] for i in range (k)]

print(random_subset_hash(10, 3))
