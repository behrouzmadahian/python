'''
Given an array of pairs, find all symmetric pairs in it.
Two pairs (a, b) and (c, d) are said to be symmetric if c is equal to b and a is equal to d.
For example (10, 20) and (20, 10) are symmetric.
'''
def symmetricPairs(a):
    h = {}
    for i in range(len(a)):
        if a[i] in h.keys():
            print(a[i], h[a[i]])
        else:
            h[(a[i][1], a[i][0])] = a[i]
a = [(1,2), (2,1), (3,4), (5,6), (6,5)]
symmetricPairs(a)