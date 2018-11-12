'''
Given two sets represented by two arrays, how to check if the given two sets are disjoint or not?
It may be assumed that the given arrays have no duplicates.
Input: set1[] = {12, 34, 11, 9, 3}
       set2[] = {2, 1, 3, 5}
Output: Not Disjoint
3 is common in two sets.
'''
def disJoint(a1,a2):
    h = {}
    for item in a1:
        h[item] = item
    for item in a2:
        try:
            h[item]
            return False
        except:
            pass
    return True
a1 = [12, 34, 11, 9, 3]
a2 = [2, 1, 30, 5]
print(disJoint(a1,a2))

