'''
Given an unsorted array with repetitions, the task is to group multiple occurrence of individual elements.
The grouping should happen in a way that the order of first occurrences of all elements is maintained.
EXAMPLE:
Input: arr[] = {5, 3, 5, 1, 3, 3}
Output:        {5, 5, 3, 3, 3, 1}
'''
def groupRepetition(a):
    h ={}
    order = []
    res = []
    for i in range(len(a)):
        try:
            h[a[i]].append(a[i])
        except:
            h[a[i]] = [a[i]]
            order.append(a[i])
    for o in order:
        res.extend(h[o])
    return res
a = [5, 3, 5, 1, 3, 3]
a1 = [4, 6, 9, 2, 3, 4, 9, 6, 10, 4]
print(groupRepetition(a))
print(groupRepetition(a1))


