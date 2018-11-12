'''
Given an array of digits representing a number, add 1 to it
example: [1,2,9] -> [1, 3, 0]
'''
def plus_one(a):
    a[-1] += 1
    for i in reversed(range(1, len(a))):
        if a[i] != 10:
            continue
        a[i] = 0
        a[i - 1] += 1
    if a[0] == 10:
        a[0] = 1
        a.append(0)
    return a

print(plus_one([9,9,9,9]))


