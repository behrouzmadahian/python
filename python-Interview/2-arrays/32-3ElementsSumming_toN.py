'''
For the first function we can use the hash table as well- here I use sorting
we can do the search in log n by binary searching using middle value!!
Sort the array outside of the functions to avoid repetition!!!
'''


def two_el_sumN(a, val):
    l, h = 0, len(a)-1
    while l < h:
        if a[l] + a[h] == val:
            return True, a[l], a[h]
        elif a[l] + a[h] > val:
            h -= 1
        else:
            l += 1
    return False, -1, -1

def three_el_sumN(a, val):
    a = sorted(a)
    for i in range(len(a)):
        two_el = two_el_sumN(a[i:], val-a[i])
        if two_el[0]:
            return True, a[i], two_el[1:]
    return False

a = [12, 3, 4, 1, 6, 9]
print(a)
print(three_el_sumN(a, 24))

