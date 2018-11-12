'''
For example, if the input array is [-1, 2, -3, 4, 5, 6, -7, 8, 9],
then the output should be [9, -7, 8, -3, 5, -1, 2, 4, 6]
'''
def arrange_around_pivot(a):
    neg_ind, pos_ind = 0, len(a)-1
    while neg_ind <pos_ind:
        if a[neg_ind]> 0:
            a[neg_ind], a[pos_ind] = a[pos_ind], a[neg_ind]
            pos_ind -=1
        else:
            neg_ind +=1
    return a, pos_ind
a = [1,-2,3,4,-5,6,7,-8,-3]
print(arrange_around_pivot(a))

def rearrange(a):
    a, pos_ind = arrange_around_pivot(a)
    print(a,pos_ind,'==')
    neg_ind =1
    while neg_ind < pos_ind and a[neg_ind] < 0:
        a[neg_ind], a[pos_ind] = a[pos_ind], a[neg_ind]
        neg_ind +=2
        pos_ind +=1
    return a
print(rearrange(a))


