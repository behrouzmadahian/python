'''
Given an array, check if there are two numbers summing to value
'''
def sumpairExist(a, value):
    a = sorted(a)
    l, r = 0, len(a)-1
    while l < r:
        if a[l] + a[r] ==value:
            return True
        elif a[l] +a[r] >value:
            r -=1
        else:
            l +=1
    return False

a = [1,2,3,4,5,6,7,10,-6]
print(sumpairExist(a, 27))



def pairsSumN1(a, val):
    a = sorted(a)
    l = 0; r = len(a)-1
    while l <r:
        if a[l] +a[r] == val:
            return a[l], a[r]
        elif a[l] +a[r] > val:
            r -=1
        else:
            l +=1
    return False
print(pairsSumN1(a, 9))
