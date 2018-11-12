'''
Given an array of distinct elements and a number x, find if there is a pair with product equal to x.
Input : arr[] = {10, 20, 9, 40};
        int x = 400;
Output : Yes

O(n)
'''
def distinctProduct(a, x):
    h = {}
    for i in range(len(a)):
        key = x //a[i]
        print(key)
        if x %a[i] ==0 and key in h:

            return True
        else:
            h[a[i]] = a[i]
    return False
a = [10, 20, 9,40]
print(distinctProduct(a, 401))
