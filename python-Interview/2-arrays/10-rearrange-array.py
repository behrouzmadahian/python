'''
given an array A rearrange it such that A[ a1<a2>a3<a4>a5...]
soulution1: sort the array and switch elements at positions even and odds starting from position 1!
example:
switch the elements as position 1 and 2!
this has time complexity O(nlogn)
soloution2:
iterate through the array and swap a[i], a[i+1] when i is even and a[i] >a[i+1]
or i is odd and a[i]<a[i+1]
time complexity O(n)
'''
def rearrange_array(a):
    for i in range(len(a)):
        print(sorted(a[i : i + 2], reverse= i % 2))
        print(a)
        a[i : i + 2] = sorted(a[i : i + 2], reverse= i % 2)
    return a
print(rearrange_array([1,23, 4,2,3,5,7,6,8]))

