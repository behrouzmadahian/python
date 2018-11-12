'''
wave like array. An array ‘arr[0..n-1]’ is sorted in wave form if arr[0] >= arr[1] <= arr[2] >= arr[3] <= arr[4] >=
'''
def sortwave(a):
    a = sorted(a)
    for i in range(0, len(a)-1,2):
        a[i], a[i+1] = a[i+1], a[i]
    return a
a = [4,3,2,7,-1]
print(sortwave(a))

def sortWave(a):
    a = sorted(a)
    for i in range(0, len(a),2):
        a[i], a[i+1] = a[i+1], a[i]
    return a
print(sortwave(a))

def sortwave1(a):
    for i in range(len(a)-1):
        a[i: i+2] = sorted(a[i:i+2], reverse=  i%2==0)
    return a

print(sortwave1(a))
