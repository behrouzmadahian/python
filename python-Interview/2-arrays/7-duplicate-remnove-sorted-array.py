'''
GIVEN A sorted array, remove duplicates, return the number of nonzero elements
exsmple: a=[1,2,3,3,4,5] -> [1,2,3,4,5,0]
'''
def dup_remove(a):
    if not a:
        return 0
    write_index = 1
    for i in range(1, len(a)):
        if a[write_index - 1] != a[i]:
            a[write_index] = a[i]
            write_index += 1
    for i in range(write_index, len(a)):
        a[i] = 0
    return a

print(dup_remove([1,1,2,2,2,3,4,5,6,6,6,7]))

def duprem2(a):
    unique_ind = 1
    for i in range(1,len(a)):
        if a[i] != a[i-1]:
            a[unique_ind] = a[i]
            unique_ind +=1
    for i in range(unique_ind, len(a)):
        a[i] =0
    return a
print(duprem2([1,1,2,2,2,3,4,5,6,6,6,7]))


'''
Given an array and a key, remove all occurrences of the element array[key]
and shift everyone to left
'''
print('-'*20)
def dup_rem_key(a, key):
    print(a, '---')
    if not a:
        return 0
    write_index = 0
    dup_cnt = 0

    for i in range(0, len(a)):
        if a[key] != a[i] :
            a[write_index] = a[i]
            write_index += 1

        elif i != key:
            dup_cnt +=1
        if write_index ==key:
            write_index += 1
    print(len(a))
    return a[:-dup_cnt]

print(dup_rem_key([2,1,1,2,2,2,3,4,5,6,6,6,7], 3))

'''
Given an array and a key, remove all occurrences of the element array[key]
and shift everyone to left
'''
def dupRemkey1(a, key):
    unique_ind = 0
    for i in range(len(a)):
        if a[i]!= key:
            a[unique_ind] =a[i]
            unique_ind += 1
    for i in range(unique_ind, len(a)):
        a[i] =0
    return a
print(dupRemkey1([2,1,1,2,2,2,3,4,5,6,6,6,7], 6))
