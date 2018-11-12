'''
Like QuickSort, Merge Sort is a Divide and Conquer algorithm. It divides input array in two halves,
calls itself for the two halves and then merges the two sorted halves.
we need two functions
1- mergeSort(arr, l,m,h) that  recursively divides the array from mid point and calls merge sort on each piece
2- merge(arr, l,m,h): given an array and the indices of two sorted sub arrays: arr[l:m], arr[m:h]
merges them taking into account that they  are sorted
'''
def merge1(a, l, m, h):
    # l: index of low, m: index of last element in L and h index of last element in R
    L = a[l:m+1]; H = a[m+1:h+1]
    l_ind = 0; r_ind = 0
    a_ind = l
    while l_ind < len(L) and r_ind < len(H):
        if L[l_ind] < H[r_ind]:
            a[a_ind] =L[l_ind]
            l_ind += 1
        else:
            a[a_ind] = H[r_ind]
            r_ind += 1
        a_ind += 1
    if l_ind < len(L):
        while l_ind < len(L):
            a[a_ind] = L[l_ind]
            l_ind += 1
            a_ind += 1
    elif r_ind < len(H):
        while r_ind < len(H):
            a[a_ind] = H[r_ind]
            r_ind += 1
            a_ind += 1
def mergeSort1(a, l, h):
    # l: index of first element, h: index of last element!
    if l <h:
        m = int((l+h)//2)
        mergeSort1(a, l, m)
        mergeSort1(a, m+1, h)
        merge1(a, l, m, h)

b =  [1,2,5,4,9,7,10,200,12,15,100,0]
mergeSort1(b,0,len(b)-1)
print(b)

def merge(a, l, m, h):
    # l: index of low, m: index of last element in L and h index of last element in R
    sub1_ind = 0;    sub2_ind = 0
    L = a[l:m+1];    R= a[m+1:h+1]
    n1 = len(L);    n2 = len(R);    k = l
    while sub1_ind < n1 and sub2_ind < n2:
        if L[sub1_ind] <= R[sub2_ind]:
            a[k] = L[sub1_ind]
            sub1_ind +=1
        else:
            a[k] = R[sub2_ind]
            sub2_ind +=1
        k +=1
    # now we look into see which subarray ind is not to the end!
    if sub1_ind < n1:
        while sub1_ind < n1:
            a[k] = L[sub1_ind]
            k +=1
            sub1_ind +=1
    else:
        while sub2_ind < n2:
            a[k] = R[sub2_ind]
            k +=1
            sub2_ind +=1
def mergeSort(a, l, h):
    # l: index of first element, h: index of last element!
    if l < h:
        m = int((l+h)//2)
        mergeSort(a, l, m)
        mergeSort(a, m+1, h)
        merge(a, l, m, h)

b =  [1,2,5,4,9,7,10,200,12,15,100,0]
mergeSort(b,0,len(b)-1)
print(b)





