'''
Given an array of integers, and a number ‘sum’, find the number
of pairs of integers in the array whose sum is equal to ‘sum’.
 arr[] = {1, 5, 7, -1},
          sum = 6
Output :  2
# we want to make sure repetitions get counted as well
'''
def nPairsGivenSum(a, x):
    ht = {}
    cnt = 0
    for item in a:
        if x-item  in ht:
            cnt += len(ht[x-item])
        try:
            ht[item].append(item)
        except:
            ht[item] =[item]
        print(cnt)
    return cnt

a = [1,1, 5, 7, -1, 5,5,5]
x = 6
print(nPairsGivenSum(a, x))
