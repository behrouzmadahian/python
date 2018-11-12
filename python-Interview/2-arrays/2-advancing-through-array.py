'''
Given an array of the steps we can take in each position of an array
we want to know if we can reach the end!
A = [ 1,2,2,0,0, 1] -> As we can see we can not reach the end!
A[1, 2,3, 0,0,1] -> we can reach the end from 3!
max position reachable form a position i: A[i] + i
'''
def can_reach_end(a):
    max_reached = 0
    last_index = len(a) - 1
    i = 0
    while i <= max_reached and max_reached < last_index:
        max_reached = max(max_reached, a[i] + i)
        i += 1
    return max_reached>= last_index

print(can_reach_end([ 1,2,2,-2,0,1] ))
print(can_reach_end([ 1,2,3,0,0, 1] ))