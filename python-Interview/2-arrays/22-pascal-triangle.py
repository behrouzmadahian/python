'''
given n, calculate the n levels of pascal triangle
the element at row i (after element 1, is sum of the two elements above
each row has one more elements than previous 1, first col and last col is one.
'''
def pascal_triangle(n):
    pascal = [[1] * (i + 1) for i in range (n)]
    for i in range(2, n):
        for j in range(1, i):
            pascal[i][j] = pascal[i-1][j-1] + pascal[i-1][j]
    return pascal

print(pascal_triangle(6))