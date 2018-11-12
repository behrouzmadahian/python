'''
given two numbers in arrays, multiply them like the way we did in middle school
note length of result is at most equal to sum of length of two arrays
'''
def multiply(a, b):
    sign = -1 if (a[0] < 0) and (b[0] > 0) or (a[0] > 0) and (b[0] < 0) else 1
    a[0] = abs(a[0]); b[0] = abs(b[0])
    result = [0] * (len(a) + len(b))
    for i in reversed(range(len(a))):
        for j in reversed(range(len(b))):
            print(j, i)
            result[i + j + 1] += a[i] * b[j]
            result[i + j] += result[i + j + 1] // 10
            result[i + j + 1] %= 10

    first_non_zero = 0
    for i in range(len(result)):
        if result[i] != 0:
            result[i] = sign * result[i]
            first_non_zero = i
            break
    return result[first_non_zero:]
print(multiply([-1,3, 1], [-5,6, 2]))