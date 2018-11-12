'''
 Fn = Fn-1 + Fn-2
 f0 =0 and f1 =1
 print nth Fib number!
'''
def fib(n):
    f1 = 0; f2 = 1
    fibArr = []
    fibArr.extend([0,1])
    for i in range(n-2):
        tmp = f1 + f2
        f1 = f2
        f2 = tmp
        fibArr.append(f2)
    return fibArr
print(fib(20))
def fib_recursive(n):
    # returns nth fib number
    if n==1 :
        return 0
    elif n==2:
        return 1
    else:
        return fib_recursive(n-1) + fib_recursive(n-2)

print(fib_recursive(20))
