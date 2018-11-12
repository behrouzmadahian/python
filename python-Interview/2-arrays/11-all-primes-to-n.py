'''
write a function that returns all prime numbers up to n:
[2, 3, ..]
soloution: keep a boolean array for all numbers to n, set to true
 if k is prime, go through the array and set all r*k entries to false
'''
import math
def isPrime(n):
    q = int(math.floor(math.sqrt(n)))
    if n ==2 or n==3:
        return True
    else:
        for i in range(2, q):
            if n % i == 0:
                return False
    return True

print(isPrime(6))

def generate_primes(n):
    is_prime_array = [True] * (n + 1)
    prime_list = []
    for i in range(2, n + 1):
        if is_prime_array[i]:
            prime_list.append(i)
            for j in range(i, n + 1, i):
                is_prime_array[j] = False
    return prime_list

print(generate_primes(10))

def generatePrimes1(n):
    isPrime = [True] * (n+1)
    primeList = []
    for i in range(2, n+1):
        if isPrime[i] == True :
            primeList.append(i)
            for j in range(i, n+1, i):
                isPrime[j] =False
    return primeList
print(generatePrimes1(10))




