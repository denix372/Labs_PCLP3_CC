def cifra_maxima(n):
  cifra = n%10
  while n:
    if cifra < n%10:
        cifra = n%10
    n = n//10
  return cifra

print(cifra_maxima(1234567)) # => 7
print(cifra_maxima(123984)) # => 9
print(cifra_maxima(2222)) # => 2
print(cifra_maxima(1)) # => 1