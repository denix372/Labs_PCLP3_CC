def suma_cifre(n):
  suma = 0
  while n:
    suma += n%10
    n = n//10
  return suma

print(suma_cifre(123)) # => 6
print(suma_cifre(2222)) # => 8
print(suma_cifre(32145)) # => 15