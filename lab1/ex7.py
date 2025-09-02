def prim(n):
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True

nr = 0
valori = input().split()
n = int(valori[0])

for x in valori[1: n+1]:
    if prim(int(x)):
        nr +=1

print(nr)