valori = input().split()
a = int(valori[0])
b = int(valori[1])

while b !=0:
    r = a%b
    a = b
    b = r

print(a)