valori = input().split()
a = int(valori[0][::-1])
b = int(valori[1][::-1])

x = a
y = b

while b !=0:
    r = a%b
    a = b
    b = r

print( (x*y)//a)