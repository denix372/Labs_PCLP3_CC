def cmmdc(a,b):
    while b !=0:
        r = a%b
        a = b
        b = r
    return a

n = int(input())

for i in range(n):
    if cmmdc(i,n) == 1:
       print(i, end = " ")