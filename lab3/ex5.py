import numpy as np

def citeste_matrice(dimensiuni):
    n, m = dimensiuni
    matrice = []
    for i in range(n):
        linie = input().strip().split()
        matrice.append([int(x) for x in linie])
    return np.array(matrice)


def kroneker(A, B):
    return np.kron(A,B)

n, m = map(int, input().split())
A = citeste_matrice((n, m))
p, q = map(int, input().split())
B = citeste_matrice((p, q))
print(kroneker(A,B))


"""sau

def produs_kronecker(A, B):
    n, m = A.shape
    p, q = B.shape
    rezultat = np.zeros((n * p, m * q), dtype=A.dtype)

    for i in range(n):
        for j in range(m):
            # Înlocuim blocul (i*p):(i+1)*p, (j*q):(j+1)*q cu A[i,j] * B
            rezultat[i*p:(i+1)*p, j*q:(j+1)*q] = A[i, j] * B

    return rezultat
"""