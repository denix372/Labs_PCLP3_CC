import numpy as np

def lu(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for i in range(n):
        for j in range(i+1, n):
            L[j,i] = U[j,i]/U[i,i]
            U[j,:] = U[j,:] - L[j,i]*U[i,:]
    return [L, U]


A = np.array([
    [2, 1, 1],
    [4, -6, 0],
    [-2, 7, 2]
])
[L,U] = lu(A)
print(" L =")
print(L)
print(" U =")
print(U)
print(" A = L * U : ")
print(L @ U)