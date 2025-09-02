import numpy as np

arr = np.arange(1,17)
arr = arr.reshape((4,4))

print(arr[1, 2:])
print(arr[0:, 1])
print(arr[2:, 2:])

for i in range(4):
    for j in range(4):
        if i % 2 == 1 and j % 2 != i % 2:
            print(arr[i,j], end=" ")

"""sau pentru ultima

rows, cols = np.indices(arr.shape)
mask = (rows % 2 == 1) & (cols % 2 != rows % 2)
print(arr[mask])

"""