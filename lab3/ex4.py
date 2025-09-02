import numpy as np
from numpy.lib.stride_tricks import as_strided

def gen_strides(a, stride_len, window_len):
    item_size = a.itemsize
    num_windows = (len(a) - window_len) // stride_len + 1

    shape = (num_windows, window_len)
    strides = (stride_len * a.itemsize, a.itemsize)

    return as_strided(a, shape, strides)

a = np.arange(10)
print(gen_strides(a, 2, 3))
