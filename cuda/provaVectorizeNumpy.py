import timeit

import numpy as np
import time
from numba import float64, vectorize,guvectorize, int64, int32, jit

sqrt2 = np.sqrt(2)
invSqrt2 = 1 / sqrt2
log2 = np.log(2)
treshold = -sqrt2 * 5
logPi = np.log(np.pi)


# @vectorize([float64(float64)],nopython=True)
# def ncdflogbc(x):
#     z = -x
#     if (x >= 0):
#         return np.log(1 + erf(x * invSqrt2)) - log2
#     if (treshold < x):
#         return np.log(1 - erf(-x * invSqrt2)) - log2
#     return -0.5 * logPi - log2 - 0.5 * z * z - np.log(z) + np.log(
#         1 - 1 / z + 3 / z ** 4 - 15 / z ** 6 + 105 / z ** 8 - 945 / z ** 10)
#
# a=np.array([[1.0,2.0],[1,1]])
# fun=ncdflogbc
# b=fun(a)
# print(b)

@guvectorize([(float64[:], int64[:], float64[:])], '(n),(n)->(n)')
def calculate_rates(coeff, species, res):
    res[0]=coeff[0]/100*species[0]*species[1]
    res[1]=coeff[1]*species[1]

@guvectorize([(int64[:], int64[:], int64[:])], '(n),()->(n)')
def g(x, y, res):
    res[0]=x[0]+y[0]
    res[1]=x[1]*y[0]

@guvectorize([(float64[:], float64[:], int32[:])], '(n),()->(n)')
def search(x, y, res):
    res[0]=x[0]>y[0]
@vectorize
def search2(x, y):
    return x/y if y!= 0 else 0

cutoffs=np.random.rand(10000,2)
true_cutoff=np.random.rand(10000,1)
selected_numpy = lambda: np.array(
           # [np.searchsorted(a, b, side='right') if b != 0 else 0 for a, b in zip(cutoffs, true_cutoff)])
            [a/b if b != 0 else 0 for a, b in zip(cutoffs, true_cutoff)])


selected_numba=lambda:  search(cutoffs, true_cutoff)
selected_vectorize=lambda:  search2(cutoffs, true_cutoff)

print(timeit.timeit(selected_numpy,number=10)/10)
#print(timeit.timeit(selected_numba,number=10)/10)
print(timeit.timeit(selected_vectorize,number=10)/10)
