from math import erf

import numpy as np
from numba import vectorize, float64, jit


@vectorize([float64(float64)])
def ncdflogbc(x):
    invSqrt2 = 0.70710678118654746
    log2 = 0.69314718055994529
    treshold = -7.0710678118654755
    z = -x
    if (x >= 0):
        return np.log(1 + erf(x * invSqrt2)) - log2
    if (treshold < x):
        return np.log(1 - erf(-x * invSqrt2)) - log2
    return -1.2655121234846454 - 0.5 * z * z - np.log(z) + np.log(
        1 - 1 / z + 3 / z ** 4 - 15 / z ** 6 + 105 / z ** 8 - 945 / z ** 10)


@vectorize
def searchsortedNew(x, y):
    return x < y if y != 0 else 0


@vectorize
def divide(x, y):
    return x / y if y != 0 else 0


@vectorize
def xor(x, y):
    return x ^ y


@vectorize
def f(x, y, th):
    return x <= th <= y


@vectorize
def diff(x, y):
    return x


def findBounds(lb, ub, xs):
    a = f(lb, ub)
    c = xor(a, np.vstack((False, a[:-1])))
    arr = c.nonzero()[0]
    if len(arr) % 2 == 1:
        arr = arr[:-1]

    value = int(len(arr) / 2)
    res = arr.reshape(value, 2)
    res[:, 1] = res[:, 1] - 1
    return xs[res]


def findBoundsIndex(lb, ub, th):
    a = f(lb, ub, th)
    c = xor(a, np.vstack((False, a[:-1])))
    arr = c.nonzero()[0]
    if len(arr) % 2 == 1:
        arr = arr[:-1]

    value = int(len(arr) / 2)
    res = arr.reshape(value, 2)
    res[:, 1] = res[:, 1] - 1
    return res


def getIndexes(matrix):
    res = list()
    for vector in matrix:
        res += range(vector[0], vector[1])
    return res


@jit
def meshy(X, Y, f):
    Z = np.empty_like(X)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])
    return Z
