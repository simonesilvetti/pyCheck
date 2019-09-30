from numba import vectorize, float64
import numpy as np
from numba.tests.test_mathlib import erf


@vectorize(nopython=True)
def f(x, y):
    return erf(x) + y

a=np.linspace(0,10,10000000)
b=np.linspace(0,89,10000000)

c=f(a,b)
#c=a+b
