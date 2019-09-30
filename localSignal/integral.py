import scipy.integrate as integrate
import numpy as np
g = lambda x:x*x-0.2
def windows(a,b):
    def fun(x):
        if x>a and x<b :
            return 1
        else:
            return 0
    return fun

signal = lambda x: x-1.80
positive = lambda x: x if x>0 else 0
negative = lambda x: x if x<0 else 0
def conv(f,a,b,s,t):
    return integrate.quad(lambda x:s(x)*f(t+a,t+b)(x),t+a,t+b)

result = conv(windows,1,2,lambda x: positive(signal(x)),0)
print(result)
result = conv(windows,1,2,lambda x: negative(signal(x)),0)
print(result)

from scipy.integrate import simps

x=np.array([1, 10])
y=np.array([1, 1])
I1 = simps(y, x)
print(I1)