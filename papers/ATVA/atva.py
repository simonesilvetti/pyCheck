import numpy as np
from scipy.integrate import odeint


def derivative(t, kernel, signal):
    mult = [1, -1] * (len(signal) // 2)
    init = next(filter(lambda i: signal[i] > t + kernel[0], range(len(signal))), -1)
    end = next(filter(lambda i: signal[i] > t + kernel[1], range(len(signal))), len(signal) - 1)
    if (init == -1):
        return 0
    if (init == 0):
        ele = list(map(lambda x: 1, np.array(signal[init:end]) - t))
        ele[0] = 0
    else:
        ele = list(map(lambda x: 1, np.array(signal[init:end]) - t))
    return np.dot(ele, mult[init:end])


# # a = derivative(1,[2,3],[0.1,2,2.5,3,3.5,3.8,4,5,6,8])
# print(derivative(-2.5,[2,3],[0,1]))

def eq(y, t):
    return derivative(t, [2, 3], [0, 1])


t = np.linspace(-3, 5, 1000)

sol = odeint(eq, 0, t)
print(sol)
