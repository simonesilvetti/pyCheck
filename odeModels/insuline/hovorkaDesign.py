from pyDOE import lhs
from scipy.optimize import minimize

from odeModels.insuline.hovorka import average, np


def findMinimum(fun, bounds, N):
    A = lhs(len(bounds), samples=N, criterion='maximin')
    vecBounds = np.array(bounds)
    min = float('Inf')
    resMin = []
    for i in range(N):
        startingPoint = vecBounds[:, 0] + A[i, :] * (vecBounds[:, 1] - vecBounds[:, 0])
        res = minimize(fun, startingPoint, method='L-BFGS-B', bounds=bnds)  # ,options={'ftol' : 0.1,'eps': 0.001})
        # res = minimize(fun, startingPoint, method='L-BFGS-B', bounds=bnds,options={'ftol' : 0.1}) #,'eps': 0.01})
        print(str(res.x) + " :--> " + str(res.fun))
        if (res.fun < min):
            resMin = res
            min = res.fun
    return resMin


def exploreMinimum(fun, bounds, N):
    A = lhs(len(bounds), samples=N, criterion='maximin')
    vecBounds = np.array(bounds)
    min = float('Inf')
    resMin = []
    for i in range(N):
        startingPoint = vecBounds[:, 0] + A[i, :] * (vecBounds[:, 1] - vecBounds[:, 0])
        res = fun(startingPoint)
        if (res < min):
            resMin = np.hstack([startingPoint, -res])
            min = res
    return resMin


rumore = list()
noise = 0
mem = 0
uNew = 0
w = 100
VG = 0.16 * w
sp = 110 * VG / 18
# initial condition

Kd = [0, -0.0602, -0.0573, -0.06002, -0.0624]
Ki = [0, -3.53e-07, -3e-07, -1.17e-07, -7.55e-07]
Kp = [0, -6.17e-04, -6.39e-04, -6.76e-04, -5.42e-04]
i = 2
optFun = lambda x: average(60, 0, 0, 1400, lambda x: 1, lambda x: x - 64, 0.8, x)
optFun2 = lambda x: average(60, 0, 0, 1400, lambda x: 1, lambda x: -x + 180, 0.7, x)
opt = lambda x: -min(optFun(x), optFun2(x))
# m = min(optFun([Kp[i], Ki[i], Kd[i]]),optFun2([Kp[i], Ki[i], Kd[i]]))
bnds = ((-1E-3, 0), (-1E-5, 0), (-1E-1, 0),)
# res = findMinimum(opt, bnds,2)
res = exploreMinimum(opt, bnds, 100)
print(res)
