import matplotlib.pyplot as plt
from pyDOE import *
from scipy.optimize import minimize

from nfm.quantitativeSML import kernel
from odeModels.insuline.simulation1Day import pidC1 as pid, simulation



def findMinimum(fun, bounds, N):
    A = lhs(len(bounds), samples=N, criterion='maximin')
    vecBounds = np.array(bounds)
    min = float('Inf')
    resMin = []
    print(":::::::::::::::::::::::::::::::::::")
    for i in range(N):
        print("-------------------------")
        startingPoint = vecBounds[:, 0] + A[i, :] * (vecBounds[:, 1] - vecBounds[:, 0])
        res = minimize(fun, startingPoint, method='L-BFGS-B', bounds=bounds,options={'eps': [2,2,1,1,1]})
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
            resMin = np.hstack([startingPoint, res])
            min = res
    return resMin


def fit(timeOfMeals, dGs, model, T, kernelFunction, T0, T1, p, atomicFunction):

    t, y = simulation(timeOfMeals, dGs, model)
    res = kernel(T, t, atomicFunction(y[:, 0]), T0, T1, p, kernelFunction)
    print(str(timeOfMeals) + " __ " + str(dGs) + " __ " + str(res))
    return res

def fitNoise(timeOfMeals, dGs, model, T, kernelFunction, T0, T1, p, atomicFunction):
    t, y = simulation(timeOfMeals, dGs, model)
    y=np.random.normal(y, 5)

    res = kernel(T, t, atomicFunction(y[:, 0]), T0, T1, p, kernelFunction)
    print(str(timeOfMeals) + " __ " + str(dGs) + " __ " + str(res))
    return res

def fitNorm(a, b,bounds,  model, T, kernelFunction, T0, T1, p, atomicFunction):
    timeOfMeals = bounds[:len(a),0] + (bounds[:len(a),1]-bounds[:len(a),0]) * a
    timeOfMeals = np.hstack([timeOfMeals,1440-sum(timeOfMeals)])
    dGs = bounds[ len(a):,0] + (bounds[ len(a):,1] - bounds[ len(a):,0]) * b
    t, y = simulation(timeOfMeals, dGs, model)
    res = kernel(T, t, atomicFunction(y[:, 0]), T0, T1, p, kernelFunction)
    print(str(timeOfMeals) + " __ " + str(dGs) + " __ " + str(res))
    return res


def violationTime(res, atomic):
    n = int((len(res.x) - 1) / 2)
    timeOfMeals = np.hstack([res.x[:n], 1440 - sum(res.x[:n])])
    dGs = res.x[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    return sum(t[i + 1] - t[i] for i in range(len(t) - 1) if atomic(y[i, 0]))

def violationTimeS(res, atomic):
    n = int((len(res) - 1) / 2)
    timeOfMeals = np.hstack([res[:n], 1440 - sum(res[:n])])
    dGs = res[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    return sum(t[i + 1] - t[i] for i in range(len(t) - 1) if atomic(y[i, 0]))

def violationSpaceS(res, atomic):
    n = int((len(res) - 1) / 2)
    timeOfMeals = np.hstack([res[:n], 1440 - sum(res[:n])])
    dGs = res[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    return min(atomic(y[i, 0]) for i in range(len(t) - 1))


def violationSpace(res, atomic):
    n = int((len(res.x) - 1) / 2)
    timeOfMeals = np.hstack([res.x[:n], 1440 - sum(res.x[:n])])
    dGs = res.x[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    return min(atomic(y[i, 0]) for i in range(len(t) - 1))


def drow(res):
    n = int((len(res.x) - 1) / 2)
    timeOfMeals = np.hstack([res.x[:n], 1440 - sum(res.x[:n])])
    dGs = res.x[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    f, (ax1, ax2) = plt.subplots(2, sharex=True)
    ax1.fill_between([t[0], t[-1]], [64, 64], [250, 250], alpha=0.5)
    ax1.step(t, y[:, 0], 'r-', label='Glucose')
    # ax1.step(ttot[::s], rum / VG, label='Noise Glucose ')

    ax1.axhline(y=100, color='k', linestyle='-')
    ax2.plot(t, y[:, -1], label='Insuline')
    ax1.legend()
    ax2.legend()
    plt.show()


def drow2(res1, res2):
    n = int((len(res1.x) - 1) / 2)
    timeOfMeals = np.hstack([res1.x[:n], 1440 - sum(res1.x[:n])])
    dGs = res1.x[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    plt.fill_between([t[0], t[-1]], [64, 64], [250, 250], alpha=0.5)
    plt.plot(t, y[:, 0], label='99')

    n = int((len(res2.x) - 1) / 2)
    timeOfMeals = np.hstack([res2.x[:n], 1440 - sum(res2.x[:n])])
    dGs = res2.x[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    plt.plot(t, y[:, 0], label='80')
    plt.legend()
    plt.show()


