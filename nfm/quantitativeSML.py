import numpy as np
from scipy import integrate


class kernelClass:
    def __init__(self, T_0=None, T_1=None, fun=None):
        self.T_0 = T_0
        self.T_1 = T_1
        self.fun = lambda x: fun((x - T_0) / (T_1 - T_0))
        self.area = integrate.quad(self.fun, 0, T_1 - T_0)[0]

    def int(self, a, b):
        return integrate.quad(self.fun, a, b)[0] / self.area


def crop(times, values, tInit, tEnd, kernel):
    for i in range(len(times)):
        if (times[i] > tInit):
            init = max(i - 1, 0)
            break
    for j in range(i, len(times)):
        if (times[j] > tEnd):
            end = j
            break
    newTimes = times[init:end]
    newTimes[0] = tInit
    #  newTimes[-1] = tEnd
    integratedTimes = np.copy(newTimes) - tInit
    for i in range(len(integratedTimes) - 1):
        integratedTimes[i] = kernel.int(newTimes[i] - tInit, newTimes[i + 1] - tInit)

    integratedTimes[-1] = kernel.int(newTimes[-1] - tInit, tEnd - tInit)
    newValues = values[init:end]
    return integratedTimes, newValues


def robustness(times, values, p):
    sum = 0
    for i in range(len(times)):
        sum = sum + times[i]
        if sum >= 1 - p:
            return values[i]


def kernel(t, times, values, T_0, T_1, p, fun):
    ker = kernelClass(T_0, T_1, fun)
    cropTimes, cropValues = crop(times, values, t + T_0, t + T_1, ker)
    sortIndexes = np.argsort(cropValues)
    sortTimes, sortValues = cropTimes[sortIndexes], cropValues[sortIndexes]
    return robustness(sortTimes, sortValues, p)


def kernelInner(t, times, values, T_0, T_1, p, fun):
    ker = kernelClass(T_0, T_1, fun)
    cropTimes, cropValues = crop(times, values, t + T_0, t + T_1, ker)

    return np.dot(cropTimes, cropValues)
    # sortIndexes = np.argsort(cropValues)
    # sortTimes, sortValues = cropTimes[sortIndexes], cropValues[sortIndexes]
    # return robustness(sortTimes, sortValues, p)

# expP = lambda x: np.exp(  x)
# expN = lambda x: np.exp(- x)
# flat = lambda x: 1
# gauss = lambda x: np.exp(-((x - 0.2) ** 2) / 0.1)
#
#
# fun = np.vectorize(lambda x: np.exp(-((x - 0.4) ** 2) / 0.02))
# # times = np.array([1.,2.,3.,4.,5.,6.])
# # values = np.array([1.,2.,3.,4.,5.,6.])
#
# x = np.linspace(0, 1, 1000)
# y = fun(x)
# finexpP = lambda t: kernel(t, x, y, 0.1, 0.3, 0.5, expP)
# finexpN = lambda t: kernel(t, x, y, 0.1, 0.3, 0.5, expN)
# finflat = lambda t: kernel(t, x, y, 0.1, 0.3, 0.5, flat)
# fingauss = lambda t: kernel(t, x, y, 0.01, 0.3, 0.5, gauss)
# finstl = lambda t: kernel(t, x, y, 0.1, 0.3, 0.0001, flat)
# glob = lambda t: kernel(t, x, y, 0.1, 0.3, 1, flat)
#
# xx = np.linspace(0, 0.6, 600)
# yy = fun(xx)
# # yy=xx+np.sin(15*xx)-0.5
#
# yy0 = [finexpP(a) for a in xx]
# yy1 = [finexpN(a) for a in xx]
# yy2 = [finflat(a) for a in xx]
# yy3 = [fingauss(a) for a in xx]
# yy4 = [finstl(a) for a in xx]
# yy5 = [glob(a) for a in xx]
#
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif')
# plt.plot(x, y, label='$s(t)$')
# plt.plot(xx, yy0, label=r'$\langle \exp(10x)_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy1, label=r'$\langle \exp(-10x)_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy2, label=r'$\langle \texttt{flat}_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy3, label=r'$\langle \texttt{gauss}_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy4, label=r'$\diamond_{[0.1,0.2]}(s(t)>0)$')
# plt.plot(xx, yy5, label=r'${[0.1,0.2]}(s(t)>0)$')
# plt.legend()
# plt.savefig('result')
# plt.show()
