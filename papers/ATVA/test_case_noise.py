import numpy as np

from nfm.quantitativeSML import kernel
from odeModels.insuline.simulation1Day import simulation, hypoGlicemia, pidC1

np.random.seed(1)

def belowThreshold(th):
    return lambda x: th-x


def gridProbFixed(p,pid, N):
    stl = list()
    stlNoise = list()
    sml = list()
    smlNoise = list()
    for i in range(N):
        t_meal1 = 300
        t_meal2 = 300
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        stl.append([kernel(0, t, belowThreshold(v)(y[:, 0]), 0, 1430, 0.0001, lambda x: 1) for v in p])
        sml.append([kernel(0, t, belowThreshold(v)(y[:, 0]), 0, 1430, 0.03, lambda x: 1) for v in p])
        # ys = np.repeat(np.random.normal(y[::5, 0], 5), 5)
        ys = np.random.normal(y[:, 0], 5)
        stlNoise.append([kernel(0, t, belowThreshold(v)(ys), 0, 1430, 0.0001, lambda x: 1) for v in p])
        smlNoise.append([kernel(0, t, belowThreshold(v)(ys), 0, 1430, 0.03, lambda x: 1) for v in p])
        print(i)
    print()
    print('stl: <>[0,24](G(t)<{}) '.format(p) + str(np.mean(np.array(stl) > 0, axis=0)))
    print('scl: <flat_[0,24],0.05>(G(t)<{}) '.format(p) + str(np.mean(np.array(sml) > 0, axis=0)))
    print('stl Noise: <>[0,24](G(t)<{}) '.format(p) + str(np.mean(np.array(stlNoise) > 0, axis=0)))
    print('scl Noise: <flat_[0,24],0.05>(G(t)<{}) '.format(p) + str(np.mean(np.array(smlNoise) > 0, axis=0)))

def countPair(pid, N):
    stlwstlNoise = 0
    stlwsmlNoise = 0
    for i in range(N):
        p = 50 + np.random.rand() * 30
        t_meal1 = np.random.normal(300, 60)
        t_meal2 = np.random.normal(300, 60)
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        stl = kernel(0, t, hypoGlicemia(p)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
        ys = np.repeat(np.random.normal(y[::5, 0], 10), 5)
        stlNoise = kernel(0, t, hypoGlicemia(p)(ys), 0, 1438, 0.9999, lambda x: 1)
        smlNoise = kernel(0, t, hypoGlicemia(p)(ys), 0, 1438, 0.97, lambda x: 1)
        stlwstlNoise = stlwstlNoise + 1 if stl * stlNoise > 0 else stlwstlNoise
        stlwsmlNoise = stlwsmlNoise + 1 if stl * smlNoise > 0 else stlwsmlNoise
        print(i)

    print('stlwstlNoise ' + str(stlwstlNoise / N))
    print('stlwsmlNoise ' + str(stlwsmlNoise / N))

# gridProbFixed([55,60,65,70],pidC1, 100)
countPair(pidC1,2000)