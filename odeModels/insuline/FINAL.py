from pyDOE import lhs

from nfm.quantitativeSML import kernel
from odeModels.insuline.hovorkaFalsification import fit, np, fitNoise
from odeModels.insuline.simulation1Day import hypoGlicemia, pidC1, simulation, hyperGlicemia

STL = lambda x: fit([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400, 0.999,
                    hypoGlicemia(70))

SML = lambda x: fit([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400, 0.95,
                    hypoGlicemia(70))

SMLNoise = lambda x: fitNoise([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400,
                              0.95,
                              hypoGlicemia(70))

STLNoise = lambda x: fitNoise([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400,
                              0.99999,
                              hypoGlicemia(70))


def gridPlain(pid, N):
    bounds = ((200, 400), (200, 400), (20, 60), (70, 120), (40, 80),)
    p = [50, 55, 60, 65, 70, 75]
    A = lhs(len(bounds), samples=N, criterion='maximin')
    vecBounds = np.array(bounds)
    stl = list()
    stlNoise = list()
    sml = list()
    smlNoise = list()
    for i in range(N):
        x = vecBounds[:, 0] + A[i, :] * (vecBounds[:, 1] - vecBounds[:, 0])
        t, y = simulation([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pid)
        stl.append([kernel(0, t, hypoGlicemia(v)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1) for v in p])
        sml.append([kernel(0, t, hypoGlicemia(v)(y[:, 0]), 0, 1438, 0.95, lambda x: 1) for v in p])
        ys = np.repeat(np.random.normal(y[::5, 0], 5), 5)
        stlNoise.append([kernel(0, t, hypoGlicemia(v)(ys), 0, 1438, 0.9999, lambda x: 1) for v in p])
        smlNoise.append([kernel(0, t, hypoGlicemia(v)(ys), 0, 1438, 0.95, lambda x: 1) for v in p])
        print(i)

    print('stl' + str(np.mean(np.array(stl) > 0, axis=0)))
    print('sml' + str(np.mean(np.array(sml) > 0, axis=0)))
    print('stlNoise' + str(np.mean(np.array(stlNoise) > 0, axis=0)))
    print('smlNoise' + str(np.mean(np.array(smlNoise) > 0, axis=0)))


def gridProb(pid, N):
    p = [60]
    stl = list()
    stlNoise = list()
    sml = list()
    smlNoise = list()
    for i in range(N):
        t_meal1 = np.random.normal(300, 60)
        t_meal2 = np.random.normal(300, 60)
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        stl.append([kernel(0, t, hypoGlicemia(v)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1) for v in p])
        sml.append([kernel(0, t, hypoGlicemia(v)(y[:, 0]), 0, 1438, 0.95, lambda x: 1) for v in p])
        #  ys = np.repeat(np.random.normal(y[::5, 0], 5), 5)
        ys = np.random.normal(y[:, 0], 5)
        stlNoise.append([kernel(0, t, hypoGlicemia(v)(ys), 0, 1438, 0.9999, lambda x: 1) for v in p])
        smlNoise.append([kernel(0, t, hypoGlicemia(v)(ys), 0, 1438, 0.95, lambda x: 1) for v in p])
        print(i)

    print('stl' + str(np.mean(np.array(stl) > 0, axis=0)))
    print('sml' + str(np.mean(np.array(sml) > 0, axis=0)))
    print('stlNoise' + str(np.mean(np.array(stlNoise) > 0, axis=0)))
    print('smlNoise' + str(np.mean(np.array(smlNoise) > 0, axis=0)))


def gridProbFixed(pid, N):
    p = [50]
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
        stl.append([kernel(0, t, hypoGlicemia(v)(y[:, 0]), 0, 1430, 0.9999, lambda x: 1) for v in p])
        sml.append([kernel(0, t, hypoGlicemia(v)(y[:, 0]), 0, 1430, 0.95, lambda x: 1) for v in p])
        # ys = np.repeat(np.random.normal(y[::5, 0], 5), 5)
        ys = np.random.normal(y[:, 0], 10)
        stlNoise.append([kernel(0, t, hypoGlicemia(v)(ys), 0, 1430, 0.9999, lambda x: 1) for v in p])
        smlNoise.append([kernel(0, t, hypoGlicemia(v)(ys), 0, 1430, 0.95, lambda x: 1) for v in p])
        print(i)

    print('stl' + str(np.mean(np.array(stl) > 0, axis=0)))
    print('sml' + str(np.mean(np.array(sml) > 0, axis=0)))
    print('stlNoise' + str(np.mean(np.array(stlNoise) > 0, axis=0)))
    print('smlNoise' + str(np.mean(np.array(smlNoise) > 0, axis=0)))


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
        smlNoise = kernel(0, t, hypoGlicemia(p)(ys), 0, 1438, 0.95, lambda x: 1)
        stlwstlNoise = stlwstlNoise + 1 if stl * stlNoise > 0 else stlwstlNoise
        stlwsmlNoise = stlwsmlNoise + 1 if stl * smlNoise > 0 else stlwsmlNoise
        print(i)

    print('stlwstlNoise ' + str(stlwstlNoise / N))
    print('stlwsmlNoise ' + str(stlwsmlNoise / N))


def countPairFixed(pid, N):
    stlwstlNoise = 0
    stlwsmlNoise = 0
    for i in range(N):
        p = 50 + np.random.rand() * 30
        t_meal1 = 300
        t_meal2 = 300
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        stl = kernel(0, t, hypoGlicemia(p)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
        ys = np.repeat(np.random.normal(y[::5, 0], 5), 5)
        stlNoise = kernel(0, t, hypoGlicemia(p)(ys), 0, 1438, 0.9999, lambda x: 1)
        smlNoise = kernel(0, t, hypoGlicemia(p)(ys), 0, 1438, 0.95, lambda x: 1)
        stlwstlNoise = stlwstlNoise + 1 if stl * stlNoise > 0 else stlwstlNoise
        stlwsmlNoise = stlwsmlNoise + 1 if stl * smlNoise > 0 else stlwsmlNoise
        print(i)

    print('stlwstlNoise ' + str(stlwstlNoise / N))
    print('stlwsmlNoise ' + str(stlwsmlNoise / N))


def violationTimeS(res, atomic, pid):
    n = int(len(res) / 2)
    timeOfMeals = res[:n]
    dGs = res[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    return sum(t[i + 1] - t[i] for i in range(len(t) - 1) if atomic(y[i, 0]) < 0)


def violationSpaceS(res, atomic, pid):
    n = int(len(res) / 2)
    timeOfMeals = res[:n]
    dGs = res[n:]
    t, y = simulation(timeOfMeals, dGs, pid)
    return min(atomic(y[i, 0]) for i in range(len(t) - 1))


def findMinimumFixed(pid, N, p):
    minSML = float('Inf')
    vSML = float('Inf')
    minSTL = float('Inf')
    vSTL = float('Inf')
    for i in range(N):
        t_meal1 = 300
        t_meal2 = 300
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        y = y[:, 0]
        stl = kernel(0, t, hypoGlicemia(p)(y), 0, 1438, 0.9999, lambda x: 1)
        sml = kernel(0, t, hypoGlicemia(p)(y), 0, 1438, 0.95, lambda x: 1)
        if (stl < minSTL):
            minSTL = stl
            vSTL = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        if (sml < minSML):
            minSML = sml
            vSML = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        print(i)

    print('vSML ' + str(vSML))
    print('vSML time Violation ' + str(violationTimeS(vSML, hypoGlicemia(p), pid)))
    print('vSML space Violation ' + str(violationSpaceS(vSML, hypoGlicemia(p), pid)))
    print('vSTL ' + str(vSTL))
    print('vSTL time Violation ' + str(violationTimeS(vSTL, hypoGlicemia(p), pid)))
    print('vSTL space Violation ' + str(violationSpaceS(vSTL, hypoGlicemia(p), pid)))


def findMinimum(pid, N, p):
    minSML = float('Inf')
    vSML = float('Inf')
    minSTL = float('Inf')
    vSTL = float('Inf')
    for i in range(N):
        t_meal1 = np.random.normal(300, 60)
        t_meal2 = np.random.normal(300, 60)
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        y = y[:, 0]
        stl = kernel(0, t, hypoGlicemia(p)(y), 0, 1438, 0.9999, lambda x: 1)
        sml = kernel(0, t, hypoGlicemia(p)(y), 0, 1438, 0.95, lambda x: 1)
        if (stl < minSTL):
            minSTL = stl
            vSTL = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        if (sml < minSML):
            minSML = sml
            vSML = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        print(i)

    print('vSML ' + str(vSML))
    print('vSML time Violation ' + str(violationTimeS(vSML, hypoGlicemia(p), pid)))
    print('vSML space Violation ' + str(violationSpaceS(vSML, hypoGlicemia(p), pid)))
    print('vSTL ' + str(vSTL))
    print('vSTL time Violation ' + str(violationTimeS(vSTL, hypoGlicemia(p), pid)))
    print('vSTL space Violation ' + str(violationSpaceS(vSTL, hypoGlicemia(p), pid)))


def findMinimumFixedHyper(pid, N, p):
    minSML = float('Inf')
    vSML = float('Inf')
    minSML2 = float('Inf')
    vSML2 = float('Inf')
    minSTL = float('Inf')
    vSTL = float('Inf')
    gauss = lambda x: np.exp(-(((x - 0.03) / 0.1) ** 2) / 0.5)
    for i in range(N):
        t_meal1 = 300
        t_meal2 = 300
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        y = y[:, 0]
        stl = kernel(0, t, hyperGlicemia(p)(y), 0, 1438, 0.9999, lambda x: 1)
        sml = kernel(0, t, hyperGlicemia(p)(y), 0, 1438, 0.70, lambda x: 1)
        sml2 = kernel(0, t, hyperGlicemia(p)(y), 0, 1438, 0.70, gauss)
        if (stl < minSTL):
            minSTL = stl
            vSTL = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        if (sml < minSML):
            minSML = sml
            vSML = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        if (sml2 < minSML2):
            minSML2 = sml2
            vSML2 = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]

        print(i)

    print('vSML ' + str(vSML))
    print('vSML time Violation ' + str(violationTimeS(vSML, hyperGlicemia(p), pid)))
    print('vSML space Violation ' + str(violationSpaceS(vSML, hyperGlicemia(p), pid)))
    print('vSTL ' + str(vSTL))
    print('vSTL time Violation ' + str(violationTimeS(vSTL, hyperGlicemia(p), pid)))
    print('vSTL space Violation ' + str(violationSpaceS(vSTL, hyperGlicemia(p), pid)))
    print('vSML2 ' + str(vSML2))
    print('vSML2 time Violation ' + str(violationTimeS(vSML2, hyperGlicemia(p), pid)))
    print('vSML2 space Violation ' + str(violationSpaceS(vSML2, hyperGlicemia(p), pid)))


def findMinimumFixedHypo(pid, N, p):
    minSML = float('Inf')
    vSML = float('Inf')
    minSML2 = float('Inf')
    vSML2 = float('Inf')
    minSTL = float('Inf')
    vSTL = float('Inf')
    gauss = lambda x: np.exp(8 * x)
    for i in range(N):
        t_meal1 = 300
        t_meal2 = 300
        t_meal3 = 1440 - t_meal1 - t_meal2
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
        y = y[:, 0]
        stl = kernel(0, t, hypoGlicemia(p)(y), 0, 1438, 0.9999, lambda x: 1)
        sml = kernel(0, t, hypoGlicemia(p)(y), 0, 1438, 0.70, lambda x: 1)
        sml2 = kernel(0, t, hypoGlicemia(p)(y), 0, 1438, 0.70, gauss)
        if (stl < minSTL):
            minSTL = stl
            vSTL = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        if (sml < minSML):
            minSML = sml
            vSML = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
        if (sml2 < minSML2):
            minSML2 = sml2
            vSML2 = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]

        print(i)

    print('vSML ' + str(vSML))
    print('vSML time Violation ' + str(violationTimeS(vSML, hypoGlicemia(p), pid)))
    print('vSML space Violation ' + str(violationSpaceS(vSML, hypoGlicemia(p), pid)))
    print('vSTL ' + str(vSTL))
    print('vSTL time Violation ' + str(violationTimeS(vSTL, hypoGlicemia(p), pid)))
    print('vSTL space Violation ' + str(violationSpaceS(vSTL, hypoGlicemia(p), pid)))
    print('vSML2 ' + str(vSML2))
    print('vSML2 time Violation ' + str(violationTimeS(vSML2, hypoGlicemia(p), pid)))
    print('vSML2 space Violation ' + str(violationSpaceS(vSML2, hypoGlicemia(p), pid)))
# findMinimumFixedHyper(pidC1, 500, 180)
# findMinimumFixedHypo(pidC1, 500, 70)

# findMinimumFixed(pidC1,2000,70)
# gridProbFixed(pidC1,1000)
