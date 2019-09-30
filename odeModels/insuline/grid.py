from pyDOE import lhs

from nfm.quantitativeSML import kernel
from odeModels.insuline.hovorkaFalsification import fit, np, fitNoise
from odeModels.insuline.simulation1Day import pidC2, hypoGlicemia, pidC1, simulation

STL = lambda x: fit([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400, 0.999,
                           hypoGlicemia(70))

SML = lambda x: fit([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400, 0.95,
                           hypoGlicemia(70))


SMLNoise = lambda x: fitNoise([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400, 0.95,
                           hypoGlicemia(70))

STLNoise = lambda x: fitNoise([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 0, 1400, 0.99999,
                           hypoGlicemia(70))


N=300
bounds = ((200, 400), (200, 400), (20, 60), (70, 120), (40, 80),)
stl60 = list()
stl65 = list()
stl70 = list()
stlNoise60 = list()
stlNoise65 = list()
stlNoise70 = list()
sml60 = list()
sml65 = list()
sml70 = list()
smlNoise60 = list()
smlNoise65 = list()
smlNoise70 = list()
A = lhs(len(bounds), samples=N, criterion='maximin')
vecBounds = np.array(bounds)
for i in range(N):
    x = vecBounds[:, 0] + A[i, :] * (vecBounds[:, 1] - vecBounds[:, 0])
    t, y = simulation([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1)
    resSTL60 = kernel(0, t, hypoGlicemia(60)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
    resSML60 = kernel(0, t, hypoGlicemia(60)(y[:, 0]), 0, 1438, 0.95, lambda x: 1)
    resSTL65 = kernel(0, t, hypoGlicemia(65)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
    resSML65 = kernel(0, t, hypoGlicemia(65)(y[:, 0]), 0, 1438, 0.95, lambda x: 1)
    resSTL70 = kernel(0, t, hypoGlicemia(70)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
    resSML70 = kernel(0, t, hypoGlicemia(70)(y[:, 0]), 0, 1438, 0.95, lambda x: 1)
    y = np.random.normal(y, 5)
    resSTLNoise60 = kernel(0, t, hypoGlicemia(60)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
    resSMLNoise60 = kernel(0, t, hypoGlicemia(60)(y[:, 0]), 0, 1438, 0.95, lambda x: 1)
    resSTLNoise65 = kernel(0, t, hypoGlicemia(65)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
    resSMLNoise65 = kernel(0, t, hypoGlicemia(65)(y[:, 0]), 0, 1438, 0.95, lambda x: 1)
    resSTLNoise70 = kernel(0, t, hypoGlicemia(70)(y[:, 0]), 0, 1438, 0.9999, lambda x: 1)
    resSMLNoise70 = kernel(0, t, hypoGlicemia(70)(y[:, 0]), 0, 1438, 0.95, lambda x: 1)
    stl60.append(resSTL60)
    stl65.append(resSTL65)
    stl70.append(resSTL70)
    sml60.append(resSML60)
    sml65.append(resSML65)
    sml70.append(resSML70)
    stlNoise60.append(resSTLNoise60)
    stlNoise65.append(resSTLNoise65)
    stlNoise70.append(resSTLNoise70)
    smlNoise60.append(resSMLNoise60)
    smlNoise65.append(resSMLNoise65)
    smlNoise70.append(resSMLNoise70)
    print(i)


print("stl60 prob: " + str(sum([1 for i in range(len(stl60)) if stl60[i]>0])/len(sml60)))
print("stlNoise60 prob: " + str(sum([1 for i in range(len(stl60)) if stlNoise60[i]>0])/len(sml60)))
print("smlNoise60 prob: " + str(sum([1 for i in range(len(stl60)) if smlNoise60[i]>0])/len(sml60)))

if(sum(np.array(stl60)>0)):
    print("stl-vs-smlNoise pair: " + str(
        sum([1 for i in range(len(sml60)) if (stl60[i] > 0 and smlNoise60[i] > 0)]) / sum(np.array(stl60) > 0)))
    print("stl-vs-stlNoise pair: " + str(
        sum([1 for i in range(len(stl60)) if (stl60[i] > 0 and stlNoise60[i] > 0)]) / sum(np.array(stl60) > 0)))



print("stl65 prob: " + str(sum([1 for i in range(len(stl65)) if stl65[i]>0])/len(sml65)))
print("stlNoise65 prob: " + str(sum([1 for i in range(len(stl65)) if stlNoise65[i]>0])/len(sml65)))
print("smlNoise65 prob: " + str(sum([1 for i in range(len(stl65)) if smlNoise65[i]>0])/len(sml65)))

if(sum(np.array(stl65)>0)):
    print("stl-vs-smlNoise pair: " + str(
        sum([1 for i in range(len(sml65)) if (stl65[i] > 0 and smlNoise65[i] > 0)]) / sum(np.array(stl65) > 0)))
    print("stl-vs-stlNoise pair: " + str(
        sum([1 for i in range(len(stl65)) if (stl65[i] > 0 and stlNoise65[i] > 0)]) / sum(np.array(stl65) > 0)))




print("stl70 prob: " + str(sum([1 for i in range(len(stl70)) if stl70[i]>0])/len(sml70)))
print("stlNoise70 prob: " + str(sum([1 for i in range(len(stl70)) if stlNoise70[i]>0])/len(sml70)))
print("smlNoise70 prob: " + str(sum([1 for i in range(len(stl70)) if smlNoise70[i]>0])/len(sml70)))

if(sum(np.array(stl70)>0)):
    print("stl-vs-smlNoise pair: " + str(
        sum([1 for i in range(len(sml70)) if (stl70[i] > 0 and smlNoise70[i] > 0)]) / sum(np.array(stl70) > 0)))
    print("stl-vs-stlNoise pair: " + str(
        sum([1 for i in range(len(stl70)) if (stl70[i] > 0 and stlNoise70[i] > 0)]) / sum(np.array(stl70) > 0)))








