import matplotlib.pyplot as plt
import numpy as np

from nfm.quantitativeSML import kernel
from odeModels.insuline.FINAL import violationTimeS, violationSpaceS
from odeModels.insuline.simulation1Day import pidC1, simulation, hyperGlicemia

np.random.seed(1)

p = 70
N = 200
pid = pidC1
minSCL = float('Inf')
vSCL = float('Inf')
minSTL = float('Inf')
vSTL = float('Inf')

# kernels
gauss = lambda x: np.exp(3 * x)
flat = lambda x: 1


def blowThreshold(th):
    return lambda x: x - th


for i in range(N):
    t_meal1 = 300
    t_meal2 = 300
    t_meal3 = 1440 - t_meal1 - t_meal2
    dg1 = np.random.normal(40, 10)
    dg2 = np.random.normal(90, 10)
    dg3 = np.random.normal(60, 10)
    t, y = simulation([t_meal1, t_meal2, t_meal3], [dg1, dg2, dg3], pid)
    y = y[:, 0]
    stl = kernel(0, t, blowThreshold(p)(y), 500, 1438, 0.93, flat)
    scl = kernel(0, t, blowThreshold(p)(y), 500, 1438, 0.93, gauss)
    if (stl < minSTL):
        minSTL = stl
        print('minSTL: ' + str(minSTL))
        vSTL = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]
    if (scl < minSCL):
        minSCL = scl
        print('minSCL: ' + str(minSCL))
        vSCL = [t_meal1, t_meal2, t_meal3, dg1, dg2, dg3]

    print(i)

print('Gauss ' + str(vSCL))
print('Gauss time Violation ' + str(violationTimeS(vSCL, blowThreshold(p), pid)))
print('Gauss space Violation ' + str(violationSpaceS(vSCL, blowThreshold(p), pid)))
print('Flat ' + str(vSTL))
print('Flat time Violation ' + str(violationTimeS(vSTL, blowThreshold(p), pid)))
print('Flat space Violation ' + str(violationSpaceS(vSTL, blowThreshold(p), pid)))

plt.axhline(y=70, color='k', linestyle='-')
t, y = simulation([300, 300, 840], vSTL[3:], pid)
plt.plot(t, y[:, 0], label='flat', color='b')
t, y = simulation([300, 300, 840], vSCL[3:], pid)
plt.plot(t, y[:, 0], label='gauss', color='r')
# t, y = simulation([300, 300, 840], vSML2[3:], pidC1)
# plt.plot(t, y[:,0], label='gauss', color='g')
plt.legend(loc='upper right', labelspacing=0, borderpad=0, fontsize=20)
plt.savefig('falsificationKernels')
plt.show()

# vSML [300, 300, 840, 57.80251362297999, 105.13309045173636, 54.395515241289154]
# vSML time Violation 173.2
# vSML space Violation -56.2982173988
# vSTL [300, 300, 840, 19.954580792687242, 107.60203251263522, 50.570424768809204]
# vSTL time Violation 113.8
# vSTL space Violation -88.6822006896
# vSML2 [300, 300, 840, 67.33413336228108, 92.8369374206408, 66.74881544121551]
# vSML2 time Violation 171.6
# vSML2 space Violation -41.0549752838
