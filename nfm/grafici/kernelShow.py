import matplotlib.pyplot as plt
import numpy as np
# fig, axes = plt.subplot(2,1)
from scipy import integrate

# axes[0].step(x,y)
# plt.show()


expP = lambda x: np.exp(3 * x)
expN = lambda x: np.exp(-3 * x)
flat = lambda x: 1
gauss = lambda x: np.exp(-((x - 0.5) ** 2) / 0.1)
intExpP = integrate.quad(expP, 0, 0.5)[0]
intexpN = integrate.quad(expN, 0, 0.5)[0]
intflat = integrate.quad(flat, 0, 0.5)[0]
ingauss = integrate.quad(gauss, 0, 0.5)[0]

expPKernel = lambda x: expP(x) / intExpP
expNKernel = lambda x: expN(x) / intexpN
flatKernel = lambda x: flat(x) / intflat
gaussKernel = lambda x: gauss(x) / ingauss

x = np.linspace(0, 0.5, 200)

plt.plot(x, [expPKernel(a) for a in x])
plt.plot(x, [expNKernel(a) for a in x])
plt.plot(x, [flatKernel(a) for a in x])
plt.plot(x, [gaussKernel(a) for a in x])
plt.show()
