import matplotlib.pyplot as plt
import numpy as np

# fig, axes = plt.subplot(2,1)
from nfm.quantitativeSML import kernelInner

n = 2
m = 100

# x = np.linspace(0, 2, 2 * n * m)
# y = np.repeat(np.tile([0, 1], n), m)

x = np.linspace(0, 3, 1000)
y = np.repeat(np.tile([0, 1], 2), 1000 / 4)
for i in range(len(x)):
    if (0.3 < x[i] < 0.9):
        y[i] = 1
    else:
        y[i] = 0
# axes[0].step(x,y)
# plt.show()


expP = lambda x: np.exp(3 * x)
expN = lambda x: np.exp(-3 * x)
flat = lambda x: 1
gauss = lambda x: np.exp(-((x - 0.5) ** 2) / 0.1)

funexpP = lambda t: kernelInner(t, x, y, 0, 0.5, 0, expP)
funexpN = lambda t: kernelInner(t, x, y, 0, 0.5, 0, expN)
funflat = lambda t: kernelInner(t, x, y, 0, 0.5, 0, flat)
fungauss = lambda t: kernelInner(t, x, y, 0, 0.5, 0, gauss)

f, axes = plt.subplots(3, sharex=True, sharey=True)
axes[0].step(x, y, color='k')
expPPoint = np.array([funexpP(a) for a in x[:750]])
expNPoint = np.array([funexpN(a) for a in x[:750]])
flatPoint = np.array([funflat(a) for a in x[:750]])
gaussPoint = np.array([fungauss(a) for a in x[:750]])

axes[1].plot(x[:750], expPPoint),  # color = 'darksalmon' )
axes[1].plot(x[:750], expNPoint),  # color = 'blue' )
axes[1].plot(x[:750], flatPoint),  # color = 'green' )
axes[1].plot(x[:750], gaussPoint)  # , color = 'yellow' )
axes[1].axhline(y=0.5, color='k', linestyle='-.')
axes[2].fill_between(x[:750], 0.77, 1, where=expPPoint >= 0.5)
axes[2].fill_between(x[:750], 0.52, 0.73, where=expNPoint >= 0.5)
axes[2].fill_between(x[:750], 0.27, 0.48, where=flatPoint >= 0.5)
axes[2].fill_between(x[:750], 0, 0.23, where=gaussPoint >= 0.58)

plt.xlim((0.0, 1.2))

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.show()
