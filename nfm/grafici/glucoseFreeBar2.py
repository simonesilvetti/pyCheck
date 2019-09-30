import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec

from odeModels.insuline.simulation1Day import simulation, pidC1

t, y = simulation([300, 300, 1400 - 300 - 300], [36.29, 86.88, 94.18], pidC1)
step = 25
s = np.repeat(np.random.normal(y[::step, 0], 10), step)
minIndex = min(len(t), len(s))
t = t[:minIndex]
s = s[:minIndex]

fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(2, 2, width_ratios=[12, 1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])
ax.axhline(y=70, color='k', linestyle='-')
ax.plot(t, s, label='G(t)')
ax.fill_between(t, 70, where=s <= 70, facecolor='red', alpha=0.3, interpolate=True)
# x.fill_between(t, 55, where=s < 180, facecolor='green',alpha=1, interpolate=True)
ax.fill_between(t, 45, where=s <= 70, facecolor='red', alpha=1, interpolate=True)
bx.fill_between([0, 1], 0.3, facecolor='red', alpha=1)
bx.set_ylim([0, 1])
ax.set_ylim([40, 250])
ax.set_xlim([0, 1440])
ax.legend()

t, s = simulation([300, 300, 1400 - 300 - 300], [36.29, 86.88, 94.18], pidC1)
s = s[:, 0]
axs = plt.subplot(gs[2])
bxs = plt.subplot(gs[3])
axs.axhline(y=70, color='k', linestyle='-')
axs.plot(t, s, label='G(t)')
axs.fill_between(t, 70, where=s <= 70, facecolor='red', alpha=0.3, interpolate=True)
# x.fill_between(t, 55, where=s < 180, facecolor='green',alpha=1, interpolate=True)
axs.fill_between(t, 45, where=s <= 70, facecolor='red', alpha=1, interpolate=True)
bxs.fill_between([0, 1], 0.3, facecolor='red', alpha=1)
bxs.set_ylim([0, 1])
axs.set_ylim([40, 250])
axs.set_xlim([0, 1440])

plt.savefig('gnoiseBoth')
plt.show()
