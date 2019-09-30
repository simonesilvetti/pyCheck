import matplotlib.pyplot as plt
from matplotlib import gridspec

from odeModels.insuline.simulation1Day import simulation, pidC1

t, y = simulation([300, 300, 1400 - 300 - 300], [59.83936429575982, 94.84665477721562, 85.44703763705998], pidC1)
s = y[:, 0]
# fig, (ax,bx )= plt.subplots(2, 1)
fig = plt.figure(figsize=(8, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[12, 1])
ax = plt.subplot(gs[0])
bx = plt.subplot(gs[1])
ax.axhline(y=180, color='k', linestyle='-')
ax.plot(t, s, label='G(t)')
ax.fill_between(t, 180, where=s >= 180, facecolor='red', alpha=0.3, interpolate=True)
# x.fill_between(t, 55, where=s < 180, facecolor='green',alpha=1, interpolate=True)
ax.fill_between(t, 55, where=s >= 180, facecolor='red', alpha=1, interpolate=True)
bx.fill_between([0, 1], 0.3, facecolor='red', alpha=1)
bx.set_ylim([0, 1])
ax.set_ylim([50, 250])
ax.set_xlim([0, 1140])
ax.legend()
plt.savefig('GnoiseFree')
plt.show()
