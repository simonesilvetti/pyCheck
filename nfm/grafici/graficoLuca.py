import matplotlib.pyplot as plt
import numpy as np

t = np.linspace(0, 1, 200)
fun = lambda x: np.exp(-(x - 0.5) ** 2 / 0.05)
y = fun(t)

fig, ax = plt.subplots(figsize=(15, 5))
plt.plot(t, y)
before = np.logical_and(np.array(t) < 0.60, np.array(t) > 0.30)
after = np.logical_and(np.array(t) < 0.65, np.array(t) > 0.35)
plt.fill_between(t, y, where=before, facecolor='red', alpha=0.4)
plt.axvline(0.30, color='red', linestyle='--', alpha=0.7)
plt.axvline(0.60, color='red', linestyle='--', alpha=0.7)
plt.fill_between(t, y, where=after, facecolor='green', alpha=0.4)
plt.axvline(0.35, color='green', linestyle='--', alpha=0.7)
plt.axvline(0.65, color='green', linestyle='--', alpha=0.7)
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.annotate(r'u0',
             xy=(0.30, -0.05),
             xytext=(0.21, -0.15),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=20)
plt.annotate(r'u0',
             xy=(0.35, -0.05),
             xytext=(0.40, -0.15),
             arrowprops=dict(facecolor='green', shrink=0.05),fontsize=20)

plt.annotate(r'u1',
             xy=(0.60, -0.05),
             xytext=(0.51, -0.15),
             arrowprops=dict(facecolor='red', shrink=0.05), fontsize=20)
plt.annotate(r'u1',
             xy=(0.65, -0.05),
             xytext=(0.70, -0.15),
             arrowprops=dict(facecolor='green', shrink=0.05),fontsize=20)

plt.show()

# step = 20
# s = np.repeat(np.random.normal(y[::step, 0], 10), step)
# minIndex = min(len(t), len(s))
# t=t[:minIndex]
# s=s[:minIndex]
#
# fig = plt.figure(figsize=(8, 6))
# gs = gridspec.GridSpec(1, 2, width_ratios=[12, 1])
# ax = plt.subplot(gs[0])
# bx = plt.subplot(gs[1])
# ax.axhline(y=50, color='k', linestyle='-')
# ax.plot(t, s, label='G(t)')
# ax.fill_between(t, 50, where=s >= 50, facecolor='red', alpha=0.3, interpolate=True)
# # x.fill_between(t, 55, where=s < 180, facecolor='green',alpha=1, interpolate=True)
# ax.fill_between(t, 35, where=s >= 50, facecolor='red', alpha=1, interpolate=True)
# bx.fill_between([0, 1], 0.95, facecolor='red', alpha=1)
# bx.set_ylim([0, 1])
# ax.set_ylim([30, 250])
# ax.set_xlim([0, 1440])
# ax.legend()
# plt.savefig('GnoiseFree')
# plt.show()
