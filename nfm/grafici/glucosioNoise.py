import matplotlib.pyplot as plt
import numpy as np

from odeModels.insuline.simulation1Day import simulation, pidC1

t, y = simulation([300, 300, 1400 - 300 - 300], [36.29, 86.88, 94.18], pidC1)
step = 2
noise = np.repeat(np.random.normal(y[::step, 0], 5), step)
minIndex = min(len(t), len(noise))
plt.axhline(y=70, color='k', linestyle='-')
plt.plot(t[:minIndex], noise[:minIndex], label='G(t)', color='r')
plt.legend()
plt.savefig('Gnoise')
plt.show()
