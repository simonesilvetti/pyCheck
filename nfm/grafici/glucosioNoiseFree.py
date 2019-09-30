import matplotlib.pyplot as plt

from odeModels.insuline.simulation1Day import simulation, pidC1

t, y = simulation([300, 300, 1400 - 300 - 300], [36.29, 86.88, 94.18], pidC1)
plt.axhline(y=70, color='k', linestyle='-')
plt.plot(t, y[:, 0], label='G(t)', color='r')
plt.legend()
plt.savefig('GnoiseFree')
plt.show()
