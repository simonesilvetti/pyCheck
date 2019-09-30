import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate

gauss = lambda x: np.exp(9 * x)

ingauss = integrate.quad(gauss, 0, 1)[0]

x = np.linspace(0, 1, 2000)
y = gauss(x) / ingauss
plt.plot(x, y)
plt.show()
