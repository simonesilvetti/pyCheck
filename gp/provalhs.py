import matplotlib.pyplot as plt
from pyDOE import *

A = lhs(2, samples=5, criterion='maximin')
plt.scatter(A[:,0], A[:,1],color='blue')
print(A)
plt.show()