import numpy as np

from nfm.quantitativeSML import kernelClass

expP = lambda x: np.exp(2 * x)
ker = kernelClass(0, 0.5, expP)
print(ker.int(0, 0.25))
