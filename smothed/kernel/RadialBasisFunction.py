from sklearn.gaussian_process.kernels import RBF

class RadialBasisFunction:

    def getDefaultParameters(self,X):
        sum = 0
        n, dim = X.shape
        for d in range(0, dim):
            max = -float('inf')
            min = float('inf')
            for i in range(0, n):
                curr = X[i][d]
                if (curr > max):
                    max = curr
                if (curr < min):
                    min = curr
            sum += (max - min) / 10.0
        lengthScale = sum / dim
        return lengthScale

    def getKernel(self,l):
        return RBF(l)









