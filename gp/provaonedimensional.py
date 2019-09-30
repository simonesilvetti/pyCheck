from scipy.optimize import minimize
import numpy as np
from scipy.stats import norm

lb = -1
ub = 1

def transform(lb,ub): return lambda x: lb+(ub-lb)/(1+np.exp(-x/2))
f = lambda x: x*x-0.7
def choose(X):
    if (len(X)==1):
        return X(1)-X(0)
    return X[:, 1] - X[:, 0]


undefined = np.array([lb,ub])
choose(undefined)
defined = np.array([])

Xtrain = np.array([[-1],[1]])
Ytrain=np.array([f(-1),f(1)])
trshold=0
from sklearn.gaussian_process import GaussianProcessRegressor
gp = GaussianProcessRegressor(n_restarts_optimizer=20)



gp.fit(Xtrain, Ytrain)
def minimo(x):
    y_pred, sigma = gp.predict(x, return_std=True)
    return max(norm.cdf((y_pred - trshold) / sigma), 1 - norm.cdf((y_pred - trshold) / sigma))
g = lambda x: minimo(transform(-1,0)(x))
res = minimize(g, (-0.5), method='CG')
print(res.fun)
print(transform(-1,0)(res.x))