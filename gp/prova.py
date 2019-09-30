import numpy as np
import matplotlib.pyplot as plt

a = np.linspace(-1, 1, num=10)
b = np.linspace(-1, 1, num=10)
g = np.meshgrid(a, b)
X = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
np.random.shuffle(X)
# ff = (lambda x: np.sin(5*x[0])+np.cos(5*x[1]))
ff = (lambda x: x[0] * x[0] + x[1] * x[1] - 1)
k = map(ff, X)
y = np.array(list(k))

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

np.random.seed(1)

# kernel = C(1.0, (1e-3, 1e3)) * RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(n_restarts_optimizer=20)
# gp = GaussianProcessRegressor()

# Fit to data using Maximum Likelihood Estimation of the parameters
Xtrain = X[:1]
Ytrain = y[:1]
Xundefined = X
Xhigh = np.array([])
Xlow = np.array([])

trshold = 0

from scipy.stats import norm
from scipy.optimize import minimize

for i in [10]:
    print('iter:'+str(i))
    a = np.linspace(-1, 1, num=i)
    b = np.linspace(-1, 1, num=i)
    g = np.meshgrid(a, b)
    X = np.append(g[0].reshape(-1, 1), g[1].reshape(-1, 1), axis=1)
    np.random.shuffle(X)
    while (True):
        if(len(X)==0):
            break
        gp.fit(Xtrain, Ytrain)
        y_pred, sigma = gp.predict(X, return_std=True)
        values = [max(norm.cdf((y_pred[i] - trshold) / sigma[i]), 1 - norm.cdf((y_pred[i] - trshold) / sigma[i])) for i
                  in
                  range(0, len(y_pred))]
        if min(values) > 0.999:
            break
        id = np.argmin(values)
        Xtrain = np.vstack([Xtrain, X[id, :]])
        Ytrain = np.append(Ytrain, ff(X[id, :]))
        X=X[np.array(range(0,len(X)))!=id]


def funzione(s):
    y_pred, sigma = gp.predict(s, return_std=True)
    return max(norm.cdf((y_pred - trshold) / sigma), 1 - norm.cdf((y_pred - trshold) / sigma))


    # y_pred, sigma = gp.predict(X, return_std=True)

#
res = minimize(funzione, np.array([-0.8, -0.8]), method='SLSQP', bounds=((-1, -0.75), (-1, -0.75)),
               options={'xtol': 1e-10, 'disp': False})

# y_pred, sigma = gp.predict(X, return_std=True)

# lower = X[y_pred + 5*sigma< trshold,:]
values = [norm.cdf((y_pred[i] - trshold) / sigma[i]) for i in range(0, len(y_pred))]
lower = X[np.array(values) > 0.999, :]
upper = X[np.array(values) < 1 - 0.999, :]

# upper = X[y_pred - 5*sigma> trshold,:]
plt.scatter(lower[:, 0], lower[:, 1], color='blue')
plt.scatter(upper[:, 0], upper[:, 1], color='red')
# colors = np.array([norm.cdf((y_pred[i] - trshold) / sigma[i]) if y_pred[i] > 0 else -norm.cdf((y_pred[i] - trshold) / sigma[i]) for i in range(0, len(y_pred))])
# plt.scatter(X[:, 0], X[:, 1], color=y_pred)
plt.scatter(Xtrain[Ytrain > 0, 0], Xtrain[Ytrain > 0, 1], s=80, facecolors='none', edgecolors='k')
plt.scatter(Xtrain[Ytrain < 0, 0], Xtrain[Ytrain < 0, 1], s=80, facecolors='none', edgecolors='k')

yReal =np.array([ff(X[i, :]) for i in range(0, len(y_pred))])
result = 1/2* (len(yReal)-np.dot(np.sign(yReal),np.sign(y_pred)))
print('errors:'+str(result))
print('evaluation:'+str(len(Ytrain)))
# print(Xtrain)
# Xtrain=np.concatenate(Xtrain,(Xundefined[0,:]))
# print(Xtrain)
# Ytrain=np.append(Ytrain,ff(Xundefined[0,:]))
# print(Xtrain)
# print(Ytrain)
# Xundefined=Xundefined[1:]
# gp.fit(Xtrain, Ytrain)
# y_pred, sigma = gp.predict(Xundefined, return_std=True)
# b = (y_pred+sigma)>trshold
# Xundefined=Xundefined[a & b,:]
#
# print(len(Xundefined))
plt.show()
# print(str(y_pred) + "\pm" + str(sigma))
