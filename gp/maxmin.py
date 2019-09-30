from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from pyDOE import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


fun = lambda x: x*x
# fun = lambda x: np.sin(3*x)
lb = -1.2
ub = 3.26

trainSetX = np.array([[lb], [ub]])
trainSetY = np.array([fun(lb), fun(ub)])
gp = GaussianProcessRegressor(n_restarts_optimizer=20)

gp.fit(trainSetX, trainSetY)


def modelMin(x):
    y_pred, sigma = gp.predict(x[0], return_std=True)
    return y_pred - 3*sigma


def modelMax(x):
    y_pred, sigma = gp.predict(x[0], return_std=True)
    return -y_pred - 3*sigma


def minimizeGP(interval):
    global bestRes
    bnds = ((interval[0], interval[1]),)
    # points = interval[0] + lhs(1, samples=10, criterion='maximin') * (interval[1] - interval[0])
    points = np.linspace(interval[0],interval[1], num=5)
    best = float('Inf')
    for i in range(0, len(points)):
        res = minimize(modelMin, points[i], method='L-BFGS-B', bounds=bnds)
        if (res.fun < best):
            best = res.fun
            bestRes = res
    min = bestRes.x
    eMin = bestRes.fun
    return min[0], eMin[0]


def maximizeGp(interval):
    global bestRes
    bnds = ((interval[0], interval[1]),)
    points = np.linspace(interval[0],interval[1], num=5)
    # points = interval[0] + lhs(1, samples=10, criterion='maximin')* (interval[1] - interval[0])
    best = float('Inf')
    for i in range(0, len(points)):
        res = minimize(modelMax, points[i], method='L-BFGS-B', bounds=bnds)
        if (res.fun < best):
            best = res.fun
            bestRes = res
    min = bestRes.x
    eMax = -bestRes.fun
    return min[0], eMax[0]


def findMax(matrix):
    values = [a[1]-a[0] for a in  matrix]
    return np.argmax(values), np.max(values)


def adjustAll(Yminmax):
    for i in range(len(sets)):
        lb = sets[i][0]
        ub = sets[i][1]
        min, eMin = minimizeGP([lb, ub])
        max, eMax = maximizeGp([lb, ub])
        Xminmax[i]=[min,max]
        Yminmax[i]=[eMin,eMax]
    return Yminmax

def dinstanceFromtS(x):
    b = [np.linalg.norm(a - x) for a in trainSetX ]
    return np.min(b)

# initialization########################
sets = list()
Xminmax = list()
Yminmax = list()
sets.append([lb, ub])
i = 0
lb = sets[i][0]
ub = sets[i][1]
min, eMin = minimizeGP([lb, ub])
max, eMax = maximizeGp([lb, ub])
Xminmax.insert(i, [min, max])
Yminmax.insert(i, [eMin, eMax])


val = 1




while(val>0.2):
    # print(val)
    i, val = findMax(Yminmax)
    print(len(trainSetX))
    lb = sets[i][0]
    ub = sets[i][1]
    xmin = Xminmax[i][0]
    xmax = Xminmax[i][1]

    c =0
    if dinstanceFromtS(xmin)>0.1:
        trainSetX = np.vstack([trainSetX, [xmin]])
        rmin = fun(xmin)
        trainSetY = np.hstack([trainSetY, rmin])
        c=1
    if dinstanceFromtS(xmax) > 0.1:
        trainSetX = np.vstack([trainSetX, [xmax]])
        rmax = fun(xmax)
        trainSetY = np.hstack([trainSetY, rmax])
        c=1

    if c==1:
        gp.fit(trainSetX, trainSetY)

    sets.pop(i)
    Xminmax.pop(i)
    Yminmax.pop(i)

    minL, eMinL = minimizeGP([lb, ub])
    maxR, eMaxR = maximizeGp([lb, ub])
    newPoint = (minL + maxR) / 2

    if minL < maxR:
        leftSet = [lb, newPoint]
        rightSet = [newPoint, ub]
    else:
        rightSet = [lb, newPoint]
        leftSet = [newPoint, ub]

    minR, eMinR = minimizeGP(rightSet)
    maxL, eMaxL = maximizeGp(leftSet)

    # Yminmax=adjustAll(Yminmax)

    sets.append(leftSet)
    Xminmax.append([minL, maxL])
    Yminmax.append([eMinL, eMaxL])

    sets.append(rightSet)
    Xminmax.append([minR, maxR])
    Yminmax.append([eMinR, eMaxR])

    if (minL == maxL or minR == maxR):
        print('=')

    if(eMaxR<eMinR or eMaxL<eMinL ):
        print('=')

Yminmax=adjustAll(Yminmax)

print(sets)
print(Xminmax)
print(Yminmax)

print(sets)
plt.figure()
plt.xlim([-1.2, 3.26])
plt.ylim([-2, 2])
currentAxis = plt.gca()
plt.scatter(trainSetX,trainSetY)
for i in range(len(sets)):
    currentAxis.add_patch(Rectangle((sets[i][0], Yminmax[i][0]), sets[i][1]-sets[i][0], Yminmax[i][1]-Yminmax[i][0], fill=None, alpha=1))
x = np.arange(-1.2, 3.26, 0.01)
y=fun(x)
mmax=[-modelMax([t]) for t in x]
mmin = [modelMin([t]) for t in x]
plt.plot(x, y)
plt.plot(x, mmax)
plt.plot(x, mmin)
plt.show()

# print(trainSetX)

