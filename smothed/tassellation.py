from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from pyDOE import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


fun = lambda x: x*x+5*np.sin(3*x)
# fun = lambda x: np.sin(3*x)


def errorSigma(x):
    y_pred, sigma = gp.predict(x[0], return_std=True)
    return -6*sigma

def maximizeGP(interval):
    global bestRes
    bnds = ((interval[0], interval[1]),)
    # points = interval[0] + lhs(1, samples=10, criterion='maximin') * (interval[1] - interval[0])
    points = np.linspace(interval[0],interval[1], num=5)
    best = float('Inf')
    for i in range(0, len(points)):
        res = minimize(errorSigma, points[i], method='L-BFGS-B', bounds=bnds)
        if (res.fun < best):
            best = res.fun
            bestRes = res
    min = bestRes.x
    eMin = -bestRes.fun
    return min[0], eMin[0]

def findInterval(gp,xs,start,t):
    c=start
    ys, sigma = gp.predict(xs[start], return_std=True)
    maxInit=ys[0] + 3 * sigma[0]
    minInit=ys[0] - 3 * sigma[0]
    maxValue=maxInit
    minValue=minInit
    c=c+1
    while True:
        maxOld = maxValue
        minOld = minValue
        ys, sigma = gp.predict(xs[c], return_std=True)
        maxValue = max(maxValue, ys[0] + 3 * sigma[0])
        minValue = min(minValue, ys[0] - 3 * sigma[0])
        if (maxValue - minValue > t or c == len(xs)-1):
            break
        # maxInit = ys[0] + 3 * sigma[0]
        # minInit = ys[0] - 3 * sigma[0]
        c=c+1
    if c == len(xs) - 1 and (maxValue - minValue <= t):
        maxOld = maxValue
        minOld = minValue
        end=c
    else:
        end=c-1
    return end,minOld,maxOld,c == len(xs) - 1

def drowSquares(fun,interval,n,t):
    global M, condition
    k=n
    xs = np.linspace(interval[0], interval[1], num=k)
    gp = GaussianProcessRegressor(n_restarts_optimizer=20)
    trainSetX = np.array([[lb], [ub]])
    trainSetY = np.array([fun(lb), fun(ub)])
    gp.fit(trainSetX, trainSetY)
    stopCondition=0
    addnumber=False
    while(True):
        if(addnumber):
            k=2*k
            xs = np.linspace(interval[0], interval[1], num=k)
        M = [0, 0, 0, 0]
        c = 0
        while c < len(xs):
            end, minOld, maxOld, condition = findInterval(gp, xs, c, t)
            if (c == end):
                if(stopCondition==c):
                    addnumber=True
                    break
                else:
                    addnumber = False
                trainSetX = np.vstack([trainSetX, [xs[c]]])
                trainSetY=np.hstack([trainSetY,fun(xs[c])])
                gp.fit(trainSetX, trainSetY)
                stopCondition=c
                break
            M = np.vstack([M, [xs[c], xs[end], minOld, maxOld]])
            c = end
            if condition:
                break
        if condition:
            break
        else:
            continue
    plt.figure()
    currentAxis = plt.gca()
    for i in range(len(M)):
        currentAxis.add_patch(
            Rectangle((M[i][0], M[i][2]), M[i][1] - M[i][0], M[i][3] - M[i][2], fill=None,
                      alpha=1))
    plt.xlim(interval)
    # xs = np.linspace(interval[0], interval[1], num=40)
    ys, sigma = gp.predict(xs[:, None], return_std=True)
    plt.plot(xs, ys + 3 * sigma)
    plt.plot(xs, ys, 'r')
    plt.plot(xs, ys - 3 * sigma)
    plt.scatter(trainSetX, trainSetY)
    plt.show(block=True)

#
# error=t+1
# while (error>t):
#     gp.fit(trainSetX, trainSetY)
#     res=maximizeGP(interval)
#     trainSetX=np.vstack([trainSetX, [res[0]]])
#     trainSetY=np.hstack([trainSetY,fun(res[0])])
#     error=res[1]




lb = -2.5
ub = 1.5
t = 0.8
interval = [lb, ub]
# trainSetX = np.array([[lb], [ub]])
# trainSetY = np.array([fun(lb), fun(ub)])
drowSquares(fun,interval,50,t)


# end,minOld,maxOld,maxInit,minInit=findInterval(gp,xs,4,0.5)
#
#
# M=[0,0,0,0]
# c=0
# initValue = xs
# maxValue = -float('Inf')
# minValue = float('Inf')
# ys, sigma = gp.predict(xs[c], return_std=True)
# maxInit = ys[0] + 3 * sigma[0]
# minInit = ys[0] - 3 * sigma[0]
# while c<len(xs):
#     init = c
#     maxValue=maxInit
#     minValue=minInit
#     while True:
#         maxOld = maxValue
#         minOld = minValue
#         ys, sigma = gp.predict(xs[c], return_std=True)
#         maxValue=max(maxValue,ys[0]+3*sigma[0])
#         minValue=min(minValue,ys[0]-3*sigma[0])
#         if(maxValue-minValue>t or c==len(xs)-1):
#             c=c-1
#             break
#         c=c+1
#     end=c
#     M=np.vstack([M, [xs[init],xs[end],minOld,maxOld]])
#     ys, sigma = gp.predict(xs[c], return_std=True)
#     maxInit = ys[0] + 3 * sigma[0]
#     minInit = ys[0] - 3 * sigma[0]
#     c=c+1
#
#
#
#
#
#
# plt.figure()
# currentAxis = plt.gca()
# for i in range(len(M)):
#     currentAxis.add_patch(
#         Rectangle((M[i][0], M[i][2]), M[i][1] - M[i][0], M[i][3] - M[i][2], fill=None,
#                   alpha=1))
# plt.xlim(interval)
# xs=np.linspace(interval[0],interval[1], num=40)
# ys, sigma = gp.predict(xs[:,None], return_std=True)
# plt.scatter(xs,ys+3*sigma)
# plt.plot(xs,ys,'r')
# plt.scatter(xs,ys-3*sigma)
# plt.scatter(trainSetX,trainSetY)
# plt.show(block=True)
