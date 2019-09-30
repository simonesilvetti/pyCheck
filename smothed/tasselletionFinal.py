import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# class BayesianRegressor():
#     def __init__(self,trainSetX,trainSetY):
#         self.trainSetX=trainSetX
#         self.trainSetY=trainSetY
#
#     def confidenceBounds(self,x,beta):
#         pass
#
#     def fit(self):
#         pass
#
#     def mean(self,x):
#         pass
#
#     def addDesign(self, e,f):
#         self.trainSetX = np.vstack([self.trainSetX, e])
#         self.trainSetY = np.hstack([self.trainSetY, f])
#
#     def getTrainSetX(self):
#         return self.trainSetX
#
#     def getTrainSetY(self):
#         return self.trainSetY



# class RaB(BayesianRegressor):
#     def __init__(self, trainSetX, trainSetY):
#         super().__init__(trainSetX, trainSetY)
#         self.gp=GaussianProcessRegressor(n_restarts_optimizer=20)
#
#     def confidenceBounds(self, x,beta):
#         ys,sigma = self.gp.predict(x, return_std=True)
#         betaSigma=beta*sigma
#         return ys-betaSigma, ys + betaSigma
#
#     def fit(self):
#         self.gp.fit(self.trainSetX, self.trainSetY)
#
#     def mean(self, x):
#         ys,sigma = self.gp.predict(x, return_std=True)
#         return ys
from pycheck.regressor import RaB


def findMaxSlope(xs,model,beta):
    index=0
    maxValue=-float('Inf')
    for i in range(len(xs)-1):
        mini,maxi=model.confidenceBounds(xs[i],beta)
        minj,maxj=model.confidenceBounds(xs[i+1],beta)
        value = max(maxi,maxj)-min(mini,minj)
        if(value>maxValue):
            maxValue=value
            index=i
    return index,maxValue

def findMaxSlope2(xs,evaluator,beta):
    th=0
    index=0
    maxValue=-float('Inf')
    for i in range(len(xs)-1):
        mini,maxi=evaluator.confidenceBounds(xs[i],beta)
        minj,maxj=evaluator.confidenceBounds(xs[i+1],beta)
        maxx =max(maxi,maxj)
        minx = min(mini,minj)
        if(minx>th or maxx<th):
            value = minx-maxx
        else:
            value=maxx-minx
        if(value>maxValue):
            maxValue=value
            index=i
    return index,maxValue


def slopy(t,xs,model,fun,beta):
    slope = float('Inf')
    while (slope > t):
        c, slope = findMaxSlope(xs,model,beta)
        print(str(c) + ':' + str(slope))
        model.addDesign(xs[c],fun(xs[c]))
        model.fit()
        # trainSetX = np.vstack([trainSetX, [xs[c]]])
        # trainSetY = np.vstack([trainSetY, function([xs[c]])])
        # fit(trainSetX, trainSetY, scale, CORRECTION, CORRECTION_FACTOR, eps_damp)


def slopy2(t,xs,model,fun,beta):
    slope = float('Inf')
    c=0
    while (slope > t):
        model.addDesign(xs[c],fun(xs[c]))
        model.fit()
        c, slope = findMaxSlope2(xs, model.getEvaluator(), beta)
        print(str(c) + ':' + str(slope))
        # c, slope = findMaxSlope2(xs, model, beta)




def findInterval(xs,start,t,model,beta):
    c=start
    lb,ub=model.confidenceBounds(xs[start],beta)
    maxInit=ub
    minInit=lb
    maxValue=maxInit
    minValue=minInit
    c=c+1
    while True:
        maxOld = maxValue
        minOld = minValue
        lb, ub = model.confidenceBounds(xs[c],beta)
        maxValue = max(maxValue, ub)
        minValue = min(minValue, lb)
        if (maxValue - minValue > t or c == len(xs)-1):
            break
        c=c+1
    if c == len(xs) - 1 and (maxValue - minValue <= t):
        maxOld = maxValue
        minOld = minValue
        end=c
    else:
        end=c-1
    return end,minOld,maxOld,c == len(xs) - 1

def findInterval2(xs, start, t, evaluator, beta):
    global maxOld, minOld
    th=0
    c=start
    lb, ub = evaluator.confidenceBounds(xs[c], beta)
    lbd, ubd = evaluator.confidenceBounds(xs[c + 1], beta)
    maxValue = max(ubd, ub)
    minValue = min(lbd, lb)
    intersec=lambda x,y:(x < th and y > th)
    if(intersec(minValue,maxValue)):
        condition=lambda x,y: intersec(x,y) and  y-x<t
    else:
        condition = lambda x,y: not intersec(x,y)
    c=c+1
    while condition(minValue,maxValue) and c<len(xs)-1:
        c+=1
        maxOld = maxValue
        minOld = minValue
        lb, ub = evaluator.confidenceBounds(xs[c], beta)
        maxValue = max(maxValue, ub)
        minValue = min(minValue, lb)
    if(c==len(xs)-1):
        end=c
    else:
        end=c-1
    return end,maxOld,minOld,c == len(xs) - 1

def findInterval2Middle(xs,start,t,model,beta):
    th=0
    c=start
    lb,ub=model.confidenceBounds(xs[start],beta)
    maxInit=ub
    minInit=lb
    maxValue=maxInit
    minValue=minInit
    c=c+1
    while True:
        maxOld = maxValue
        minOld = minValue
        lb, ub = model.confidenceBounds(xs[c],beta)
        maxValue = max(maxValue, ub)
        minValue = min(minValue, lb)
        if ((minValue>th or maxValue<th or maxValue-minValue>=t)  or c == len(xs)-1):
            break
        c=c+1
    if c == len(xs) - 1 and (maxValue>=th and minValue<=th):
        maxOld = maxValue
        minOld = minValue
        end=c
    else:
        end=c-1
    return end,minOld,maxOld,c == len(xs) - 1


def boxy(xs,model,t,beta):
    c=0
    M = [0, 0, 0, 0]
    while c < len(xs):
        end, minOld, maxOld, condition = findInterval(xs, c, t, model,beta)
        M = np.vstack([M, [xs[c], xs[end], minOld, maxOld]])
        c = end
        if condition:
            break
    return M[1:]

def boxy2(xs,evaluator,t,beta):
    c=0
    M = [0, 0, 0, 0]
    while c < len(xs):
        end, minOld, maxOld, condition = findInterval2(xs, c, t, evaluator,beta)
        M = np.vstack([M, [xs[c], xs[end], minOld, maxOld]])
        c = end
        if condition:
            break
    return M[1:]

def drowBoxes(M,xs,model,beta):
    plt.figure()
    currentAxis = plt.gca()
    for i in range(len(M)):
        currentAxis.add_patch(
            Rectangle((M[i][0], M[i][2]), M[i][1] - M[i][0], M[i][3] - M[i][2], fill=None,
                      alpha=1))
    plt.xlim([xs[0],xs[-1]])
    evaluator = model.getEvaluator()
    lb, ub = evaluator.confidenceBounds(xs[:, None],beta)
    ys = evaluator.mean(xs[:, None])
    plt.plot(xs, ub)
    plt.plot(xs, ys, 'r')
    plt.plot(xs, lb)
    plt.scatter(model.getTrainSetX(), model.getTrainSetY())
    plt.show(block=True)


if __name__ == '__main__':
    lb = -2.5
    ub = 3.5
    t = 0.8
    beta=3
    interval = [lb, ub]
    fun = lambda x: x * x + 5 * np.sin(3 * x)
    trainSetX = np.array([np.linspace(interval[0], interval[1], 5)]).T
    trainSetY = np.array([fun(x[0]) for x in trainSetX])

    # trainSetX = np.array([[lb], [ub]])
    # trainSetY = np.array([fun(lb), fun(ub)])
    xs= np.linspace(interval[0], interval[1], num=200)
    model = RaB(trainSetX,trainSetY)
    model.fit()
    # slopy(t,xs,model,fun,beta)
    # M=boxy(xs,model,t,beta)
    # drowBoxes(M, xs, model,beta)

    slopy2(t, xs, model, fun, beta)
    M = boxy2(xs, model.getEvaluator(), t, beta)
    drowBoxes(M, xs, model, beta)
