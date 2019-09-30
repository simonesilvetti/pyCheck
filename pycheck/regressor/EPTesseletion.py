import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

from pycheck.regressor import EP
from pycheck.regressor import simulate
from pycheck.regressor.supportFunction import findBounds, getIndexes


class TessellationTh1DParallel:
    def __init__(self, interval, fun, model, trainSetX, trainSetY, beta, threshold, eps, nToTrain,nToTest, mult):
        self.interval = interval
        self.fun = fun
        self.model = model
        self.trainSetX = trainSetX
        self.trainSetY = trainSetY
        self.beta = beta
        self.threshold = threshold
        self.eps = eps
        self.nToTest = nToTest
        self.nToTrain = nToTrain
        self.mult = mult



    def execute(self):
        if (len(self.trainSetX)!=0):
            self.trainSetX = np.vstack((self.trainSetX, np.array([np.linspace(self.interval[0], self.interval[1], self.nToTrain)]).T))
            self.trainSetY = np.hstack((self.trainSetY, np.array([[self.fun(x[0])] for x in self.trainSetX])))
        else:
            self.trainSetX = np.array([np.linspace(self.interval[0], self.interval[1], self.nToTrain)]).T
            self.trainSetY = np.array([[self.fun(x[0])] for x in self.trainSetX])
        model.setTrainSet(self.trainSetX, self.trainSetY)
        model.fit()
        xs = np.linspace(self.interval[0], self.interval[1], num=self.nToTest)
        lb, ub = model.getEvaluator().confidenceBounds(xs[:, None], self.beta)
        return findBounds(lb[:, None], ub[:, None],xs)


    def tessellation(self):
        a = time.time()
        t = self.t
        beta = self.beta
        interval = self.interval
        model = self.model
        trainSetX = np.array([np.linspace(interval[0], interval[1], self.n)]).T
        trainSetY = np.array([[self.fun(x[0])] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = np.linspace(interval[0], interval[1], num=300)
        # xs = np.arange(interval[0], interval[1],0.0015)
        res = self.slopy(t, xs, model, self.fun, beta)
        b = time.time()
        M = self.boxy(xs, model.getEvaluator(), res + 0.005, beta)
        print(b - a)
        self.drowBoxes(M, xs, model, beta)

    def boxy(self, xs, evaluator, t, beta):
        c = 0
        M = [0, 0, 0, 0]
        while c < len(xs):
            end, minOld, maxOld, condition = self.findInterval(xs, c, t, evaluator, beta)
            M = np.vstack([M, [xs[c], xs[end], minOld, maxOld]])
            c = end
            if condition:
                break
        return M[1:]

    def drowBoxes(self, M, xs, model, beta):
        plt.figure()
        currentAxis = plt.gca()
        for i in range(len(M)):
            currentAxis.add_patch(
                Rectangle((M[i][0], M[i][2]), M[i][1] - M[i][0], M[i][3] - M[i][2], fill=None,
                          alpha=1))
        plt.xlim([xs[0], xs[-1]])
        evaluator = model.getEvaluator()
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        ys = evaluator.mean(xs[:, None])
        plt.plot(xs, ub)
        plt.plot(xs, ys, 'r')
        plt.plot(xs, lb)
        plt.plot([xs[0], xs[-1]], [self.th, self.th])
        plt.scatter(model.getTrainSetX(), model.getTrainSetY())
        plt.show(block=True)

    def findInterval(self, xs, start, t, evaluator, beta):
        global maxOld, minOld
        th = self.th
        c = start
        lb, ub = evaluator.confidenceBounds(xs[c], beta)
        lbd, ubd = evaluator.confidenceBounds(xs[c + 1], beta)
        maxValue = max(ubd, ub)
        minValue = min(lbd, lb)
        intersec = lambda x, y: (x < th and y > th)
        if (intersec(minValue, maxValue)):
            condition = lambda x, y: intersec(x, y) and y - x < t
        else:
            condition = lambda x, y: not intersec(x, y)
        c = c + 1
        while condition(minValue, maxValue) and c < len(xs) - 1:
            c += 1
            maxOld = maxValue
            minOld = minValue
            lb, ub = evaluator.confidenceBounds(xs[c], beta)
            maxValue = max(maxValue, ub)
            minValue = min(minValue, lb)
        if (c == len(xs) - 1):
            end = c
        else:
            end = c - 1
        return end, maxOld, minOld, c == len(xs) - 1

    def slopy(self, t, xs, model, fun, beta):
        slope = float('Inf')
        value = float('Inf')
        c = [0]
        # while (slope > t):
        while (value > t):
            for e in c:
                model.addDesign(xs[e], fun(xs[e]))
            model.fit()
            c, slope, counter = self.findMaxSlopeMulti(xs, model.getEvaluator(), beta)
            # print(str(c) + ':' + str(slope))
            value = (counter + 1) / len(xs)
            # print(str(c) + ':' + str(slope))
            print(str(c) + ':' + str(value))
        return slope

    def findMaxSlope(self, xs, evaluator, beta):
        th = self.th
        index = 0
        maxValue = -float('Inf')
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        findBounds(lb, ub)
        for i in range(len(xs) - 1):
            # mini, maxi = evaluator.confidenceBounds(xs[i], beta)
            # minj, maxj = evaluator.confidenceBounds(xs[i + 1], beta)
            maxx = max(ub[i], ub[i + 1])
            minx = min(lb[i], lb[i + 1])
            if (minx > th or maxx < th):
                value = minx - maxx
            else:
                value = maxx - minx
            if (value > maxValue):
                maxValue = value
                index = i
        return index, maxValue

    def findMaxSlopeMulti(self, xs, evaluator, beta):
        th = self.th
        counter = 0
        index = 0
        maxValue = -float('Inf')
        value = np.zeros(len(xs) - 1)
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        res = findBounds(lb[:, None], ub[:, None])
        # getIndexes(res)
        # for i in range(len(xs) - 1):
        for i in getIndexes(res):
            # mini, maxi = evaluator.confidenceBounds(xs[i], beta)
            # minj, maxj = evaluator.confidenceBounds(xs[i + 1], beta)
            maxx = max(ub[i], ub[i + 1])
            minx = min(lb[i], lb[i + 1])
            if (minx > th or maxx < th):
                value[i] = minx - maxx
            else:
                value[i] = maxx - minx
                counter = counter + 1
        index = np.argsort(value)
        return index[-17:-1:2], value[index[-1]], counter
        # return index[-6:-1], value[index[-1]]


class TessellationTh2DParallel:
    def __init__(self, interval, fun, model, beta, th, t):
        self.interval = interval
        self.fun = fun
        self.model = model
        self.beta = beta
        self.th = th
        self.t = t

    def tessellation(self):
        a = time.time()
        t = self.t
        beta = self.beta
        interval = self.interval
        model = self.model

        trainSetX = np.array([[i, j] for i in np.linspace(interval[0][0], interval[1][0], 35) for j in
                              np.linspace(interval[0][1], interval[1][1], 35)])
        trainSetY = np.array([[self.fun(x)] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = [[[i, j] for i in np.linspace(interval[0][0], interval[1][0], 200)] for j in
              np.linspace(interval[0][1], interval[1][1], 200)]
        # xs = np.linspace(interval[0], interval[1], num=200)
        lb, ub, c, xs, model = self.slopy(t, xs, model, self.fun, beta)

        # x=np.linspace(interval[0][0], interval[1][0], 200)
        # y=np.linspace(interval[0][0], interval[1][0], 200)
        # X, Y = np.meshgrid(x, y)
        # l=lambda x: model.getEvaluator().thr( x, self.th)
        # Z=meshy(X,Y,l)
        # #levels = np.array([0, 2.5, 3.5, 10])
        # plt.figure()
        # plt.contourf(X, Y, Z)
        # plt.show(block=True)

        b = time.time()
        print((b - a) / 60.0)
        fig = plt.figure()
        currentAxis = plt.gca()
        for i in range(len(xs) - 1):
            for j in range(len(xs[i]) - 1):
                l1, l2, l3, l4 = lb[i, j], lb[i + 1, j], lb[i, j + 1], lb[i + 1, j + 1]
                u1, u2, u3, u4 = ub[i, j], ub[i + 1, j], ub[i, j + 1], ub[i + 1, j + 1]
                up = min(l1, l2, l3, l4) > th
                down = max(u1, u2, u3, u4) < th
                if (up):
                    currentAxis.add_patch(
                        Rectangle((xs[i][j][0], xs[i][j][1]), xs[i][j + 1][0] - xs[i][j][0],
                                  xs[i + 1][j][1] - xs[i][j][1],
                                  facecolor="red",
                                  alpha=0.5
                                  ))
                elif (down):
                    currentAxis.add_patch(
                        Rectangle((xs[i][j][0], xs[i][j][1]), xs[i][j + 1][0] - xs[i][j][0],
                                  xs[i + 1][j][1] - xs[i][j][1],
                                  facecolor="blue",
                                  alpha=0.5))
                else:
                    currentAxis.add_patch(
                        Rectangle((xs[i][j][0], xs[i][j][1]), xs[i][j + 1][0] - xs[i][j][0],
                                  xs[i + 1][j][1] - xs[i][j][1],
                                  facecolor="yellow",
                                  alpha=0.5))
        plt.xlim([interval[0][0], interval[1][0]])
        plt.ylim([interval[0][1], interval[1][1]])
        plt.scatter(c[:, 0], c[:, 1], c='k')
        fig.savefig('rectFig.png', dpi=90, bbox_inches='tight')
        plt.show(block=True)
        print("Done")


        # M = self.boxy(xs, model.getEvaluator(), t, beta)
        # self.drowBoxes(M, xs, model, beta)

    def slopy(self, t, xs, model, fun, beta):
        slope = float('Inf')
        value = float('Inf')
        c = [1, 1]
        while (value > t):
            model.addDesign(xs[c[0]][c[1]], fun(xs[c[0]][c[1]]))
            model.addDesign(xs[c[0] + 1][c[1]], fun(xs[c[0] + 1][c[1]]))
            model.addDesign(xs[c[0]][c[1] + 1], fun(xs[c[0]][c[1] + 1]))
            model.addDesign(xs[c[0] + 1][c[1] + 1], fun(xs[c[0] + 1][c[1] + 1]))
            model.fit()
            c, slope, lb, ub, counter = self.findMaxSlope(xs, model.getEvaluator(), beta)
            value = (counter + 1) / (len(xs) * len(xs))
            print(str(c) + ':' + str(value))
        return lb, ub, model.getTrainSetX(), xs, model

    def findMaxSlope(self, xs, evaluator, beta):
        th = self.th
        counter = 0
        index = 0
        maxValue = -float('Inf')
        lb = np.zeros([len(xs), len(xs[0])])
        ub = np.zeros([len(xs), len(xs[0])])
        for i in range(len(xs)):
            lb[i], ub[i] = evaluator.confidenceBounds(xs[i], beta)
        for i in range(len(xs) - 1):
            for j in range(len(xs[i]) - 1):
                maxx = max(ub[i, j], ub[i + 1, j], ub[i, j + 1], ub[i + 1, j + 1])
                minx = min(lb[i, j], lb[i + 1, j], lb[i, j + 1], lb[i + 1, j + 1])
                if (minx > th or maxx < th):
                    value = minx - maxx
                else:
                    value = maxx - minx
                    counter = counter + 1
                if (value > maxValue):
                    maxValue = value
                    index = [i, j]
        return index, maxValue, lb, ub, counter


t = 0.2
th = 0.1
beta = 3
model = EP()
model.setScale(1000)
trajectoriesNumber = 1000
timeEnd = 125
# fun = lambda x: run_simulation([x, 0.05], trajectoriesNumber, timeEnd)
# interval = [0.005, 0.3]
# fun = lambda x: simulate([x, 0.05],trajectoriesNumber)
# tas = TessellationTh1DParallel(interval, fun, model, beta, th, t)

interval = [0.005,0.3]
fun = lambda x: simulate([x, 0.05], trajectoriesNumber)
mult=1
ti = time.time()
tas = TessellationTh1DParallel(interval, fun, model,np.array([]),np.array([]), beta, th, t,50,200,mult)
res= tas.execute()
value = 0
for a in res:
    value = value +(a[1]-a[0])
area =value / (interval[1] - interval[0])
print(area)
tas = TessellationTh1DParallel(res[0], fun, model,np.array([]),np.array([]), beta, th, t,50,200,(res[0][1]-res[0][0])/(interval[1]-interval[0]))
newRes= tas.execute()
res=np.delete(res, 0 , axis=0)
res=np.vstack((res,newRes))
value = 0
for a in res:
    value = value +(a[1]-a[0])
area =value / (interval[1] - interval[0])
print(area)
tas = TessellationTh1DParallel(res[0], fun, model,np.array([]),np.array([]), beta, th, t,50,200,(res[0][1]-res[0][0])/(interval[1]-interval[0]))
newRes= tas.execute()
res=np.delete(res, 0 , axis=0)
res=np.vstack((res,newRes))
value = 0
for a in res:
    value = value +(a[1]-a[0])
area =value / (interval[1] - interval[0])
print(area)

te = time.time()

print((ti-te)/60)
print(res)

