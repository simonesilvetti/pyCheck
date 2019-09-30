import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import norm

from pycheck.regressor import EP
from pycheck.regressor import simulate
from pycheck.regressor.supportFunction import findBounds, getIndexes, findBoundsIndex


class TessellationTh1DParallel:
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
        trainSetX = np.array([np.linspace(interval[0], interval[1], 60)]).T
        trainSetY = np.array([[self.fun(x[0])] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = np.linspace(interval[0], interval[1], num=600)
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
        while True:
            c, slope, value = self.findMaxSlopeMulti(xs, model.getEvaluator(), beta)
            print(str(c) + ':' + str(value))
            if value < t:
                break
            for e in c:
                model.addDesign(xs[int(e)], fun(xs[int(e)]))
            model.fit()
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

        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        res = findBoundsIndex(lb[:, None], ub[:, None], self.th)
        indexes = getIndexes(res)
        value = np.zeros(len(indexes))
        index = np.zeros(len(indexes))
        c = 0
        for i in indexes:
            maxx = max(ub[i], ub[i + 1])
            minx = min(lb[i], lb[i + 1])
            if (minx > th or maxx < th):
                value[c] = minx - maxx
                index[c] = i
            else:
                value[c] = maxx - minx
                index[c] = i
            c = c + 1
        argsort = np.argsort(value)
        index = index[argsort]
        area = np.sum((xs[res][:, 1] - xs[res][:, 0])) / (self.interval[1] - self.interval[0])
        return index[-40:-1:2], value[argsort[-1]], area


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

        trainSetX = np.array([[i, j] for i in np.linspace(interval[0][0], interval[1][0], 20) for j in
                              np.linspace(interval[0][1], interval[1][1], 20)])
        trainSetY = np.array([[self.fun(x)] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = [[[i, j] for i in np.linspace(interval[0][0], interval[1][0], 100)] for j in
              np.linspace(interval[0][1], interval[1][1], 100)]
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
        plt.xlabel('k_i')
        plt.ylabel('k_r')
        fig.savefig('rectFig.png', dpi=90, bbox_inches='tight')
        plt.show(block=True)
        print("Done")


        # M = self.boxy(xs, model.getEvaluator(), t, beta)
        # self.drowBoxes(M, xs, model, beta)

    def slopy(self, t, xs, model, fun, beta):
        while (True):
            c, slope, lb, ub, counter = self.findMaxSlopeMulti(xs, model.getEvaluator(), beta)
            value = (counter + 1) / (len(xs) * len(xs))
            print(str(c) + ':' + str(value))
            if (value <= t):
                break
            for e in c:
                model.addDesign(xs[e[0]][e[1]], fun(xs[e[0]][e[1]]))
               # model.addDesign(xs[e[0]+1][e[1]], fun(xs[e[0]+1][e[1]]))
               # model.addDesign(xs[e[0]][e[1]+1], fun(xs[e[0]][e[1]+1]))
               # model.addDesign(xs[e[0]+1][e[1]+1], fun(xs[e[0]+1][e[1]+1]))

            model.fit()

        return lb, ub, model.getTrainSetX(), xs, model

    def findMaxSlopeMulti(self, xs, evaluator, beta):
        th = self.th
        counter = 0
        maxValue = -float('Inf')
        lb = np.zeros([len(xs), len(xs[0])])
        ub = np.zeros([len(xs), len(xs[0])])
        value = list()
        index = list()
        for i in range(len(xs)):
            lb[i], ub[i] = evaluator.confidenceBounds(xs[i], beta)
        for i in range(len(xs) - 1):
            for j in range(len(xs[i]) - 1):
                maxx = max(ub[i, j], ub[i + 1, j], ub[i, j + 1], ub[i + 1, j + 1])
                minx = min(lb[i, j], lb[i + 1, j], lb[i, j + 1], lb[i + 1, j + 1])
                if (minx <= th and maxx >= th):
                    value.append(min(th-minx,maxx-th))
                    index.append([i,j])
        #argsort = np.argsort(value)
        index = np.array(index)
        np.random.shuffle(index)
        return index[1:100], maxValue, lb, ub, len(index)

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


class TessellationTh1DParallelGP:
    def __init__(self, interval, fun, model, beta, th, t):
        self.interval = interval
        self.fun = fun
        self.model = model
        self.beta = beta
        self.th = norm.ppf(th)
        self.t = t

    def tessellation(self):
        a = time.time()
        t = self.t
        beta = self.beta
        interval = self.interval
        model = self.model
        trainSetX = np.array([np.linspace(interval[0], interval[1], 40)]).T
        trainSetY = np.array([[self.fun(x[0])] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = np.linspace(interval[0], interval[1], num=400)
        res = self.slopy(t, xs, model, self.fun, beta)
        b = time.time()
        # M = self.boxy(xs, model.getEvaluator(),  res + 0.005, beta)
        print(b - a)
        self.drowAllBoxes(xs, model, beta)

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
        plt.plot([xs[0], xs[-1]], [norm.cdf(self.th), norm.cdf(self.th)])
        plt.scatter(model.getTrainSetX(), model.getTrainSetY())
        plt.show(block=True)

    def drowAllBoxes(self, xs, model, beta):
        th = norm.cdf(self.th)
        evaluator = model.getEvaluator()
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        plt.figure()
        currentAxis = plt.gca()
        for i in range(len(xs) - 1):
            maxx = max(ub[i], ub[i + 1])
            minn = min(lb[i], lb[i + 1])
            if (minn > th):
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              #facecolor="red", alpha=0.5,edgecolor='k'))
                              facecolor="red", alpha=0.5))
            elif (maxx < th):
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              #facecolor="blue", alpha=0.5,edgecolor='k'))
                              facecolor="blue", alpha=0.5))
            else:
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              #facecolor="yellow", alpha=0.5,edgecolor='k'))
                              facecolor="yellow", alpha=0.5))

        plt.xlim([xs[0], xs[-1]])
        ys = evaluator.mean(xs[:, None])
        plt.plot(xs, ub)
        plt.plot(xs, ys, 'r')
        plt.plot(xs, lb)
        plt.xlabel('k_i')
        plt.ylabel('P')
        plt.plot([xs[0], xs[-1]], [norm.cdf(self.th), norm.cdf(self.th)])
        plt.scatter(model.getTrainSetX(), model.getTrainSetY())
        plt.show(block=True)

    def findInterval(self, xs, start, t, evaluator, beta):
        global maxOld, minOld
        th = norm.cdf(self.th)
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
        while True:
            c, slope, value = self.findMaxSlopeMulti(xs, model.getEvaluator(), beta)
            print(str(c) + ':' + str(value))
            if value < t:
                break
            for e in c:
                model.addDesign(xs[int(e)], fun(xs[int(e)]))
            model.fit()
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

        # lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        lb, ub = evaluator.confidenceBoundsGP(xs[:, None], beta)
        res = findBoundsIndex(lb[:, None], ub[:, None], self.th)
        indexes = getIndexes(res)
        value = np.zeros(len(indexes))
        index = np.zeros(len(indexes))
        c = 0
        for i in indexes:
            maxx = max(ub[i], ub[i + 1])
            minx = min(lb[i], lb[i + 1])
            if (minx > th or maxx < th):
                value[c] = minx - maxx
                index[c] = i
            else:
                value[c] = maxx - minx
                index[c] = i
            c = c + 1
        argsort = np.argsort(value)
        index = index[argsort]
        area = np.sum((xs[res][:, 1] - xs[res][:, 0])) / (self.interval[1] - self.interval[0])
        return index[-40:-1:2], value[argsort[-1]], area

class TessellationTh1DParallelGPTest:
    def __init__(self, interval, fun, model, beta, th, t):
        self.interval = interval
        self.fun = fun
        self.model = model
        self.beta = beta
        self.th = norm.ppf(th)
        self.t = t

    def tessellation(self):
        t = self.t
        beta = self.beta
        interval = self.interval
        model = self.model
        trainSetX = np.array([np.linspace(interval[0], interval[1], 40)]).T
        trainSetY = np.array([[self.fun(x[0])] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = np.linspace(interval[0], interval[1], num=400)
        res = self.slopy(t, xs, model, self.fun, beta)
        return model,xs


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
        plt.plot([xs[0], xs[-1]], [norm.cdf(self.th), norm.cdf(self.th)])
        plt.scatter(model.getTrainSetX(), model.getTrainSetY())
        plt.show(block=True)

    def drowAllBoxes(self, xs, model, beta):
        th = norm.cdf(self.th)
        evaluator = model.getEvaluator()
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        plt.figure()
        currentAxis = plt.gca()
        for i in range(len(xs) - 1):
            maxx = max(ub[i], ub[i + 1])
            minn = min(lb[i], lb[i + 1])
            if (minn > th):
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              #facecolor="red", alpha=0.5,edgecolor='k'))
                              facecolor="red", alpha=0.5))
            elif (maxx < th):
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              #facecolor="blue", alpha=0.5,edgecolor='k'))
                              facecolor="blue", alpha=0.5))
            else:
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              #facecolor="yellow", alpha=0.5,edgecolor='k'))
                              facecolor="yellow", alpha=0.5))

        plt.xlim([xs[0], xs[-1]])
        ys = evaluator.mean(xs[:, None])
        plt.plot(xs, ub)
        plt.plot(xs, ys, 'r')
        plt.plot(xs, lb)
        plt.xlabel('k_i')
        plt.ylabel('P')
        plt.plot([xs[0], xs[-1]], [norm.cdf(self.th), norm.cdf(self.th)])
        plt.scatter(model.getTrainSetX(), model.getTrainSetY())
        plt.show(block=True)

    def findInterval(self, xs, start, t, evaluator, beta):
        global maxOld, minOld
        th = norm.cdf(self.th)
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
        while True:
            c, slope, value = self.findMaxSlopeMulti(xs, model.getEvaluator(), beta)
            print(str(c) + ':' + str(value))
            if value < t:
                break
            for e in c:
                model.addDesign(xs[int(e)], fun(xs[int(e)]))
            model.fit()
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

        # lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        lb, ub = evaluator.confidenceBoundsGP(xs[:, None], beta)
        res = findBoundsIndex(lb[:, None], ub[:, None], self.th)
        indexes = getIndexes(res)
        value = np.zeros(len(indexes))
        index = np.zeros(len(indexes))
        c = 0
        for i in indexes:
            maxx = max(ub[i], ub[i + 1])
            minx = min(lb[i], lb[i + 1])
            if (minx > th or maxx < th):
                value[c] = minx - maxx
                index[c] = i
            else:
                value[c] = maxx - minx
                index[c] = i
            c = c + 1
        argsort = np.argsort(value)
        index = index[argsort]
        area = np.sum((xs[res][:, 1] - xs[res][:, 0])) / (self.interval[1] - self.interval[0])
        return index[-40:-1:2], value[argsort[-1]], area


t = 0.1
th = 0.1
beta = 3
model = EP()
model.setScale(1000)
trajectoriesNumber = 1000
timeEnd = 125
#fun = lambda x: run_simulation([x, 0.05], trajectoriesNumber, timeEnd)
#case1
# interval = [0.005, 0.3]
# fun = lambda x: simulate([0.05, x], trajectoriesNumber)
# tas = TessellationTh1DParallelGP(interval, fun, model, beta, th, t)
# tas.tessellation()
#case2
# interval = [0.005, 0.2]
# fun = lambda x: simulate([x,0.12], trajectoriesNumber)
# tas = TessellationTh1DParallelGP(interval, fun, model, beta, th, t)
# tas.tessellation()

#case3
#interval = [[0.005, 0.005], [0.2, 0.3]]
#fun = lambda x: simulate(x, trajectoriesNumber)
#tas = TessellationTh2DParallel(interval, fun, model, beta, th, t)
#tas.tessellation()

#caseScalability
interval = [0.005, 0.3]
fun = lambda x: simulate([0.05, x], trajectoriesNumber)
tas = TessellationTh1DParallelGPTest(interval, fun, model, beta, th, t)
m,xs =tas.tessellation()
test = np.linspace(0.01, 0.07, num=300)

fun = lambda x: simulate([0.05, x], 10000)

real = [fun(e) for e in test]
estimated = [m.getEvaluator().confidenceBounds(e,3) for e in test]

count = 0
for i in range(len(real)):
    if(estimated[i][0]>th and real[i]<th ):
        count=count+1
    elif (estimated[i][1]<th and  real[i]>th):
        count = count + 1

print(count/len(real))


# m,xs =tas.tessellation()
# test = np.linspace(0.01, 0.07, num=100)
#
# b = m.getEvaluator().confidenceBounds(xs[:,None], 2)
# difflb = b[0][1:] -  b[0][:-1]
# diffub= b[1][1:] -  b[1][:-1]
# gllb = max(difflb)
# glub=max(diffub)




