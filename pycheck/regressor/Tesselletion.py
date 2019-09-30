import matplotlib.pyplot as plt
import numpy as np
import time
from matplotlib.patches import Rectangle


class TessellationTh1D:
    def __init__(self, interval, fun, model, beta, th, t):
        self.interval = interval
        self.fun = fun
        self.model = model
        self.beta = beta
        self.th = th
        self.t = t

    def tessellation(self):
        t = self.t
        beta = self.beta
        interval = self.interval
        model = self.model
        trainSetX = np.array([np.linspace(interval[0], interval[1], 20)]).T
        trainSetY = np.array([[self.fun(x[0])] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = np.linspace(interval[0], interval[1], num=300)
        self.slopy(t, xs, model, self.fun, beta)
        M = self.boxy(xs, model.getEvaluator(), t, beta)
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
        step = (self.interval[1]-self.interval[0])/len(xs)
        c = 0
        #while (slope > t):
        while (value > t):
            model.addDesign(xs[c], fun(xs[c]))
            model.fit()
            c, slope,counter = self.findMaxSlope(xs, model.getEvaluator(), beta)
            value = step*(counter+1)
           # print(str(c) + ':' + str(slope))
            print(str(c) + ':' + str(value))

    def findMaxSlope(self, xs, evaluator, beta):
        counter=0
        th = self.th
        index = 0
        maxValue = -float('Inf')
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        for i in range(len(xs) - 1):
            # mini, maxi = evaluator.confidenceBounds(xs[i], beta)
            # minj, maxj = evaluator.confidenceBounds(xs[i + 1], beta)
            maxx = max(ub[i], ub[i + 1])
            minx = min(lb[i], lb[i + 1])
            if (minx > th or maxx < th):
                value = minx - maxx
            else:
                value = maxx - minx
                counter=counter+1
            if (value > maxValue):
                maxValue = value
                index = i
        return index, maxValue,counter

    def findMaxSlopeMulti(self, xs, evaluator, beta):
        th = self.th
        index = 0
        maxValue = -float('Inf')
        value = np.zeros(len(xs) - 1)
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        for i in range(len(xs) - 1):
            # mini, maxi = evaluator.confidenceBounds(xs[i], beta)
            # minj, maxj = evaluator.confidenceBounds(xs[i + 1], beta)
            maxx = max(ub[i], ub[i + 1])
            minx = min(lb[i], lb[i + 1])
            if (minx > th or maxx < th):
                value[i] = minx - maxx
            else:
                value[i] = maxx - minx
        index = np.argsort(value)
        return index[-5:-1], value[index[-1]]


class TessellationMinMax1D:
    def __init__(self, interval, fun, model, beta, t):
        self.interval = interval
        self.fun = fun
        self.model = model
        self.beta = beta
        self.t = t

    def tessellation(self):
        t = self.t
        beta = self.beta
        interval = self.interval
        model = self.model
        trainSetX = np.array([np.linspace(interval[0], interval[1], 20)]).T
        trainSetY = np.array([[self.fun(x[0])] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = np.linspace(interval[0], interval[1], num=500)
        self.slopy(t, xs, model, self.fun, beta)
        M = self.boxy(xs, model.getEvaluator(), t, beta)
        self.drowBoxes(M, xs, model, beta)

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
        plt.scatter(model.getTrainSetX(), model.getTrainSetY())
        plt.show(block=True)

    def boxy(self, xs, model, t, beta):
        c = 0
        M = [0, 0, 0, 0]
        while c < len(xs):
            end, minOld, maxOld, condition = self.findInterval(xs, c, t, model, beta)
            M = np.vstack([M, [xs[c], xs[end], minOld, maxOld]])
            c = end
            if condition:
                break
        return M[1:]

    def findInterval(self, xs, start, t, model, beta):
        c = start
        lb, ub = model.confidenceBounds(xs[start], beta)
        maxInit = ub
        minInit = lb
        maxValue = maxInit
        minValue = minInit
        c = c + 1
        while True:
            maxOld = maxValue
            minOld = minValue
            lb, ub = model.confidenceBounds(xs[c], beta)
            maxValue = max(maxValue, ub)
            minValue = min(minValue, lb)
            if (maxValue - minValue > t or c == len(xs) - 1):
                break
            c = c + 1
        if c == len(xs) - 1 and (maxValue - minValue <= t):
            maxOld = maxValue
            minOld = minValue
            end = c
        else:
            end = c - 1
        return end, minOld, maxOld, c == len(xs) - 1

    def slopy(self, t, xs, model, fun, beta):
        slope = float('Inf')
        while (slope > t):
            c, slope = self.findMaxSlope(xs, model.getEvaluator(), beta)
            print(str(c) + ':' + str(slope))
            model.addDesign(xs[c], fun(xs[c]))
            model.fit()

    def findMaxSlope(self, xs, evaluator, beta):
        index = 0
        maxValue = -float('Inf')
        for i in range(len(xs) - 1):
            mini, maxi = evaluator.confidenceBounds(xs[i], beta)
            minj, maxj = evaluator.confidenceBounds(xs[i + 1], beta)
            value = max(maxi, maxj) - min(mini, minj)
            if (value > maxValue):
                maxValue = value
                index = i
        return index, maxValue


class TessellationTh2D:
    def __init__(self, interval, fun, model, beta, th, t):
        self.interval = interval
        self.fun = fun
        self.model = model
        self.beta = beta
        self.th = th
        self.t = t

    def tessellation(self):
        t = self.t
        beta = self.beta
        interval = self.interval
        model = self.model

        trainSetX = np.array([[i, j] for i in np.linspace(interval[0][0], interval[1][0], 10) for j in
                     np.linspace(interval[0][1], interval[1][1], 10)])
        trainSetY = np.array([[self.fun(x)] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = [[[i, j] for i in np.linspace(interval[0][0], interval[1][0], 50)] for j in
              np.linspace(interval[0][1], interval[1][1], 50)]
        # xs = np.linspace(interval[0], interval[1], num=200)
        return self.slopy(t, xs, model, self.fun, beta)
        #M = self.boxy(xs, model.getEvaluator(), t, beta)
        #self.drowBoxes(M, xs, model, beta)

    # def boxy(self,xs, evaluator, t, beta):
    #     c = 0
    #     M = [0, 0, 0, 0]
    #     while c < len(xs):
    #         end, minOld, maxOld, condition = self.findInterval(xs, c, t, evaluator, beta)
    #         M = np.vstack([M, [xs[c], xs[end], minOld, maxOld]])
    #         c = end
    #         if condition:
    #             break
    #     return M[1:]

    # def drowBoxes(self,M, xs, model, beta):
    #     plt.figure()
    #     currentAxis = plt.gca()
    #     for i in range(len(M)):
    #         currentAxis.add_patch(
    #             Rectangle((M[i][0], M[i][2]), M[i][1] - M[i][0], M[i][3] - M[i][2], fill=None,
    #                       alpha=1))
    #     plt.xlim([xs[0], xs[-1]])
    #     evaluator = model.getEvaluator()
    #     lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
    #     ys = evaluator.mean(xs[:, None])
    #     plt.plot(xs, ub)
    #     plt.plot(xs, ys, 'r')
    #     plt.plot(xs, lb)
    #     plt.plot([xs[0],xs[-1]], [self.th,self.th])
    #     plt.scatter(model.getTrainSetX(), model.getTrainSetY())
    #     plt.show(block=True)

    # def findInterval(self,xs, start, t, evaluator, beta):
    #     global maxOld, minOld
    #     th = self.th
    #     c = start
    #     lb, ub = evaluator.confidenceBounds(xs[c], beta)
    #     lbd, ubd = evaluator.confidenceBounds(xs[c + 1], beta)
    #     maxValue = max(ubd, ub)
    #     minValue = min(lbd, lb)
    #     intersec = lambda x, y: (x < th and y > th)
    #     if (intersec(minValue, maxValue)):
    #         condition = lambda x, y: intersec(x, y) and y - x < t
    #     else:
    #         condition = lambda x, y: not intersec(x, y)
    #     c = c + 1
    #     while condition(minValue, maxValue) and c < len(xs) - 1:
    #         c += 1
    #         maxOld = maxValue
    #         minOld = minValue
    #         lb, ub = evaluator.confidenceBounds(xs[c], beta)
    #         maxValue = max(maxValue, ub)
    #         minValue = min(minValue, lb)
    #     if (c == len(xs) - 1):
    #         end = c
    #     else:
    #         end = c - 1
    #     return end, maxOld, minOld, c == len(xs) - 1

    def slopy(self, t, xs, model, fun, beta):
        slope = float('Inf')
        c = [1,1]
        while (slope > t):
            model.addDesign(xs[c[0]][c[1]], fun(xs[c[0]][c[1]]))
            model.addDesign(xs[c[0]+1][c[1]], fun(xs[c[0]+1][c[1]]))
            model.addDesign(xs[c[0]][c[1]+1], fun(xs[c[0]][c[1]+1]))
            model.addDesign(xs[c[0]+1][c[1]+1], fun(xs[c[0]+1][c[1]+1]))
            model.fit()
            c, slope, lb, ub = self.findMaxSlope(xs, model.getEvaluator(), beta)
            print(str(c) + ':' + str(slope))
        return lb, ub,model.getTrainSetX(),xs

    def findMaxSlope(self, xs, evaluator, beta):
        th = self.th
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
                if (value > maxValue):
                    maxValue = value
                    index = [i,j]
        return index, maxValue, lb, ub

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
        trainSetX = np.array([np.linspace(interval[0], interval[1], 40)]).T
        trainSetY = np.array([[self.fun(x[0])] for x in trainSetX])
        model.setTrainSet(trainSetX, trainSetY)
        model.fit()
        xs = np.linspace(interval[0], interval[1], num=200)
        #xs = np.arange(interval[0], interval[1],0.0015)
        res=self.slopy(t, xs, model, self.fun, beta)
        b = time.time()
        M = self.boxy(xs, model.getEvaluator(), res, beta)
        print(b-a)
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
        #while (slope > t):
        while (value > t):
            for e in c:
                model.addDesign(xs[e], fun(xs[e]))
            model.fit()
            c, slope,counter = self.findMaxSlopeMulti(xs, model.getEvaluator(), beta)
            #print(str(c) + ':' + str(slope))
            value = (counter + 1)/len(xs)
            # print(str(c) + ':' + str(slope))
            print(str(c) + ':' + str(value))
        return slope

    def findMaxSlope(self, xs, evaluator, beta):
        th = self.th
        index = 0
        maxValue = -float('Inf')
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
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
        counter=0
        index = 0
        maxValue = -float('Inf')
        value = np.zeros(len(xs) - 1)
        lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        for i in range(len(xs) - 1):
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
        return index[-17:-1:2], value[index[-1]],counter
        #return index[-6:-1], value[index[-1]]


