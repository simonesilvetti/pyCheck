import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.stats import norm

from pycheck.regressor.supportFunction import getIndexes, findBoundsIndex


class TessellationThreshold1D:
    def __init__(self, interval, spf, model, beta, th, volumeEps):
        self.interval = interval
        self.spf = spf
        self.model = model
        self.beta = beta
        self.th = th
        self.volumeEps = volumeEps

    def tessellation(self):
        timeInit = time.time()
        trainSetX = np.array([np.linspace(self.interval[0], self.interval[1], 40)]).T
        trainSetY = np.array([[self.spf(x[0])] for x in trainSetX])
        self.model.setTrainSet(trainSetX, trainSetY)
        self.model.fit()
        hGrid = np.linspace(self.interval[0], self.interval[1], num=400)
        self.BayesianParamterSynthesis(hGrid)
        timeEnd = time.time()
        print("Execution Time:" + str(timeEnd - timeInit))
        # Plot
        self.drowAllBoxes(hGrid)
        return self.model, hGrid

    def drowAllBoxes(self, xs):
        th = self.th
        evaluator = self.model.getEvaluator()
        lb, ub = evaluator.confidenceBounds(xs[:, None], self.beta)
        fig = plt.figure()
        currentAxis = plt.gca()
        for i in range(len(xs) - 1):
            maxx = max(ub[i], ub[i + 1])
            minn = min(lb[i], lb[i + 1])
            if (minn > th):
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              # facecolor="red", alpha=0.5,edgecolor='k'))
                              facecolor="red", alpha=0.5))
            elif (maxx < th):
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              # facecolor="blue", alpha=0.5,edgecolor='k'))
                              facecolor="blue", alpha=0.5))
            else:
                currentAxis.add_patch(
                    Rectangle((xs[i], min(lb[i], lb[i + 1])), xs[i + 1] - xs[i],
                              max(ub[i], ub[i + 1]) - min(lb[i], lb[i + 1]),
                              # facecolor="yellow", alpha=0.5,edgecolor='k'))
                              facecolor="yellow", alpha=0.5))

        plt.xlim([xs[0], xs[-1]])
        ys = evaluator.mean(xs[:, None])
        plt.plot(xs, ub)
        plt.plot(xs, ys, 'r')
        plt.plot(xs, lb)
        plt.xlabel('k_i')
        plt.ylabel('P')
        plt.plot([xs[0], xs[-1]], [self.th, self.th])
        plt.scatter(self.model.getTrainSetX(), self.model.getTrainSetY())
        fig.savefig('threshold1DFigure.png', dpi=90, bbox_inches='tight')
        plt.show(block=True)

    def BayesianParamterSynthesis(self, xs):
        while True:
            regions, slope, value = self.findUndefinedRegions(xs, self.model.getEvaluator(), self.beta)
            if value < self.volumeEps:
                break
            for region in regions:
                self.model.addDesign(xs[int(region)], self.spf(xs[int(region)]))
            self.model.fit()
        return slope

    def findUndefinedRegions(self, xs, evaluator, beta):
        th = norm.ppf(self.th)

        # lb, ub = evaluator.confidenceBounds(xs[:, None], beta)
        lb, ub = evaluator.confidenceBoundsGP(xs[:, None], beta)
        res = findBoundsIndex(lb[:, None], ub[:, None], th)
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


class TessellationThreshold2D:
    def __init__(self, domain, spf, model, beta, threshold, volumeEps):
        self.domain = domain
        self.spf = spf
        self.model = model
        self.beta = beta
        self.threshold = threshold
        self.volumeEps = volumeEps

    def tessellation(self):
        fig = plt.figure()
        timeInit = time.time()
        trainSetX = np.array([[i, j] for i in np.linspace(self.domain[0][0], self.domain[1][0], 20) for j in
                              np.linspace(self.domain[0][1], self.domain[1][1], 20)])
        trainSetY = np.array([[self.spf(x)] for x in trainSetX])
        self.model.setTrainSet(trainSetX, trainSetY)
        self.model.fit()
        xs = [[[i, j] for i in np.linspace(self.domain[0][0], self.domain[1][0], 100)] for j in
              np.linspace(self.domain[0][1], self.domain[1][1], 100)]
        lb, ub, c, xs, model = self.BayesianParamterSynthesis(xs)
        timeEnd = time.time()
        print("Execution Time:" + str(timeEnd - timeInit))
        currentAxis = plt.gca()
        for i in range(len(xs) - 1):
            for j in range(len(xs[i]) - 1):
                l1, l2, l3, l4 = lb[i, j], lb[i + 1, j], lb[i, j + 1], lb[i + 1, j + 1]
                u1, u2, u3, u4 = ub[i, j], ub[i + 1, j], ub[i, j + 1], ub[i + 1, j + 1]
                up = min(l1, l2, l3, l4) > self.threshold
                down = max(u1, u2, u3, u4) < self.threshold
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
        plt.xlim([self.domain[0][0], self.domain[1][0]])
        plt.ylim([self.domain[0][1], self.domain[1][1]])
        plt.scatter(c[:, 0], c[:, 1], c='k')
        plt.xlabel('k_i')
        plt.ylabel('k_r')
        fig.savefig('threshold2DFigure.png', dpi=90, bbox_inches='tight')
        # plt.show(block=True)

    def BayesianParamterSynthesis(self, xs):
        while True:
            c, slope, lb, ub, counter = self.findUndefinedRegions(xs)
            value = (counter + 1) / (len(xs) * len(xs))
            if value <= self.volumeEps:
                break
            for e in c:
                self.model.addDesign(xs[e[0]][e[1]], self.spf(xs[e[0]][e[1]]))
            self.model.fit()

        return lb, ub, self.model.getTrainSetX(), xs, self.model

    def findUndefinedRegions(self, xs):
        maxValue = -float('Inf')
        lb = np.zeros([len(xs), len(xs[0])])
        ub = np.zeros([len(xs), len(xs[0])])
        value = list()
        index = list()
        for i in range(len(xs)):
            lb[i], ub[i] = self.model.getEvaluator().confidenceBounds(xs[i], self.beta)
        for i in range(len(xs) - 1):
            for j in range(len(xs[i]) - 1):
                maxx = max(ub[i, j], ub[i + 1, j], ub[i, j + 1], ub[i + 1, j + 1])
                minx = min(lb[i, j], lb[i + 1, j], lb[i, j + 1], lb[i + 1, j + 1])
                if minx <= self.threshold <= maxx:
                    value.append(min(self.threshold - minx, maxx - self.threshold))
                    index.append([i, j])
        index = np.array(index)
        np.random.shuffle(index)
        return index[1:100], maxValue, lb, ub, len(index)
