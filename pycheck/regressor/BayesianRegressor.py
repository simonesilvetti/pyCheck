import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor

from pycheck.regressor.EP import ExpectedPropagation


class BayesianRegressor():
    def setTrainSet(self, trainSetX, trainSetY):
        self.trainSetX = trainSetX
        self.trainSetY = trainSetY

    def fit(self):
        pass

    def addDesign(self, e, f):
        self.trainSetX = np.vstack([self.trainSetX, e])
        self.trainSetY = np.vstack([self.trainSetY, f])

    def getTrainSetX(self):
        return self.trainSetX

    def getTrainSetY(self):
        return self.trainSetY

    def getEvaluator(self):
        pass


class Evaluator:
    def mean(self, x):
        pass

    def confidenceBounds(self, x, beta):
        pass

    def confidenceBoundsGP(self, x, beta):
        pass

    def thr(self, x, thr):
        pass


class RaB(BayesianRegressor):
    def setTrainSet(self, trainSetX, trainSetY):
        self.trainSetX = trainSetX
        self.trainSetY = trainSetY
        self.model=GaussianProcessRegressor(n_restarts_optimizer=20)

    def fit(self):
        self.model.fit(self.trainSetX, self.trainSetY.T[0])

    def getEvaluator(self):
        return RaBEvaluator(self.model)



class RaBEvaluator(Evaluator):
    def __init__(self,gp):
        self.gp=gp

    def mean(self, x):
        ys, sigma = self.gp.predict(x, return_std=True)
        return ys

    def confidenceBounds(self, x, beta):
        ys, sigma = self.gp.predict(x, return_std=True)
        betaSigma = beta * sigma
        return ys - betaSigma, ys + betaSigma

class EP(BayesianRegressor):
    def setTrainSet(self, trainSetX, trainSetY):
        super().setTrainSet(trainSetX, trainSetY)
        self.model=ExpectedPropagation(self.scale)

    def fit(self):
        self.model.fit(self.trainSetX, self.trainSetY)

    def getEvaluator(self):
        return EPEvaluator(self.model)

    def setScale(self,scale):
        self.scale=scale


class EPEvaluator(Evaluator):
    def __init__(self,model):
        self.model=model

    def mean(self, x):
        a, b = self.model.latentPrediction(x)
        return self.model.getProbability(a, b)

    def confidenceBounds(self, x, beta):
        a, b = self.model.latentPrediction(x)
        bounds = self.model.getBounds(a, b, beta)
        return bounds[0, :], bounds[1, :]

    def confidenceBoundsGP(self, x, beta):
        a, b = self.model.latentPrediction(x)
        return np.array([a - beta * np.sqrt(b),a + beta * np.sqrt(b)])

    def thr(self, x, thr):
        v=self.mean(x)
        s=v-self.confidenceBounds(x, 1)[0]
        return (v-thr)/s

