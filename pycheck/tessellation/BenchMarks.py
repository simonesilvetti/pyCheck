"""
Bencharms of the Paper L.Bortolussi, S.Silvetti,"Bayesian Statistical
Parameter Synthesis for Linear Temporal Properties of Stochastic Models"
submetted to TACAS 2018

- There are one function for each of the case in the paper, Table 1 (case1(),case2(),case3()).

- The execution time is showed in console.

- Depending on the dimensionality of the space $ ($ = 1 or 2, notice that case1() and case2() are 1 dimensional. On the contrary,
case3() is 2 Dimensional)the file threshold$DFigure.png will be created.

- There is one function for the Accuracy Test of Case 1 (accuracyCase1()) which prints in console
the missclassified percentage of points.

TBN: The STL formula is \phi = (I>0) \,\mathcal{U}_{[100,120]}\, (I=0) . We check it with a dedicated monitoring
monitor algorithm which checks if the reactions end between 100 and 120. It corresponds to simulate.py:67
which is contained in the package pycheck/regressor.


Further details:
http://simonesilvetti.com/pycheck/
https://bitbucket.org/xfde/pycheck/
email: simone.silvetti@gmail.com
"""
import numpy as np

from pycheck.regressor.simulate import simulate
from pycheck.tessellation.Tessellation import TessellationThreshold1D, TessellationThreshold2D
from pycheck.regressor.BayesianRegressor import EP


def case1():
    volumeEps = 0.1
    threshold = 0.1
    beta = 3
    model = EP()
    model.setScale(1000)
    trajectoriesNumber = 1000
    # timeEnd = 125
    domain = [0.005, 0.3]
    fun = lambda x: simulate([0.05, x], trajectoriesNumber)
    tas = TessellationThreshold1D(domain, fun, model, beta, threshold, volumeEps)
    model = tas.tessellation()



def case2():
    volumeEps = 0.1
    threshold = 0.1
    beta = 3
    model = EP()
    model.setScale(1000)
    trajectoriesNumber = 1000
    #timeEnd = 125
    domain = [0.005, 0.2]
    fun = lambda x: simulate([x, 0.12], trajectoriesNumber)
    tas = TessellationThreshold1D(domain, fun, model, beta, threshold, volumeEps)
    tas.tessellation()


def case3():
    volumeEps = 0.1
    threshold = 0.1
    beta = 3
    model = EP()
    model.setScale(1000)
    trajectoriesNumber = 1000
    #timeEnd = 125
    domain = [[0.005, 0.005], [0.2, 0.3]]
    fun = lambda x: simulate(x, trajectoriesNumber)
    tas = TessellationThreshold2D(domain, fun, model, beta, threshold, volumeEps)
    tas.tessellation()

def accuracyCase1():
    volumeEps = 0.1
    threshold = 0.1
    beta = 3
    model = EP()
    model.setScale(1000)
    trajectoriesNumber = 1000
    #timeEnd = 125
    domain = [0.005, 0.3]
    fun = lambda x: simulate([0.05, x], trajectoriesNumber)
    tas = TessellationThreshold1D(domain, fun, model, beta, threshold, volumeEps)
    model,xs = tas.tessellation()
    testGrid = np.linspace(0.01, 0.07, num=300)
    accurateFunction = lambda x: simulate([0.05, x], 10000)

    accurateValues = [accurateFunction(e) for e in testGrid]
    estimated = [model.getEvaluator().confidenceBounds(e, 3) for e in testGrid]

    count = 0
    for i in range(len(accurateValues)):
        if (estimated[i][0] > threshold and accurateValues[i] < threshold):
            count = count + 1
        elif (estimated[i][1] < threshold and accurateValues[i] > threshold):
            count = count + 1
    print("Missclassifed Points:" + str(count*100 / len(accurateValues))+"%")
    print("______________________________________________")


def derivativeCase1():
    volumeEps = 0.1
    threshold = 0.1
    beta = 3
    model = EP()
    model.setScale(1000)
    trajectoriesNumber = 1000
    # timeEnd = 125
    domain = [0.005, 0.3]
    fun = lambda x: simulate([0.05, x], trajectoriesNumber)
    tas = TessellationThreshold1D(domain, fun, model, beta, threshold, volumeEps)
    m, xs = tas.tessellation()
    lbub = m.getEvaluator().confidenceBounds(xs[:, None], 2)
    """
    METHOD 1: LOCAL LISPSCHITZ BOUND
    """
    localDerivativeLB = lbub[0][1:] - lbub[0][:-1]
    localDerivativeUB = lbub[1][1:] - lbub[1][:-1]
    """
    METHOD 2: GLOBAL LISPSCHITZ BOUND
    """
    globalDerivativeLB = max(localDerivativeLB)
    globalDerivativeUB = max(localDerivativeUB)

    lbub = m.getEvaluator().confidenceBounds(xs[:, None], 3)
    changeStatus = 0
    for i in range(len(xs) - 1):
        if (lbub[0][i] > threshold and lbub[0][i] - localDerivativeLB[i] < threshold):
            changeStatus = changeStatus + 1
        elif (lbub[1][i] < threshold and lbub[1][i] + localDerivativeUB[i] > threshold):
            changeStatus = changeStatus + 1

    print("LOCAL LISPSCHITZ (points changing status):" + str(changeStatus * 100 / (len(xs) - 1)) + "%")
    print("______________________________________________")

    changeStatus = 0
    for i in range(len(xs) - 1):
        if (lbub[0][i] > threshold and lbub[0][i] - globalDerivativeLB < threshold):
            changeStatus = changeStatus + 1
        elif (lbub[1][i] < threshold and lbub[1][i] + globalDerivativeUB > threshold):
            changeStatus = changeStatus + 1
    print("GLOBAL LISPSCHITZ (points changing status):" + str(changeStatus * 100 / (len(xs) - 1)) + "%")
    print("______________________________________________")




if __name__ == '__main__':
   case1()
    #case2()
    #case3()
    #accuracyCase1()
    #derivativeCase1()
