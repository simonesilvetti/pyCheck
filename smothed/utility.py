from scipy.stats import norm
import numpy as np
from scipy.special import erf
import stochpy
from antlr4 import FileStream
import matplotlib.pyplot as plt

from scipy.optimize import minimize
from sklearn.gaussian_process.kernels import RBF

from pycheck.semantics.STL.BooleanSemantics import BooleanSemantics
from pycheck.semantics.STL.STLLexer import STLLexer, CommonTokenStream
from pycheck.semantics.STL.STLParser import STLParser
from pycheck.series.TimeSeries import TimeSeries

CORRECTION_FACTOR = 1
CORRECTION = 1E-4
eps_damp = 0.5
invC = []
mu_tilde = []
trainSetX=[]
trainSetY=[]
# kernel = lambda x: 0
# scale = 1

def getProbability(mean,variance):
    return norm.cdf(mean/np.sqrt(1+variance))

def getBounds(mean,variance,beta):
    return norm.cdf(np.tile(1 / np.sqrt(1+variance), (trainSetX.shape[1], 1)) * [mean - beta * np.sqrt(variance), mean + beta * np.sqrt(variance)])

def doTraining():
    gauss = expectationPropagation(1e-6)
    v_tilde=gauss.Term[:,0]
    tau_tilde=gauss.Term[:,1]
    diag_sigma_tilde=1/tau_tilde
    global mu_tilde
    mu_tilde=v_tilde * diag_sigma_tilde
    sigma_tilde = np.diag(diag_sigma_tilde)
    global invC
    invC =  np.linalg.solve(gauss.C+sigma_tilde, np.eye(len(mu_tilde)))
    return invC,mu_tilde,sigma_tilde

def expectationPropagation(tolerance):
    gauss = Gauss()
    p=kernel(trainSetX)
    gauss.C=p

    gauss.C= gauss.C + CORRECTION * np.eye(len(gauss.C))
    gauss.LC= np.linalg.cholesky(gauss.C)
    gauss.LC_t = gauss.LC.transpose()
    gauss_LC_diag =np.diag(gauss.LC)
    logdet_LC = 2 * np.sum(np.log(gauss_LC_diag))
    logZprior = 0.5 * logdet_LC
    n = len(trainSetX)
    logZterms = np.zeros(shape=(n,1))
    logZloo = np.zeros(shape=(n,1))
    Term = np.zeros(shape=(n,2))
    appo,gauss = computeMarginalMoments(gauss,Term,logdet_LC)

    # Stuff related to the likelihood
    gauss.LikPar_p= trainSetY * scale
    gauss.LikPar_q= np.ones(shape=(n,1))*scale-gauss.LikPar_p
    NODES = 96
    gauss.xGauss=np.zeros(shape=(NODES,1))
    gauss.wGauss=np.zeros(shape=(NODES,1))
    gauss.xGauss, gauss.wGauss = gausshermite(NODES, gauss.xGauss, gauss.wGauss)
    gauss.logwGauss= np.log(gauss.wGauss)
    # for (int i = 0; i < gauss.gauss.logwGauss.getLength(); i++)
    # gauss.gauss.logwGauss.put(i, Math.log(gauss.gauss.wGauss.get(i)));

    # initialize cycle control
    MaxIter = 1000
    tol = tolerance
    logZold = 0
    logZ = 2 * tol
    steps = 0
    logZappx = 0
    while ((np.abs(logZ - logZold) > tol) & (steps < MaxIter)):
         # cycle control
        steps = steps + 1
        logZold = logZ
        cavGauss = computeCavities(gauss, -Term)
        #
        # // [Term, logZterms, logZloo] = EPupdate(cavGauss, gauss.LikFunc, y,
        #                                 // Term, eps_damp);
        update = ep_update(cavGauss, Term, eps_damp,gauss.LikPar_p, gauss.LikPar_q, gauss.xGauss,gauss.logwGauss)
        Term = update.TermNew
        logZterms = update.logZterms
        logZloo = update.logZ

        logZappx, gauss = computeMarginalMoments(gauss, Term,logdet_LC)
        logZ = logZterms.sum() + logZappx

    #finishing
    logZ = logZ - logZprior
    gauss.logZloo = np.sum(logZloo)
    gauss.logZappx = logZappx
    gauss.logZterms = logZterms
    gauss.logZ = logZ
    gauss.Term = Term
    return gauss


def computeMarginalMoments(gauss, Term,logdet_LC):
    # // (repmat(Term(:,2),1,N).*Gauss.LC)
    N = len(Term)
    tmp = np.tile(Term[:, 1], (N, 1)).T * gauss.LC
    A = np.matrix.dot(gauss.LC_t,tmp) + np.eye(N) * CORRECTION_FACTOR
    # // Serious numerical stability issue with the calculation
    # // of A (i.e. A = LC' * tmp + I)
    # // as it does not appear to be PD for large amplitudes
    gauss.L = np.linalg.cholesky(A)
    # // Gauss.W = Gauss.L\(Gauss.LC');
    gauss.W = np.linalg.solve(gauss.L, gauss.LC_t)
    # // Gauss.diagV = sum(Gauss.W.*Gauss.W,1)';
    tmp = gauss.W * gauss.W
    # gauss.diagV = np.zeros(shape=(N, 1))
    # for (int i = 0; i < N; i++)
    # gauss.diagV.put(i, tmp.getColumn(i).sum());
    gauss.diagV = np.array([np.sum(tmp, 0)]).T
    # // or
    # // gauss.diagV = gauss.W.transpose().mmul(gauss.W).diag();
    #
    # // Gauss.m = Gauss.W'*(Gauss.W*Term(:,1));
    tmp = np.dot(gauss.W,Term[:,0])
    gauss.m = np.array([np.dot(gauss.W.T,tmp)]).T
    # // logdet = -2*sum(log(diag(Gauss.L))) + 2*sum(log(diag(Gauss.LC)));
    logdet = 0
    sum = 0
    tmp = np.diag(gauss.L)
    # for (int i = 0; i < tmp.getLength(); i++)
    # sum += Math.log(tmp.get(i));
    # logdet += -2 * sum;
    logdet += -2.0* np.sum(np.log(tmp))
    # // sum = 0;
    # // tmp = gauss.LC.diag();
    # // for (int i = 0; i < tmp.getLength(); i++)
    # // sum += Math.log(tmp.get(i));

    logdet += logdet_LC

    # // logZappx = 0.5*(Gauss.m'*Term(:,1) + logdet);
    logZappx = 0.5 * (np.dot(gauss.m.transpose(),Term[:,0]) + logdet)
    return logZappx , gauss



def gausshermite(n, x, w):
    x0 = np.zeros(shape=(len(x),1))
    w0 = np.zeros(shape=(len(w),1))
    m = int((n + 1) / 2)
    z = 0
    pp = 0
    p1 = 0
    p2 = 0
    p3 = 0
    for i in range(0,m):
        if (i == 0):
            z = np.sqrt(2 * n + 1) - 1.85575*(2 * n + 1)**(-0.16667)
        elif(i == 1):
            z = z - 1.14 * n**0.426 / z
        elif(i == 2):
            z = 1.86 * z - 0.86 * x0[0]
        elif (i == 3):
            z = 1.91 * z - 0.91 * x0[1]
        else:
            z = 2.0 * z - x0[i-2]

        for its in range(0,10):
            p1 = 1 / np.sqrt(np.sqrt(np.pi))
            p2 = 0
            for j in range (1,n+1):
                p3 = p2
                p2 = p1
                a = z * np.sqrt(2 / j) * p2
                b = np.sqrt((j - 1) / j) * p3
                p1 = a - b

            pp = np.sqrt(2 * n) * p2
            z1 = z
            z = z1 - p1 / pp
            if (np.abs(z - z1) < 2.2204e-16):
                break

        x0[i]=z
        x0[n - 1 - i] =  -z
        w0[i]=2 / (pp * pp)
        w0[n - 1 - i]=w0[i]

    w0 = w0/np.sqrt(np.pi)
    x0 = x0*np.sqrt(2)
    x0 = np.sort(x0)[::-1]
    x = x0
    w = w0
    return x,w

def computeCavities(gauss,Term):
    cavGauss = CavGauss()
    # // C = Gauss.diagV;
    C = gauss.diagV
    # // s = 1./(1 + Term(:,2).*C)
    appo = np.array([a*b for a,b in zip(Term[:,1],C)])
    s = np.ones(shape=(len(C), 1)) / (appo + 1)
    # // CavGauss.diagV = s. * C;
    cavGauss.diagV = s*C
    # // CavGauss.m = s. * (Gauss.m + Term(:, 1).*C);
    appo = np.array([a * b for a, b in zip(Term[:, 0], C)])
    cavGauss.m = s* (gauss.m +appo)
    return cavGauss

def ep_update(cavGauss,Term,eps_damp,LikPar_p, LikPar_q, xGauss, wGauss):
    update = EPupdate()
    Cumul = np.zeros(shape=(len(LikPar_p),2))
    logZ, Cumul = GaussHermiteNQ(LikPar_p, LikPar_q, cavGauss.m,cavGauss.diagV, xGauss, wGauss, Cumul)
    update.logZ = np.array([logZ]).T
    m2 = cavGauss.m*cavGauss.m
    logV=np.log(cavGauss.diagV)
    cumul1=np.array([Cumul[:,0]*Cumul[:,0]]).T
    cumul2 = np.log(np.array([Cumul[:,1]]).T)
    tmp = m2/(cavGauss.diagV)+logV-(cumul1/np.array([Cumul[:,1]]).T+cumul2)
    update.logZterms = update.logZ+tmp*0.5
    ones = np.ones(shape=(len(LikPar_p),1))
    TermNew = np.zeros(shape=(len(LikPar_p),2))
    c1 = np.array([Cumul[:,0]/Cumul[:,1]]).T-cavGauss.m/cavGauss.diagV
    c2 = ones/ np.array([Cumul[:,1]]).T-ones/cavGauss.diagV
    TermNew[:,0] =c1[:,0]
    TermNew[:, 1] = c2[:,0]
    TermNew = (1 - eps_damp) * Term + eps_damp * TermNew
    update.TermNew = TermNew
    return update


def GaussHermiteNQ(FuncPar_p,FuncPar_q,m,v,xGH,logwGH,Cumul):
    stdv = np.sqrt(v)
    Nnodes = len(xGH)
    tmp = np.dot(stdv,xGH.transpose())
    Y = tmp + np.tile(m,(1, Nnodes))
    tmp = logprobitpow(Y, FuncPar_p, FuncPar_q)
    G = tmp+np.tile(logwGH.transpose(),(len(tmp),1))
    maxG = np.max(G,1)
    G = G - np.tile(maxG, (Nnodes,1)).T
    expG = np.exp(G)
    denominator = np.sum(expG, 1)
    logZ = maxG + np.log(denominator)
    deltam = stdv* (np.dot(expG,xGH))/ np.array([denominator]).T
    appo = m + deltam
    Cumul[:, 0] = appo[:,0]
    appo=v* np.dot(expG,xGH**2)/ np.array([denominator]).T - deltam**2
    Cumul[:, 1] =appo[:,0]
    return logZ , Cumul

def logprobitpow(X,LikPar_p, LikPar_q):
    n = X.shape[0]
    m = X.shape[1]
    Y=np.zeros(shape=(n,m))
    for i in range(0,n):
        for j in range(0,m):
            Y[i][j] = ncdflogbc(X[i][j])
    Za =  Y*np.tile(LikPar_p,(1, m))
    Y = np.zeros(shape=(n, m))
    for i in range(0, n):
        for j in range(0, m):
            Y[i][j] = ncdflogbc(-X[i][j])
    Zb = Y * np.tile(LikPar_q, (1, m))
    return Za+Zb

def ncdflogbc(x):
    sqrt2 = np.sqrt(2)
    invSqrt2 = 1 / sqrt2
    log2 = np.log(2)
    treshold = -sqrt2 * 5
    z = -x
    if (x >= 0):
        return np.log(1 + erf(x * invSqrt2)) - log2
    if (treshold < x):
        return np.log(1 - erf(-x * invSqrt2)) - log2
    return -0.5*np.log(np.pi)- log2-0.5*z*z- np.log(z)+ np.log(1 - 1 / z + 3 /z**4 - 15 /z**6+ 105 / z**8 - 945 / z**10)

def latentPrediction(Xs):
    kss = np.diag(kernel(Xs))
    ks = kernel(Xs, trainSetX)
    # if (invC == null | | mu_tilde == null | | trainingSet.isModified())
    #     doTraining();
    tmp = np.dot(ks,invC)
    fs = np.dot(tmp,mu_tilde)
    vfs = kss-(np.diag(np.dot(tmp,ks.transpose())))
    return fs,vfs

def getMarginalLikelihood():
    gauss = expectationPropagation(1e-3)
    return gauss.logZ

def objectivefunction(l):
    global kernel
    r = RBF(l)
    kernel = r
    return getMarginalLikelihood()


def getDefaultHyperarametersRBF(X,Y):
    signal = 0.5*np.max(Y)-np.min(X)
    sum = 0
    n, dim = X.shape
    for d in range(0,dim):
        max = -float('inf')
        min = float('inf')
        for i in range(0,n):
            curr = X[i][d]
            if (curr > max):
                max = curr
            if (curr < min):
                min = curr
        sum += (max - min) / 10.0
    lengthScale= sum / dim
    return signal,lengthScale







class EPupdate():
    def __init__(self):
        self.TermNew = None
        self.logZterms = None
        self.logZ = None

class CavGauss():
    def __init__(self):
        self.diagV = None
        self.m = None

class Gauss():
    def __init__(self):
        self.LikPar_p = None
        self.xGauss = None
        self.wGauss = None
        self.logwGauss = None
        self.C = []
        self.LC = None
        self.LC_t = None
        self.L = None
        self.W = None
        self.diagV = None
        self.m = None
        self.logZloo = None
        self.logZappx = None
        self.logZterms = None
        self.logZ = None
        self.Term = None


def function(value):
    smod = stochpy.SSA()
    smod.Model('epidemic.psc')
    smod.ChangeParameter('ki',value[0])
    smod.DoStochSim(end=130, mode='time', trajectories=150,quiet=True)
    input_stream = FileStream('t.expr')
    lexer = STLLexer(input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = STLParser(token_stream)
    tree = parser.prog()
    count = 0
    for i in range(1, 151):
        smod.GetTrajectoryData(i)
        # if(max(smod.data_stochsim.time)<130):
        #     print("CAZZO")
        visitor = BooleanSemantics(
            timeSeries=TimeSeries(smod.data_stochsim.species_labels, smod.data_stochsim.time, smod.data_stochsim.species.T))
        count += 1 if visitor.visit(tree) else 0
    return count/150


#
#
#
# global trainSetX
# trainSetX = np.array([np.linspace(0.005, 0.3, num=20)]).T
# global trainSetY
# trainSetY =np.array([np.array([function(x) for x in trainSetX])]).T
# # value =np.array([[0.2],[0.25],[0.26],[0.5],[0.9]])
# global scale
# scale =500
# aa,bb =getDefaultHyperarametersRBF(trainSetX, trainSetY)
#
# res = minimize(objectivefunction, bb, method='L-BFGS-B', bounds=((0.5*bb,2*bb),))
# global kernel
# r = RBF(res.x)
# print(r)
# kernel = r
# invC,mu_tilde,sigma_tilde = doTraining()
# xs = [np.linspace(0.005, 0.3, num=200)]
# a,b=latentPrediction(np.array(xs).T)
# prob = getProbability(a,b)
# bounds = getBounds(a,b,2)
# print(prob)
# print(bounds)
#
# plt.scatter(trainSetX, trainSetY)
# plt.plot(xs[0], prob)
# plt.plot(xs[0], bounds[0,:])
# plt.plot(xs[0], bounds[1,:])
#
# plt.show(block=True)

def fit():
    aa, bb = getDefaultHyperarametersRBF(trainSetX, trainSetY)

    res = minimize(objectivefunction, bb, method='L-BFGS-B', bounds=((0.5 * bb, 2 * bb),))
    global kernel
    r = RBF(res.x)
    print(r)
    kernel = r
    global invC,mu_tilde,sigma_tilde
    invC, mu_tilde, sigma_tilde = doTraining()




from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from pyDOE import *
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle




def modelMin(x):
    a, b = latentPrediction(x)
    # prob = getProbability(a, b)
    bounds = getBounds(a, b, 3)
    # y_pred, sigma = gp.predict(x[0], return_std=True)
    return bounds[0,:]


def modelMax(x):
    a, b = latentPrediction(x)
    # prob = getProbability(a, b)
    bounds = getBounds(a, b, 3)
    # y_pred, sigma = gp.predict(x[0], return_std=True)
    return -bounds[1, :]

def model(x):
    a, b = latentPrediction(x)
    prob = getProbability(a, b)
    return prob


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





# fun = lambda x: np.sin(3*x)
glb = 0.005
gub = 0.3
# global trainSetX
# global trainSetY
trainSetX = np.array([[glb],[gub]])
trainSetX = np.array([np.linspace(glb,gub,5)]).T
trainSetY =np.array([np.array([function(x) for x in trainSetX])]).T

scale = 150
fit()
# gp = GaussianProcessRegressor(n_restarts_optimizer=20)
#
# gp.fit(trainSetX, trainSetY)

# initialization########################
sets = list()
Xminmax = list()
Yminmax = list()
sets.append([glb, gub])
i = 0
lb = sets[i][0]
ub = sets[i][1]
min, eMin = minimizeGP([lb, ub])
max, eMax = maximizeGp([lb, ub])
Xminmax.insert(i, [min, max])
Yminmax.insert(i, [eMin, eMax])


val = 1


while(val > 0.3):
    while (val > 0.3):
        # print(val)
        i, val = findMax(Yminmax)
        print(str(len(trainSetX)) + '-->' + str(val))
        lb = sets[i][0]
        ub = sets[i][1]
        xmin = Xminmax[i][0]
        xmax = Xminmax[i][1]

        c = 0
        if dinstanceFromtS(xmin) > 0:
            trainSetX = np.vstack([trainSetX, [xmin]])
            rmin = function([xmin])
            trainSetY = np.vstack([trainSetY, rmin])
            c = 1
        if dinstanceFromtS(xmax) > 0:
            trainSetX = np.vstack([trainSetX, [xmax]])
            rmax = function([xmax])
            trainSetY = np.vstack([trainSetY, rmax])
            c = 1

        if c == 1:
            fit()

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

        if (eMaxR < eMinR or eMaxL < eMinL):
            print('=')
    Yminmax = adjustAll(Yminmax)
Yminmax=adjustAll(Yminmax)

print(sets)
print(Xminmax)
print(Yminmax)

print(sets)
plt.figure()
plt.xlim([glb,gub])
plt.ylim([0, 1])
currentAxis = plt.gca()
plt.scatter(trainSetX,trainSetY)
for i in range(len(sets)):
    currentAxis.add_patch(Rectangle((sets[i][0], Yminmax[i][0]), sets[i][1]-sets[i][0], Yminmax[i][1]-Yminmax[i][0], fill=None, alpha=1))
x = np.linspace(glb, gub, num=150)
y=np.array([model(e) for e in x])
mmax=np.array([-modelMax(t) for t in x])
mmin = np.array([modelMin(t) for t in x])
plt.plot(x, y[:,0])
plt.plot(x, mmax[:,0])
plt.plot(x, mmin[:,0])
plt.show(block=True)

# print(trainSetX)

