import numpy as np
from antlr4 import InputStream

from pycheck.semantics.STL.BooleanSemantics import BooleanSemantics
from pycheck.semantics.STL.STLLexer import STLLexer, CommonTokenStream
from pycheck.semantics.STL.STLParser import STLParser
from pycheck.series.TimeSeries import TimeSeries

bufo = '(G_[0.953476412294208, 0.958673749884154] (flow <= 5199))'
# G[Tl_46, Tu_46] ((flowp <= Theta_2 | flowp >= Theta_3))  :: -0.15833333134651184 :: [0.2467857559801895, 1.2429362954319914, 423.082299469462, 647.387175283398]
# G[Tl_28, Tu_28] (flow <= Theta_0)  :: -0.19166666666666668 :: [0.39040647079280066, 1.1426717951942855, 7488.690705190697]
C0 = '(G_[0.20444603129532193, 1.1824754112132077]((FP <= 211)|(F<=7929)))';  # 218
C1 = '(G_[0.2156655513407244, 1.2854440796586268](FP<435.4669505464228))';  # 453
C2 = '(G_[0.44456879127815185, 1.1746676625606733] (F <=7879.376953095606))'  # 172  #(0.16319444444444445, 0.8611111111111112, 0.1875)
C3 = '(G_[0.7586673462600042, 1.2104521794463543](FP >=344.68)))'  # 68 sec (0.1909722222222222, 0.8055555555555556, 0.1875)
C4 = '(G_[0.31553893791535953, 1.3018450301946094]((FP <=453.14848474300516)&(F<=8682.20101044784)))'  # 115 sec (0.19791666666666666, 0.8611111111111112, 0.2569444444444444)
# C5 = '(G_[0.35397927825119385, 1.1434551526372856] ((FP >=1183.94) | (F <= 7754.50)))' #73 sec (0.1701388888888889, 0.8472222222222222, 0.1875)
C5 = '(G_[0.4300706986194084, 1.210676971306759] ((F >=3601) & (F <= 8161.675961119892)))'  # 210 sec (0.1701388888888889, 0.8472222222222222, 0.1875)
0.35397927825119


# np.mean([0.17708333333333334,0.16319444444444445,0.1909722222222222,0.19791666666666666,0.1701388888888889])
# np.std([0.17708333333333334,0.16319444444444445,0.1909722222222222,0.19791666666666666,0.1701388888888889])
# mcr mean = 0.1798
# mcr std = 0.012

# np.mean([0.8680555555555556,0.8611111111111112,0.8055555555555556,0.8611111111111112,0.8472222222222222])
# np.std([0.8680555555555556,0.8611111111111112,0.8055555555555556,0.8611111111111112,0.8472222222222222])

# good mean = 0.84861111111111109
# good std = 0.022566773346211006

# np.mean([0.2222222222222222,0.1875,0.25694444444444,0.1875,0.1875])
# np.std([0.2222222222222222,0.1875,0.25694444444444,0.1875,0.1875])

# bad mean =  0.20
# bad std = 0.02


def generateTreeFormula(formula):
    input_stream = InputStream(formula + '\n')
    lexer = STLLexer(input=input_stream)
    token_stream = CommonTokenStream(lexer)
    parser = STLParser(token_stream)
    return parser.prog()


def mcr(time, variables, goodTrajectories, badTrajectories, formula):
    good = 0
    bad = 0
    totLength = len(goodTrajectories) + len(badTrajectories)
    objectiveTree = generateTreeFormula(formula)
    for i in range(len(goodTrajectories)):
        serie = TimeSeries(variables, time, goodTrajectories[i])
        visitor = BooleanSemantics(timeSeries=serie)
        good = good + visitor.visit(objectiveTree)
    for i in range(len(badTrajectories)):
        serie = TimeSeries(variables, time, badTrajectories[i])
        visitor = BooleanSemantics(timeSeries=serie)
        bad = bad + visitor.visit(objectiveTree)
    mcr = ((len(goodTrajectories) - good) + bad) / (totLength)
    return mcr, good / len(goodTrajectories), bad / len(badTrajectories)


def mcr2(time, variables, goodTrajectories, badTrajectories, formula, nfold, fold):
    good = 0
    bad = 0
    length = len(goodTrajectories)
    goodTrajectories = goodTrajectories[int(nfold * length / fold):int((nfold + 1) * length / fold)]
    badTrajectories = badTrajectories[int(nfold * length / fold):int((nfold + 1) * length / fold)]
    totLength = len(goodTrajectories) + len(badTrajectories)
    objectiveTree = generateTreeFormula(formula)
    for i in range(len(goodTrajectories)):
        serie = TimeSeries(variables, time, goodTrajectories[i])
        visitor = BooleanSemantics(timeSeries=serie)
        good = good + visitor.visit(objectiveTree)
    for i in range(len(badTrajectories)):
        serie = TimeSeries(variables, time, badTrajectories[i])
        visitor = BooleanSemantics(timeSeries=serie)
        bad = bad + visitor.visit(objectiveTree)
    mcr = ((len(goodTrajectories) - good) + bad) / (totLength)
    return mcr, good / len(goodTrajectories), bad / len(badTrajectories)


time = np.genfromtxt('time.csv', delimiter=',')
flow = np.genfromtxt('flow.csv', delimiter=',')
flowP = np.genfromtxt('flowP.csv', delimiter=',')
pressure = np.genfromtxt('pressure.csv', delimiter=',')
pressureP = np.genfromtxt('pressureP.csv', delimiter=',')

goodTrajectories = [np.array([flow[i, :], flowP[i, :], pressure[i, :], pressureP[i, :]]) for i in
                    range(int(len(flow) / 2))]
badTrajectories = [np.array([flow[i, :], flowP[i, :], pressure[i, :], pressureP[i, :]]) for i in
                   range(int(len(flow) / 2), len(flow))]

# print(mcr(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, bufo))
# print(mcr(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, '(({})&({}))'.format(nostra1,nostra2)))
# print(mcr(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, bufo))
print(mcr2(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, C0, 0, 6))
print(mcr2(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, C1, 1, 6))
print(mcr2(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, C2, 2, 6))
print(mcr2(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, C3, 3, 6))
print(mcr2(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, C4, 4, 6))
print(mcr2(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, C5, 5, 6))

# print(mcr(time, ['F', 'FP', 'P', 'PP'], goodTrajectories, badTrajectories, nostra2))
