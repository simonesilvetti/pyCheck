import numpy as np
from antlr4 import InputStream
import matplotlib.pyplot as plt
from pycheck.semantics.STL.BooleanSemantics import BooleanSemantics
from pycheck.semantics.STL.RobSemantics import RobSemantics
from pycheck.semantics.STL.STLLexer import STLLexer, CommonTokenStream
from pycheck.semantics.STL.STLParser import STLParser
from pycheck.semantics.STL.ZeroSemantics import ZeroSemantics
from pycheck.series.TimeSeries import TimeSeries
from scipy.integrate import simps


def percentile(N, P):
    """
    Find the percentile of a list of values

    @parameter N - A list of values.  N must be sorted.
    @parameter P - A float value from 0.0 to 1.0

    @return - The percentile of the values.
    """
    N.sort()
    n = int(round(P * len(N) + 0.5))
    return N[n-1]

def percentil(s,p):
    for i in range(0, s.shape[1] - 1):
        s[0,i]= s[0,i+1]-s[0,i]
    s=s[:,0:-1]
    index = s[1,:].argsort()
    e=s[:,index]
    area=0
    val =0
    c=0
    for i in range(0,e.shape[1]-1):
        area+=e[0,i]
    while ((val/area)<p and c<s.shape[1]):
        val += e[0, c]
        c+=1
    if (val/area)>p:
        return (e[1,c-1]+e[1,c])/2
    else:
        return e[1,c]

def percentilI(s,p):
    for i in range(0, s.shape[1] - 1):
        s[0,i]= s[0,i+1]-s[0,i]
    s=s[:,0:-1]
    index = s[1,:].argsort()
    e=s[:,index]
    area=0
    val =0
    c=0
    for i in range(0,e.shape[1]-1):
        if e[1,i]>0:
            area += e[1,i]*e[0, i]
        else:
            area += -e[1,i]*e[0, i]
    while ((val/area)<p and c<s.shape[1]):
        if e[1,c]>0:
            val += e[1,c]*e[0, c]
        else:
            val += -e[1,c]*e[0, c]
        c+=1
    if (val/area)>p:
        return (e[1,c-1]+e[1,c])/2
    else:
        return e[1,c]


t = np.linspace(1,4,101)
s=2*np.sin(t)
g=-2*np.sin(t)
print(percentil(np.array([t,s]),0.3))
print(percentil(np.array([t,s]),0.2))
print(-percentil(np.array([t,g]),0.8))
# print(np.percentile(s[0:-1],0,interpolation='midpoint'))
# print(-np.percentile(g[0:-1],100,interpolation='midpoint'))

# print(min(s[1:-1]))
# print(min(s[1:-1]))
# print(-percentil(np.array([t,g]),0.7))
# plt.plot(t,s)
# plt.plot(t,g)
# plt.show()
#
# print(np.percentile(s,100))
# g=np.cos(t)
# f = lambda x: np.exp(-x*x/0.3)
# h = lambda x: 2*np.exp(-x*x/0.02)
# s=f(t)
# g=h(t)
# print(-np.percentile(-s,39))
# print(np.percentile(s,100-39))
# print(np.percentile(s,100-39))
# s2=s-np.percentile(s,100-39)
# print(np.percentile(s2,100-39))
# # print(np.percentile(g,70))
# from scipy.integrate import simps
# I1 = simps(s, t)
#
# input_stream = InputStream('G_[0,1] (S > G) \n')
# lexer = STLLexer(input=input_stream)
# token_stream = CommonTokenStream(lexer)
# parser = STLParser(token_stream)
# objectiveTree = parser.prog()
#
# serie = TimeSeries(['S','G'], t, np.vstack([s,g]))
# zero = np.zeros(len(t))
# rob = np.zeros(len(t))
# bol = np.zeros(len(t))
# c=0
# for time in t[:-10]:
#     visitor = ZeroSemantics(timeSeries=serie, currentState=time)
#     zero[c] = visitor.visit(objectiveTree)
#     visitor = RobSemantics(timeSeries=serie, currentState=time)
#     rob[c] = visitor.visit(objectiveTree)
#     visitor = BooleanSemantics(timeSeries=serie, currentState=time)
#     bol[c] = visitor.visit(objectiveTree)
#     c+=1
#
# f, (ax0,ax1, ax2, ax3) = plt.subplots(4, sharex=True)
# ax0.plot(t,s)
# ax0.plot(t,g)
# ax1.plot(t,zero)
# # ax1.pcolor(xx,yy,zZ, vmin=np.min(-1), vmax=np.max(1))
# # ax1.set_title('zero')
# ax2.plot(t,rob)
# ax3.plot(t, bol)# ax3.pcolormesh(xx,yy,objZeroSemantics)
# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
#
# plt.show()

