import matplotlib.pyplot as plt
import numpy as np
from antlr4 import InputStream
from scipy.integrate import odeint

from pycheck.semantics.STL.BooleanSemantics import BooleanSemantics
from pycheck.semantics.STL.RobSemantics import RobSemantics
from pycheck.semantics.STL.STLLexer import STLLexer, CommonTokenStream
from pycheck.semantics.STL.STLParser import STLParser
from pycheck.semantics.STL.ZeroSemantics import ZeroSemantics
from pycheck.series.TimeSeries import TimeSeries


def dominates(a,b,i,j,constraint):
    if(a[constraint]>0 and b[constraint]>0):
        return (a[i]>b[i] and a[j]>=b[j]) or (a[i]>=b[i] and a[j]>b[j])
    elif (a[constraint]<0 and b[constraint]<0):
        return a[constraint]>=b[constraint]
    else:
        return (a[constraint]>0)

def isdominated(a,b,i,j,constraint):
    if (a[constraint] > 0 and b[constraint] > 0):
        return (a[i] < b[i] and a[j] <= b[j]) or (a[i] <= b[i] and a[j] < b[j])
    elif (a[constraint] < 0 and b[constraint] < 0):
        return (a[constraint] < b[constraint])
    else:
        return (a[constraint] > 0)





# def paretoDominateElement(pareto,element,i,j):
#     for e in pareto:
#         if (dominates(e,element,i,j)):
#             return True
#     return False

def adjustPareto(pareto,element,ii,jj,constraint):
    indexes = np.array([])
    if(element[constraint]<0):
        return pareto
    for i in range(0,len(pareto)):
        if(dominates(element,pareto[i,:],ii,jj,constraint)):
            indexes=np.append(indexes,i)
        elif isdominated(element,pareto[i,:],ii,jj,constraint):
            return pareto
    if (len(indexes)>0):
        pareto = np.delete(pareto, indexes, axis=0)

    pareto=np.vstack([pareto,element])
    return pareto

def findPareto(set,ii,jj,constraint):
    pareto = np.array([set[0,:]])
    set=np.delete(set, (0), axis=0)
    for i in range(0,len(set)):
        pareto = adjustPareto(pareto,set[i,:],ii,jj,constraint)
        # if (not paretoDominateElement(pareto,set[0,:],ii,jj)):
        #     pareto = np.vstack([pareto,set[0,:]])
        # set=np.delete(set, (0), axis=0)
    return pareto


def incFF(KBCp,dCp,y0,time,m):
    def fHp(n,a,K,X):
        return a*1/(1+pow(K/X,n))
    def fHm(n,a,K,X):
        return a*1/(1+pow(X/K,n))
        # return a*1/(pow(K,n)+pow(X,n))

    def incFFEquation(y, t,S,bA,dA,bB,KAB,dB,bC,KAC,KBC,dC,n):
        xA,xB,xC = y
        dydt = [bA*S-dA*xA,fHp(n,bB,KAB,xA)-dB*xB , fHp(n,bC,KAC,xA)*fHm(n,1,KBC,xB)-dC*xC]
        # dydt = [0,fHp(n,bB,KAB,xA)-dB*xB , fHp(n,bC,KAC,xA)*fHm(n,1,KBC,xB)-dC*xC]
        return dydt
    S=1
    bA=1
    dA=1
    bB=1
    KAB=1
    dB=1
    bC=1
    KAC=1
    KBC=KBCp
    dC=dCp
    n=2
    t = np.linspace(time[0], time[1], m)
    sol = odeint(incFFEquation, y0, t, args=(S,bA,dA,bB,KAB,dB,bC,KAC,KBC,dC,n))
    return t,sol


objInput = InputStream('(F_[0,3] (C > 0.12))\n')
lexer = STLLexer(input=objInput)
token_stream = CommonTokenStream(lexer)
parser = STLParser(token_stream)
objectiveTree = parser.prog()

constraintInput = InputStream('(G_[10,15](C < 0.03))\n')
lexer = STLLexer(input=constraintInput)
token_stream = CommonTokenStream(lexer)
parser = STLParser(token_stream)
constraintTree = parser.prog()

x1 = np.linspace(0.01, 1, 40)
x2 = np.linspace(0.01, 1, 40)
xx, yy = np.meshgrid(x1, x2)
objZeroSemantics=np.zeros(xx.shape)
objQuantiativeSemantics=np.zeros(xx.shape)
consQuantiativeSemantics=np.ones(xx.shape)
booleanSemantics=np.zeros(xx.shape)
booleanFromQuantiative=np.zeros(xx.shape)

matrix=np.zeros([xx.shape[0]*xx.shape[1],5])
c=0
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        print(xx.shape[0],xx.shape[1], i,j)
        t, sol = incFF(xx[i, j], yy[i, j], [1, 0, 0], [0, 20], 100)
        serie = TimeSeries(['A', 'B', 'C'], t, sol.T)
        visitor = ZeroSemantics(timeSeries=serie)
        objZeroSemantics[i, j] = visitor.visit(objectiveTree)
        visitor = RobSemantics(timeSeries=serie)
        objQuantiativeSemantics[i, j] = visitor.visit(objectiveTree)
        visitor = RobSemantics(timeSeries=serie)
        consQuantiativeSemantics[i, j] = visitor.visit(constraintTree)
        visitor = BooleanSemantics(timeSeries=serie)
        booleanSemantics[i, j] = visitor.visit(objectiveTree)
        matrix[c,:]=[xx[i,j], yy[i,j], objZeroSemantics[i, j], objQuantiativeSemantics[i, j],consQuantiativeSemantics[i, j]]
        c+=1
        # print( zQ[i, j])
        # visitor = FilterSemantics(series=serie)
        # zF[i, j] = visitor.visit(tree)

pareto = findPareto(matrix,2,3,4)
plt.scatter(matrix[:,2],matrix[:,3])
plt.show()
plt.scatter(pareto[:,2],pareto[:,3])
plt.show()
# plt.scatter(pareto[:,2],pareto[:,3])
# plt.show()
# plt.pcolor(xx,yy,zZ)

# for e in pareto:
#     t, sol = incFF(e[0], e[1], [1, 0, 0], [0, 20], 300)
#     plt.plot(t, sol[:, 2],label=str(e))
# plt.legend(loc='best')
# plt.show()


f, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, sharey=True)
ax1.pcolormesh(xx,yy,objQuantiativeSemantics)
# ax1.pcolor(xx,yy,zZ, vmin=np.min(-1), vmax=np.max(1))
ax1.set_title('')
ax2.pcolormesh(xx,yy,objZeroSemantics)
ax3.pcolormesh(xx,yy,booleanSemantics)
# ax3.pcolormesh(xx,yy,objZeroSemantics)
f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

plt.show()

# plt.subplot(3, 1, 1)
# plt.pcolormesh(xx,yy,zZ,vmin=0)
# plt.title('Zero semantics')
# # set the limits of the plot to the limits of the data
# plt.colorbar()
#
# plt.subplot(3, 1, 2)
# plt.pcolormesh(xx,yy,zQ,vmin=0)
# plt.title('Space semantics')
# # set the limits of the plot to the limits of the data
# plt.colorbar()
#
# plt.subplot(3, 1, 3)
# plt.pcolormesh(xx,yy,zB)
# plt.title('Boolean semantics')
# # set the limits of the plot to the limits of the data
# plt.colorbar()
#
#
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# # surf = ax.plot_surface(xx, yy, zZ, cmap=cm.coolwarm,
# #                        linewidth=0, antialiased=False)
# # fig.colorbar(surf, shrink=0.5, aspect=5)
# # plt.show()
#
# # X, Y, Z = axes3d.get_test_data(0.05)
#
# # Plot a basic wireframe.
# ax.plot_wireframe(xx, yy, zZ, rstride=1, cstride=1)
#
# plt.show()

# ax1.plot_surface(xx,yy,zZ)
# ax2.plot_surface(xx,yy,zQ)
# ax3.plot_surface(xx,yy,zF)

# Fine-tune figure; make subplots close to each other and hide x ticks for
# all but bottom plot.



# input_stream = InputStream('(G_[15,20]C < 0.1)\n')
# lexer = STLLexer(input=input_stream)
# token_stream = CommonTokenStream(lexer)
# parser = STLParser(token_stream)
# tree = parser.prog()
#
# t, sol = incFF(0.17, 0.38, [1,0,0], [0, 20], 100)
# # t, sol = incFF(0.4, 1, [0,0,0], [0, 20], 100)
# serie = StochSerie(['A', 'B', 'C'], t, sol)
# visitor = BooleanSemantics(series=serie)
# print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
# visitor = QuantitativeSemantics(series=serie)
# print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
# # visitor = FilterSemantics(series=serie)
# # print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
# visitor = ZeroSemantics(series=serie)
# print(str(input_stream)+ '--> ' + str(visitor.visit(tree)))
#
#
# import matplotlib.pyplot as plt
# plt.plot(t, sol[:, 0], 'r', label='A')
# plt.plot(t, sol[:, 1], 'b', label='B')
# plt.plot(t, sol[:, 2], 'g', label='C')
# plt.legend(loc='best')
# plt.xlabel('t')
# plt.grid()
# plt.show()


# def toggle(y, t, a1, b1,a2,b2):
#     x1,x2=y
#     N=x1+x2
#     dydt=[x1-x1*a1*pow(N,b1+1)/(pow(N,b1)+pow(x2,b1)),x2-x2*a2*pow(N,b2+1)/(pow(N,b2)+pow(x1,b2))]
#     return dydt
#
# def bistable(y, t, k1,k2,k3,k4):
#     x1,x2=y
#     dydt=[2*k1*x2-k2*x1*x1-k3*x1*x2-k4*x1,k2*x1*x1-k1*x2]
#     return dydt

# a1=0.069
# b1=0.25
# a2=0.166
# b2=0.25
#
# k1=8.46056267132
# k2=1
# k3=1
# k4=1.5

# sol = odeint(toggle, y0, t, args=(a1, b1,a2,b2))

# sol = odeint(bistable, y0, t, args=(k1, k2,k3,k4))

# sol = odeint(bistable, y0, t, args=(k1, k2,k3,k4))