import stochpy
import numpy as np

def findFirst(x,pred):
    for i in range(len(x)):
        if(pred(x[i])):
            return i
    return None


smod = stochpy.SSA()
smod.Model('epidemic.psc')
# smod.DoStochSim(end=100,mode='time',trajectories=1)
smod.ChangeParameter('ks', 0.1)
smod.ChangeParameter('ki', 0.1)
smod.ChangeParameter('kn', 0.3)
NTOT = 500
T = [0, 200]
smod.DoStochSim(end=T[1], mode='time', trajectories=1)
time = smod.data_stochsim.time
y = smod.data_stochsim.species[:, 0]
W=[0,30]
kernel = lambda x : np.exp(x/W[1])/np.e
atomicPredicate = lambda x : 1 if x>40 else 0
atomicPredicateVect = np.vectorize( atomicPredicate)

actualTime=time[0]
actualIndex=0
finalIndex=findFirst(time, lambda t : t > W[1])
finalTime=actualTime+30

val = np.dot(atomicPredicateVect(y[1:finalIndex - 1]), kernel(time[2:finalIndex]) - kernel(time[1:finalIndex - 1]))
hmax = min(time[actualIndex+1]-actualTime,finalTime-time[finalIndex-1])
p=0.2
B=kernel(0)-1+p
C=-val+kernel(time[actualIndex+1]-actualTime)
A=  kernel( W[1] - time[finalIndex-1]+actualTime)

delta = B**2-4*A*C

sol1 = (- B + np.sqrt(delta))/(4*A*C)
sol2 = (- B - np.sqrt(delta))/(4*A*C)

print(sol1)
print(sol2)
print(hmax)
