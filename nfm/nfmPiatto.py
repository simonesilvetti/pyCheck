import stochpy
import numpy as np
import matplotlib.pyplot as plt

def findFirst(x,pred):
    for i in range(len(x)):
        if(pred(x[i])):
            return i-1
    return None


def monitor(tInit, tFinal, time, y, W, atomicPredicate, p):
    atomicPredicateVect = np.vectorize(atomicPredicate)
    ti = tInit + W[0]
    tf = tInit + W[1]
    indexTi = findFirst(time, lambda t: t > ti)
    indexTf = findFirst(time, lambda t: t > tf)
    val = np.dot(atomicPredicateVect(y[indexTi:indexTf]),
                 np.hstack([time[indexTi + 1:indexTf],tf]) - np.hstack(
                     [ti, time[indexTi + 1:indexTf]]))
    val = val/( W[1]- W[0]) - p
    solt = [tInit]
    solx = [val > 0]
    hcount=0
    while (True):
        vi = atomicPredicate(y[indexTi])
        vf = atomicPredicate(y[indexTf])
        if (tf == time[indexTf+1]):
            vf = atomicPredicate(y[indexTf+1])
        if (vi!=vf):
            h = -val*((W[1]-W[0])) / (-vi + vf)
        else:
            h = min(time[indexTi + 1] - ti, time[indexTf + 1] - tf) + 1
        if (solt[-1] + hcount > tFinal):
            break
        if (time[indexTi + 1] - ti < time[indexTf+1] - tf):
            hmin = time[indexTi + 1] - ti
            if (h > 0 and h < hmin):
                ti = ti + h
                tf = tf + h
                solt.append(solt[-1] + hcount)
                hcount=0
                solx.append(not solx[-1])
                hmin = h
            else:
                ti = time[indexTi + 1]
                tf = tf + hmin
                indexTi = indexTi + 1
        else:
            hmin = time[indexTf+1] - tf
            if (h > 0 and h < hmin):
                ti = ti + h
                tf = tf + h
                solt.append(solt[-1] + hcount)
                solx.append(not solx[-1])
                hmin = h
            else:
                tf = time[indexTf+1]
                indexTf = indexTf + 1
                ti = ti + hmin
        hcount = hcount+hmin
        val = val + ((-vi + vf) * hmin)/(W[1]-W[0])


    return np.hstack([solt,solt[-1] + hcount]), solx



smod = stochpy.SSA()
smod.Model('epidemic.psc')
# smod.DoStochSim(end=100,mode='time',trajectories=1)
smod.ChangeParameter('ks', 0.1)
smod.ChangeParameter('ki', 0.1)
smod.ChangeParameter('kn', 0.3)
NTOT = 500
T = [0, 300]
smod.DoStochSim(end=T[1], mode='time', trajectories=1)
time = smod.data_stochsim.time
y = smod.data_stochsim.species[:, 0]
W=[0,50]

atomicPredicate = lambda x : 1 if x>50 else 0

time1=[0,1,2,3,4,5,6,7,8,9,10]
y1 = [1,1,1,0,0,1,0,1,1,0,1]
plt.plot(time, np.vectorize(atomicPredicate)(y))
#plt.show(block=True)
plt.show()
a,b, = monitor(0,200,time,y,W,atomicPredicate,1)

print(a)
print(b)














