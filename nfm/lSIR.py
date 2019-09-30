import os
import matplotlib.pyplot as plt
import numpy as np
import stochpy
from scipy import signal

np.random.exponential(1)


def glitchesPoisson(l, T, m):
    t = T[0] + np.random.exponential(l)
    times = list()
    hight = list()
    while (t < T[1]):
        times.append(t)
        hight.append(np.sign(2 * np.random.rand() - 1) * m)
        # hight.append(- m)
        t = t + np.random.exponential(l)
    return times, hight


def gaussian(times, hight, delta, x):
    res = 0
    for i in range(len(times)):
        res = res + hight[i] * np.exp(-((x - times[i]) / delta) ** 2)
    return res

def medfilter(t,f,delta,index):
    t_inxed=t[index]
    iBefore=index-1
    while(iBefore>0 and t_inxed - t[iBefore]<delta):
        iBefore=iBefore-1
    iAfter = index + 1
    while (iAfter < len(t)) and t[iAfter] - t_inxed < delta :
        iAfter = iAfter + 1
    return np.median(f[iBefore+1:iAfter-1])

def initFilter(t,delta):
    t_inxed=t[0]
    iAfter = 1
    while (iAfter < len(t)) and t[iAfter] - t_inxed < delta :
        iAfter = iAfter + 1
    return iAfter

def endFilter(t,delta):
    t_inxed=t[:-1]
    iBefore=-2
    while(iBefore>0 and t_inxed - t[iBefore]<delta):
        iBefore=iBefore-1
    return iBefore


l = 10
T = [0, 200]
hight = 20
delta = 0.25
# a, b = glitchesPoisson(l, T, hight)
p = delta * 1 / l * (T[1] - T[0])/2
print('p: ' + str(p))
# print(p / (T[1] - T[0]))
# print(b)
smod = stochpy.SSA()
smod.Model('epidemic.psc', dir=os.path.dirname(__file__) + '/model')
# smod.DoStochSim(end=100,mode='time',trajectories=1)
smod.ChangeParameter('ks', 0.2)
smod.ChangeParameter('ki', 0.1)
smod.ChangeParameter('kn', 0.3)
NTOT = 200
smod.DoStochSim(end=T[1], mode='time', trajectories=NTOT)
meanyy1 = 0
meanyyf1 = 0
meanyyf2 = 0
meanyyfSort = 0

probyy1 = 0
probyyf1 = 0
probyyf2 = 0
probyyfSort = 0

# value = round(p) + (round(p) % 2 == 0)

N=NTOT
for i in range(1, NTOT):
    a, b = glitchesPoisson(1, T, 10)
    #normalFun = lambda x: gaussian(a, b, delta, x)

    # p = 3*delta * 1 / l * (T[1] - T[0])
    smod.GetTrajectoryData(i)
    time = smod.data_stochsim.time
    yy1 = smod.data_stochsim.species[:, 0]
    if(len(yy1)<100):
        N = N - 1
        continue
    # min_yy1 =min((y for (x,y) in zip(time,yy1) if x>30))
    min_yy1 = min(yy1)
    meanyy1 = meanyy1 + min_yy1
    probyy1 = probyy1 + (min_yy1 > 40)
    # print("minyy1: "+str(min_yy1))
    a, b = glitchesPoisson(l, T, hight)
    noiseFun = lambda x: gaussian(a, b, delta, x)
    yyf1 = yy1 + noiseFun(time)
    # min_yyf1 =min((y for (x,y) in zip(time,yyf1) if x>30))
    min_yyf1 = min(yyf1)
    meanyyf1 = meanyyf1 + min_yyf1
    probyyf1 = probyyf1 + (min_yyf1 > 40)
    # value = round(3*delta/100*len(time)) + (round(3*delta/100*len(time)) % 2 == 0)
    #  print("minyyf1: "+str(min_yyf1))
    #yyf2 = signal.medfilt(yyf1, 11)
    yyf2=[medfilter(time,yyf1,5*delta,i) for i in range(len(time))]
    # min_yyf2 =min((y for (x,y) in zip(time,yyf2) if x>30))
    min_yyf2 = min(yyf2[initFilter(time,3*delta):endFilter(time,3*delta)])
    meanyyf2 = meanyyf2 + min_yyf2
    probyyf2 = probyyf2 + (min_yyf2>40)
    #   print("minyyf2: "+str(min_yyf2))
    sorted = np.argsort(yyf1[:-1])
    yyf1Sort = yyf1[sorted]
    proc = time[1:] - time[:-1]
    timeSort = proc[sorted]
    v = 0
    i = 0
    while (v < p and i < len(timeSort)):
        v = v + timeSort[i]
        i = i + 1
    meanyyfSort = meanyyfSort + yyf1Sort[i - 1]
    probyyfSort = probyyfSort + (yyf1Sort[i - 1] > 40)

print('meanyy1: ' + str(meanyy1 / (N - 1)))
print('meanyyf1: ' + str(meanyyf1 / (N - 1)))
print('meanyyf2: ' + str(meanyyf2 / (N - 1)))
print('meanyyfSort: ' + str(meanyyfSort / (N - 1)))

print('probyy1: ' + str(probyy1 / (N - 1)))
print('probyyf1: ' + str(probyyf1 / (N - 1)))
print('probyyf2: ' + str(probyyf2 / (N - 1)))
print('probyyfSort: ' + str(probyyfSort / (N - 1)))

a, b = glitchesPoisson(1, T, 10)
#normalFun = lambda x: gaussian(a, b, delta, x)

time = smod.data_stochsim.time
yy1 = smod.data_stochsim.species[:, 0]
a, b = glitchesPoisson(l, T, hight)
noiseFun = lambda x: gaussian(a, b, delta, x)
yyf1 = yy1 + noiseFun(time)
# yyf2 = signal.medfilt(yyf1, round(p) + (round(p) % 2 == 0))
yyf2=[medfilter(time,yyf1,5*delta,i) for i in range(len(time))]
yyf3 = signal.medfilt(yyf1, round(2*p) + (round(2*p) % 2 == 0))
yyf4 = signal.medfilt(yyf1, round(3*p) + (round(3*p) % 2 == 0))

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.plot(time, yy1)
plt.plot(time, yyf1)
plt.plot(time, yyf2)
plt.plot(time, yyf3)
plt.plot(time, yyf4)
plt.show(block=True)

# plt.plot(time, yy1)
# plt.plot(time, yyf1)
# yyf2 = signal.medfilt(yyf1,15)
# min_yyf2 =min((y for (x,y) in zip(time,yyf2) if x>100))
# print("minyyf2: "+str(min_yyf2))
#
# #smod.data_stochsim.species_labels, smod.data_stochsim.time, smod.data_stochsim.species
# time = smod.data_stochsim.time
# yy1 = smod.data_stochsim.species[:, 0]
# min_yy1 =min((y for (x,y) in zip(time,yy1) if x>100))
# print("minyy1: "+str(min_yy1))
# fun = lambda x :  gaussian(a,b,delta,x)
# yyf1 = yy1+fun(time)
# min_yyf1 =min((y for (x,y) in zip(time,yyf1) if x>100))
# print("minyyf1: "+str(min_yyf1))
#
# plt.plot(time, yy1)
# plt.plot(time, yyf1)
# yyf2 = signal.medfilt(yyf1,15)
# min_yyf2 =min((y for (x,y) in zip(time,yyf2) if x>100))
# print("minyyf2: "+str(min_yyf2))
#
# # yy2 = smod.data_stochsim.species[:, 1]
# plt.plot(time, yyf2)
# plt.show(block=True)
# fun = lambda x :  gaussian(a,b,delta,x)+np.sin(10*x)
# xx=np.linspace(T[0],T[1],1000)
# yy=fun(xx)
# yf = signal.medfilt(yy,101)
# plt.plot(xx,yy)
# plt.plot(xx,yf)
# plt.show()
