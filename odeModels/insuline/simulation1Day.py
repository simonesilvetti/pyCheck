from scipy.integrate import odeint

from odeModels.insuline.hovorka import modelPID, np


def simulation(timeOfMeals, dGs, model):
    ttot = [0]
    ytot = [97.77, 19.08024, 3.0525, 3.0525, 0.033551, 0.01899, 0.03128, 0.02681, 0.0, 0, 0, 0]
    x0 = ytot
    for i in range(len(timeOfMeals)):
        x0[8] = 0
        x0[9] = dGs[i]
        time = np.arange(0, timeOfMeals[i], 0.2)
        #t = timeOfMeals[i]
        y = odeint(model, x0, time)
        x0 = y[-1, :]
        ytot = np.vstack([ytot, y])
        ttot = np.hstack([ttot, time + ttot[-1]])
    return ttot, ytot


Kd = [0, -0.0602, -0.0573, -0.06002, -0.0624]
Ki = [0, -3.53e-07, -3e-07, -1.17e-07, -7.55e-07]
Kp = [0, -6.17e-04, -6.39e-04, -6.76e-04, -5.42e-04]

pidC1 = lambda x, t: modelPID(x, t, Kp[1], Ki[1], Kd[1])
pidC2 = lambda x, t: modelPID(x, t, Kp[2], Ki[2], Kd[2])
pidC3 = lambda x, t: modelPID(x, t, Kp[3], Ki[3], Kd[3])
pidC4 = lambda x, t: modelPID(x, t, Kp[4], Ki[4], Kd[4])

pidC1Noise = lambda x, t: modelPID(x, t, Kp[1], Ki[1], Kd[1])
pidC2Noise = lambda x, t: modelPID(x, t, Kp[2], Ki[2], Kd[2])
pidC3Noise= lambda x, t: modelPID(x, t, Kp[3], Ki[3], Kd[3])
pidC4Noise= lambda x, t: modelPID(x, t, Kp[4], Ki[4], Kd[4])


# pid5 = lambda x, t: modelPID(x, t, -0.1, 0, -0.1)
# pid6 = lambda x, t: modelPID(x, t, -5.12720526e-04, -1.16330231e-06, -6.54656447e-02)
# pid7 = lambda x, t: modelPID(x, t, -6.16146031e-04, -3.90083101e-07, -6.75289141e-02)
def hyperGlicemia(th):
    return lambda x : th-x
def hypoGlicemia(th):
    return lambda x : x-th
