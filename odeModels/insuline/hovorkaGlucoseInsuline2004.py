import matplotlib.pyplot as plt
from scipy.integrate import odeint

from nfm.quantitativeSML import *


def modelPID(x, t, Kp, Ki, Kd):
    w = 100
    ka1 = 0.006  #
    ka2 = 0.06  #
    ka3 = 0.03  #
    kb1 = 0.0034  #
    kb2 = 0.056  #
    kb3 = 0.024  #
    u_b = 0.0555
    tmaxI = 55  #
    VI = 0.12 * w  #
    ke = 0.138  #
    k12 = 0.066  #
    VG = 0.16 * w  #
    # G = Q1 / VG
    F01 = 0.0097 * w  #
    FR = 0
    EGP0 = 0.0161 * w  #
    AG = 0.8  #
    Gmolar = 180.1559
    tmaxG = 40  #
    sp = 110 * VG / 18

    Q1, Q2, S1, S2, I, x1, x2, x3, tau, Dg, Ie, u = x
    dQ1 = - F01 - x1 * Q1 + k12 * Q2 - FR + EGP0 * (1 - x3) + (x[9] * AG * 1000 / Gmolar) * tau * np.exp(
        -tau / tmaxG) / (tmaxG ** 2)
    dQ2 = x1 * Q1 - (k12 + x2) * Q2
    dIe = sp - Q1
    de = - dQ1
    uNew = max(Ki * Ie + Kp * dIe + Kd * de, 0)
    dS1 = uNew + u_b - S1 / tmaxI
    dS2 = (S1 - S2) / tmaxI
    dI = S2 / (tmaxI * VI) - ke * I
    dx1 = - ka1 * x1 + kb1 * I
    dx2 = - ka2 * x2 + kb2 * I
    dx3 = - ka3 * x3 + kb3 * I
    dtau = 1
    dDg = 0
    dxdt = [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dtau, dDg, dIe, uNew - u]

    return dxdt


def modelPIDwNoise(x, t, Kp, Ki, Kd):
    global mem, uNew, noise, rumore
    w = 100
    ka1 = 0.006  #
    ka2 = 0.06  #
    ka3 = 0.03  #
    kb1 = 0.0034  #
    kb2 = 0.056  #
    kb3 = 0.024  #
    u_b = 0.0555
    tmaxI = 55  #
    VI = 0.12 * w  #
    ke = 0.138  #
    k12 = 0.066  #
    VG = 0.16 * w  #
    # G = Q1 / VG
    F01 = 0.0097 * w  #
    FR = 0
    EGP0 = 0.0161 * w  #
    AG = 0.8  #
    Gmolar = 180.1559
    tmaxG = 40  #
    sp = 110 * VG / 18

    Q1, Q2, S1, S2, I, x1, x2, x3, tau, Dg, Ie, time, u = x
    dQ1 = - F01 - x1 * Q1 + k12 * Q2 - FR + EGP0 * (1 - x3) + (x[9] * AG * 1000 / Gmolar) * tau * np.exp(
        -tau / tmaxG) / (tmaxG ** 2)
    dQ2 = x1 * Q1 - (k12 + x2) * Q2
    dIe = sp - Q1 + noise * np.cos(time / 10)
    de = - dQ1 - noise * np.sin(time / 10)
    rumore.append([time, Q1 - noise * np.cos(time / 10)])
    if (tau > mem):
        mem = tau + 1 + max(0, np.random.normal(10, 5))
        noise = max(min(20, np.random.normal(0, 10)), -20)

    uNew = max(Ki * Ie + Kp * dIe + Kd * de, 0)
    dS1 = uNew + u_b - S1 / tmaxI
    dS2 = (S1 - S2) / tmaxI
    dI = S2 / (tmaxI * VI) - ke * I
    dx1 = - ka1 * x1 + kb1 * I
    dx2 = - ka2 * x2 + kb2 * I
    dx3 = - ka3 * x3 + kb3 * I
    dtau = 1
    dDg = 0
    dTime = 1
    dxdt = [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dtau, dDg, dIe, dTime, uNew - u]

    return dxdt


def modelPIDwNoiseFunction(x, t, Kp, Ki, Kd, fun):
    global mem, uNew, noise, rumore
    w = 100
    ka1 = 0.006  #
    ka2 = 0.06  #
    ka3 = 0.03  #
    kb1 = 0.0034  #
    kb2 = 0.056  #
    kb3 = 0.024  #
    u_b = 0.0555
    tmaxI = 55  #
    VI = 0.12 * w  #
    ke = 0.138  #
    k12 = 0.066  #
    VG = 0.16 * w  #
    # G = Q1 / VG
    F01 = 0.0097 * w  #
    FR = 0
    EGP0 = 0.0161 * w  #
    AG = 0.8  #
    Gmolar = 180.1559
    tmaxG = 40  #
    sp = 110 * VG / 18

    Q1, Q2, S1, S2, I, x1, x2, x3, tau, Dg, Ie, time, u = x
    dQ1 = - F01 - x1 * Q1 + k12 * Q2 - FR + EGP0 * (1 - x3) + (x[9] * AG * 1000 / Gmolar) * tau * np.exp(
        -tau / tmaxG) / (tmaxG ** 2)
    dQ2 = x1 * Q1 - (k12 + x2) * Q2
    dIe = sp - (Q1 + noise * fun(time))
    de = - (dQ1 + noise * (fun(time + 1E-10) - fun(time)) / 1E-10)
    rumore.append([time, Q1 + noise * fun(time)])
    if (tau > mem):
        mem = tau + 1 + max(0, np.random.normal(10, 2))
        noise = max(min(20, np.random.normal(0, 10)), -20)

    uNew = max(Ki * Ie + Kp * dIe + Kd * de, 0)
    dS1 = uNew + u_b - S1 / tmaxI
    dS2 = (S1 - S2) / tmaxI
    dI = S2 / (tmaxI * VI) - ke * I
    dx1 = - ka1 * x1 + kb1 * I
    dx2 = - ka2 * x2 + kb2 * I
    dx3 = - ka3 * x3 + kb3 * I
    dtau = 1
    dDg = 0
    dTime = 1
    dxdt = [dQ1, dQ2, dS1, dS2, dI, dx1, dx2, dx3, dtau, dDg, dIe, dTime, uNew - u]

    return dxdt


rumore = list()
noise = 0
mem = 0
uNew = 0
w = 100
VG = 0.16 * w
sp = 110 * VG / 18
# initial condition

Kd = [0, -0.0602, -0.0573, -0.06002, -0.0624, -3.86119113e-07]
Ki = [0, -3.53e-07, -3e-07, -1.17e-07, -7.55e-07, -1.32620078e-07]
Kp = [0, -6.17e-04, -6.39e-04, -6.76e-04, -5.42e-04, -3.30338580e-07]

i = 3
dg1 = np.random.normal(40, 10)
dg2 = np.random.normal(90, 10)
dg3 = np.random.normal(60, 10)

# dg1 = 40
# dg2 = 90
# dg3 = 60

# x0 = [97.77, 19.08024, 3.0525, 3.0525, 0.033551, 0.01899, 0.03128, 0.02681, 0.0, dg1, 0,0.0555,0];
x0 = [97.77, 19.08024, 3.0525, 3.0525, 0.033551, 0.01899, 0.03128, 0.02681, 0.0, dg1, 0, 0];

# time points
t_offset = 0
t_sleep = 540
t_meal1 = np.random.normal(300, 60)
t_meal2 = np.random.normal(300, 60)
t_meal3 = 1440 - t_meal1 - t_meal2

t1 = np.arange(0, t_meal1, 0.2)
t2 = np.arange(0, t_meal2, 0.2)
t3 = np.arange(0, t_meal3, 0.2)

noiseFun = lambda x: np.sin(x / 9)
y = odeint(modelPID, x0, t1, args=(Kp[i], Ki[i], Kd[i]))
mem = 0
ytot = y
ttot = t1
ystart = y[-1, :]
ystart[8] = 0
ystart[9] = dg2
y = odeint(modelPID, ystart, t2, args=(Kp[i], Ki[i], Kd[i]))
mem = 0
ytot = np.vstack([ytot, y])
ttot = np.hstack([ttot, t2 + ttot[-1]])
ystart = y[-1, :]
ystart[8] = 0
ystart[9] = dg3

y = odeint(modelPID, ystart, t3, args=(Kp[i], Ki[i], Kd[i]))
ytot = np.vstack([ytot, y])
ttot = np.hstack([ttot, t3 + ttot[-1]])

# plot results
f, (ax1, ax2) = plt.subplots(2, sharex=True)
ax1.fill_between([ttot[0], ttot[-1]], [4, 4], [16, 16], alpha=0.5)
# ax1.plot(ttot, ytot[:, 0] / VG, 'r-', label='Glucose')

s = int(len(ytot) * 5 / 1440)
rum = np.random.normal(ytot[::s, 0], 10)

ax1.step(ttot, ytot[:, 0] / VG, 'r-', label='Glucose')
ax1.step(ttot[::s], rum / VG, label='Noise Glucose ')

ax1.axhline(y=sp / VG, color='k', linestyle='-')
# ax1.xlabel('time')
# ax1.ylabel('y(t)')
# ax1.legend()
# ax1.xlabel('Time (min)')
# ax1.ylabel('BG (mmol/L)')
ax2.plot(ttot, ytot[:, -1], label='Insuline')
ax1.legend()
ax2.legend()
plt.show()
print(kernel(0, ttot, ytot[:, 0] / VG - 4, 0, 1300, 0.001, lambda x: 1))
