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


def average(pid,N, T, T0, T1, kernelFunction, atomicFunction, p, K):
    #  print(".")
    result = []
    for j in range(N):
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        x0 = [97.77, 19.08024, 3.0525, 3.0525, 0.033551, 0.01899, 0.03128, 0.02681, 0.0, dg1, 0, 0];
        # time points
        t_meal1 = np.random.normal(300, 60)
        t_meal2 = np.random.normal(300, 60)
        # t_meal1=300
        # t_meal2=300
        t_meal3 = 1440 - t_meal1 - t_meal2

        t1 = np.arange(0, t_meal1, 0.2)
        t2 = np.arange(0, t_meal2, 0.2)
        t3 = np.arange(0, t_meal3, 0.2)
        y = odeint(pid, x0, t1)
        ytot = y
        ttot = t1
        ystart = y[-1, :]
        ystart[8] = 0
        ystart[9] = dg2
        y = odeint(pid, ystart, t2)
        ytot = np.vstack([ytot, y])
        ttot = np.hstack([ttot, t2 + ttot[-1]])
        ystart = y[-1, :]
        ystart[8] = 0
        ystart[9] = dg3

        y = odeint(pid, ystart, t3)
        ytot = np.vstack([ytot, y])
        ttot = np.hstack([ttot, t3 + ttot[-1]])
        result = np.hstack([result, kernel(T, ttot, atomicFunction(ytot[:, 0]), T0, T1, p, kernelFunction)])
    # r = np.percentile(result, 20)
    # print(str(K) + "__" + str(p) + "__" + str(r))
    return np.mean(result),np.sqrt(np.var(result))

def prob(pid,N, T, T0, T1, kernelFunction, atomicFunction, p, K):
    #  print(".")
    result = []
    for j in range(N):
        dg1 = np.random.normal(40, 10)
        dg2 = np.random.normal(90, 10)
        dg3 = np.random.normal(60, 10)
        x0 = [97.77, 19.08024, 3.0525, 3.0525, 0.033551, 0.01899, 0.03128, 0.02681, 0.0, dg1, 0, 0];
        # time points
        t_meal1 = np.random.normal(300, 60)
        t_meal2 = np.random.normal(300, 60)
        t_meal3 = 1440 - t_meal1 - t_meal2

        t1 = np.arange(0, t_meal1, 0.2)
        t2 = np.arange(0, t_meal2, 0.2)
        t3 = np.arange(0, t_meal3, 0.2)
        y = odeint(pid, x0, t1)
        ytot = y
        ttot = t1
        ystart = y[-1, :]
        ystart[8] = 0
        ystart[9] = dg2
        y = odeint(pid, ystart, t2)
        ytot = np.vstack([ytot, y])
        ttot = np.hstack([ttot, t2 + ttot[-1]])
        ystart = y[-1, :]
        ystart[8] = 0
        ystart[9] = dg3

        y = odeint(pid, ystart, t3)
        ytot = np.vstack([ytot, y])
        ttot = np.hstack([ttot, t3 + ttot[-1]])
        result = np.hstack([result, kernel(T, ttot, atomicFunction(ytot[:, 0]), T0, T1, p, kernelFunction)>0])
    # r = np.percentile(result, 20)
    # print(str(K) + "__" + str(p) + "__" + str(r))
    return np.mean(result)


# plot results
# f, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.fill_between([ttot[0], ttot[-1]], [4, 4], [16, 16], alpha=0.5)
# ax1.plot(ttot, ytot[:, 0] / VG, 'r-', label='Glucose')
#
# ax1.axhline(y=sp / VG, color='k', linestyle='-')
# # ax1.xlabel('time')
# # ax1.ylabel('y(t)')
# # ax1.legend()
# # ax1.xlabel('Time (min)')
# # ax1.ylabel('BG (mmol/L)')
# ax2.plot(ttot, ytot[:, -1], label='Insuline')
# ax1.legend()
# ax2.legend()
# plt.show()
#  bnds = ((interval[0], interval[1]),)
#     # points = interval[0] + lhs(1, samples=10, criterion='maximin') * (interval[1] - interval[0])
#     points = np.linspace(interval[0],interval[1], num=5)
#     best = float('Inf')
#     for i in range(0, len(points)):
#         res = minimize(modelMin, points[i], method='L-BFGS-B', bounds=bnds)

# print(str(i)+"------------")
#     print("SML mean : " + str(np.mean(resultSML)))
#     print("SML sigma : " + str(np.var(resultSML)))
#     print("STL mean : " + str(np.mean(resultSTL)))
#     print("STL sigma : " + str(np.var(resultSTL)))
#     print("-------------")
