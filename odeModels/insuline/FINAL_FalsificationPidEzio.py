
from odeModels.insuline.hovorkaFalsification import *
from odeModels.insuline.simulation1Day import pidC4, hyperGlicemia, hypoGlicemia, pidC4, pidC1, pidC2

# hypo60_p95 = lambda x: fit([300,300, 1440-600],[ x[0], x[1], x[2]], pidC4, 0, lambda x: 1, 0, 1400, 0.90,
#                            hypoGlicemia(60))
#
# hypo60_p999 = lambda x: fit([300,300,1440-600],[ x[0], x[1], x[2]], pidC4, 0, lambda x: 1, 0, 1400, 0.999,
#                            hypoGlicemia(60))
#
hypo60_p95 = lambda x: fit([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC2, 0, lambda x: 1, 0, 1400, 0.90,
                           hypoGlicemia(60))

hypo60_p999 = lambda x: fit([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC2, 0, lambda x: 1, 0, 1400, 0.999,
                           hypoGlicemia(60))

bnds = ((200, 400), (200, 400), (20, 60), (70, 120), (40, 80),)
# hypo60_p95 = lambda x: fitNorm([x[0], x[1]], [x[2], x[3], x[4]],np.array(bnds1), pidC4, 0, lambda x: 1, 0, 1400, 0.90,
#                            hypoGlicemia(60))
#
# hypo60_p999 = lambda x: fitNorm([x[0], x[1]],[x[2], x[3], x[4]],np.array(bnds1), pidC4, 0, lambda x: 1, 0, 1400, 0.999,
#                            hypoGlicemia(60))



# kernel99 = lambda x: fit([x[0], x[1], 1440 - (x[0] + x[1])], [x[2], x[3], x[4]], pidC1, 0, lambda x: 1, 900, 1400, 0.99,
#                          lambda x: x - 64)
# print(a([300,300,300,90,60,80]))




# bndsa = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1),)
# bnds = ((300, 500), (300, 500), (50, 150), (50, 150), (50, 150),)
#bndsFixed = ((20, 60), (40, 80), (50, 120),)
# res = minimize(kernel80, [300, 300, 90, 60, 80], method='L-BFGS-B', bounds=bnds, options={'eps':1})
# res_hypo60_p95 = exploreMinimum(hypo60_p95, bnds, 100)
# res_hypo60_p999 = exploreMinimum(hypo60_p999, bnds, 100)
# bnds = ((0, 1), (0, 1), (0, 1), (0, 1), (0, 1),)

np.random.seed(2)
res_hypo60_p95 = exploreMinimum(hypo60_p95, bnds, 300)
np.random.seed(2)
res_hypo60_p999 = exploreMinimum(hypo60_p999, bnds, 300)


# res99 = findMinimum(kernel99, bnds, 10)
# res80 = exploreMinimum(kernel80, bnds, 100)
# res99 = exploreMinimum(kernel99, bnds, 100)

print("SML-------------")
print(res_hypo60_p95)
print(violationTimeS(res_hypo60_p95[:-1], lambda x: hypoGlicemia(60)(x) < 0))
print(violationSpaceS(res_hypo60_p95[:-1], hypoGlicemia(60)))
print("STL-------------")
print(res_hypo60_p999)
print(violationTimeS(res_hypo60_p999[:-1], lambda x: hypoGlicemia(60)(x) < 0))
print(violationSpaceS(res_hypo60_p999[:-1], hypoGlicemia(60)))

# drow2(res_hypo60_p95, res_hypo60_p999)
# drow(res)

# t,y = simulation([180,180,300,1440-180-180-300],[60, 10,150,0],pid3)
#
# w=100
# VG = 0.16 * w
# sp = 110 * VG / 18
# f, (ax1, ax2) = plt.subplots(2, sharex=True)
# ax1.fill_between([t[0], t[-1]], [4*VG, 4*VG], [16*VG, 16*VG], alpha=0.5)
# s = int(len(t)*5/1440)
# rum=  np.random.normal(y[::s,0],15)
#
# ax1.step(t, y[:, 0], 'r-', label='Glucose')
# ax1.step(t[::s], rum , label='Noise Glucose ')
#
# ax1.axhline(y=sp, color='k', linestyle='-')
#
# ax2.plot(t, y[:, -1], label='Insuline')
# ax1.legend()
# ax2.legend()
# plt.show()
