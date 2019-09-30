import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from nfm.quantitativeSML import *
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
# rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

expP = lambda x: np.exp(x)
expN = lambda x: np.exp(- x)
flat = lambda x: 1
gauss = lambda x: np.exp(-((x - 0.2) ** 2) / 0.1)

fun = np.vectorize(lambda x: np.exp(-((x - 0.4) ** 2) / 0.02))

# times = np.array([1.,2.,3.,4.,5.,6.])
# values = np.array([1.,2.,3.,4.,5.,6.])

x = np.linspace(0, 10, 1000)
y = fun(x)




# finexpP = lambda t: kernel(t, x, y, 0.1, 0.3, 0.5, expP)
# finexpN = lambda t: kernel(t, x, y, 0.1, 0.3, 0.5, expN)
# finflat = lambda t: kernel(t, x, y, 0.1, 0.3, 0.5, flat)
# fingauss = lambda t: kernel(t, x, y, 0.01, 0.3, 0.5, gauss)
# finstl = lambda t: kernel(t, x, y, 0.1, 0.3, 0.0001, flat)
# glob = lambda t: kernel(t, x, y, 0.1, 0.3, 1, flat)

# xx = np.linspace(0, 0.6, 600)
# yy = fun(xx)
yy=x+np.sin(15*x)*2-0.5

# yy0 = [finexpP(a) for a in xx]
# yy1 = [finexpN(a) for a in xx]
# yy2 = [finflat(a) for a in xx]
# yy3 = [fingauss(a) for a in xx]
# yy4 = [finstl(a) for a in xx]
# yy5 = [glob(a) for a in xx]

plt.rc('text', usetex=True)
plt.rc('font', weight='semibold')
plt.rc('xtick.major', size=5, pad=7)
# plt.rc('xtick', labelsize=15)
plt.figure()
#plt.plot(x, y, label='$s(t)$')
y = np.sin(2*x)-0.4
linef = plt.plot(x, y)
plt.setp(linef, color='b', linewidth = 6.0,label='$s(t)$')
#plt.plot(xx, yy0, label=r'$\langle \exp(10x)_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy1, label=r'$\langle \exp(-10x)_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy2, label=r'$\langle \texttt{flat}_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy3, label=r'$\langle \texttt{gauss}_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy4, label=r'$\diamond_{[0.1,0.2]}(s(t)>0)$')
# plt.plot(xx, yy5, label=r'${[0.1,0.2]}(s(t)>0)$')
plt.axhline(linewidth=4, color='k')
plt.axhspan(-0.4, 0, facecolor='#2ca02c', alpha=0.9)
# plt.axvspan(1.25, 1.55, facecolor='0.7', alpha=0.5)
plt.axis([0, 10, -1.5, 1.5],linewidth=4,fontsize=18)
plt.plot(x, np.choose(y<=0,[0,-10]), 'ro',linewidth=1.0)
plt.xlabel('time t', fontsize=18, color='black')
plt.ylabel('signal x', fontsize=18, fontweight='bold', color='black')
plt.tight_layout()
#plt.annotate('local max', xy=(0.3, 1), xytext=(0.7, 0.5),
#           arrowprops=dict(facecolor='black', shrink=0.05),
#            )
#plt.legend()
#plt.title('Easy as 1, 2, 3')
plt.savefig('percquant')
plt.show(block=False)
plt.get_current_fig_manager().window.wm_geometry("+100+100")


