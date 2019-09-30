# matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib import rc
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from nfm.quantitativeSML import *

rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
## for Palatino and other serif fonts use:
rc('font',**{'family':'serif','serif':['Palatino']})
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

#matplotlib.rcParams.update({'font.size': 22})
plt.rc('text', usetex=True)
plt.rc('font', weight='semibold')
plt.rc('xtick.major', size=7, pad=7)
# plt.rc('xtick', labelsize=15)
fig, ax = plt.subplots(figsize=(6, 5))
#plt.plot(x, y, label='$s(t)$')
#y =np.sin(2*x)
y =np.cos(x)
r=0.535
yt= y-r
linef = plt.plot(x, y)
plt.setp(linef, color='m', linewidth = 4.0,label='$s(t)$')
lineft = plt.plot(x, yt)
plt.setp(lineft, color='b', linewidth = 4.0,label='$s(t)-r$')
#plt.plot(xx, yy0, label=r'$\langle \exp(10x)_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy1, label=r'$\langle \exp(-10x)_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy2, label=r'$\langle \texttt{flat}_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy3, label=r'$\langle \texttt{gauss}_{[0.1,0.2]},0.5\rangle (s(t)>0)$')
# plt.plot(xx, yy4, label=r'$\diamond_{[0.1,0.2]}(s(t)>0)$')
# plt.plot(xx, yy5, label=r'${[0.1,0.2]}(s(t)>0)$')
plt.axhline(linewidth=2, color='k')
plt.axhspan(-r, 0, facecolor='#2ca02c', alpha=0.8,label='$r$')
#plt.axvspan(1.25, 1.55, facecolor='0.7', alpha=0.5)
plt.axis([0, 3, -1.5, 1.5],linewidth=4,fontsize=22)
plt.plot(x, np.choose(yt<=0,[0,-10]), '--', color='0.2', linewidth = 2.0)
plt.plot(x, np.choose(yt<=0,[-1.45,-10]), 'r',linewidth = 6.0)
plt.plot(x, np.choose(y<=0,[0,-10]), '--', color='0.2', linewidth = 2.0)
plt.plot(x, np.choose(y<=0,[-1.5,-10]), 'y',linewidth = 6.0)
# plt.xlabel('time t', fontsize=18, color='black')
# plt.ylabel('signal x', fontsize=18, fontweight='bold', color='black')
plt.tight_layout()

#plt.annotate('local max', xy=(0.3, 1), xytext=(0.7, 0.5),
#           arrowprops=dict(facecolor='black', shrink=0.05),
#            )
plt.legend()
ax.legend(loc='upper right', labelspacing=0, borderpad=0,fontsize=20 )
#plt.title('Easy as 1, 2, 3')
plt.savefig('percquant')
#plt.show(block=True)
plt.get_current_fig_manager().window.wm_geometry("+100+100")


