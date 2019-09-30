from scipy.fftpack import fft,ifft
from scipy.signal import medfilt
import numpy as np
from scipy.interpolate import interp1d
# Number of sample points
N = 600
# sample spacing
T = 1.0 / 800.0
x = np.linspace(0.0, N*T, N)
y1 = np.repeat([0., 1., 0.], 100)
y2 = np.repeat([0., -1., 0.], 100)
# y=np.concatenate((y1,y2), axis=0)
import matplotlib.pyplot as plt
f=lambda x: np.exp(-(x-0.2)*(x-0.2)/0.0002)+ np.exp(-(x-0.25)*(x-0.25)/0.0002)+np.exp(-(x-0.3)*(x-0.3)/0.0002)
fpicco1=lambda x: np.exp(-(x-0.2)*(x-0.2)/0.00002)
fpicco2=lambda x: np.exp(-(x-0.5)*(x-0.5)/0.000002)
# a=[0,19,20,25,26,28,30,35,37,89,91,100]
# b=[0,0,1,1,0,0,1,1,0,0,1,1]
a=[0,19,20,60,61,100]
b=[0,0,1,1,0,0]
# a=[0,9,10,19,20,39,40,49,50,69,70,99,100]
# b=[0,0,1,1,0,0,1,1,0,0,1,1,0]
x = np.linspace(0.0, 100, 100)
y = interp1d(a,b)(x)
# plt.plot(x,y)
sum = 0;
for i in range(0,70):
    ynex = medfilt(y, [2*i+1])
    sum += np.dot(x[1:] - x[0:-1], ynex[:-1])
val=sum/50
print(val)

from scipy.integrate import simps
import scipy.integrate as integrate





# ypicco1 = fpicco1(x)
# ypicco2 = fpicco2(x)
# ftpicco1 = fft(ypicco1)
# ftpicco2 = fft(ypicco2)
# ft=fft(y)
# # plt.plot(x,ft)
# ft[ft<15]=0
# ift=ifft(ft)
# plt.plot(x,y)
# plt.plot(x,ynex)
# plt.show()
# y=np.concatenate((y,-y), axis=0)
# N=2*N
# x = np.linspace(0.0, N*T, N)
# plt.plot(x,y)
# plt.show()
# y = np.sin(50.0 * 2.0*np.pi*x) + 0.5*np.sin(80.0 * 2.0*np.pi*x)
# ft = fft(y)-2*fft(ypicco)
# # ft[0:200]=0
# # ft[400:600]=0
# # ft[1000:1200]=0
# # ft[1190:1200]=0
# xf = np.linspace(0.0, 1.0/(2.0*T), N//2)
# # plt.plot(xf, 2.0/N * np.abs(ft[0:N//2]))
# plt.plot(x, np.abs(ft))
# # plt.plot(x, ft)
# plt.grid()
# plt.show()
#
# yy = ifft(ft)
# plt.plot(x,yy)
# plt.show()
