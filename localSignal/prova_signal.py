# Smooth a square pulse using a Hann window:

import numpy as np
from scipy import signal

sig = np.repeat([0., 1., 0.], 100)
win = signal.hann(50)
filtered = signal.convolve(sig, win, mode='same') / sum(win)

def windows(a,b):
    def fun(x):
        if x>a and x<b :
            return 1
        else:
            return 0
    return fun
p=windows(50,100)
newsignals=np.array([p(x) for x in  np.linspace(0.0, 300, num=300)])
filtered = signal.convolve(sig, newsignals, mode='same') / sum(newsignals)
import matplotlib.pyplot as plt
fig, (ax_orig, ax_win, ax_filt) = plt.subplots(3, 1, sharex=True)
ax_orig.plot(sig)
ax_orig.set_title('Original pulse')
ax_orig.margins(0, 0.1)
ax_win.plot(newsignals)
# ax_win.plot(win)
ax_win.set_title('Filter impulse response')
ax_win.margins(0, 0.1)
ax_filt.plot(filtered)
ax_filt.set_title('Filtered signal')
ax_filt.margins(0, 0.1)
fig.tight_layout()
fig.show()
plt.show(block=True)
p=windows(1,3)
print(p(4))