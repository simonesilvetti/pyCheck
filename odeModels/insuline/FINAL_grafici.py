import matplotlib.pyplot as plt

from odeModels.insuline.FINAL import violationTimeS, violationSpaceS
from odeModels.insuline.simulation1Day import pidC1, simulation, hypoGlicemia, hyperGlicemia


def printPidC1FlasificationHypo70FIXED():
    t, y = simulation([300, 300, 1400 - 300 - 300], [36.29, 86.88, 94.18], pidC1)
    print("SML-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 36.29, 86.88, 94.18], lambda x: hypoGlicemia(70)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 36.29, 86.88, 94.18], hypoGlicemia(70), pidC1))

    plt.plot(t, y[:, 0], label='SML',color='b')

    t, y = simulation([300, 300, 1400 - 300 - 300], [39.1, 125.1, 60.65], pidC1)
    print("STL-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 39.1, 125.1, 60.65], lambda x: hypoGlicemia(70)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 39.1, 125.1, 60.65], hypoGlicemia(70), pidC1))

    plt.axhline(y=70, color='k', linestyle='-')
    plt.plot(t, y[:, 0], label='STL',color='r')
    # plt.legend()
    plt.legend(loc='upper right', labelspacing=0, borderpad=0, fontsize=20)
    plt.savefig('printPidC1FlasificationHypo70FIXED')
    plt.show()


def printPidC1FlasificationHypo70():
    t, y = simulation([228, 399, 1400 - 228 - 399], [30.16, 117, 68.77], pidC1)

    plt.plot(t, y[:, 0], label='SML',color='b')

    t, y = simulation([330, 337, 1400 - 330 - 337], [44.1, 116.9, 50.09], pidC1)
    plt.axhline(y=70, color='k', linestyle='-')
    plt.plot(t, y[:, 0], label='STL',color='r')
    plt.legend()
    plt.savefig('printPidC1FlasificationHypo70')
    plt.show()



# vSML [300, 300, 840, 59.83936429575982, 94.84665477721562, 85.44703763705998]
# vSML time Violation 215.6
# vSML space Violation -40.045665756
# vSTL [300, 300, 840, 36.690820906878955, 115.61229751503811, 59.44655916098702]
# vSTL time Violation 110.4
# vSTL space Violation -86.7666129019

def printPidC1FlasificationfixedHyper180_p70():
    t, y = simulation([300, 300, 840], [59.83936429575982, 94.84665477721562, 85.44703763705998], pidC1)
    print("SML-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 59.83936429575982, 94.84665477721562, 85.44703763705998],
                         lambda x: hyperGlicemia(180)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 36.29, 86.88, 94.18], hyperGlicemia(180), pidC1))

    plt.plot(t, y[:, 0], label='SML', color='b')
    t, y = simulation([300, 300, 840], [36.690820906878955, 115.61229751503811, 59.44655916098702], pidC1)
    print("STL-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 36.690820906878955, 115.61229751503811, 59.44655916098702],
                         lambda x: hyperGlicemia(180)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 36.690820906878955, 115.61229751503811, 59.44655916098702],
                          hyperGlicemia(180), pidC1))
    plt.axhline(y=180, color='k', linestyle='-')
    plt.plot(t, y[:, 0], label='STL', color='r')
    plt.legend(loc='upper right', labelspacing=0, borderpad=0, fontsize=20)
    plt.savefig('printPidC1FlasificationHyper70')
    plt.show()


def printPidC1FlasificationfixedHyper180_p702():
    t, y = simulation([300, 300, 840], [300, 300, 840, 14.745483910545722, 89.78098679980081, 79.81647263283439], pidC1)
    print("SML2-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 14.745483910545722, 89.78098679980081, 79.81647263283439],
                         lambda x: hyperGlicemia(180)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 14.745483910545722, 89.78098679980081, 79.81647263283439],
                          hyperGlicemia(180), pidC1))

    plt.plot(t, y[:, 0], label='SML', color='b')
    t, y = simulation([300, 300, 840], [56.902960373467124, 89.7764259713761, 76.46204712860708], pidC1)
    print("STL-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 56.902960373467124, 89.7764259713761, 76.46204712860708],
                         lambda x: hyperGlicemia(180)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 56.902960373467124, 89.7764259713761, 76.46204712860708],
                          hyperGlicemia(180), pidC1))
    plt.axhline(y=180, color='k', linestyle='-')
    plt.plot(t, y[:, 0], label='STL', color='r')
    plt.legend(loc='upper right', labelspacing=0, borderpad=0, fontsize=20)
    plt.savefig('printPidC1FlasificationHyper7000')
    plt.show()
    # vSML2[300, 300, 840, 32.71264989821074, 94.10286887354053, 88.58499812278913]
    # vSML2
    # time
    # Violation
    # 68.2
    # vSML2
    # space
    # Violation - 19.0798070243



def plot():
    t, y = simulation([300, 300, 1400 - 300 - 300], [36.29, 86.88, 94.18], pidC1)
    print("SML-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 36.29, 86.88, 94.18], lambda x: hypoGlicemia(70)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 36.29, 86.88, 94.18], hypoGlicemia(70), pidC1))

    plt.plot(t, y[:, 0], label='SML', color='b')

    t, y = simulation([300, 300, 1400 - 300 - 300], [39.1, 125.1, 60.65], pidC1)
    print("STL-------------")
    print(violationTimeS([300, 300, 1400 - 300 - 300, 39.1, 125.1, 60.65], lambda x: hypoGlicemia(70)(x), pidC1))
    print(violationSpaceS([300, 300, 1400 - 300 - 300, 39.1, 125.1, 60.65], hypoGlicemia(70), pidC1))

    plt.axhline(y=70, color='k', linestyle='-')
    plt.plot(t, y[:, 0], label='STL', color='r')
    plt.legend()
    plt.savefig('printPidC1FlasificationHypo70FIXED')
    plt.show()


# printPidC1FlasificationHypo70FIXED()
# printPidC1FlasificationfixedHyper180_p70()
# printPidC1FlasificationfixedHyper180_p702

t, y = simulation([300, 300, 840], [69.59089636408751, 95.03987167490442, 60.94354520310543], pidC1)
plt.plot(t[:-2000], y[:-2000, 0], label='gauss', color='b')
t, y = simulation([300, 300, 840], [51.9428755377552, 75.69758196233069, 83.92426841287391], pidC1)
plt.axhline(y=180, color='k', linestyle='-')
plt.plot(t[:-2000], y[:-2000, 0], label='flat', color='r')
plt.legend(loc='upper right', labelspacing=0, borderpad=0, fontsize=20)
plt.savefig('printPidC1FlasificationHyper7000')
plt.show()
