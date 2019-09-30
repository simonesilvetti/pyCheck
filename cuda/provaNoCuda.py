import numpy as np
import time


def VectorAdd(a, b,c):
    for i in range(len(a)):
        c[i]=a[i]+b[i]
def main():
    N = 82000000

    A = np.ones(N, dtype=np.float32)
    B = np.ones(N, dtype=np.float32)
    C = np.ones(N, dtype=np.float32)

    start = time.time()
    VectorAdd(A,B,C)
    #C=A+B
    vector_add_time = time.time() - start

    print ("C[:5] = " + str(C[:5]))
    print ("C[-5:] = " + str(C[-5:]))

    print( "VectorAdd took for % seconds" % vector_add_time)

if __name__=='__main__':
    main()