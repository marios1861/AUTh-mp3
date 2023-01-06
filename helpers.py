import numpy as np
from scipy.fft import dct, idct
from scipy.sparse import lil_matrix, csr_matrix


def frameDCT(Y: np.ndarray) -> np.ndarray:
    return dct(Y.reshape((-1, 1)), axis=0)


def DCTpower(c: np.ndarray) -> np.ndarray:
    return 10 * np.log10(c**2)


def Dksparse(Kmax: int) -> csr_matrix:
    D = lil_matrix((Kmax, Kmax))

    def getDeltaK(k):
        if k <= 2:
            return 1
        elif k > 2 and k < 282:
            return 2
        elif k >= 282 and k < 570:
            return 14
        else:
            return 28

    for i in range(Kmax):
        deltaK = getDeltaK(i)
        if i - deltaK < 0:
            D[i, 0 : i + deltaK + 1] = 1
        elif i + deltaK >= Kmax:
            D[i, i - deltaK : Kmax] = 1
        else:
            D[i, i - deltaK : i + deltaK + 1] = 1
    return D.tocsr()


def STinit(c: np.ndarray, D: csr_matrix) -> np.ndarray:
    power = DCTpower(c)
    ST = []
    B = D.tocoo()
    for i, j in zip(B.row, B.col):
        if j >= i - 1 and j <= i + 1:
            D[i, j] = power[j]
        else:
            D[i, j] = power[j] + 7
    for i, k in enumerate(D.argmax(axis=1)):
        if i == k:
            ST.append(i)
    return np.array(ST)


def Hz2Barks(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def iframeDCT(c: np.ndarray, N: int, M: int) -> np.ndarray:
    return idct(c, axis=0).reshape((N, M))


a = np.random.rand(10, 10)
b = frameDCT(a)
D = Dksparse(10 * 10 - 1)
print(STinit(b, D))
import matplotlib.pyplot as plt

plt.figure()
plt.spy(D)
plt.show()
