import numpy as np
from scipy.fft import dct, idct
from scipy.sparse import coo_matrix, find


def frameDCT(Y: np.ndarray) -> np.ndarray:
    return dct(Y.reshape((-1, 1)), axis=0)


def DCTpower(c: np.ndarray) -> np.ndarray:
    return 10 * np.log10(c**2)


def Dksparse(Kmax: int) -> coo_matrix:
    rows1 = np.arange(0, Kmax)
    columns1 = np.arange(0, Kmax)
    rows2 = np.arange(0, Kmax - 1)
    columns2 = np.arange(1, Kmax)
    rows3 = np.arange(1, Kmax)
    columns3 = np.arange(0, Kmax - 1)
    data = np.ones(3 * Kmax - 2)
    rows = np.hstack([rows1, rows2, rows3])
    cols = np.hstack([columns1, columns2, columns3])
    return coo_matrix((data, (rows, cols)), shape=(Kmax, Kmax))


def STinit(c: np.ndarray, D: coo_matrix) -> np.ndarray:
    (I, J, V) = find(D)
    power = DCTpower(c)

    for (row, column, value) in zip(I, J, V):
        print(value)
        value = 5
        print(D(row, column))

def iframeDCT(c: np.ndarray, N: int, M: int) -> np.ndarray:
    return idct(c, axis=0).reshape((N, M))

a = np.random.rand(5, 5)
b = frameDCT(a)
D = Dksparse(5 * 5 - 1)
STinit(b, D)