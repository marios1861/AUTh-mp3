from typing import Tuple
import numpy as np
from scipy.fft import dct, idct
from scipy.sparse import lil_matrix, csr_matrix


def frameDCT(Y: np.ndarray) -> np.ndarray:
    return dct(Y.reshape((-1, 1)), axis=0)


def iframeDCT(c: np.ndarray, N: int, M: int) -> np.ndarray:
    return idct(c, axis=0).reshape((N, M))


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


def MaskPower(c: np.ndarray, ST: np.ndarray) -> np.ndarray:
    power = DCTpower(c)
    max_j = len(c)
    P_m = np.full_like(ST, fill_value=0, dtype=np.float64)
    for i, k in enumerate(ST):
        P_m[i] = 10 * np.log10(
            np.sum(10 ** (0.1 * power[max(k - 1, 0) : min(k + 1, max_j - 1) + 1]))
        )
    return P_m


def Hz2Barks(f):
    return 13 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500) ** 2)


def STreduction(ST, c, Tq) -> Tuple[np.ndarray, np.ndarray]:
    B = 689
    N = 36
    maskers = MaskPower(c, ST)
    new_ST = []
    new_maskers = []
    # Keep audible maskers
    for i, k in enumerate(ST):
        if maskers[i] >= Tq[k]:
            new_ST.append(k)
            new_maskers.append(maskers[i])

    # Keep the strongest out of two maskers
    new_ST = np.array(new_ST)
    new_maskers = np.array(new_maskers)
    bark_ST = Hz2Barks(new_ST * B / N)
    weak_ST_i = [1]
    while len(weak_ST_i) > 0:
        weak_ST_i.clear()
        for i, bark1 in enumerate(
            bark_ST[:-1]
        ):  # skip final term (will be compared with previous term)
            if i in weak_ST_i:
                continue
            if abs(bark1 - bark_ST[i + 1]) < 0.5:
                if new_maskers[i] < new_maskers[i + 1]:
                    weak_ST_i.append(i)
                else:
                    weak_ST_i.append(i + 1)
        new_ST = np.delete(new_ST, weak_ST_i)
        new_maskers = np.delete(new_maskers, weak_ST_i)
        bark_ST = np.delete(bark_ST, weak_ST_i)

    return (new_ST, new_maskers)


def SpreadFunc(ST: np.ndarray, PM: np.ndarray, Kmax) -> np.ndarray:
    B = 689
    N = 36
    SF = np.full((Kmax + 1, len(ST)), 0, np.float64)
    ST_len = len(ST)
    for i in range(0, Kmax + 1):
        for k in range(0, ST_len):
            dz = Hz2Barks(i * B / N) - Hz2Barks(ST[k] * B / N)
            if dz >= -3 and dz < -1:
                SF[i, k] = 17 * dz - 0.4 * PM[k] + 11
            elif dz >= -1 and dz < 0:
                SF[i, k] = (0.4 * PM[k] + 6) * dz
            elif dz >= 0 and dz < 1:
                SF[i, k] = -17 * dz
            elif dz >= 1 and dz < 8:
                SF[i, k] = (0.15 * PM[k] - 17) * dz - 0.15 * PM[k]
    return SF


def Masking_Thresholds(ST: np.ndarray, PM: np.ndarray, Kmax: np.ndarray) -> np.ndarray:
    B = 689
    N = 36
    SF = SpreadFunc(ST, PM, Kmax)
    ST_len = len(ST)
    TM = np.zeros((Kmax + 1, ST_len))

    for i in range(0, Kmax + 1):
        TM[i, :] = PM - 0.275 * Hz2Barks(ST * B / N) + SF[i, :] - 6.025

    return TM


def Global_Masking_Thresholds(Ti: np.ndarray, Tq: np.ndarray) -> np.ndarray:
    Tg = np.zeros((Ti.shape[0]), np.float64)

    for i in range(0, (Ti.shape[0])):
        Tg[i] = 10 * np.log10(10 ** (0.1 * Tq[i]) + np.sum(10 ** (0.1 * Ti[i, :])))
    return Tg


def psycho(c: np.ndarray, D: np.ndarray) -> np.ndarray:
    Kmax = D.shape[0]
    ST = STinit(c, D)
    Tq = np.load("./Tq.npy")[0, :]
    (ST, PM) = STreduction(ST, c, Tq)
    Ti = Masking_Thresholds(ST, PM, Kmax)
    return Global_Masking_Thresholds(Ti, Tq)


def main():
    np.random.seed(0)
    a = np.random.rand(36, 32)
    c = frameDCT(a)  # 1152 (32 * 36)
    D = Dksparse(36 * 32 - 1)
    # print(DCTpower(b).dtype)
    ST = STinit(c, D)
    # print(ST)
    # print(MaskPower(c, ST))
    Tq = np.load("./Tq.npy")
    Tq = Tq[0, :]
    (ST, PM) = STreduction(ST, c, Tq)
    # print(ST)
    Ti = Masking_Thresholds(ST, PM, 32 * 36 - 1)
    print(Global_Masking_Thresholds(Ti, Tq))
    print(psycho(c, D))
    import matplotlib.pyplot as plt  # noqa

    plt.figure()
    plt.spy(D)
    plt.show()


if __name__ == "__main__":
    main()
