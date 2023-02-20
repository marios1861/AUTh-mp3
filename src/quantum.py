from typing import Tuple
from matplotlib import pyplot as plt
import numpy as np
from psycho import DCTpower, Dksparse, frameDCT, psycho


def critical_bands(K: np.ndarray) -> np.ndarray:
    """create the critical band dictionary"""
    B = 689
    N = 36
    cb = np.zeros((K), dtype=int)
    for k in range(0, K):
        f = k * B / N
        # band 1
        if f < 100:
            cb[k] = 1
        elif f < 200:
            cb[k] = 2
        elif f < 300:
            cb[k] = 3
        elif f < 400:
            cb[k] = 4
        elif f < 510:
            cb[k] = 5
        elif f < 630:
            cb[k] = 6
        elif f < 770:
            cb[k] = 7
        elif f < 920:
            cb[k] = 8
        elif f < 1080:
            cb[k] = 9
        elif f < 1270:
            cb[k] = 10
        elif f < 1480:
            cb[k] = 11
        elif f < 1720:
            cb[k] = 12
        elif f < 2000:
            cb[k] = 13
        elif f < 2320:
            cb[k] = 14
        elif f < 2700:
            cb[k] = 15
        elif f < 3150:
            cb[k] = 16
        elif f < 3700:
            cb[k] = 17
        elif f < 4400:
            cb[k] = 18
        elif f < 5300:
            cb[k] = 19
        elif f < 6400:
            cb[k] = 20
        elif f < 7700:
            cb[k] = 21
        elif f < 9500:
            cb[k] = 22
        elif f < 12000:
            cb[k] = 23
        elif f < 15500:
            cb[k] = 24
        else:
            cb[k] = 25
    return cb


def DCT_band_scale(c: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    cs = np.zeros((len(c)))
    sc = np.zeros((25))
    cba = critical_bands(len(c))

    def scale_fun(c_val):
        return abs(c_val) ** (3 / 4)

    scale_vec_fun = np.vectorize(scale_fun)
    # for each band find the scale factor
    for i in range(1, 26):
        mask_i = np.equal(cba, i)
        sc[i - 1] = np.amax(scale_vec_fun(c[mask_i]))
    # for each DCT coeff in c, find the encoded coeff
    for i, val in enumerate(c):
        cs[i] = np.sign(val) * scale_fun(val) / sc[cba[i] - 1]
    return cs, sc


def quantizer(x: np.ndarray, b: int) -> np.ndarray:
    wb = 1 / (2 ** (b - 1))
    zones = [((i) * wb, (i + 1) * wb) for i in range(0, 2 ** (b - 1), 1)]
    zones[0] = (-wb, wb)
    reverse = zones[1:]
    reverse = [(-b, -a) for a, b in reverse]
    reverse.reverse()
    zones = reverse + zones
    zones = np.array(zones, dtype=float)
    symbols = np.arange(-(2 ** (b - 1) - 1), 2 ** (b - 1), 1, dtype=int)
    quantized_x = np.zeros_like(x, dtype=int)
    # for each value
    for i, val in enumerate(x):
        # find the interval it belongs to and write the corresponding symbol
        for sym, (l, h) in zip(symbols, zones):
            if (val >= l) and (val <= h):
                quantized_x[i] = sym
    return quantized_x


def dequantizer(quantized_x: np.ndarray, b: int) -> np.ndarray:
    wb = 1 / (2 ** (b - 1))
    zones = [((i) * wb, (i + 1) * wb) for i in range(0, 2 ** (b - 1), 1)]
    zones[0] = (-wb, wb)
    reverse = zones[1:]
    reverse = [(-b, -a) for a, b in reverse]
    reverse.reverse()
    zones = reverse + zones
    zones = np.array(zones, dtype=float)
    symbols = np.arange(-(2 ** (b - 1) - 1), 2 ** (b - 1), 1, dtype=int)
    dequantized_x = np.zeros_like(quantized_x, dtype=float)
    # for each symbol
    for i, val in enumerate(quantized_x):
        # estimate the value as the median of the interval
        for sym, (l, h) in zip(symbols, zones):
            if val == sym:
                dequantized_x[i] = (l + h) / 2
    return dequantized_x


def all_bands_quantizer(
    c: np.ndarray, Tg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    Tg = np.nan_to_num(Tg, nan=np.inf)
    cs, sc = DCT_band_scale(c)
    cba = critical_bands(len(c))
    B = np.zeros_like(sc, dtype=int)
    symbs = np.zeros_like(cs, dtype=int)
    # for each band
    for i, sc_i in enumerate(sc):
        mask_i = np.equal(cba, i + 1)
        c_i = c[mask_i]
        cs_i = cs[mask_i]
        Tg_i = Tg[mask_i]
        # find the required bits
        for bits in range(1, 8):
            quantized_band = quantizer(cs_i, bits)
            dequantized_band = dequantizer(quantized_band, bits)
            coeff = np.sign(dequantized_band) * np.absolute(
                dequantized_band * sc_i
            ) ** (4 / 3)
            errP = DCTpower(np.absolute(coeff - c_i))
            # if bits are enough for the threshold, save the bits used, and the symbols
            if np.all(errP <= Tg_i):
                B[i] = bits
                symbs[mask_i] = quantized_band
                break

    return symbs, sc, B


def all_bands_dequantizer(
    syms: np.ndarray, B: np.ndarray, SF: np.ndarray
) -> np.ndarray:
    cba = critical_bands(len(syms))
    xh = np.zeros_like(syms, dtype=float)
    # for each band
    for i in range(25):
        mask_i = np.equal(cba, i + 1)
        syms_i = syms[mask_i]
        dequantized_band = dequantizer(syms_i, B[i])
        xh[mask_i] = np.sign(dequantized_band) * np.absolute(
            dequantized_band * SF[i]
        ) ** (4 / 3)

    return xh


def main():
    np.random.seed(0)
    a_raw = 100 * np.random.rand(36, 32) - 50
    a = frameDCT(a_raw)[:, 0]
    DCT_band_scale(a)
    # arr = np.vstack((-np.random.rand(5, 1), np.random.rand(5, 1)))
    # print(arr)
    # quantized_arr = quantizer(arr, 8)
    # print(quantized_arr)
    # dequantize_arr = dequantizer(quantized_arr, 8)
    # print(dequantize_arr)
    D = Dksparse(36 * 32 - 1)
    Tg = psycho(a, D)
    syms, sf, bits = all_bands_quantizer(a, Tg)
    ahat = all_bands_dequantizer(syms, bits, sf)
    plt.figure()
    plt.plot(range(len(ahat)), ahat, a)
    plt.show()


if __name__ == "__main__":
    main()
