from math import floor
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import upfirdn
from tqdm import tqdm
import numpy as np

from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
from psycho import Dksparse, frameDCT, iframeDCT, psycho
from quantum import RLD, RLE, all_bands_dequantizer, all_bands_quantizer, huff, ihuff


def MP3codec(
    wavin: Path, h: np.ndarray, M: int, N: int, calc_SNR: bool = False
) -> np.ndarray:
    _, data = wavfile.read(wavin)
    HFREQS, BITS, SFS = MP3coder(wavin, h, M, N)
    xhat = MP3decoder(h, M, N, HFREQS, BITS, SFS)

    # scale back xhat and fix time delay caused by filter size
    xhat *= data.max() / xhat.max()
    xhat = xhat[511:-512].astype(np.int16)

    if calc_SNR:

        def signalPower(x):
            return np.mean(x**2)

        def SNR(signal, noise):
            powS = signalPower(signal)
            powN = signalPower(noise)
            return 10 * np.log10(abs((powS - powN) / powN))

        signal = data[: xhat.size]
        # this is assumed to be the noise of the original signal data
        noise: np.ndarray = signal - xhat
        print(f"{SNR(signal.astype(np.float32), noise.astype(np.float32))=} dB")

    return xhat


def MP3coder(
    wavin: Path,
    h: np.ndarray,
    M: int,
    N: int,
    save_file: Path = Path("./huff_frames.bin"),
) -> np.ndarray:
    sample_rate, data = wavfile.read(wavin)
    H = make_mp3_analysisfb(h, M)
    Y = [upfirdn(h_i, data, down=M) for h_i in H.T]
    framed_data = [
        [y[i * N : N * (i + 1)] for y in Y] for i in range(0, floor(Y[0].size / N))
    ]
    D = Dksparse(M * N - 1)
    HFREQS = []
    SFS = []
    BITS = []
    with open("./huff_frames.bin", "w") as f:
        for frame in tqdm(framed_data):
            frame = np.vstack(frame).T
            Yc = frameDCT(frame)
            Tg = psycho(Yc, D)
            syms, sf, bits = all_bands_quantizer(Yc, Tg)
            SFS.append(sf)
            BITS.append(bits)
            rle = RLE(syms, len(syms))
            huff_vec, huff_freq = huff(rle)
            HFREQS.append(huff_freq)
            f.write(huff_vec + "\n")

    return HFREQS, BITS, SFS


def MP3decoder(
    h: np.ndarray,
    M: int,
    N: int,
    HFREQS: list,
    BITS: list,
    SFS: list,
    save_file: Path = Path("./huff_frames.bin"),
) -> np.ndarray:
    G = make_mp3_synthesisfb(h, M)

    Yh_tot = []
    with open("./huff_frames.bin", "r") as f:
        for huff_vec, huff_freq, bits, sf in tqdm(zip(f, HFREQS, BITS, SFS)):
            rle = ihuff(huff_vec, huff_freq)
            syms = RLD(rle, M * N)
            xh = all_bands_dequantizer(syms, bits, sf)
            Yh = iframeDCT(xh, N, M)
            Yh_tot.append(Yh)
    Yh_tot = np.vstack(Yh_tot)
    R = np.vstack([upfirdn(g_i, y_i, up=M) for y_i, g_i in zip(Yh_tot.T, G.T)]).T
    xhat = np.sum(R, axis=1)
    return xhat
