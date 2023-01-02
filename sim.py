import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import upfirdn
from math import floor
from nothing import donothing, idonothing


def codec0(
    wavin: Path, h: np.ndarray, M: int, N: int, calc_SNR=False
) -> Tuple[np.ndarray, np.ndarray]:
    _, data = wavfile.read(wavin)
    Ytot = coder0(wavin, h, M, N)
    xhat = decoder0(Ytot, h, M, N)

    # scale back xhat and fix time delay caused by filter size
    xhat *= data.max() / xhat.max()
    xhat = xhat[511:-512].astype(np.int16)

    if calc_SNR:
        err: np.ndarray = data[:xhat.size] - xhat
        plt.plot(err, label="Reconstruction error")
        mean = err.mean()
        sd = err.std()
        SNR = 10 * np.log10(abs(mean / sd))
        print(f"{SNR=} dB")

    return (Ytot, xhat)


def coder0(wavin: Path, h: np.ndarray, M: int, N: int) -> np.ndarray:
    sample_rate, data = wavfile.read(wavin)
    H = make_mp3_analysisfb(h, M)
    Y = [upfirdn(h_i, data, down=M) for h_i in H.T]
    framed_data = [
        [y[i * N : N * (i + 1)] for y in Y] for i in range(0, floor(Y[0].size / N))
    ]
    Ytot = []
    for frame in framed_data:
        frame = np.vstack(frame).T
        Yc = donothing(frame)
        Ytot.append(Yc)
    Ytot = np.vstack(Ytot)
    return Ytot


def decoder0(Ytot: np.ndarray, h: np.ndarray, M: int, N: int) -> np.ndarray:
    G = make_mp3_synthesisfb(h, M)
    framed_data = [
        Ytot[i * N : N * (i + 1), :] for i in range(0, floor(Ytot.shape[0] / N))
    ]
    Yh_tot = []
    for frame in framed_data:
        Yh = idonothing(frame)
        Yh_tot.append(Yh)
    Yh_tot = np.vstack(Yh_tot)
    R = np.vstack([upfirdn(g_i, y_i, up=M) for y_i, g_i in zip(Yh_tot.T, G.T)]).T
    xhat = np.sum(R, axis=1)
    return xhat


def main(args):
    M = 32
    N = 36
    sample_rate = 44100
    h = np.load(Path("h.npy"), allow_pickle=True).item()["h"].flatten()

    if args.plot:
        # plot transfer functions of h_i (in dB)
        H = make_mp3_analysisfb(h, M)
        fig1 = plt.figure(num=0)
        fig2 = plt.figure(num=1)
        # calculate frequency axis
        cutoff_db = -30
        h_freq = np.fft.rfftfreq(H.shape[0], 1.0 / sample_rate)
        h_bark = 13 * np.arctan(0.00076 * h_freq) + 3.5 * np.arctan(
            (h_freq / 7500) ** 2
        )
        for i in range(0, M):
            # calculate FFT
            # since signal is real, we use rfft
            h_fft = np.fft.rfft(H[:, i])
            # cutoff at cutoff_db to remove low magnitude noise from plot
            h_dB = np.maximum(20 * np.log10(np.absolute(h_fft)), cutoff_db)
            plt.figure(num=0)
            plt.plot(h_freq, h_dB)
            plt.figure(num=1)
            plt.plot(h_bark, h_dB)
        plt.figure(num=0)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude [dB]")
        fig1.suptitle("Magnitude of transfer functions over Hz", fontsize=14)
        plt.figure(num=1)
        plt.xlabel("Frequency [bark]")
        plt.ylabel("Magnitude [dB]")
        fig2.suptitle("Magnitude of transfer functions over barks", fontsize=14)

        plt.show()

    Ytot, xhat = codec0(Path(args.file), h, M, N, calc_SNR=args.snr)

    plt.legend()
    plt.show()
    wavfile.write(Path(f"modified_{args.file}"), sample_rate, xhat)


def parse_opt():
    parser = argparse.ArgumentParser(description="A wav to mp3 converter")
    parser.add_argument(
        "--file", type=str, default="myfile.wav", help="The wav file to be used"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="if provided, plot transfer functions of filters",
    )
    parser.add_argument("--snr", action="store_true", help="if provided, calculate SNR")

    return parser.parse_args()


if __name__ == "__main__":
    main(parse_opt())
