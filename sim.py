from pathlib import Path
import numpy as np
from mp3 import make_mp3_analysisfb, make_mp3_synthesisfb
import matplotlib.pyplot as plt


def main():
    M = 32
    h = np.load(Path("h.npy"), allow_pickle=True).item()['h'].flatten()
    H = make_mp3_analysisfb(h, M)
    G = make_mp3_synthesisfb(h, M)
    # plot transfer functions of h_i (in dB)
    fig1 = plt.figure(num=0)
    fig2 = plt.figure(num=1)
    cutoff_db = -30
    # calculate frequency axis
    sample_rate = 44100
    h_freq = np.fft.rfftfreq(H.shape[0], 1. / sample_rate)
    h_bark = 13 * np.arctan(0.00076 * h_freq) + 3.5 * np.arctan((h_freq / 7500) ** 2) 
    print(h_bark.shape, h_freq.shape)
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


if __name__ == "__main__":
    main()
