from time import sleep

import librosa
import numpy as np
import sounddevice as sd
from sbe_system.SpeechBandwidthExtension.sbe import SpeechBandwidthExtension
import matplotlib.pyplot as plt

if __name__ == "__main__":
    filepath = "../data/GeorgeManos/sample2.wav"
    sbe = SpeechBandwidthExtension(filepath)
    sig, fs = sbe.produce_wideband_speech()
    S_wb, fs_wb = librosa.load(filepath, sr=16000)

    sig_interp = sbe.upsample_signal()

    # # Narrowband signal:
    # sd.play(sbe.S_nb, sbe.fs_nb)
    # sleep(len(sbe.S_nb) / sbe.fs_nb + 1)

    # # Wideband signal:
    # sd.play(S_wb, fs_wb)
    # sleep(len(S_wb) / fs_wb + 1)

    sig = np.concatenate([sig, np.zeros((abs(len(sig) - len(S_wb))))])

    # # Reconstructed wideband:
    # sd.play(sig, fs)
    # sleep(len(sig) / fs + 1)

    t_nb = np.arange(start=0, stop=len(sbe.S_nb)) / sbe.fs_nb
    t_wb = np.arange(start=0, stop=len(S_wb)) / fs_wb

    plt.figure()
    plt.plot(t_nb, sbe.S_nb, label='Orig (NB)')
    plt.plot(t_wb, S_wb, label='Orig (WB)')
    plt.plot(t_wb, sig_interp, label='Interpolated')
    plt.plot(t_wb, sig, label='Reconstructed')
    plt.legend()
    plt.show()
