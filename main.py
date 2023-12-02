import os
from time import sleep

import librosa
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as ss
import sounddevice as sd
import soundfile as sf

from speech_bandwidth_expansion import SpeechBandwidthExtension


def stft_mag(sig, fs):
    return abs(ss.stft(sig, fs)[2])


def spectral_convergence(X, X_hat):
    return np.linalg.norm(x=(X - X_hat), ord="fro") / np.linalg.norm(x=X, ord="fro")


def spectral_distance(X, X_hat):
    return np.linalg.norm(x=(X - X_hat), ord=1)


def evaluate():
    # Metrics
    MSD = spectral_distance(
        X=stft_mag(sig=S_wb, fs=fs_wb), X_hat=stft_mag(sig=sig_interp, fs=fs_interp)
    )
    print("Interpolated Signal MSD:", MSD)
    MSD = spectral_distance(
        X=stft_mag(sig=S_wb, fs=fs_wb), X_hat=stft_mag(sig=sig, fs=fs)
    )
    print("SBE Signal MSD:", MSD)

    SC = spectral_convergence(
        X=stft_mag(sig=S_wb, fs=fs_wb), X_hat=stft_mag(sig=sig_interp, fs=fs_interp)
    )
    print("Interpolated Signal SC:", SC)
    SC = spectral_convergence(
        X=stft_mag(sig=S_wb, fs=fs_wb), X_hat=stft_mag(sig=sig, fs=fs)
    )
    print("SBE Signal SC:", SC)

    MEL = spectral_distance(
        X=librosa.feature.melspectrogram(y=S_wb, sr=fs_wb),
        X_hat=librosa.feature.melspectrogram(y=sig_interp, sr=fs_interp),
    )
    print("Interpolated Signal MEL:", MEL)

    MEL = spectral_distance(
        X=librosa.feature.melspectrogram(y=S_wb, sr=fs_wb),
        X_hat=librosa.feature.melspectrogram(y=sig, sr=fs),
    )
    print("SBE Signal MEL:", MEL)

    t_nb = np.arange(start=0, stop=len(S_nb)) / fs_nb
    t_wb = np.arange(start=0, stop=len(S_wb)) / fs_wb

    plt.figure()
    plt.subplot(2, 2, 1)
    plt.plot(t_nb, S_nb)
    plt.title("Orig (NB)")
    plt.xlabel("Time (s)")

    plt.subplot(2, 2, 2)
    plt.plot(t_wb, S_wb)
    plt.title("Orig (WB)")
    plt.xlabel("Time (s)")

    plt.subplot(2, 2, 3)
    plt.plot(t_wb, sig_interp)
    plt.title("Interpolated")
    plt.xlabel("Time (s)")

    plt.subplot(2, 2, 4)
    plt.plot(t_wb, sig)
    plt.title("Reconstructed")
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(f"plots/" + file_name + "/waveforms.png")
    # %%
    t_wb = np.arange(start=0, stop=len(S_wb)) / fs_wb

    plt.figure()
    plt.plot(t_wb, S_wb, label="Orig (WB)")
    plt.plot(t_wb, sig_interp, label="Interpolated")
    plt.plot(t_wb, sig, label="Reconstructed")
    plt.legend()
    plt.savefig(f"plots/" + file_name + "/waveforms_merged.png")
    plt.show()

    plt.figure()
    plt.subplot(2, 2, 1)
    # Plot the spectrogram
    plt.specgram(S_nb, Fs=fs_nb)
    plt.title("nb_signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 2)
    # Plot the spectrogram
    plt.specgram(S_wb, Fs=fs_wb)
    plt.title("wb_signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 3)
    # Plot the spectrogram
    plt.specgram(sig, Fs=fs)
    plt.title("sbe_signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    # Plot the spectrogram
    plt.specgram(sig_interp, Fs=fs_interp)
    plt.title("interpolated_signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"plots/" + file_name + "/spectrograms.png")
    plt.show()


if __name__ == "__main__":
    data_dir = "data/TIMIT"
    plot_dir = "plots"
    speech_dir = "output_speechfiles"
    for file in os.listdir(data_dir):
        print("-----------------------------------")
        print("Processing file:", file)
        file_name = file.split(".")[0]
        filepath = os.path.join(data_dir, file)

        # Create result directories if they don't exist
        plot_directory = os.path.join(plot_dir, file_name)
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        speech_directory = os.path.join(speech_dir, file_name)
        if not os.path.exists(speech_directory):
            os.makedirs(speech_directory)
        # Speech Bandwidth Expansion
        sbe = SpeechBandwidthExtension(filepath)
        sig, fs = sbe.produce_wideband_speech()
        S_wb, fs_wb = librosa.load(filepath, sr=16000)
        S_nb, fs_nb = sbe.S_nb, sbe.fs_nb
        sig_interp, fs_interp = sbe.upsample_signal()

        # Add zeros to make all signals the same length
        S_wb = np.concatenate([S_wb, np.zeros((abs(len(sig_interp) - len(S_wb))))])
        sig = np.concatenate([sig, np.zeros((abs(len(sig) - len(S_wb))))])
        assert len(sig) == len(S_wb) == len(sig_interp)
        # Listen to the result
        sd.play(sig, fs)
        sleep(len(sig) / fs + 1)
        sf.write(os.path.join(speech_directory, "sbe.wav"), sig, fs)
        evaluate()
