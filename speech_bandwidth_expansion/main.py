from time import sleep

import librosa
import numpy as np
import scipy.signal as ss
import sounddevice as sd

from speech_bandwidth_expansion.sbe_system.utils.features import levinson


def fbf_sequence(filepath, window_length=30, OrderLPC=10):
    sig, fs = librosa.load(filepath, sr=16000)

    sd.play(sig, fs)
    sleep(len(sig) / fs + 1)

    Buffer = 0
    synth_signal = np.zeros(sig.size)

    wl_samples = int(np.ceil(window_length * fs / 1000))
    shift = int(np.floor(wl_samples / 2))
    window = ss.windows.hann(wl_samples + 2, sym=True)[1:-1]

    Lsig = len(sig)
    sig_pos = 0
    save_pos = 0
    Nfr = int(np.floor((Lsig - wl_samples) / shift)) + 1

    for _ in range(Nfr):
        sigLPC = window * sig[sig_pos : sig_pos + wl_samples]

        en = sum(sigLPC**2)
        r: np.ndarray = ss.correlate(sigLPC, sigLPC)
        lags = ss.correlation_lags(wl_samples, wl_samples)
        r: np.ndarray = r[int(len(lags) / 2) :]
        a, _, _ = levinson(r=r, order=OrderLPC)
        G = np.sqrt(sum(a * r[: OrderLPC + 1].T))
        ex = ss.lfilter(a, 1, sigLPC)

        s = ss.lfilter([G], a, ex)
        ens = sum(s**2)
        g = np.sqrt(en / ens)
        s = s * g
        s[:shift] = s[:shift] + Buffer
        synth_signal[save_pos : save_pos + shift] = s[:shift]
        Buffer = s[shift:wl_samples]

        sig_pos += shift
        save_pos += shift

    sd.play(synth_signal, fs)
    sleep(len(synth_signal) / fs + 1)
    return synth_signal


if __name__ == "__main__":
    target_sample_rate = 8000
    filepath = "data/TIMIT_new/timit_speech_db/timit/train/dr1/fcjf0/sa1.wav"
    fbf_sequence(filepath=filepath)
    # sig, fs = librosa.load(filepath)
    # target_size = int(len(sig) * target_sample_rate / fs)
    #
    # sd.play(sig, fs)
    # sleep(len(sig) / fs + 1)
    # sig = ss.resample(sig, target_size)
    # fs = target_sample_rate
    # # sd.play(sig, fs)
    # # sleep(len(sig) / fs + 1)
    # # sbe = SpeechBandwidthExtension(lpc_order=8)
    # # sbe.produce_wideband_speech(S_nb=sig)
    # y, sr = librosa.load(filepath)
    # a = librosa.lpc(y, order=2)
    # b = np.hstack([[0], -1 * a[1:]])
    # y_hat = ss.lfilter(b, [1], y)
    # fig, ax = plt.subplots()
    # ax.plot(y)
    # ax.plot(y_hat, linestyle='--')
    # ax.legend(['y', 'y_hat'])
    # ax.set_title('LP Model Forward Prediction')
    # plt.show()
