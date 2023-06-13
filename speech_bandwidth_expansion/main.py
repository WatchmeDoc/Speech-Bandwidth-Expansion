from time import sleep

import librosa
import sounddevice as sd
from sbe_system.SpeechBandwidthExtension.sbe import SpeechBandwidthExtension

if __name__ == "__main__":
    filepath = "../data/GeorgeManos/sample2.wav"
    sbe = SpeechBandwidthExtension(filepath)
    sig, fs = sbe.produce_wideband_speech()
    # Narrowband signal:
    sd.play(sbe.S_nb, sbe.fs_nb)
    sleep(len(sbe.S_nb) / sbe.fs_nb + 1)

    S_wb, fs_wb = librosa.load(filepath, sr=16000)

    # Wideband signal:
    sd.play(S_wb, fs_wb)
    sleep(len(S_wb) / fs_wb + 1)

    # Reconstructed wideband:
    sd.play(sig, fs)
    sleep(len(sig) / fs + 1)
