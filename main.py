from time import sleep
import librosa
import numpy as np
import sounddevice as sd
from speech_bandwidth_expansion.sbe_system.SpeechBandwidthExtension.sbe import SpeechBandwidthExtension
import matplotlib.pyplot as plt
import scipy.signal as ss
import soundfile as sf

if __name__ == "__main__":
    file = 'arctic_bdl1_snd_norm'
    filepath = f"data/TIMIT/{file}.wav"

    sbe = SpeechBandwidthExtension(filepath, window_length=80)
    sig, fs = sbe.produce_wideband_speech()
    S_wb, fs_wb = librosa.load(filepath, sr=16000)
    S_nb, fs_nb = sbe.S_nb, sbe.fs_nb
    sig_interp, fs_interp = sbe.upsample_signal()

    sd.play(sig, fs)
    sf.write(str(f'output_speechfiles/{file}/sbe1.wav'), sig, fs)
    sleep(len(sig) / fs + 1)
