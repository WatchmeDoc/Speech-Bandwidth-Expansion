import soundfile as sf
import sounddevice as sd
from time import sleep

if __name__ == "__main__":
    sig, fs = sf.read('data/TIMIT_new/timit_speech_db/timit/train/dr1/fcjf0/sa1.wav')
    sd.play(sig, fs)
    sleep(len(sig) / fs)
