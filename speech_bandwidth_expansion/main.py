from sbe_system.SpeechBandwidthExtension.sbe import SpeechBandwidthExtension


if __name__ == "__main__":
    sbe = SpeechBandwidthExtension('../data/GeorgeManos/sample2.wav')
    sig = sbe.produce_wideband_speech()
