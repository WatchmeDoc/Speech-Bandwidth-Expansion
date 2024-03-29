from typing import Tuple

import librosa
import numpy as np
import scipy.signal as ss
from scipy.ndimage.interpolation import shift

from speech_bandwidth_expansion.utils.features import levinson


class SpeechBandwidthExtension:
    def __init__(
        self,
        filepath,
        sr: int = 8000,
        upsample_order=2,
        window_length=60,
        lpc_order=16,
        shift_interpolation=0.25,
        corr_delta=0.01,
        cutoff_freq=3900,
        filter_order=20,
        preemphasis_coefficient=0.67,
    ):
        """
        Speech Bandwidth Expansion System as described by the patent of David Malah
        :param filepath: Path to the input narrowband speech signal
        :param sr: Sampling Rate of the input narrowband speech signal
        :param upsample_order: Upsampling factor
        :param lpc_order: Linear Prediction Coefficients Order
        :param window_length: Window Length in ms
        :param shift_interpolation: Shift interpolation factor
        :param corr_delta: Correlation Delta
        :param cutoff_freq: High-Pass Filter Cutoff Frequency
        :param filter_order: High-Pass Filter Order
        :param preemphasis_coefficient: Preemphasis Coefficient
        """
        self._lpc_order = lpc_order
        self.S_nb, self.fs_nb = librosa.load(filepath, sr=sr)
        self.window_length = window_length
        self._shift_interpolation = shift_interpolation
        self._orig_sr = sr
        self._upsample_order = upsample_order
        self._fc = cutoff_freq
        self._filter_order = filter_order
        self.fs_wb: int = int(self.fs_nb * self._upsample_order)
        self._delta = corr_delta
        self._preemph_coeff = preemphasis_coefficient

    def produce_wideband_speech(self) -> Tuple[np.ndarray, int]:
        """
        Performs frame-by-frame analysis and applies on each frame the bandwidth expansion algorithm as described
        by David Malah.
        :return: Output Wideband Speech Signal
        """
        synth_signal = np.zeros(len(self.S_nb) * self._upsample_order)

        wl_samples = int(np.ceil(self.window_length * self.fs_nb / 1000))
        shift = int(np.floor(wl_samples / 2))
        window = ss.windows.hann(wl_samples + 2, sym=True)[1:-1]
        Buffer = 0

        Lsig = len(self.S_nb)
        sig_pos = 0
        save_pos = 0
        Nfr = int(np.floor((Lsig - wl_samples) / shift)) + 1
        wb_shift = shift * self._upsample_order

        for _ in range(Nfr):
            sigslice = self.S_nb[sig_pos : sig_pos + wl_samples]
            sigLPC = window * sigslice
            s_wb = self._d_malah_algorithm(S_nb=sigLPC)

            # Overlap Add
            s_wb[:wb_shift] = s_wb[:wb_shift] + Buffer
            synth_signal[save_pos : save_pos + wb_shift] = s_wb[:wb_shift]
            Buffer = s_wb[wb_shift : wb_shift + wl_samples]
            sig_pos += shift
            save_pos += wb_shift
        return synth_signal, self.fs_wb

    def upsample_signal(self):
        """
        Simply interpolates without bandwidth expansion algorithm the configured speech signal file
        """
        return self._signal_interpolation(S_nb=self.S_nb), self.fs_wb

    def _d_malah_algorithm(self, S_nb: np.ndarray) -> np.ndarray:
        """
        Applies Speech Bandwidth Extension as described on the patent to the input narrowband speech signal.
        :param S_nb: Input Narrowband Speech Signal Frame
        :return: Output Wideband Speech Signal Frame
        """
        sig_preemph = librosa.effects.preemphasis(S_nb, coef=self._preemph_coeff)
        ex, a_nb, k_nb, G = self._lpc_analysis(S_nb=sig_preemph)
        r_nb = -k_nb
        A_nb = self._area_coeff_computation(r_nb=r_nb)
        A_wb = self._area_shifted_interpolation(A_nb=A_nb)

        S_tilde = self._signal_interpolation(S_nb=S_nb)
        A_nb_2 = self._square_up(a_nb=a_nb)
        a_wb = self._area_coeff_to_lpc(A_wb=A_wb)

        r_nb = self._inverse_filtering(S_tilde=S_tilde, A_nb_2=A_nb_2)
        r_wb = self._extract_frame_mean(r_nb=abs(r_nb))
        Y_wb = self._wideband_lpc_synthesis(r_wb=r_wb, a_wb=a_wb)
        S_hb = self._hpf_and_gain(Y_wb=Y_wb, G_wb=G)

        S_wb = S_hb + S_tilde
        return S_wb

    def _lpc_analysis(self, S_nb: np.ndarray):
        """
        Applies LPC analysis to the input Speech Signal
        :param S_nb: Input Narrowband Speech Signal
        :return: ex, a, k, G: excitation signal ex, LPC Coefficients a, reflection coefficients k, gain G
        """

        r: np.ndarray = ss.correlate(S_nb, S_nb)
        r: np.ndarray = r[int(len(r) / 2) :]
        r[0] += 1 + self._delta
        a, _, k = levinson(r=r, order=self._lpc_order)
        G = np.sqrt(sum(a * r[: self._lpc_order + 1].T))
        ex = ss.lfilter(a, 1, S_nb)

        return ex, a, k, G

    def _area_coeff_computation(self, r_nb):
        """
        Computes area coefficients from the Linear Prediction Coefficients
        :param r_nb: Input Narrowband speech partial correlation coefficients (parcors)
        :return: Narrowband Log-Area Coefficients A_nb
        """
        return np.log((1 - r_nb) / (1 + r_nb))

    def _area_shifted_interpolation(self, A_nb):
        """
        Interpolates the input Area Coefficients
        :param A_nb: Area Coefficients 1xm vector
        :return: Area Coefficients cubic spline shifted vector (interpolated)
        """
        return shift(input=A_nb, shift=self._shift_interpolation)

    def _signal_interpolation(self, S_nb):
        """
        Upsamples the input narrowband speech signal
        :param S_nb: Input Narrowband Speech Signal
        :return: Interpolated Narrowband Speech Signal
        """
        return ss.resample(S_nb, len(S_nb) * self._upsample_order)

    def _square_up(self, a_nb):
        """
        Squares up the linear prediction coefficients
        :param a_nb: Narrowband Speech Signal LPCs
        :return: A(z^2)
        """
        return np.insert(arr=a_nb, obj=list(range(1, len(a_nb))), values=0)

    def _area_coeff_to_lpc(self, A_wb):
        """
        Computes LPC parameters from Log-Area coefficients
        :param A_wb: Interpolated Log-Area Coefficients
        :return: Wideband LPC parameters
        """
        N = len(A_wb)
        LPCMatrix = np.zeros(shape=(N, N))
        k = (1 - np.exp(A_wb)) / (1 + np.exp(A_wb))
        for i in range(N):
            LPCMatrix[i, i] = k[i]
            for j in range(i):
                LPCMatrix[i, j] = LPCMatrix[i - 1, j] - k[i] * LPCMatrix[i - 1, i - j]
        a_wb = (-1) * LPCMatrix[N - 1, 1:]
        a_wb = np.insert(arr=a_wb, obj=0, values=1)
        return a_wb

    def _inverse_filtering(self, S_tilde, A_nb_2):
        """
        Applies inverse filtering on the interpolated narrowband speech signal
        :param S_tilde: Interpolated Narrowband Speech Signal
        :param A_nb_2: Squared Up Narrowband Speech Signal LPCs
        :return: Interpolated Narrowband excitation (or residual) signal
        """
        return ss.lfilter(b=A_nb_2, a=1, x=S_tilde)

    def _extract_frame_mean(self, r_nb):
        """
        Removes frame mean from the input Narrowband residual signal
        :param r_nb: Narrowband residual signal
        :return: Wideband residual signal
        """
        return r_nb - np.mean(r_nb)

    def _wideband_lpc_synthesis(self, r_wb, a_wb):
        """
        Using the wideband residual signal and the wideband LP coefficients synthesizes a wideband signal
        :param r_wb: Wideband residual signal
        :param a_wb: Wideband Linear Prediction Coefficients
        :return: Y_wb
        """
        return ss.lfilter(b=[1.0], a=a_wb, x=r_wb)

    def _hpf_and_gain(self, Y_wb, G_wb):
        """
        Applies High-Pass butterworth filter and multiplied by Gain value to the input wideband speech signal
        :param Y_wb: Wideband generated Speech Signal Frame
        :param G_wb: Gain resulted from LPC Analysis
        :return: High-band speech signal
        """

        nyq = 0.5 * self.fs_wb
        normal_cutoff = self._fc / nyq
        b, a = ss.butter(self._filter_order, normal_cutoff, btype="high", analog=False)
        return G_wb * ss.filtfilt(b, a, Y_wb)
