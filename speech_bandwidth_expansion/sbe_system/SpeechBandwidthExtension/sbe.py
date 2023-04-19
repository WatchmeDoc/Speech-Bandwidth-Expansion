import librosa
import numpy as np


class SpeechBandwidthExtension:
    def __init__(self, lpc_order=8):
        self._lpc_order = lpc_order

    def produce_wideband_speech(self, S_nb: np.ndarray) -> np.ndarray:
        """
        Applies Speech Bandwidth Extension as described on the patent to the input narrowband speech signal.
        :param S_nb: Input Narrowband Speech Signal Frame
        :return: Output Wideband Speech Signal
        """
        a_nb = self._lpc_analysis(S_nb=S_nb)
        A_nb = self._area_coeff_computation(a_nb=a_nb)
        A_wb = self._area_shifted_interpolation(A_nb=A_nb)

        S_tilde = self._signal_interpolation(S_nb=S_nb)
        A_nb_2 = self._square_up(a_nb=a_nb)
        a_wb = self._area_coeff_to_lpc(A_wb=A_wb)

        r_nb = abs(self._inverse_filtering(S_tilde=S_tilde, A_nb_2=A_nb_2))
        r_wb = self._extract_frame_mean(r_nb=r_nb)
        Y_wb = self._wideband_lpc_synthesis(r_wb=r_wb, a_wb=a_wb)
        S_hb = self._hpf_and_gain(Y_wb=Y_wb)

        S_wb = S_hb + S_tilde
        return S_wb

    def _lpc_analysis(self, S_nb: np.ndarray):
        """
        Applies LPC analysis to the input Speech Signal
        :param S_nb: Input Narrowband Speech Signal
        :return: Narrowband Speech Linear Prediction Coefficients a_nb
        """
        a = librosa.lpc(S_nb, order=self._lpc_order)
        a_nb = np.hstack([[0], -1 * a[1:]])
        return a_nb

    def _area_coeff_computation(self, a_nb):
        """
        Computes area coefficients from the Linear Prediction Coefficients
        :param a_nb: Input Narrowband speech linear prediction coefficients
        :return: Narrowband Area Coefficients A_nb
        """
        pass

    def _area_shifted_interpolation(self, A_nb):
        """
        Interpolates the input Area Coefficients
        :param A_nb: Area Coefficients 1xm vector
        :return: Area Coefficients 1x2m vector (interpolated)
        """
        pass

    def _signal_interpolation(self, S_nb):
        """
        Interpolates the input narrowband speech signal
        :param S_nb: Input Narrowband Speech Signal
        :return: Interpolated Narrowband Speech Signal
        """
        pass

    def _square_up(self, a_nb):
        """
        Squares up the linear prediction coefficients
        :param a_nb: Narrowband Speech Signal LPCs
        :return: A(z^2)
        """
        pass

    def _area_coeff_to_lpc(self, A_wb):
        """
        Computes LPC parameters from Area coefficients
        :param A_wb: Interpolated Area Coefficients
        :return: Wideband LPC parameters
        """
        pass

    def _inverse_filtering(self, S_tilde, A_nb_2):
        """
        Applies inverse filtering on the interpolated narrowband speech signal
        :param S_tilde: Interpolated Narrowband Speech Signal
        :param A_nb_2: Squared Up Narrowband Speech Signal LPCs
        :return: Interpolated Narrowband excitation (or residual) signal
        """
        pass

    def _extract_frame_mean(self, r_nb):
        """
        Removes frame mean from the input Narrowband residual signal
        :param r_nb: Narrowband residual signal
        :return: Wideband residual signal
        """
        pass

    def _wideband_lpc_synthesis(self, r_wb, a_wb):
        """
        Using the wideband residual signal and the wideband LP coefficients synthesizes a wideband signal
        :param r_wb: Wideband residual signal
        :param a_wb: Wideband Linear Prediction Coefficients
        :return: Y_wb
        """
        pass

    def _hpf_and_gain(self, Y_wb):
        """
        Applies High-Pass filter and computes Gain to the input wideband speech signal
        :param Y_wb:
        :return: High-band speech signal
        """
        pass
