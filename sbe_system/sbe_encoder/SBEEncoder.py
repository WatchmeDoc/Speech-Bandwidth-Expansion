class SBEEncoder:

    def __init__(self):
        pass

    def encode_speech(self, S_wb):
        """
        Applies Speech Bandwidth Extension encoding to the provided wideband speech signal.
        :param S_wb: Wide-band speech signal
        :return: Coefficients in LSF representation, Gain
        """
        # Upper branch:
        S_nb = self._decimation(S_wb=S_wb)
        e_nb = self._nb_analysis(S_nb=S_nb)
        e_hat = self._wideband_excitation_generation(e_nb=e_nb)

        # Middle branch:
        a_hb = self._selective_lp(S_wb=S_wb)
        w_hb = self._lpc_to_lsf(a_hb=a_hb)
        w_hat = self._lsf_quantization(w_hb=w_hb)
        a_hat = self._wideband_lpc_codebook(w_hb=w_hat)
        S_tilde = self._wideband_synthesis(a_wb=a_hat, e_wb=e_hat)

        # Lower Branch
        g_hb = self._gain_estimation(S_wb=S_wb, S_tilde=S_tilde)
        g_hat = self._gain_quantization(g_hb=g_hb)

        return w_hat, g_hat

    def _decimation(self, S_wb):
        """

        :param S_wb: Wide-band speech signal
        :return: Narrow-band speech signal
        """
        return None

    def _nb_analysis(self, S_nb):
        """

        :param S_nb: Narrow-band speech signal
        :return:
        """
        return None

    def _wideband_excitation_generation(self, e_nb):
        """

        :param e_nb: Narrow-band excitation signal
        :return: Reconstructed wideband excitation signal
        """
        return None

    def _selective_lp(self, S_wb):
        """

        :param S_wb: Wide-band speech signal
        :return: High-band LP coefficients
        """
        return None

    def _lpc_to_lsf(self, a_hb):
        """

        :param a_hb: High-band LP coefficients
        :return: High-band LP coefficients in Line Spectral Frequencies (LSF) representation
        """
        return None

    def _lsf_quantization(self, w_hb):
        """

        :param w_hb: High-band LP coefficients in LSF representation
        :return: Quantized high-band LP coefficients in LSF representation
        """
        return None

    def _wideband_lpc_codebook(self, w_hb):
        """

        :param w_hb: Quantized high-band LP coefficients in LSF representation
        :return: Wide-band LP coefficients
        """
        return None

    def _wideband_synthesis(self, a_wb, e_wb):
        """

        :param a_wb: Wide-band LP coefficients
        :param e_wb: Reconstructed wideband excitation signal
        :return: Wide-band reconstructed speech signal
        """
        return None

    def _gain_estimation(self, S_wb, S_tilde):
        """

        :param S_wb: Wide-band speech signal
        :param S_tilde: Wide-band reconstructed speech signal
        :return: High-band gain parameter
        """
        return None

    def _gain_quantization(self, g_hb):
        """

        :param g_hb: High-band gain parameter
        :return: Quantized high-band gain parameter
        """
        return None
