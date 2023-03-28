# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 20:41:58 2021

@author: George Kafentzis
"""
import numpy as np
import sys
from scipy.fftpack import fft
import matplotlib.pyplot as plt
from scipy.signal import lfilter
from scipy.fftpack import dct
import librosa as lr

eps = sys.float_info.epsilon


def fbf_feature_extraction(signal, fs, window, step):
    """
    This function implements the shor-term windowing process.
    For each short-term window a set of features is extracted.
    This results to a sequence of feature vectors, stored in a np matrix.
    ARGUMENTS
        signal:         the input signal samples
        fs:             the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:           the short-term window step (in samples)

    RETURNS
        features (numpy.ndarray):        contains features
                                         (n_feats x numOfShortTermWindows)
        feature_names (python list):     contains feature names
                                         (n_feats x numOfShortTermWindows)
    """

    window = int(window)
    step = int(step)

    # signal normalization
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)

    signal = dc_normalize(signal)

    number_of_samples = len(signal)  # total number of samples
    current_position = 0
    count_fr = 0
    NFFT = 1024

    # LPC order
    order = fs / 1000 + 2

    n_mfcc_feats = 13
    n_lpc_feats = order + 1
    n_spectral_feats = 4
    n_total_feats = n_lpc_feats + n_mfcc_feats + n_spectral_feats

    # define list of feature names
    feature_names = ["LPCcoef_{0:d}".format(lpcc_i)
                     for lpcc_i in range(1, order + 2)]
    feature_names += ["F1", "F2", "BW1", "BW2"]
    feature_names += ["mfcc_{0:d}".format(mfcc_i)
                      for mfcc_i in range(1, n_mfcc_feats + 1)]

    # compute the triangular filter banks used in the mfcc calculation
    fbank, freqs = mfcc_filter_banks(sampling_rate=fs, num_fft=NFFT // 2)

    features = []
    # for each short-term window to end of signal
    while current_position + window - 1 < number_of_samples:
        count_fr += 1
        # get current window
        x = signal[current_position:current_position + window]

        # initialize feature vector
        feature_vector = np.zeros((n_total_feats, 1))

        # update window position
        current_position = current_position + step

        # get fft magnitude
        fft_mag = np.abs(fft(x, NFFT))

        # normalize fft
        fft_mag = fft_mag[0:int(NFFT / 2)]
        fft_mag = fft_mag / len(fft_mag)

        # LPC coefffs
        feature_vector[0:n_lpc_feats, 0] = lpc(x, order)
        # formants (first two Formants and their BWs)
        F1F2, BWs = formants(x, order, fs)
        [feature_vector[n_lpc_feats + n_spectral_feats], feature_vector[n_lpc_feats + n_spectral_feats + 1]] = F1F2
        [feature_vector[n_lpc_feats + n_spectral_feats + 2], feature_vector[n_lpc_feats + n_spectral_feats + 3]] = BWs
        # MFCCs
        mffc_feats_end = n_total_feats
        feature_vector[n_total_feats - n_mfcc_feats:mffc_feats_end, 0] = mfcc(fft_mag, fbank, n_mfcc_feats).copy()

        features.append(feature_vector)

    features = np.concatenate(features, 1)

    return features, feature_names


""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""                     		   FEATURES                                      """
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def dc_normalize(sig_array):
    """Removes DC and normalizes to -1, 1 range"""
    sig_array_norm = sig_array.copy()
    sig_array_norm -= sig_array_norm.mean()
    sig_array_norm /= abs(sig_array_norm).max() + 1e-10
    return sig_array_norm


def formants(x, order, fs):
    """Compute formants using LPC method.
    Formants are computed by solving for the LPC polynomial and 
    selecting roots that have positive imaginary part, corresponding to
    poles on the complex plane. 
    Parameters
    ----------
    x: array_like
        input signal
    fs: int
        sampling rate
    """
    N = len(x)
    w = np.hamming(N)

    # Apply window and high pass filter
    x_win = x * w
    x_pre = lfilter([1], [1., 0.63], x_win)

    # Get LPC
    A, e, k = lpc(x_pre, order)

    # Get roots.
    rts = np.roots(A)
    rts = [r for r in rts if np.imag(r) > 0]

    # Get angles and mags
    angz = np.arctan2(np.imag(rts), np.real(rts))
    mags = np.abs(rts)

    # Get frequencies and bandwidths
    frqs = angz * (fs / (2 * np.pi))
    bws = -np.log(mags) * (fs / np.pi)
    bws = [x for _, x in sorted(zip(frqs, bws))]
    frqs = sorted(frqs)

    # return frqs, bws
    return frqs[0:2], bws[0:2]


def lpc(signal, order):
    """Compute the Linear Prediction Coefficients.
    Return the order + 1 LPC coefficients for the signal. c = lpc(x, k) will
    find the k+1 coefficients of a k order linear filter:
      xp[n] = -c[1] * x[n-2] - ... - c[k-1] * x[n-k-1]
    Such as the sum of the squared-error e[i] = xp[i] - x[i] is minimized.
    Parameters
    ----------
    signal: array_like
        input signal
    order : int
        LPC order (the output will have order + 1 items)
    """

    if order > signal.size:
        raise ValueError("Input signal must have a length >= lpc order")

    if order > 0:
        p = order + 1
        r = np.zeros(p, dtype=signal.dtype)
        # Number of non zero values in autocorrelation one needs for p LPC
        # coefficients
        x = np.correlate(signal, signal, 'full')
        r = x[signal.size - 1:signal.size + order]
        a, e, k = levinson(r, order)
        return a, e, k
    else:
        return np.ones(1, dtype=signal.dtype)


def levinson(r, order):
    """Levinson-Durbin recursion, to efficiently solve symmetric linear systems
    with toeplitz structure.
    Parameters
    ---------
        r : array-like
            input array to invert (since the matrix is symmetric Toeplitz, the
            corresponding pxp matrix is defined by p items only). Generally the
            autocorrelation of the signal for linear prediction coefficients
            estimation. The first item must be a non zero real.
    order : integer
            the order of the predictor
    """
    r = np.atleast_1d(r)
    if r.ndim > 1:
        raise ValueError("Only rank 1 are supported for now.")

    n = r.size
    if order > n - 1:
        raise ValueError("Order should be <= size-1")
    elif n < 1:
        raise ValueError("Cannot operate on empty array !")

    if not np.isreal(r[0]):
        raise ValueError("First item of input must be real.")
    elif not np.isfinite(1 / r[0]):
        raise ValueError("First item should be != 0")

    # Estimated coefficients
    a = np.empty(order + 1, r.dtype)
    # temporary array
    t = np.empty(order + 1, r.dtype)
    # Reflection coefficients
    k = np.empty(order, r.dtype)

    a[0] = 1.
    e = r[0]

    for i in range(1, order + 1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i - j]
        k[i - 1] = -acc / e
        a[i] = k[i - 1]

        for j in range(order):
            t[j] = a[j]

        for j in range(1, i):
            a[j] += k[i - 1] * np.conj(t[i - j])

        e *= 1 - k[i - 1] * np.conj(k[i - 1])

    return a, e, k


def mfcc_filter_banks(sampling_rate, num_fft, lowfreq=133.33, linc=200 / 3,
                      logsc=1.0711703, num_lin_filt=13, num_log_filt=27):
    """
    Computes the triangular filterbank for MFCC computation 
    This function is taken from the scikits.talkbox library (MIT Licence):
    https://pypi.python.org/pypi/scikits.talkbox
    """

    if sampling_rate < 8000:
        nlogfil = 5

    # Total number of filters
    num_filt_total = num_lin_filt + num_log_filt

    # Compute frequency points of the triangle:
    frequencies = np.zeros(num_filt_total + 2)
    frequencies[:num_lin_filt] = lowfreq + np.arange(num_lin_filt) * linc
    frequencies[num_lin_filt:] = frequencies[num_lin_filt - 1] * logsc ** \
                                 np.arange(1, num_log_filt + 3)
    heights = 2. / (frequencies[2:] - frequencies[0:-2])

    # Compute filterbank coeff (in fft domain, in bins)
    fbank = np.zeros((num_filt_total, num_fft))
    nfreqs = np.arange(num_fft) / (1. * num_fft) * sampling_rate

    for i in range(num_filt_total):
        low_freqs = frequencies[i]
        cent_freqs = frequencies[i + 1]
        high_freqs = frequencies[i + 2]

        lid = np.arange(np.floor(low_freqs * num_fft / sampling_rate) + 1,
                        np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        dtype=np.int)
        lslope = heights[i] / (cent_freqs - low_freqs)
        rid = np.arange(np.floor(cent_freqs * num_fft / sampling_rate) + 1,
                        np.floor(high_freqs * num_fft / sampling_rate) + 1,
                        dtype=np.int)
        rslope = heights[i] / (high_freqs - cent_freqs)
        fbank[i][lid] = lslope * (nfreqs[lid] - low_freqs)
        fbank[i][rid] = rslope * (high_freqs - nfreqs[rid])

    return fbank, frequencies


def mfcc(fft_magnitude, fbank, num_mfcc_feats):
    """
    Computes the MFCCs of a frame, given the fft mag
    ARGUMENTS:
        fft_magnitude:  fft magnitude abs(FFT)
        fbank:          filter bank 
    RETURN
        ceps:           MFCCs (13 element vector)
    Note:    MFCC calculation is, in general, taken from the 
             scikits.talkbox library (MIT Licence),
    #    with a small number of modifications
    """

    mspec = np.log10(np.dot(fft_magnitude, fbank.T) + eps)
    ceps = dct(mspec, type=2, norm='ortho', axis=-1)[:num_mfcc_feats]
    return ceps
