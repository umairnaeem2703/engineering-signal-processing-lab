"""
Reusable spectral analysis utilities for Stage 2:
Frequency Analysis using FFT.

Includes:
- single-sided FFT amplitude spectrum
- phase spectrum
- peak detection
- Welch PSD
- harmonic detection
"""

import numpy as np
from scipy.signal import find_peaks, welch


def single_sided_fft_amplitude(x, fs):
    """
    Compute the single-sided FFT amplitude spectrum.

    Parameters
    ----------
    x : array-like
        Time-domain signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs_pos : ndarray
        Positive frequency axis in Hz.
    amplitude : ndarray
        Single-sided amplitude spectrum.
    """
    x = np.asarray(x)
    N = len(x)

    if N == 0:
        raise ValueError("Input signal x must not be empty.")

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    positive_mask = freqs >= 0
    freqs_pos = freqs[positive_mask]
    X_pos = X[positive_mask]

    amplitude = np.abs(X_pos) / N

    # For a real-valued signal, the negative-frequency half is the mirror
    # of the positive-frequency half. Since we keep only the positive half,
    # we double non-DC amplitudes.
    if len(amplitude) > 2:
        amplitude[1:] *= 2

    return freqs_pos, amplitude


def single_sided_fft_phase(x, fs):
    """
    Compute the single-sided FFT phase spectrum.

    Parameters
    ----------
    x : array-like
        Time-domain signal.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    freqs_pos : ndarray
        Positive frequency axis in Hz.
    phase : ndarray
        Phase spectrum in radians.
    """
    x = np.asarray(x)
    N = len(x)

    if N == 0:
        raise ValueError("Input signal x must not be empty.")

    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(N, d=1 / fs)

    positive_mask = freqs >= 0

    freqs_pos = freqs[positive_mask]
    X_pos = X[positive_mask]

    phase = np.angle(X_pos)

    return freqs_pos, phase


def detect_frequency_peaks(freqs, amplitude, min_height=0.1, min_distance=None):
    """
    Detect dominant peaks in an amplitude spectrum.

    Parameters
    ----------
    freqs : array-like
        Frequency axis in Hz.
    amplitude : array-like
        Amplitude spectrum.
    min_height : float
        Minimum peak height.
    min_distance : int or None
        Minimum number of samples between neighboring peaks.

    Returns
    -------
    peak_freqs : ndarray
        Frequencies of detected peaks.
    peak_amps : ndarray
        Amplitudes of detected peaks.
    properties : dict
        Properties returned by scipy.signal.find_peaks.
    """
    freqs = np.asarray(freqs)
    amplitude = np.asarray(amplitude)

    peak_indices, properties = find_peaks(
        amplitude,
        height=min_height,
        distance=min_distance,
    )

    peak_freqs = freqs[peak_indices]
    peak_amps = amplitude[peak_indices]

    return peak_freqs, peak_amps, properties


def compute_psd_welch(x, fs, nperseg=512, window="hann"):
    """
    Estimate power spectral density using Welch's method.

    Parameters
    ----------
    x : array-like
        Time-domain signal.
    fs : float
        Sampling frequency in Hz.
    nperseg : int
        Length of each Welch segment.
    window : str
        Window type passed to scipy.signal.welch.

    Returns
    -------
    freqs : ndarray
        Frequency axis in Hz.
    psd : ndarray
        Power spectral density values.
    """
    x = np.asarray(x)

    if len(x) == 0:
        raise ValueError("Input signal x must not be empty.")

    freqs, psd = welch(
        x,
        fs=fs,
        window=window,
        nperseg=nperseg,
    )

    return freqs, psd


def detect_harmonics(peak_freqs, peak_amps, fundamental=None, tolerance=0.05):
    """
    Detect harmonic relationships among frequency peaks.

    Parameters
    ----------
    peak_freqs : array-like
        Detected peak frequencies in Hz.
    peak_amps : array-like
        Peak amplitudes.
    fundamental : float or None
        Fundamental frequency. If None, the lowest detected peak is used.
    tolerance : float
        Allowed error in frequency ratio.

    Returns
    -------
    harmonics : list of tuples
        Each tuple is:
        (frequency, amplitude, harmonic_number)
    """
    peak_freqs = np.asarray(peak_freqs)
    peak_amps = np.asarray(peak_amps)

    if len(peak_freqs) == 0:
        return []

    sorted_indices = np.argsort(peak_freqs)
    peak_freqs = peak_freqs[sorted_indices]
    peak_amps = peak_amps[sorted_indices]

    if fundamental is None:
        fundamental = peak_freqs[0]

    if fundamental <= 0:
        raise ValueError("Fundamental frequency must be positive.")

    harmonics = []

    for f, a in zip(peak_freqs, peak_amps):
        ratio = f / fundamental
        nearest_integer = round(ratio)
        error = abs(ratio - nearest_integer)

        if nearest_integer >= 1 and error <= tolerance:
            harmonics.append((f, a, nearest_integer))

    return harmonics


def frequency_resolution(fs, n_samples):
    """
    Compute FFT frequency spacing.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    n_samples : int
        Number of samples.

    Returns
    -------
    df : float
        Frequency spacing in Hz.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    return fs / n_samples
